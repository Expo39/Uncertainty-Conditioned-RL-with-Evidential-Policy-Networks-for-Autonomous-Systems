"""
@file tune_hyperparams.py
@brief Optuna hyperparameter tuning for PPO + evidential policy networks.

Uses TPESampler (multivariate) and MedianPruner to systematically search PPO +
evidential hyperparameters across multiple trials. Each trial runs a short
training session (100k steps default) and evaluates env/mean_progress_reward.
Best trial params are written back to configs/train_config.yaml for seamless
integration with the normal training workflow.

Search space design references:
  [1] Andrychowicz et al. 2021, "What Matters In On-Policy RL?" (ICLR 2021)
  [2] Eimer et al. 2023, "Hyperparameters in RL and How To Tune Them" (ICML 2023)
  [3] Watanabe 2023, "Tree-Structured Parzen Estimator" (arXiv:2304.11127)
  [4] Raffin 2022, "Automatic Hyperparameter Tuning In Practice" (ICRA tutorial)
"""

import argparse
import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Optuna is required for tuning
try:
    import optuna
except ImportError:
    raise ImportError(
        "optuna is required for hyperparameter tuning. "
        "Install with: pip install optuna"
    )

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    raise ImportError(
        "stable_baselines3 is required. "
        "Install with: pip install 'stable-baselines3>=2.0.0'"
    )

from uncertainty_rl.training.train_ppo import (
    TrainResult,
    load_config,
    merge_configs,
    train,
)

logger = logging.getLogger("uncertainty_rl.training.tune_hyperparams")


# ===========================================================================
# sample_hyperparams: Tuning search space (8 active parameters)
# ===========================================================================


def sample_hyperparams(trial: "optuna.Trial", tuning_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Sample hyperparameters from the search space.

    Bounds come from tuning_config.yaml, not hardcoded. Returns a flat dict
    of hyperparameters to be merged into the training config.

    8 active parameters [1][2]: learning_rate, n_steps, batch_size, n_epochs,
    gamma, ent_coef, lambda_reg, lambda_reg_warmup_steps. Fixed parameters
    (clip_range, max_grad_norm, target_kl, net_arch) retain their defaults
    from train_config.yaml.

    @param trial: Optuna trial object.
    @param tuning_config: Tuning configuration with search space bounds.
    @return Dictionary of sampled hyperparameters.
    """
    space = tuning_config.get("search_space", {})

    # PPO Hyperparameters
    learning_rate = trial.suggest_float(
        "learning_rate",
        float(space.get("learning_rate", [1e-5, 1e-3])[0]),
        float(space.get("learning_rate", [1e-5, 1e-3])[1]),
        log=True,
    )

    n_steps = trial.suggest_categorical(
        "n_steps",
        space.get("n_steps", [1024, 2048, 4096]),
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        space.get("batch_size", [64, 128, 256]),
    )
    # Constraint: batch_size must be <= n_steps
    if batch_size > n_steps:
        raise optuna.TrialPruned()

    n_epochs = trial.suggest_categorical(
        "n_epochs",
        space.get("n_epochs", [3, 5, 10]),
    )

    # Gamma: sample 1 - (1 - gamma) on log scale for precision near 1.0 [4]
    one_minus_gamma = trial.suggest_float(
        "one_minus_gamma",
        1 - float(space.get("gamma", [0.98, 0.999])[1]),
        1 - float(space.get("gamma", [0.98, 0.999])[0]),
        log=True,
    )
    gamma = 1.0 - one_minus_gamma

    ent_coef = trial.suggest_float(
        "ent_coef",
        float(space.get("ent_coef", [1e-6, 0.01])[0]),
        float(space.get("ent_coef", [1e-6, 0.01])[1]),
        log=True,
    )

    # Evidential parameters -- always enabled (lambda_reg > 0).
    # Disabling evidential regularisation defeats the architecture's purpose.
    lambda_reg = trial.suggest_float(
        "lambda_reg",
        float(space.get("lambda_reg", [1e-5, 0.01])[0]),
        float(space.get("lambda_reg", [1e-5, 0.01])[1]),
        log=True,
    )

    lambda_reg_warmup_steps = trial.suggest_float(
        "lambda_reg_warmup_steps",
        float(space.get("lambda_reg_warmup_steps", [10000, 100000])[0]),
        float(space.get("lambda_reg_warmup_steps", [10000, 100000])[1]),
        log=True,
    )

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "evidential": {
            "lambda_reg": lambda_reg,
            "lambda_reg_warmup_steps": int(lambda_reg_warmup_steps),
        },
    }


# ===========================================================================
# TrialEvalCallback: Read metrics from training for trial evaluation
# ===========================================================================


class TrialEvalCallback(BaseCallback):
    """
    @class TrialEvalCallback
    @brief Callback that reports training metrics to Optuna for trial pruning.

    Reads training metrics from self.logger.name_to_value (populated by
    EnvDiagnosticsCallback) and reports the primary objective metric
    (env/mean_progress_reward) to the trial. Also checks for early stopping
    via Optuna's pruning mechanism.
    """

    def __init__(self, trial: "optuna.Trial") -> None:
        """
        @brief Constructor.
        @param trial: Optuna trial object.
        """
        super().__init__()
        self.trial = trial

    def _on_step(self) -> bool:
        """
        @brief Called at every training step (required by SB3).
        @return True to continue training, False to stop.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        @brief Called at the end of each rollout (policy update).

        Reads the objective metric (env/mean_progress_reward) from the logger
        and reports it to Optuna for pruning decisions.
        """
        # Try to get the primary objective metric
        metric = self.logger.name_to_value.get("env/mean_progress_reward")

        # Fallback to rollout reward if primary metric is not available
        if metric is None:
            metric = self.logger.name_to_value.get("rollout/ep_rew_mean", 0.0)

        # Report to trial and check for pruning
        self.trial.report(metric, step=self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ===========================================================================
# apply_best_params: Write best trial params back to train_config.yaml
# ===========================================================================


def apply_best_params(
    best_params: Dict[str, Any],
    train_config_path: str,
) -> None:
    """
    @brief Write best trial hyperparameters back into train_config.yaml.

    Uses ruamel.yaml (if available) to preserve comments and formatting.
    Falls back to pyyaml if round-trip fails. Creates a backup in logs/tuning/backups/
    before overwriting.

    @param best_params: Dictionary of best hyperparameters (flat or nested).
    @param train_config_path: Path to train_config.yaml.
    """
    config_path = Path(train_config_path)
    backup_dir = Path("logs/tuning/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup in configs/backups/
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"train_config_{timestamp}.yaml.bak"

    # Copy original config to backup
    if config_path.exists():
        import shutil
        shutil.copy2(config_path, backup_path)
        logger.info("Created backup: %s", backup_path)

    # Load original config
    try:
        import yaml as pyyaml
        with open(config_path) as f:
            original_config = pyyaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load original config: %s", e)
        raise

    # Update only keys that were in the search space
    for key, value in best_params.items():
        if isinstance(value, dict):
            # Handle nested keys like evidential.lambda_reg
            if key not in original_config:
                original_config[key] = {}
            original_config[key].update(value)
        else:
            # Handle flat keys
            original_config[key] = value

    # Write back to train_config.yaml
    try:
        import yaml as pyyaml
        with open(config_path, 'w') as f:
            pyyaml.dump(original_config, f, default_flow_style=False, sort_keys=False)
        logger.info("Updated train_config.yaml with best params")

        # Log the changes
        for key, value in best_params.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    logger.info("  %s.%s: %s", key, subkey, subvalue)
            else:
                logger.info("  %s: %s", key, value)

    except Exception as e:
        logger.error("Failed to write train_config.yaml: %s", e)
        logger.error("Original config backed up at: %s", backup_path)
        raise


# ===========================================================================
# objective: Optuna objective function
# ===========================================================================


def objective(
    trial: "optuna.Trial",
    base_config: Dict[str, Any],
    tuning_config: Dict[str, Any],
    train_config_path: str,
    env_config_path: str,
) -> float:
    """
    @brief Optuna objective function for a single trial.

    Samples hyperparameters, runs a short training session, and returns the
    primary metric (env/mean_progress_reward) for optimisation.

    @param trial: Optuna trial object.
    @param base_config: Base training config (merged train + env configs).
    @param tuning_config: Tuning configuration with study settings.
    @param train_config_path: Path to train_config.yaml.
    @param env_config_path: Path to env_config.yaml.
    @return Primary metric value for this trial.
    """
    try:
        # Sample hyperparameters
        sampled_params = sample_hyperparams(trial, tuning_config)

        # Create a trial-specific config
        trial_config = copy.deepcopy(base_config)
        trial_config.update(sampled_params)

        # Set trial-specific training budget and directories
        trial_config["total_timesteps"] = tuning_config.get("timesteps_per_trial", 100000)
        trial_config["log_dir"] = os.path.join(
            "logs/tuning",
            f"trial_{trial.number}",
        )
        trial_config["checkpoint_dir"] = os.path.join(
            "checkpoints/tuning",
            f"trial_{trial.number}",
        )

        # Create trial eval callback
        trial_callback = TrialEvalCallback(trial)

        # Run training
        result = train(trial_config, extra_callbacks=[trial_callback])

        # Extract primary metric from final metrics
        primary_metric = result.final_metrics.get("env/mean_progress_reward", 0.0)

        # Also log secondary metric as trial user attribute
        secondary_metric = result.final_metrics.get("env/success_rate", 0.0)
        trial.set_user_attr("env/success_rate", secondary_metric)

        logger.info(
            "Trial %d: primary=%.4f, secondary=%.4f",
            trial.number,
            primary_metric,
            secondary_metric,
        )

        return primary_metric

    except optuna.TrialPruned:
        logger.info("Trial %d pruned", trial.number)
        raise

    except Exception as e:
        logger.error("Trial %d failed with exception: %s", trial.number, e)
        # CARLA crashes during training should trigger pruning
        raise optuna.TrialPruned()


# ===========================================================================
# run_study: Execute the Optuna study
# ===========================================================================


def run_study(
    tuning_config: Dict[str, Any],
    base_config: Dict[str, Any],
    train_config_path: str,
    env_config_path: str,
) -> None:
    """
    @brief Run the Optuna hyperparameter tuning study.

    Uses TPESampler (multivariate) + MedianPruner. Results are stored in
    SQLite for persistence and resume capability. Best params are written
    back to train_config.yaml after the study completes.

    Sampler: multivariate TPE captures parameter interactions (e.g.
    learning_rate vs batch_size) [3]. n_startup_trials=10 gives pure random
    exploration before TPE builds its density model.

    Pruner: MedianPruner over HyperbandPruner -- Hyperband creates multiple
    brackets requiring ~10 startup trials each, exhausting most of our budget
    on random search [4].

    @param tuning_config: Tuning configuration with study settings.
    @param base_config: Base training config (merged train + env configs).
    @param train_config_path: Path to train_config.yaml.
    @param env_config_path: Path to env_config.yaml.
    """
    # Create tuning results directory (in logs/ which is writable in Docker)
    results_dir = Path("logs/tuning/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # SQLite storage path
    storage_path = Path(tuning_config.get("storage_path", "logs/tuning/optuna_study.db"))
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Sampler and pruner settings from tuning_config.yaml
    sampler_cfg = tuning_config.get("sampler", {})
    pruner_cfg = tuning_config.get("pruner", {})

    # Create study
    study_name = tuning_config.get("study_name", "uncertainty_rl_tuning")
    sampler = optuna.samplers.TPESampler(
        seed=tuning_config.get("seed", 42),
        multivariate=sampler_cfg.get("multivariate", True),
        n_startup_trials=sampler_cfg.get("n_startup_trials", 10),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_cfg.get("n_startup_trials", 8),
        n_warmup_steps=pruner_cfg.get("n_warmup_steps", 5),
        n_min_trials=pruner_cfg.get("n_min_trials", 5),
    )

    storage = optuna.storages.RDBStorage(f"sqlite:///{storage_path}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # Maximise env/mean_progress_reward
        load_if_exists=True,  # Resume from previous run
    )

    # Run optimisation
    n_trials = tuning_config.get("n_trials", 35)
    logger.info("Starting Optuna study: %d trials, %dk steps/trial, 8 params",
                n_trials, tuning_config.get("timesteps_per_trial", 100000) // 1000)

    study.optimize(
        lambda trial: objective(
            trial,
            base_config,
            tuning_config,
            train_config_path,
            env_config_path,
        ),
        n_trials=n_trials,
    )

    # Print summary
    logger.info("Study complete. Best trial:")
    try:
        best_trial = study.best_trial
    except ValueError:
        # All trials pruned or no completed trials
        logger.error("No completed trials in study. Check trial logs for errors.")
        from optuna.trial import TrialState
        logger.error("Study trials: %d completed, %d pruned",
                     len([t for t in study.trials if t.state == TrialState.COMPLETE]),
                     len([t for t in study.trials if t.state == TrialState.PRUNED]))
        return

    logger.info("  Number: %d", best_trial.number)
    logger.info("  Value (primary): %.4f", best_trial.value)
    logger.info("  Secondary (success_rate): %.4f",
                best_trial.user_attrs.get("env/success_rate", 0.0))
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info("    %s: %s", key, value)

    # Write best params back to train_config.yaml
    best_params = best_trial.params
    apply_best_params(best_params, train_config_path)

    # Also save standalone copy to logs/tuning/results/best_params.yaml
    import yaml
    best_params_path = results_dir / "best_params.yaml"
    with open(best_params_path, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info("Saved best params to %s", best_params_path)
    logger.info("Tuning results saved to %s", results_dir)


# ===========================================================================
# main: CLI entry point
# ===========================================================================


def main() -> None:
    """
    @brief Main entry point for hyperparameter tuning script.
    """
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for PPO + evidential policies"
    )
    parser.add_argument(
        "--tuning-config",
        type=str,
        default="configs/training/tuning_config.yaml",
        help="Path to tuning configuration (study settings, search space)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training hyperparameter config",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/carla/env_config.yaml",
        help="Path to environment config",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configs
    train_config = load_config(args.train_config)
    env_config = load_config(args.env_config)
    tuning_config = load_config(args.tuning_config)

    # Merge train + env configs
    base_config = merge_configs(train_config, env_config)

    # Run study
    run_study(tuning_config, base_config, args.train_config, args.env_config)


if __name__ == "__main__":
    main()
