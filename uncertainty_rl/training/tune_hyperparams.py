"""
@file tune_hyperparams.py
@brief Optuna tuning of the structural PPO hyperparameters.

Uses TPESampler (multivariate) and MedianPruner to search STRUCTURAL PPO
params only - learning_rate/ent_coef are stage-owned schedules. Evaluates
env/success_rate, with mean_progress_reward as an early tiebreaker.

@note Never run. Every reported result uses the committed defaults in
      train_config.yaml, because tuning per arm would make the configuration a
      fifth variable in the 2x2 ablation. Retained for future work.
"""

import argparse
import copy
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:
    from yaml import SafeLoader as _YamlLoader  # type: ignore[assignment]
import yaml

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
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    raise ImportError(
        "stable_baselines3 is required. "
        "Install with: pip install 'stable-baselines3>=2.0.0'"
    )

from uncertainty_rl.training.train_ppo import (
    DEFAULT_BASELINE,
    DEFAULT_STAGE,
    load_config,
    load_env_config,
    make_env,
    merge_configs,
    train,
)
from uncertainty_rl.utils.config_merge import apply_baseline

logger = logging.getLogger("uncertainty_rl.training.tune_hyperparams")


# Tiebreaker scale: any non-zero success_rate dominates a progress-based fallback.
_TIEBREAK_SCALE: float = 1e-3


def _composite_objective(
    success_rate: Optional[float],
    progress: Optional[float],
) -> float:
    """
    @brief Compute the Optuna trial objective from training metrics.
    @param success_rate: env/success_rate in [0, 1], or None if unavailable.
    @param progress: env/mean_progress_reward, or None if unavailable.
    @return success_rate when > 0, else _TIEBREAK_SCALE * progress (or 0.0).
    """
    if success_rate is None or success_rate == 0.0:
        return 0.0 if progress is None else _TIEBREAK_SCALE * float(progress)
    return float(success_rate)


def sample_hyperparams(
    trial: "optuna.Trial", tuning_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Sample hyperparameters from the search space.

    @param trial: Optuna trial object.
    @param tuning_config: Tuning configuration with search space bounds.
    @return Dictionary of sampled hyperparameters.
    """
    space = tuning_config.get("search_space", {})

    # Structural PPO params only: learning_rate / ent_coef are stage-owned and
    # evidential.* is ablation-specific, so neither is tuned.
    # @see docs/detailed_notes/training/ablation_hpo_methodology.md
    gamma_range = space.get("gamma", [0.98, 0.999])
    gae_range = space.get("gae_lambda", [0.90, 0.98])
    clip_range_bounds = space.get("clip_range", [0.1, 0.3])
    vf_coef_range = space.get("vf_coef", [0.25, 1.0])
    grad_norm_range = space.get("max_grad_norm", [0.3, 2.0])

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

    # Gamma: sample 1 - (1 - gamma) on log scale for precision near 1.0
    one_minus_gamma = trial.suggest_float(
        "one_minus_gamma",
        1 - float(gamma_range[1]),
        1 - float(gamma_range[0]),
        log=True,
    )
    gamma = 1.0 - one_minus_gamma

    gae_lambda = trial.suggest_float(
        "gae_lambda",
        float(gae_range[0]),
        float(gae_range[1]),
    )

    clip_range = trial.suggest_float(
        "clip_range",
        float(clip_range_bounds[0]),
        float(clip_range_bounds[1]),
    )

    vf_coef = trial.suggest_float(
        "vf_coef",
        float(vf_coef_range[0]),
        float(vf_coef_range[1]),
    )

    max_grad_norm = trial.suggest_float(
        "max_grad_norm",
        float(grad_norm_range[0]),
        float(grad_norm_range[1]),
    )

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
    }


class TrialEvalCallback(BaseCallback):
    """
    @class TrialEvalCallback
    @brief Callback that reports training metrics to Optuna for trial pruning.

    Reads training metrics from self.logger.name_to_value (populated by
    EnvDiagnosticsCallback) and reports the composite objective
    (env/success_rate primary, env/mean_progress_reward tiebreaker) to the
    trial. Also checks for early stopping via Optuna's pruning mechanism.
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

        Reports a composite objective to Optuna for pruning decisions:
        env/success_rate as the dominant signal, with mean_progress_reward
        scaled into a small positive range as a tiebreaker before any
        success has been observed.
        """
        metric = _composite_objective(
            self.logger.name_to_value.get("env/success_rate"),
            self.logger.name_to_value.get("env/mean_progress_reward"),
        )
        self.trial.report(metric, step=self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"train_config_{timestamp}.yaml.bak"

    # Copy original config to backup
    if config_path.exists():
        shutil.copy2(config_path, backup_path)
        logger.info("Created backup: %s", backup_path)

    # Load original config
    try:
        with open(config_path) as f:
            original_config = yaml.load(f, Loader=_YamlLoader)
    except Exception as e:
        logger.error("Failed to load original config: %s", e)
        raise

    for key, value in best_params.items():
        if isinstance(value, dict):
            if key not in original_config:
                original_config[key] = {}
            original_config[key].update(value)
        else:
            original_config[key] = value

    # Write back to train_config.yaml
    try:
        with open(config_path, "w") as f:
            yaml.dump(original_config, f, default_flow_style=False, sort_keys=False)
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


def objective(
    trial: "optuna.Trial",
    base_config: Dict[str, Any],
    tuning_config: Dict[str, Any],
    train_config_path: str,
    env_config_path: str,
    shared_vec_env: Optional[DummyVecEnv] = None,
) -> float:
    """
    @brief Optuna objective function for a single trial.
    @param trial: Optuna trial object.
    @param base_config: Base training config (merged train + env configs).
    @param tuning_config: Tuning configuration with study settings.
    @param train_config_path: Path to train_config.yaml.
    @param env_config_path: Path to env_config.yaml.
    @param shared_vec_env: Pre-built CARLA vec env reused across all trials to
           avoid the CARLA destroy/respawn cycle that breaks the EKF; each
           trial wraps it in a FRESH VecNormalize so stats never leak between
           trials.
    @return Composite objective value (env/success_rate primary,
            env/mean_progress_reward tiebreaker) for this trial.
    """
    try:
        # Sample hyperparameters
        sampled_params = sample_hyperparams(trial, tuning_config)

        # Create a trial-specific config
        trial_config = copy.deepcopy(base_config)
        trial_config.update(sampled_params)

        # Fresh per-trial VecNormalize over the shared CARLA env, constructed
        # EXACTLY as train_ppo.py builds it, with this trial's sampled gamma -
        # tuning against any other normalisation setup ranks configs on
        # dynamics the real training run never sees.
        trial_env: Optional[VecNormalize] = None
        if shared_vec_env is not None:
            trial_env = VecNormalize(
                shared_vec_env,
                norm_obs=False,
                norm_reward=True,
                clip_reward=20.0,
                gamma=float(trial_config.get("gamma", 0.99)),
            )

        # train() nests every run as <root>/<baseline_name>/<run_leaf>/, so a
        # deterministic per-trial leaf gives logs/tuning/<baseline>/trial_<N>/ -
        # the same per-baseline layout as a normal training run.
        trial_config["total_timesteps"] = tuning_config.get(
            "timesteps_per_trial", 100000
        )
        trial_config["log_dir"] = "logs/tuning"
        trial_config["checkpoint_dir"] = "checkpoints/tuning"
        trial_config["run_leaf"] = f"trial_{trial.number}"

        # Create trial eval callback
        trial_callback = TrialEvalCallback(trial)

        # Run training with the shared env (env stays alive between trials so
        # CARLA sensors + EKF do not need to re-warm).
        result = train(
            trial_config,
            extra_callbacks=[trial_callback],
            env=trial_env,
        )

        success_rate = float(result.final_metrics.get("env/success_rate", 0.0))
        progress = float(result.final_metrics.get("env/mean_progress_reward", 0.0))
        primary_metric = _composite_objective(success_rate, progress)

        trial.set_user_attr("env/success_rate", success_rate)
        trial.set_user_attr("env/mean_progress_reward", progress)

        logger.info(
            "Trial %d: success_rate=%.4f, progress=%.4f, objective=%.4f",
            trial.number,
            success_rate,
            progress,
            primary_metric,
        )

        return primary_metric

    except optuna.TrialPruned:
        logger.info("Trial %d pruned", trial.number)
        raise

    except Exception as e:
        logger.error("Trial %d failed with exception: %s", trial.number, e)
        # CARLA crashes during training should trigger pruning
        raise optuna.TrialPruned()


def run_study(
    tuning_config: Dict[str, Any],
    base_config: Dict[str, Any],
    train_config_path: str,
    env_config_path: str,
    baseline_name: Optional[str] = None,
) -> None:
    """
    @brief Run the Optuna hyperparameter tuning study.

    Uses TPESampler (multivariate) + MedianPruner. Results are stored in
    SQLite for persistence and resume capability. When `baseline_name` is set,
    the study name, storage DB, and best-params output are all suffixed with
    the baseline name, so each ablation baseline gets an INDEPENDENT study and
    best-params file, leaving train_config.yaml untouched. When `baseline_name`
    is None, the single study tunes the train_config defaults and writes best
    params back into train_config.yaml directly.

    @param tuning_config: Tuning configuration with study settings.
    @param base_config: Base training config (merged train + env [+ baseline]).
    @param train_config_path: Path to train_config.yaml.
    @param env_config_path: Path to env_config.yaml.
    @param baseline_name: Ablation baseline this study tunes, or None for the
                          shared-defaults single study.
    """
    # Create tuning results directory (in logs/ which is writable in Docker)
    results_dir = Path("logs/tuning/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-baseline study name, storage DB, and best-params filename so the four
    # ablation studies never collide in one Optuna DB.
    suffix = f"_{baseline_name}" if baseline_name else ""

    # SQLite storage path
    storage_path = Path(
        tuning_config.get("storage_path", "logs/tuning/optuna_study.db")
    )
    if baseline_name:
        storage_path = storage_path.with_name(
            f"{storage_path.stem}{suffix}{storage_path.suffix}"
        )
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    # Sampler and pruner settings from tuning_config.yaml
    sampler_cfg = tuning_config.get("sampler", {})
    pruner_cfg = tuning_config.get("pruner", {})

    # `tuning_seed` is distinct from train_config's `seed` so the tuning split
    # is disjoint from the evaluation split. Required (no fallback) so a missing
    # seed fails loud rather than silently using a magic number.
    if "tuning_seed" not in tuning_config:
        raise KeyError(
            "tuning_config.yaml must define 'tuning_seed' (the Optuna sampler seed)."
        )
    study_name = tuning_config.get("study_name", "uncertainty_rl_tuning") + suffix
    sampler = optuna.samplers.TPESampler(
        seed=tuning_config["tuning_seed"],
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
        direction="maximize",  # Maximise env/success_rate (primary objective).
        load_if_exists=True,  # Resume from previous run
    )

    # Run optimisation
    n_trials = tuning_config.get("n_trials", 40)
    logger.info(
        "Starting Optuna study '%s': %d trials, %dk steps/trial",
        study_name,
        n_trials,
        tuning_config.get("timesteps_per_trial", 100000) // 1000,
    )

    # Build the env once and share it across every trial. A destroy+respawn
    # cycle of CARLA sensors between trials breaks the ROS bridge so the EKF
    # stops publishing /odometry/filtered and every subsequent trial times
    # out in _wait_for_covariance.
    n_workers = base_config.get("parallel_workers", 1)
    logger.info(
        "Creating shared training environment (%d worker(s)) for all trials...",
        n_workers,
    )
    # bay_margin is defined per stage in configs/deployment/sim/curriculum/
    # stage<N>.yaml; .get guards a missing key with the env's own default.
    tune_bay_margin = float(base_config.get("bay_margin", 0.0))
    # Only the CARLA vec env is shared; each trial wraps it in its own
    # VecNormalize (see objective) so reward-normalisation running stats and
    # the sampled gamma stay per-trial, exactly as a fresh training run.
    shared_vec_env = DummyVecEnv(
        [
            make_env(base_config, bay_margin=tune_bay_margin, rank=i)
            for i in range(n_workers)
        ]
    )

    try:
        study.optimize(
            lambda trial: objective(
                trial,
                base_config,
                tuning_config,
                train_config_path,
                env_config_path,
                shared_vec_env=shared_vec_env,
            ),
            n_trials=n_trials,
        )
    finally:
        logger.info("Closing shared training environment.")
        shared_vec_env.close()

    # Print summary
    logger.info("Study complete. Best trial:")
    try:
        best_trial = study.best_trial
    except ValueError:
        # All trials pruned or no completed trials
        logger.error("No completed trials in study. Check trial logs for errors.")
        from optuna.trial import TrialState

        logger.error(
            "Study trials: %d completed, %d pruned",
            len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            len([t for t in study.trials if t.state == TrialState.PRUNED]),
        )
        return

    logger.info("  Number: %d", best_trial.number)
    logger.info("  Objective: %.4f", best_trial.value)
    logger.info(
        "  env/success_rate: %.4f", best_trial.user_attrs.get("env/success_rate", 0.0)
    )
    logger.info(
        "  env/mean_progress_reward: %.4f",
        best_trial.user_attrs.get("env/mean_progress_reward", 0.0),
    )
    logger.info("  Params:")
    for key, value in best_trial.params.items():
        logger.info("    %s: %s", key, value)

    best_params = best_trial.params

    # A per-baseline study writes to a per-baseline file and leaves the shared
    # train_config.yaml untouched - otherwise multiple baselines would overwrite
    # each other's hyperparameters in one file. The no-baseline study instead
    # writes directly into train_config.yaml.
    if baseline_name:
        logger.info(
            "Per-baseline study: leaving %s untouched; writing best params to a "
            "baseline-specific file.",
            train_config_path,
        )
    else:
        apply_best_params(best_params, train_config_path)

    # Save standalone copy (per-baseline filename when tuning a baseline).
    best_params_path = results_dir / f"best_params{suffix}.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.info("Saved best params to %s", best_params_path)
    logger.info("Tuning results saved to %s", results_dir)


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
        default="configs/deployment/sim/env_config.yaml",
        help="Path to environment config",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help=(
            "Curriculum stage (1..N). Deep-merges "
            "configs/deployment/sim/curriculum/stage<N>.yaml over the env config "
            "so tuning runs at that stage's difficulty. Stage 1 (the default) is "
            "the recommended tuning stage: trials train from scratch on a "
            "100k-step budget, and stage 1 is the only stage where that budget "
            "yields a non-zero success-rate objective (later stages would leave "
            "only the progress tiebreaker)."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help=(
            "Path to a baseline override config (configs/baselines/*.yaml). When "
            "set, the study tunes that baseline's observation/policy configuration "
            "and writes its best params to a per-baseline file, giving the 2x2 "
            "ablation one independent study each. Omit to tune the shared "
            "hyperparameters at the full-method observation space and write back to "
            "train_config.yaml."
        ),
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve effective stage / baseline the same way train_ppo.py does:
    # difficulty lives only in the stage files and obs/policy flags only in the
    # baseline files, so omitting a flag defaults to stage 1 and the full method.
    stage = args.stage if args.stage is not None else DEFAULT_STAGE
    baseline_path = args.baseline if args.baseline is not None else DEFAULT_BASELINE

    train_config = load_config(args.train_config)
    env_config = load_env_config(args.env_config, stage=stage)
    tuning_config = load_config(args.tuning_config)

    # Merge train + env configs
    base_config = merge_configs(train_config, env_config)

    # The write-back target depends on whether --baseline was EXPLICIT: an
    # explicit baseline gets its own study and per-baseline best-params file
    # (baseline_name set); the default full-method case writes back to
    # train_config.yaml (baseline_name None).
    baseline_override = load_config(baseline_path)
    apply_baseline(base_config, baseline_override)
    baseline_name: Optional[str] = (
        baseline_override.get("baseline_name", Path(baseline_path).stem)
        if args.baseline is not None
        else None
    )
    logger.info(
        "Tuning stage %d, baseline '%s' (include_covariance=%s, policy_type=%s)",
        stage,
        baseline_override.get("baseline_name", Path(baseline_path).stem),
        base_config.get("include_covariance"),
        base_config.get("policy_type"),
    )

    # Run study
    run_study(
        tuning_config,
        base_config,
        args.train_config,
        args.env_config,
        baseline_name=baseline_name,
    )


if __name__ == "__main__":
    main()
