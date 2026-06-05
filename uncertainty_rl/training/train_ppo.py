"""
@file train_ppo.py
@brief Training script for uncertainty-conditioned RL with PPO and evidential policies.

This module provides training functionality using Stable-Baselines3's PPO algorithm
with evidential actor networks for autonomous parking.
"""

import argparse
import dataclasses
import logging
import os
import random
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Deque, List, Optional

import numpy as np

try:
    from yaml import CSafeLoader as _YamlLoader
except ImportError:
    from yaml import SafeLoader as _YamlLoader  # type: ignore[assignment]
import yaml

try:
    import gymnasium as gym
except ImportError:
    gym = None  # type: ignore[assignment,misc]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment,misc]

try:
    from stable_baselines3.common.callbacks import EvalCallback  # noqa: F401
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.ppo import PPO
except ImportError:
    BaseCallback = None  # type: ignore[assignment,misc]
    CallbackList = None  # type: ignore[assignment,misc]
    CheckpointCallback = None  # type: ignore[assignment,misc]
    configure = None  # type: ignore[assignment,misc]
    DummyVecEnv = None  # type: ignore[assignment,misc]
    VecNormalize = None  # type: ignore[assignment,misc]
    PPO = None  # type: ignore[assignment,misc]
    EvalCallback = None  # type: ignore[assignment,misc]

from uncertainty_rl.utils.bay_success import BaySuccessTracker
from uncertainty_rl.utils.constants import TRAINING_BAY_MARGIN

try:
    # make_env is re-exported here so existing imports
    # (`from uncertainty_rl.training.train_ppo import make_env`) keep
    # working; the canonical owner is uncertainty_rl.envs.factory.
    from uncertainty_rl.envs import CARLAParkingEnv, make_env  # noqa: F401
    from uncertainty_rl.networks import EvidentialActorCriticPolicy, EvidentialPPO
except ImportError:
    CARLAParkingEnv = None  # type: ignore[assignment,misc]
    make_env = None  # type: ignore[assignment,misc]
    EvidentialActorCriticPolicy = None  # type: ignore[assignment,misc]
    EvidentialPPO = None  # type: ignore[assignment,misc]

logger = logging.getLogger("uncertainty_rl.training.train_ppo")

# Suppress Gymnasium's float64->float32 precision warning for unbounded obs.
# spaces.Box with low/high=±inf always triggers this; it is harmless.
warnings.filterwarnings(
    "ignore",
    message=".*Box.*precision lowered.*",
    category=UserWarning,
)


@dataclasses.dataclass
class TrainResult:
    """
    @class TrainResult
    @brief Encapsulates the results of a training run.

    Returned by train() to allow programmatic access to final metrics and paths
    for integration with external optimisation frameworks (e.g. Optuna).
    """

    final_metrics: Dict[str, float]
    """Final metrics from model.logger.name_to_value (e.g. env/success_rate)."""

    model_path: str
    """Absolute path to the saved final model."""

    log_dir: str
    """Absolute path to the TensorBoard logs directory."""

    env: Optional["VecNormalize"] = None
    """Training environment (only populated when the caller passed an external
    env in, so it can be reused across trials). None when train() created and
    closed the env itself."""


def linear_schedule(
    initial_value: float, final_value: float = 0.0
) -> Callable[[float], float]:
    """
    @brief Linear schedule decaying from initial_value to final_value.
    @param initial_value: Value at the start of training (progress_remaining=1.0).
    @param final_value: Value at the end of training (progress_remaining=0.0).
           Defaults to 0.0.
    @return Callable that takes progress_remaining (1.0 -> 0.0) and returns the
            interpolated value.
    """

    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return func


def load_config(config_path: str) -> Dict[str, Any]:
    """
    @brief Load configuration from YAML file.
    @param config_path: Path to configuration file.
    @return Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.load(f, Loader=_YamlLoader)
    return config


def _deep_merge(
    base: Dict[str, Any], override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Recursively merge override into base; override wins on scalar keys.
    @param base: Lower-precedence dict (mutated in place and returned).
    @param override: Higher-precedence dict whose values take priority.
    @return The merged dict.

    @note A nested dict on both sides is merged key-by-key rather than replaced
          wholesale. This lets a higher-precedence config add or override
          individual keys inside a shared block (e.g. env_config adding
          ros2.carla_recovery without dropping ros2.covariance_timeout defined
          in agent_config). A plain {**base, **override} would discard every
          base key under any block the override also defines.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def merge_configs(
    train_config: Dict[str, Any], env_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Merge training and environment configs into a single dict.

    @param train_config: Training hyperparameters from train_config.yaml.
    @param env_config: Environment settings from env_config.yaml.
    @return Merged configuration dictionary.
    """
    merged = {**env_config, **train_config}
    return merged


def load_env_config(env_config_path: str) -> Dict[str, Any]:
    """
    @brief Load and merge the environment config with shared deployment configs.


    @param env_config_path: Path to the CARLA environment config YAML.
    @return Fully merged environment configuration dictionary.
    """
    with open(env_config_path) as f:
        env_config: Dict[str, Any] = yaml.load(f, Loader=_YamlLoader) or {}

    deployment_dir = Path(env_config_path).parent.parent

    sensor_cfg: Dict[str, Any] = {}
    sensor_cfg_path = deployment_dir / "sensor_config.yaml"
    if sensor_cfg_path.exists():
        with open(sensor_cfg_path) as f:
            sensor_cfg = yaml.load(f, Loader=_YamlLoader) or {}

    agent_cfg: Dict[str, Any] = {}
    agent_cfg_path = deployment_dir / "agent_config.yaml"
    if agent_cfg_path.exists():
        with open(agent_cfg_path) as f:
            agent_cfg = yaml.load(f, Loader=_YamlLoader) or {}

    # Merge: sensor_config < agent_config < env_config (env wins on conflict).
    # Deep merge so a higher-precedence file can override individual keys inside
    # a shared nested block (e.g. env_config's ros2.carla_recovery layered onto
    # agent_config's ros2.covariance_timeout) instead of replacing the whole
    # block. A shallow {**a, **b} would silently drop the agent_config ros2 keys.
    merged: Dict[str, Any] = {}
    _deep_merge(merged, sensor_cfg)
    _deep_merge(merged, agent_cfg)
    _deep_merge(merged, env_config)

    # Inject sensor mounts into carla_sensors so the CARLA spawner gets them.
    for sensor_name, sensor_data in sensor_cfg.get("sensors", {}).items():
        if "mount" in sensor_data and sensor_name in merged.get("carla_sensors", {}):
            merged["carla_sensors"][sensor_name]["mount"] = sensor_data["mount"]

    return merged


class EnvDiagnosticsCallback(BaseCallback):
    """
    @class EnvDiagnosticsCallback
    @brief SB3 callback that logs per-episode environment diagnostics to TensorBoard.

    The per-step means (pos/orientation/speed/progress) are averaged over every
    step in a rollout, so they are already smooth. The terminal rates (success,
    collision, out-of-bounds, timeout) are sparse - one PPO rollout holds only a
    handful of terminal episodes (~n_steps / max_steps), so a per-rollout rate
    has a noise band of tens of percentage points and cannot be read as policy
    quality. To make the rates legible, terminal outcomes are kept in a rolling
    window of the last `outcome_window` episodes and the logged rate is the mean
    over that window. This is a logging change only - it does not touch training
    dynamics.
    """

    def __init__(self, outcome_window: int = 50) -> None:
        """
        @brief Initialise accumulators.
        @param outcome_window: Number of most-recent terminal episodes the
               success/collision/oob/timeout rates are averaged over. Larger
               values trade reporting latency for a tighter noise band on the rate.
        """
        super().__init__(verbose=0)
        self._pos_sum = 0.0
        self._ori_sum = 0.0
        self._spd_sum = 0.0
        self._prog_sum = 0.0
        self._step_count = 0
        # Rolling per-episode terminal outcomes (1.0 / 0.0 flags) over the last
        # `outcome_window` episodes, so the logged rate reflects policy quality
        # rather than the handful of episodes that happened to end this rollout.
        self._success_hist: Deque[float] = deque(maxlen=outcome_window)
        self._collision_hist: Deque[float] = deque(maxlen=outcome_window)
        self._oob_hist: Deque[float] = deque(maxlen=outcome_window)
        self._timeout_hist: Deque[float] = deque(maxlen=outcome_window)

    def _on_step(self) -> bool:
        """
        @brief Accumulate diagnostics from the latest env step.
        @return Always True (training continues).
        """
        for info in self.locals.get("infos", []):
            self._pos_sum += info.get("pos_error", 0.0)
            self._ori_sum += info.get("orientation_error", 0.0)
            self._spd_sum += info.get("speed", 0.0)
            self._prog_sum += info.get("progress_reward", 0.0)
            self._step_count += 1
            # Read each flag once and reuse for both is_terminal and individual recording.
            success = info.get("success", False)
            collision = info.get("collision", False)
            oob = info.get("oob", False)
            timeout = info.get("timeout", False)
            if success or collision or oob or timeout:
                self._success_hist.append(float(success))
                self._collision_hist.append(float(collision))
                self._oob_hist.append(float(oob))
                self._timeout_hist.append(float(timeout))
        return True

    def _on_rollout_end(self) -> None:
        """
        @brief Flush accumulated diagnostics to TensorBoard at rollout end.
        """
        if self._step_count:
            inv = 1.0 / self._step_count
            self.logger.record("env/mean_pos_error_m", self._pos_sum * inv)
            self.logger.record("env/mean_orientation_error_rad", self._ori_sum * inv)
            self.logger.record("env/mean_speed_ms", self._spd_sum * inv)
            self.logger.record("env/mean_progress_reward", self._prog_sum * inv)

        # Terminal rates are averaged over the rolling window (the deques retain
        # the last `outcome_window` episodes across rollouts), so they are not
        # reset here - only the per-step sums below are.
        if self._success_hist:
            inv_t = 1.0 / len(self._success_hist)
            self.logger.record("env/success_rate", sum(self._success_hist) * inv_t)
            self.logger.record("env/collision_rate", sum(self._collision_hist) * inv_t)
            self.logger.record("env/oob_rate", sum(self._oob_hist) * inv_t)
            self.logger.record("env/timeout_rate", sum(self._timeout_hist) * inv_t)

        # Reset per-step accumulators for the next rollout window.
        self._pos_sum = 0.0
        self._ori_sum = 0.0
        self._spd_sum = 0.0
        self._prog_sum = 0.0
        self._step_count = 0


class BaySuccessCallback(BaseCallback):
    """
    @class BaySuccessCallback
    @brief Accumulates per-bay success counts over training and dumps to CSV.

    Unlike the rolling-window rates in EnvDiagnosticsCallback, this tracks
    cumulative attempts and successes per target bay across the whole run, so a
    run on the random-bay curriculum (Stage 2 onwards) can be inspected for
    which specific bays the policy can and cannot park in. Output goes to
    outputs/bay_successes/training/<run_name>/ as bay_successes.csv plus a
    run_info.txt header. This is a logging change only - it does not touch
    training dynamics.
    """

    def __init__(
        self,
        output_dir: Path,
        run_info: Dict[str, Any],
        dump_freq_steps: int = 50000,
    ) -> None:
        """
        @brief Initialise the per-bay tracker.
        @param output_dir: Directory to write bay_successes.csv + run_info.txt.
        @param run_info: Key/value pairs written to run_info.txt.
        @param dump_freq_steps: Flush the CSV to disk every N env steps so a
               run can be inspected mid-training; a final dump also runs on end.
        """
        super().__init__(verbose=0)
        self._tracker = BaySuccessTracker()
        self._output_dir = output_dir
        self._run_info = run_info
        self._dump_freq_steps = dump_freq_steps
        self._next_dump = dump_freq_steps

    def _on_step(self) -> bool:
        """
        @brief Record terminal episodes against their target bay.
        @return Always True (training continues).
        """
        for info in self.locals.get("infos", []):
            success = info.get("success", False)
            collision = info.get("collision", False)
            timeout = info.get("timeout", False)
            if not (success or collision or timeout):
                continue
            target_bay = info.get("target_bay", {})
            self._tracker.record(
                bay_id=str(target_bay.get("bay_id", "")),
                success=bool(success),
                bay_type=str(target_bay.get("bay_type", "")),
            )
        if self.num_timesteps >= self._next_dump:
            self._dump()
            self._next_dump += self._dump_freq_steps
        return True

    def _on_training_end(self) -> None:
        """
        @brief Final flush of per-bay counts at the end of training.
        """
        self._dump()

    def _dump(self) -> None:
        """
        @brief Write the running per-bay counts and run_info to disk.
        """
        info = dict(self._run_info)
        info["total_attempts"] = self._tracker.total_attempts
        info["timesteps_at_dump"] = self.num_timesteps
        self._tracker.dump(self._output_dir, run_info=info)


def train(
    config: Dict[str, Any],
    extra_callbacks: Optional[List[BaseCallback]] = None,
    env: Optional["VecNormalize"] = None,
    resume_from: Optional[str] = None,
) -> TrainResult:
    """
    @brief Train the uncertainty-conditioned RL agent with PPO.
    @param config: Fully-resolved configuration dictionary. All operational
           settings (seed, log_dir, etc.) and hyperparameters are read from
           this dict. CLI arguments override YAML values before this is called.
    @param extra_callbacks: Optional list of additional callbacks to append to
           the training callback list (e.g. for Optuna trial evaluation).
    @param env: Pre-built VecNormalize env to reuse across calls. When None,
           train() creates and closes its own env. Optuna tuning passes a
           shared env so the CARLA actors and ROS bridge stay warm between
           trials (a destroy+respawn cycle breaks the EKF, see
           tune_hyperparams.run_study).
    @param resume_from: Optional path to checkpoint directory (contains
           final_model.zip and vec_normalize.pkl). When provided, loads the
           saved model and environment normalisation from this checkpoint and
           continues training with reset_num_timesteps=False. If None, trains
           from scratch.
    @return TrainResult containing final metrics, model path, and log directory.
    """
    # Resolve operational settings from config
    seed = config.get("seed", 42)
    total_timesteps = config.get("total_timesteps", 1000000)

    # Build a run name that uniquely identifies this configuration so each
    # training run gets its own TensorBoard subdirectory under logs/.
    # Format: <baseline_name>_seed<N>_<DDMMYYYY-HHMM>
    # baseline_name is set explicitly in baseline override configs; for ad-hoc
    # runs it is derived from policy_type and observation flags.
    policy_type = config.get("policy_type", "evidential")
    include_cov = config.get("include_covariance", True)
    include_obs = config.get("include_obstacle_obs", True)
    _default_run_name = (
        f"{policy_type}"
        f"_cov{'on' if include_cov else 'off'}"
        f"_obs{'on' if include_obs else 'off'}"
    )
    baseline_name = config.get("baseline_name", _default_run_name)
    _timestamp = datetime.now().strftime("%d%m%Y-%H%M")
    run_name = f"{baseline_name}_seed{seed}_{_timestamp}"

    _base_log_dir = config.get("log_dir", "./logs")
    _base_checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    log_dir = os.path.join(_base_log_dir, run_name)
    checkpoint_dir = os.path.join(_base_checkpoint_dir, run_name)
    # eval_freq and n_eval_episodes are read here for when eval_env is re-enabled.
    # Currently eval_env is always None (see comment below).
    eval_freq = config.get("eval_freq", 10000)
    n_eval_episodes = config.get("n_eval_episodes", 10)

    # Set random seeds. All three global RNGs are seeded: torch (network
    # init + action sampling), numpy (module-level np.random), and the stdlib
    # random module. SB3's PPO(seed=seed) additionally seeds each vectorised
    # env's Gymnasium np_random (with a per-rank offset), which drives the
    # per-episode task selection (spawn point, target bay, NPC placement).
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create training environment - one worker per CARLA instance.
    # parallel_workers > 1 requires docker-compose.parallel.yml (see make docker-train-parallel).
    own_env: bool = env is None
    if own_env:
        n_workers: int = config.get("parallel_workers", 1)
        logger.info(f"Creating training environment ({n_workers} worker(s))...")
        train_vec_env = DummyVecEnv(
            [
                make_env(config, bay_margin=TRAINING_BAY_MARGIN, rank=i)
                for i in range(n_workers)
            ]
        )

        # norm_reward divides rewards by a running estimate of the discounted-
        # return std so the critic predicts a unit-variance target. The reward
        # range is near-bimodal (large terminal bonuses dominate the per-step
        # shaping), so an unnormalised critic struggles to track it.
        # clip_reward raised from SB3's default 10.0 to 20.0 so the +50
        # success terminal is not clipped during the early-training window
        # where the reward-normalisation running_std is still small.
        env = VecNormalize(
            train_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=20.0,
        )
    else:
        logger.info("Reusing existing training environment (caller-owned).")

    # Evaluation environment is disabled when using a single CARLA instance
    # in synchronous mode.  Two clients calling world.tick() on the same
    # server causes double-advancing of the simulation clock and deadlocks.
    # Evaluation is handled separately via make docker-eval after training.
    eval_env: Optional[VecNormalize] = None

    # Shared policy kwargs for both standard and evidential policies
    _activation_map: Dict[str, Any] = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    activation_fn = _activation_map.get(
        config.get("activation", "relu").lower(), torch.nn.ReLU
    )
    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("net_arch", [256, 256]),
            vf=config.get("net_arch", [256, 256]),
        ),
        activation_fn=activation_fn,
    )

    # Shared PPO hyperparameters. learning_rate decays linearly to
    # learning_rate_final (default 0.0). The optional floor exists for
    # experiments that need late-stage updates, but the recommended schedule
    # is decay-to-zero so a converged policy stops being perturbed by noise.
    lr_initial = config.get("learning_rate", 3e-4)
    lr_final = config.get("learning_rate_final", 0.0)
    lr_schedule = linear_schedule(lr_initial, lr_final)

    # ent_coef is a linear DECAY schedule, not a constant. In the evidential
    # policy the action sampling std IS sqrt(aleatoric), so the entropy bonus
    # is a direct pressure on the action std. A constant value cannot allow
    # the policy to both explore early and commit late, so we decay to a small
    # non-zero floor.
    ent_coef_initial = config.get("ent_coef", 0.01)
    ent_coef_final = config.get("ent_coef_final", 0.0005)
    ent_coef_schedule = linear_schedule(ent_coef_initial, ent_coef_final)

    ppo_kwargs = dict(
        env=env,
        learning_rate=lr_schedule,
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 256),
        n_epochs=config.get("n_epochs", 5),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        clip_range_vf=config.get("clip_range_vf", None),
        ent_coef=ent_coef_schedule,
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        target_kl=config.get("target_kl", 0.02),
        policy_kwargs=policy_kwargs,
        verbose=config.get("verbose", 1),
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Create agent based on policy_type config (read once at run-name build above)
    logger.info("Initialising %s PPO agent...", policy_type)

    model: PPO
    if resume_from is not None:
        # Load model from checkpoint. Resume path may point to:
        # 1. final_model (from a completed training run)
        # 2. A directory containing periodic checkpoints (ppo_*.zip files)
        logger.info("Resuming from checkpoint: %s", resume_from)

        # Find the latest checkpoint (either final_model or the highest-step
        # intermediate checkpoint).
        checkpoint_model_path: Optional[str] = None
        checkpoint_vec_norm_path: Optional[str] = None

        final_model = os.path.join(resume_from, "final_model")
        final_vec_norm = os.path.join(resume_from, "vec_normalize.pkl")

        # SB3 saves models as <name>.zip, so the on-disk file is
        # final_model.zip even though SB3's load() takes the extension-less
        # path. Test for the .zip explicitly: os.path.exists("final_model")
        # is False when only "final_model.zip" is present, which would
        # otherwise silently skip the final model and fall through to a
        # (lexically mis-sorted) periodic checkpoint.
        if os.path.exists(final_model + ".zip") or os.path.exists(final_model):
            checkpoint_model_path = final_model
            checkpoint_vec_norm_path = (
                final_vec_norm if os.path.exists(final_vec_norm) else None
            )
            logger.info("Found final_model checkpoint")
        else:
            # Look for the latest periodic checkpoint by step count. Sort
            # numerically on the embedded step integer, NOT lexically: a
            # lexical sort ranks "950000" above "4000000" (because "9" > "4")
            # and would resume from a far earlier checkpoint than intended.
            import glob

            pattern = os.path.join(resume_from, "ppo_uncertainty_rl_*_steps.zip")

            def _step_of(path: str) -> int:
                return int(path.split("_steps.zip")[0].split("_")[-1])

            checkpoints = sorted(glob.glob(pattern), key=_step_of)
            if checkpoints:
                checkpoint_model_path = checkpoints[-1]
                step_count = str(_step_of(checkpoint_model_path))
                vec_norm_path = os.path.join(
                    resume_from,
                    f"ppo_uncertainty_rl_vecnormalize_{step_count}_steps.pkl",
                )
                checkpoint_vec_norm_path = (
                    vec_norm_path if os.path.exists(vec_norm_path) else None
                )
                logger.info(
                    "Found periodic checkpoint at %s steps: %s",
                    step_count,
                    checkpoint_model_path,
                )
            else:
                raise ValueError(
                    f"No checkpoint found in {resume_from}. "
                    f"Expected either final_model or ppo_uncertainty_rl_*_steps.zip"
                )

        # Determine which model class to load based on policy_type
        if policy_type == "evidential":
            model = EvidentialPPO.load(checkpoint_model_path)
        elif policy_type == "standard":
            model = PPO.load(checkpoint_model_path)
        else:
            raise ValueError(
                f"Unknown policy_type '{policy_type}'. "
                f"Expected 'evidential' or 'standard'."
            )

        # Load environment normalisation statistics if we have a new env to wrap.
        if (
            own_env
            and checkpoint_vec_norm_path
            and os.path.exists(checkpoint_vec_norm_path)
        ):
            logger.info(
                "Loading VecNormalize from checkpoint: %s", checkpoint_vec_norm_path
            )
            env = VecNormalize.load(checkpoint_vec_norm_path, env)
        elif not own_env:
            logger.warning(
                "VecNormalize not reloaded - caller-owned env is being used."
            )
        elif not checkpoint_vec_norm_path:
            logger.warning("VecNormalize checkpoint not found")

        # Attach the environment to the model for continued training
        model.set_env(env)

        # Re-bind the decay schedules from the current config. ent_coef is
        # not a first-class SB3 schedule, so PPO.load() does not restore the
        # original closure; without this re-bind a resumed run keeps a stale
        # ent_coef that never reaches the decay floor.
        model.lr_schedule = lr_schedule
        model.ent_coef = ent_coef_schedule
    else:
        # Create fresh agent
        if policy_type == "evidential":
            evidential_config = config.get("evidential", {})
            lambda_reg = evidential_config.get("lambda_reg", 0.01)
            lambda_reg_warmup_steps = evidential_config.get(
                "lambda_reg_warmup_steps", 50000
            )
            use_uncertainty_conditioning = evidential_config.get(
                "use_uncertainty_conditioning", False
            )
            # Forwarded into policy_kwargs so the policy can wire the
            # dual-encoder actor when requested.
            ppo_kwargs["policy_kwargs"][
                "use_uncertainty_conditioning"
            ] = use_uncertainty_conditioning
            model = EvidentialPPO(
                policy=EvidentialActorCriticPolicy,
                lambda_reg=lambda_reg,
                lambda_reg_warmup_steps=lambda_reg_warmup_steps,
                **ppo_kwargs,
            )
        elif policy_type == "standard":
            model = PPO(
                policy="MlpPolicy",
                **ppo_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown policy_type '{policy_type}'. "
                f"Expected 'evidential' or 'standard'."
            )

    # Set up SB3 logger (TensorBoard + stdout). Named sb3_logger to avoid
    # shadowing the module-level Python logger.
    sb3_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(sb3_logger)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 50000),
        save_path=checkpoint_dir,
        name_prefix="ppo_uncertainty_rl",
        save_vecnormalize=True,
    )

    # Per-bay success accounting. Cumulative attempts/successes per target bay
    # over the whole run, dumped to outputs/bay_successes/training/<run_name>/
    # so a random-bay run can be inspected for which bays the policy can park.
    _bay_output_dir = Path("./outputs/bay_successes/training") / run_name
    _bay_run_info: Dict[str, Any] = {
        "run_name": run_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "resumed_from": resume_from if resume_from is not None else "(fresh)",
        "started": datetime.now().strftime("%d-%m-%Y %H:%M"),
    }

    callbacks = [
        checkpoint_callback,
        EnvDiagnosticsCallback(
            outcome_window=config.get("success_rate_window", 50)
        ),
        BaySuccessCallback(
            output_dir=_bay_output_dir,
            run_info=_bay_run_info,
            dump_freq_steps=config.get("checkpoint_freq", 50000),
        ),
    ]
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=checkpoint_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    # Append extra callbacks (e.g. for Optuna trial evaluation)
    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    callback_list = CallbackList(callbacks)

    # Train the agent (wrapped in try/finally for CARLA crash safety)
    try:
        logger.info("Starting training for %d timesteps...", total_timesteps)
        # Disable progress bar when env is caller-owned (e.g. Optuna tuning).
        # tqdm[rich] leaks a "live display" between trials when model.learn()
        # is called repeatedly in one process, so the second call onwards
        # raises "Only one live display may be active at once".
        reset_num_ts = resume_from is None
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=config.get("log_interval", 10),
            progress_bar=own_env,
            reset_num_timesteps=reset_num_ts,
        )

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "final_model")
        model.save(final_model_path)
        assert env is not None
        env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))

        logger.info("Training complete. Model saved to %s", final_model_path)

    finally:
        # Clean up (always runs, even if CARLA crashes during training).
        if own_env:
            assert env is not None
            env.close()
        if eval_env is not None:
            eval_env.close()

    # Collect final metrics from logger
    final_metrics = dict(model.logger.name_to_value)

    return TrainResult(
        final_metrics=final_metrics,
        model_path=final_model_path,
        log_dir=log_dir,
        env=None if own_env else env,
    )


def main() -> None:
    """
    @brief Main entry point for training script.

    CLI arguments override values from the YAML config file. The config file
    is the single source of truth; CLI args are convenience overrides for
    per-run settings (e.g. --seed 7 for a specific ablation run).
    """
    parser = argparse.ArgumentParser(
        description=("Train uncertainty-conditioned RL agent for autonomous parking")
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help=(
            "Path to training hyperparameter config "
            "(PPO, network, evidential settings)"
        ),
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/deployment/sim/env_config.yaml",
        help="Path to environment config (CARLA, sensors, parking scenarios)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory from config",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory from config",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Override evaluation frequency from config",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from checkpoint directory (contains final_model.zip and vec_normalize.pkl)",
    )

    args = parser.parse_args()

    # Load and merge configs, then apply CLI overrides.
    # load_env_config() merges sensor_config.yaml (shared keys) with env_config.yaml
    # (CARLA-specific keys) so all consumers see a single unified dict.
    config = merge_configs(
        load_config(args.train_config), load_env_config(args.env_config)
    )
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.eval_freq is not None:
        config["eval_freq"] = args.eval_freq
    if args.n_eval_episodes is not None:
        config["n_eval_episodes"] = args.n_eval_episodes
    if args.seed is not None:
        config["seed"] = args.seed

    # Configure logging after all overrides are applied.
    _log_level = logging.DEBUG if config.get("debug", False) else logging.INFO
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Pass resume checkpoint path if provided.
    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
