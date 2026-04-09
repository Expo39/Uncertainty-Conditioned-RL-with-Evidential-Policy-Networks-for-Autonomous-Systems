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
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
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
    # Reserved: re-enable when multi-instance CARLA eval is supported.
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

try:
    from uncertainty_rl.envs import CARLAParkingEnv
    from uncertainty_rl.networks import EvidentialActorCriticPolicy, EvidentialPPO
except ImportError:
    CARLAParkingEnv = None  # type: ignore[assignment,misc]
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


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    @brief Linear learning rate schedule decaying to zero.
    @param initial_value: Initial learning rate.
    @return Callable that takes progress_remaining (1.0 -> 0.0) and returns LR.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def make_env(
    config: Dict[str, Any],
    rank: int = 0,
    carla_sensors_override: Optional[Dict[str, Any]] = None,
) -> Callable[[], gym.Env]:
    """
    @brief Create a callable that returns a new environment instance.
    @param config: Configuration dictionary.
    @param rank: Environment rank for seeding.
    @param carla_sensors_override: Override sensor noise config (for evaluation).
    @return Callable that creates and returns a CARLAParkingEnv instance.
    """

    def _init() -> gym.Env:
        # Each worker connects to its own CARLA server on a separate port and host.
        # Port stride is 1000 so worker ports (2000, 3000, ...) never collide
        # with CARLA's 3-port block (world+streaming+RPC on base, base+1, base+2).
        # Host: container names uncertainty-rl-carla-0, uncertainty-rl-carla-1, ...
        # Container names (not service names) are used because each worker is a
        # separate compose invocation sharing the same bridge network -- Docker DNS
        # resolves container names across compose projects on a shared network.
        worker_port = config.get("carla_port", 2000) + rank * 1000
        worker_host = f"uncertainty-rl-carla-{rank}"

        # Per-worker EKF state file so each worker reads from its own ros2-bridge.
        # Worker 0 uses the default path (backward compatible with single-instance).
        # Worker 1+ derive: ekf_state.json -> ekf_state_1.json, ekf_state_2.json, etc.
        ros2_config = config.get("ros2", {}).copy()
        if rank > 0:
            base_ekf = ros2_config.get(
                "ekf_state_file", "/workspace/outputs/ekf_state.json"
            )
            p = Path(base_ekf)
            ros2_config["ekf_state_file"] = str(p.parent / f"{p.stem}_{rank}{p.suffix}")

        # Per-worker vis_history file so make visualise WORKER=N shows the right env.
        # Worker 0 uses the CARLAParkingEnv default (outputs/vis_history.jsonl).
        # Worker N writes to outputs/vis_history_N.jsonl.
        vis_path: Optional[str] = (
            None if rank == 0 else f"outputs/vis_history_{rank}.jsonl"
        )

        env = CARLAParkingEnv(
            carla_host=worker_host,
            carla_port=worker_port,
            town=config.get("town", "FlatPlane"),
            max_steps=config.get("max_steps", 500),
            ros2_config=ros2_config,
            carla_sensors_config=(
                carla_sensors_override
                if carla_sensors_override is not None
                else config.get("carla_sensors", {})
            ),
            parking_scenarios_config=config.get("parking_scenarios", {}),
            include_covariance=config.get("include_covariance", True),
            include_obstacle_obs=config.get("include_obstacle_obs", True),
            carla_timestep=config.get("carla_timestep", 0.05),
            debug=config.get("debug", False),
            map_load_sleep=config.get("map_load_sleep", 5.0),
            action_repeat=config.get("action_repeat", 1),
            no_rendering_mode=config.get("no_rendering_mode", False),
            max_ego_speed_ms=config.get("max_ego_speed_ms", 6.0),
            use_extra_spawns=config.get("use_extra_spawns", False),
            gnss_noise_profiles_path=config.get("gnss_noise_profiles", None),
            vis_output_path=vis_path,
            real_world_deployment=config.get("real_world_deployment", False),
            real_world_datum_path=config.get("real_world_datum", None),
            actuation_calibration_path=config.get("actuation_calibration", None),
        )
        return env

    return _init


def load_config(config_path: str) -> Dict[str, Any]:
    """
    @brief Load configuration from YAML file.
    @param config_path: Path to configuration file.
    @return Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


def merge_configs(
    train_config: Dict[str, Any], env_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Merge training and environment configs into a single dict.

    env_config keys take precedence for environment settings. train_config
    keys take precedence for training hyperparameters. In practice they have
    no overlapping keys so this is a simple union.

    @param train_config: Training hyperparameters from train_config.yaml.
    @param env_config: Environment settings from env_config.yaml.
    @return Merged configuration dictionary.
    """
    merged = {**env_config, **train_config}
    return merged


def load_env_config(env_config_path: str) -> Dict[str, Any]:
    """
    @brief Load and merge the environment config with the shared sensor config.

    sensor_config.yaml (configs/deployment/sensor_config.yaml) is the single
    source of truth for all parameters shared between simulation and the real
    vehicle: sensor mounts, agent parameters (max_steps, action_repeat,
    max_ego_speed_ms), observation flags, ROS 2 settings, and deployment flags.

    env_config.yaml (configs/deployment/sim/env_config.yaml) contains only
    CARLA-specific parameters: noise profiles, simulation timing, scenario
    geometry. Keys in env_config take precedence over sensor_config on conflict.

    Sensor mounts from sensor_config.sensors.*.mount are injected into
    carla_sensors.*.mount so the CARLA spawner receives them.

    @param env_config_path: Path to the CARLA environment config YAML.
    @return Fully merged environment configuration dictionary.
    """
    with open(env_config_path) as f:
        env_config: Dict[str, Any] = yaml.safe_load(f) or {}

    # Resolve sensor_config.yaml relative to deployment/ (one level up from sim/).
    sensor_cfg_path = Path(env_config_path).parent.parent / "sensor_config.yaml"
    sensor_cfg: Dict[str, Any] = {}
    if sensor_cfg_path.exists():
        with open(sensor_cfg_path) as f:
            sensor_cfg = yaml.safe_load(f) or {}

    # sensor_config provides the base; env_config overrides with sim-specific keys.
    merged: Dict[str, Any] = {**sensor_cfg, **env_config}

    # Inject sensor mounts into carla_sensors so the CARLA spawner gets them.
    for sensor_name, sensor_data in sensor_cfg.get("sensors", {}).items():
        if "mount" in sensor_data and sensor_name in merged.get("carla_sensors", {}):
            merged["carla_sensors"][sensor_name]["mount"] = sensor_data["mount"]

    return merged


class EnvDiagnosticsCallback(BaseCallback):
    """
    @class EnvDiagnosticsCallback
    @brief SB3 callback that logs per-episode environment diagnostics to TensorBoard.

    Reads from the `info` dict returned by `CARLAParkingEnv.step()` and records
    rolling means of reward components (position error, orientation error, speed,
    progress reward) plus episode outcome rates (success, collision, timeout).
    These appear under the `env/` namespace in TensorBoard alongside SB3's
    built-in `rollout/` and `train/` metrics.

    Logged at every rollout collection step (i.e. every `n_steps` environment
    steps), matching the cadence of SB3's own metric dumps.
    """

    def __init__(self) -> None:
        """@brief Initialise accumulators."""
        super().__init__(verbose=0)
        self._ep_pos_errors: List[float] = []
        self._ep_orientation_errors: List[float] = []
        self._ep_speeds: List[float] = []
        self._ep_progress_rewards: List[float] = []
        self._ep_successes: List[float] = []
        self._ep_collisions: List[float] = []
        self._ep_timeouts: List[float] = []

    def _on_step(self) -> bool:
        """
        @brief Accumulate diagnostics from the latest env step.
        @return Always True (training continues).
        """
        # self.locals["infos"] is a list of info dicts, one per parallel env.
        for info in self.locals.get("infos", []):
            self._ep_pos_errors.append(float(info.get("pos_error", 0.0)))
            self._ep_orientation_errors.append(
                float(info.get("orientation_error", 0.0))
            )
            self._ep_speeds.append(float(info.get("speed", 0.0)))
            self._ep_progress_rewards.append(float(info.get("progress_reward", 0.0)))
            # Episode-terminal flags -- count when episode ended.
            is_terminal = (
                info.get("success", False)
                or info.get("collision", False)
                or info.get("timeout", False)
            )
            if is_terminal:
                self._ep_successes.append(1.0 if info.get("success", False) else 0.0)
                self._ep_collisions.append(1.0 if info.get("collision", False) else 0.0)
                self._ep_timeouts.append(1.0 if info.get("timeout", False) else 0.0)
        return True

    def _on_rollout_end(self) -> None:
        """
        @brief Flush accumulated diagnostics to TensorBoard at rollout end."""
        if self._ep_pos_errors:
            self.logger.record(
                "env/mean_pos_error_m",
                float(np.mean(self._ep_pos_errors)),
            )
            self.logger.record(
                "env/mean_orientation_error_rad",
                float(np.mean(self._ep_orientation_errors)),
            )
            self.logger.record(
                "env/mean_speed_ms",
                float(np.mean(self._ep_speeds)),
            )
            self.logger.record(
                "env/mean_progress_reward",
                float(np.mean(self._ep_progress_rewards)),
            )

        if self._ep_successes:
            self.logger.record(
                "env/success_rate",
                float(np.mean(self._ep_successes)),
            )
            self.logger.record(
                "env/collision_rate",
                float(np.mean(self._ep_collisions)),
            )
            self.logger.record(
                "env/timeout_rate",
                float(np.mean(self._ep_timeouts)),
            )

        # Reset accumulators for the next rollout window.
        self._ep_pos_errors.clear()
        self._ep_orientation_errors.clear()
        self._ep_speeds.clear()
        self._ep_progress_rewards.clear()
        self._ep_successes.clear()
        self._ep_collisions.clear()
        self._ep_timeouts.clear()


def train(
    config: Dict[str, Any],
    extra_callbacks: Optional[List[BaseCallback]] = None,
) -> TrainResult:
    """
    @brief Train the uncertainty-conditioned RL agent with PPO.
    @param config: Fully-resolved configuration dictionary. All operational
           settings (seed, log_dir, etc.) and hyperparameters are read from
           this dict. CLI arguments override YAML values before this is called.
    @param extra_callbacks: Optional list of additional callbacks to append to
           the training callback list (e.g. for Optuna trial evaluation).
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

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create training environment -- one worker per CARLA instance.
    # parallel_workers > 1 requires docker-compose.parallel.yml (see make docker-train-parallel).
    n_workers: int = config.get("parallel_workers", 1)
    logger.info(f"Creating training environment ({n_workers} worker(s))...")
    train_vec_env = DummyVecEnv([make_env(config, rank=i) for i in range(n_workers)])

    # Normalise observations but not rewards - reward components will be
    # manually scaled via potential-based shaping (see reward TODO in config)
    env = VecNormalize(
        train_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

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

    # Shared PPO hyperparameters
    lr_initial = config.get("learning_rate", 3e-4)
    lr_schedule = linear_schedule(lr_initial)

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
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        target_kl=config.get("target_kl", 0.02),
        policy_kwargs=policy_kwargs,
        verbose=config.get("verbose", 1),
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Create agent based on policy_type config
    policy_type = config.get("policy_type", "evidential")
    logger.info("Initialising %s PPO agent...", policy_type)

    model: PPO
    if policy_type == "evidential":
        evidential_config = config.get("evidential", {})
        lambda_reg = evidential_config.get("lambda_reg", 0.01)
        lambda_reg_warmup_steps = evidential_config.get(
            "lambda_reg_warmup_steps", 50000
        )
        use_uncertainty_conditioning = evidential_config.get(
            "use_uncertainty_conditioning", False
        )
        # Forward conditioning flag into policy_kwargs so the policy can wire the
        # dual-encoder actor (UncertaintyConditionedActor) when requested.
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

    callbacks = [checkpoint_callback, EnvDiagnosticsCallback()]
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
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=config.get("log_interval", 10),
            progress_bar=True,
        )

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))

        logger.info("Training complete. Model saved to %s", final_model_path)

    finally:
        # Clean up (always runs, even if CARLA crashes during training)
        env.close()
        if eval_env is not None:
            eval_env.close()

    # Collect final metrics from logger
    final_metrics = dict(model.logger.name_to_value)

    return TrainResult(
        final_metrics=final_metrics,
        model_path=final_model_path,
        log_dir=log_dir,
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

    args = parser.parse_args()

    # Load and merge configs, then apply CLI overrides.
    # load_env_config() merges sensor_config.yaml (shared keys) with env_config.yaml
    # (CARLA-specific keys) so all consumers see a single unified dict.
    config = merge_configs(load_config(args.train_config), load_env_config(args.env_config))
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

    train(config)


if __name__ == "__main__":
    main()
