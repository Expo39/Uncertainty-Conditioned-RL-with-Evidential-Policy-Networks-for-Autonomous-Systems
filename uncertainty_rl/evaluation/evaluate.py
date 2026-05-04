"""
@file evaluate.py
@brief Evaluation script for trained agents across physical conditions.

This module provides comprehensive evaluation of trained agents under varying
physical conditions (sensor noise, traffic density) that produce different
EKF uncertainty levels.
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

try:
    import torch as th
except ImportError:
    th = None  # type: ignore[assignment,misc]

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    PPO = None  # type: ignore[assignment,misc]
    DummyVecEnv = None  # type: ignore[assignment,misc]
    VecNormalize = None  # type: ignore[assignment,misc]

try:
    from uncertainty_rl.envs import CARLAParkingEnv
    from uncertainty_rl.envs.safety_wrapper import SafetyWrapper
    from uncertainty_rl.networks.sb3_integration import EvidentialPPO
except ImportError:
    CARLAParkingEnv = None  # type: ignore[assignment,misc]
    SafetyWrapper = None  # type: ignore[assignment,misc]
    EvidentialPPO = None  # type: ignore[assignment,misc]

logger = logging.getLogger("uncertainty_rl.evaluation")


@dataclass
class EvaluationMetrics:
    """
    @class EvaluationMetrics
    @brief Container for evaluation metrics.

    Fields cover per-condition success rate, reward, step counts, final pose
    errors (reserved; not populated by current env), and per-step evidential
    uncertainty estimates (evidential policy only).
    """

    success_rate: float = 0.0
    average_reward: float = 0.0
    average_steps: float = 0.0
    position_errors: List[float] = field(default_factory=list)
    orientation_errors: List[float] = field(default_factory=list)
    epistemic_uncertainties: List[float] = field(default_factory=list)
    aleatoric_uncertainties: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        @brief Convert metrics to dictionary.
        @return Dictionary of metrics.
        """
        def _mean_std(lst: List[float]) -> Tuple[float, float]:
            if not lst:
                return 0.0, 0.0
            arr = np.asarray(lst)
            return float(arr.mean()), float(arr.std())

        pos_mean, pos_std = _mean_std(self.position_errors)
        ori_mean, ori_std = _mean_std(self.orientation_errors)
        epi_mean, _ = _mean_std(self.epistemic_uncertainties)
        ale_mean, _ = _mean_std(self.aleatoric_uncertainties)

        return {
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "mean_position_error": pos_mean,
            "std_position_error": pos_std,
            "mean_orientation_error": ori_mean,
            "std_orientation_error": ori_std,
            "mean_epistemic_uncertainty": epi_mean,
            "mean_aleatoric_uncertainty": ale_mean,
        }


def _scale_sensor_noise(
    base_sensors: Dict[str, Any],
    imu_multiplier: float,
) -> Dict[str, Any]:
    """
    @brief Scale base sensor noise parameters by the condition-specific IMU multiplier.
    @param base_sensors: Base sensor config from train_config.yaml.
    @param imu_multiplier: Multiplier for all IMU noise stddev values.
    @return New sensor config dict with scaled noise values.
    """
    base_imu: Dict[str, Any] = base_sensors.get("imu", {})
    scaled_imu = {
        k: (v * imu_multiplier if "stddev" in k else v)
        for k, v in base_imu.items()
    }
    return {**base_sensors, "imu": scaled_imu}


def make_eval_env(
    condition: Dict[str, Any],
    config: Dict[str, Any],
    base_sensors: Dict[str, Any],
    env_config: Optional[Dict[str, Any]] = None,
) -> DummyVecEnv:
    """
    @brief Create evaluation environment for a specific physical condition.
    @param condition: Condition dict with noise multipliers and traffic counts.
    @param config: Evaluation configuration dictionary.
    @param base_sensors: Base sensor noise config from env_config.yaml.
    @param env_config: Environment config for parking_scenarios and obs flags.
    @return Vectorised evaluation environment.
    """
    # Scale sensor noise by condition multipliers
    scaled_sensors = _scale_sensor_noise(
        base_sensors,
        imu_multiplier=condition.get("imu_noise_multiplier", 1.0),
    )

    # Build parking_scenarios_config: NPC counts and lot layout from condition
    # overrides + training defaults. eval_config.yaml uses num_patrol_vehicles
    # (not num_vehicles) to match CARLAParkingEnv's parking_scenarios_config keys.
    base_scenarios: Dict[str, Any] = (
        dict(env_config.get("parking_scenarios", {})) if env_config is not None else {}
    )
    floor_plans: Dict[str, Any] = base_scenarios.get("floor_plans", {})

    # Perimeter cone flag: per-condition > eval_config global > env_config default.
    _cones_fallback = base_scenarios.get(
        "spawn_perimeter_cones", config.get("spawn_perimeter_cones", False)
    )
    spawn_cones: bool = condition.get("spawn_perimeter_cones", _cones_fallback)

    # In evaluation, occupancy is fixed per condition (min == max).
    # eval_config.yaml uses bay_occupancy_rate (a single value); training uses
    # bay_occupancy_min/max for the per-episode uniform resample range.
    bay_occupancy: float = condition.get(
        "bay_occupancy_rate", base_scenarios.get("bay_occupancy_max", 0.6)
    )

    parking_config: Dict[str, Any] = {
        "spawn_perimeter_cones": spawn_cones,
        "num_patrol_vehicles_max": condition.get("num_patrol_vehicles", 0),
        "pedestrian_spawn_probability": condition.get(
            "pedestrian_spawn_probability", 1.0
        ),
        "bay_occupancy_min": bay_occupancy,
        "bay_occupancy_max": bay_occupancy,
        "floor_plans": floor_plans,
    }
    # Allow per-condition floor plan override (e.g. OOD evaluation)
    if "floor_plan" in condition:
        floor_plan_name: str = condition["floor_plan"]
        if floor_plan_name in floor_plans:
            parking_config["floor_plans"] = {
                floor_plan_name: floor_plans[floor_plan_name]
            }

    # Observation flags and env-specific settings resolved once from env_config
    _ec = env_config if env_config is not None else {}
    include_covariance: bool = _ec.get("include_covariance", True)
    include_obstacle_obs: bool = _ec.get("include_obstacle_obs", True)
    use_extra_spawns: bool = _ec.get("use_extra_spawns", False)
    gnss_profiles_path: Optional[str] = _ec.get("gnss_noise_profiles", None)

    # GNSS noise multiplier override: locks tier for this eval condition.
    gnss_override: Optional[float] = condition.get("gnss_noise_multiplier", None)

    # debug: per-step DebugLogger diagnostics - off by default, same as training.
    debug: bool = config.get("debug", False)

    # SafetyWrapper parameters from agent_config.yaml
    aleatoric_scaling: float = float(config.get("safety_aleatoric_scaling", 0.5))
    handoff_threshold: float = float(config.get("safety_handoff_threshold", 5.0))

    def _init() -> Any:
        base_env: Any = CARLAParkingEnv(
            carla_host=config.get("carla_host", "localhost"),
            carla_port=config.get("carla_port", 2000),
            town=config.get("town", "FlatPlane"),
            max_steps=config.get("max_steps", 500),
            ros2_config=config.get("ros2", {}),
            carla_sensors_config=scaled_sensors,
            parking_scenarios_config=parking_config,
            include_covariance=include_covariance,
            include_obstacle_obs=include_obstacle_obs,
            use_extra_spawns=use_extra_spawns,
            gnss_noise_profiles_path=gnss_profiles_path,
            gnss_noise_multiplier_override=gnss_override,
            debug=debug,
        )
        return SafetyWrapper(
            base_env,
            aleatoric_scaling=aleatoric_scaling,
            handoff_threshold=handoff_threshold,
        )

    env = DummyVecEnv([_init])
    return env


def evaluate_agent(
    model: PPO,
    env: Union[DummyVecEnv, VecNormalize],
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
) -> EvaluationMetrics:
    """
    @brief Evaluate agent performance.
    @param model: Trained PPO model.
    @param env: Evaluation environment.
    @param n_episodes: Number of evaluation episodes.
    @param deterministic: Use deterministic actions.
    @param render: Render episodes.
    @return EvaluationMetrics object with results.

    @note For EvidentialPPO models, uses get_action_with_uncertainty() to
          collect per-step epistemic and aleatoric uncertainty estimates.
          Success is determined from the environment's info dict (set by
          CARLAParkingEnv.step()) rather than re-computing from final state.
    """
    metrics = EvaluationMetrics()

    episode_rewards = np.empty(n_episodes, dtype=np.float64)
    episode_steps = np.empty(n_episodes, dtype=np.int32)
    success_flags = np.zeros(n_episodes, dtype=bool)

    # Detect evidential policy once; hoist method references out of the loop.
    is_evidential = isinstance(model, EvidentialPPO) and hasattr(
        model.policy, "get_action_with_uncertainty"
    )

    # Bind a single step function to eliminate the per-step branch.
    _StepReturn = Tuple[np.ndarray, float, np.ndarray, List[Dict[str, Any]]]
    step_fn: Callable[[np.ndarray], _StepReturn]
    if is_evidential:
        _get_action_with_uncertainty = (
            model.policy.get_action_with_uncertainty  # type: ignore[union-attr]
        )
        _set_uncertainty = env.env_method

        def step_fn(obs: np.ndarray) -> _StepReturn:  # type: ignore[misc]
            obs_tensor = th.as_tensor(obs)
            action_tensor, uncertainty_dict = _get_action_with_uncertainty(
                obs_tensor, deterministic=deterministic
            )
            action = action_tensor.cpu().numpy()
            epistemic = float(uncertainty_dict["epistemic"].mean().item())
            aleatoric = float(uncertainty_dict["aleatoric"].mean().item())
            metrics.epistemic_uncertainties.append(epistemic)
            metrics.aleatoric_uncertainties.append(aleatoric)
            _set_uncertainty("set_uncertainty", epistemic, aleatoric)
            next_obs, reward, done, infos = env.step(action)
            return next_obs, float(reward[0]), done, infos
    else:
        def step_fn(obs: np.ndarray) -> _StepReturn:  # type: ignore[misc]
            action, _states = model.predict(obs, deterministic=deterministic)
            next_obs, reward, done, infos = env.step(action)
            return next_obs, float(reward[0]), done, infos

    for episode in range(n_episodes):
        obs: np.ndarray = env.reset()
        episode_reward = 0.0
        steps = 0
        done = np.array([False])

        while not done[0]:
            obs, step_reward, done, infos = step_fn(obs)
            episode_reward += step_reward
            steps += 1
            if done[0]:
                # DummyVecEnv.step() returns (obs, rewards, dones, infos) - 4 elements.
                success_flags[episode] = infos[0].get("success", False)
                if render:
                    env.render()

        episode_rewards[episode] = episode_reward
        episode_steps[episode] = steps

    metrics.success_rate = float(success_flags.sum()) / n_episodes * 100.0
    metrics.average_reward = float(episode_rewards.mean())
    metrics.average_steps = float(episode_steps.mean())

    return metrics


def evaluate_across_conditions(
    model_path: str,
    eval_config_path: str,
    env_config_path: str,
    train_config_path: str,
    n_episodes: int = 0,
    output_dir: str = "./evaluation_results",
) -> pd.DataFrame:
    """
    @brief Evaluate agent across different physical conditions.
    @param model_path: Path to trained model.
    @param eval_config_path: Path to evaluation configuration file.
    @param env_config_path: Path to environment config (sensors, parking scenarios).
    @param train_config_path: Path to training config (policy_type for model loading).
    @param n_episodes: Episodes per condition. 0 means read from eval_config
        (n_episodes key), falling back to 100.
    @param output_dir: Directory to save results.
    @return DataFrame with evaluation results.
    """
    from uncertainty_rl.training.train_ppo import load_env_config

    # Load configurations
    with open(eval_config_path, "r") as f:
        eval_config: Dict[str, Any] = yaml.safe_load(f)

    # load_env_config merges sensor_config.yaml (shared keys) with env_config.yaml
    # (CARLA-specific keys) so env_config is the single unified config for the env.
    env_config: Dict[str, Any] = load_env_config(env_config_path)

    with open(train_config_path, "r") as f:
        train_config: Dict[str, Any] = yaml.safe_load(f)

    base_sensors = env_config.get("carla_sensors", {})
    conditions = eval_config.get("eval_conditions", [])
    # n_episodes: caller can override; fall back to eval_config, then hard default.
    n_episodes = n_episodes or int(eval_config.get("n_episodes", 100))

    # Load model
    logger.info("Loading model from %s...", model_path)
    # Load model: use EvidentialPPO when train_config specifies policy_type=evidential
    # so that isinstance(model, EvidentialPPO) is True and uncertainty is collected.
    policy_type = train_config.get("policy_type", "evidential")
    if policy_type == "evidential":
        model: PPO = EvidentialPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    # Load normalisation statistics if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    results = []
    vec_normalize_exists = os.path.exists(vec_normalize_path)
    if not vec_normalize_exists:
        logger.warning(
            "No VecNormalize stats found at %s. "
            "Running without observation normalisation.",
            vec_normalize_path,
        )
    deterministic: bool = eval_config.get("deterministic", True)

    for condition in conditions:
        name = condition.get("name", "unknown")
        description = condition.get("description", "")
        logger.info("Evaluating condition: %s - %s", name, description)

        # Create environment for this condition
        base_env = make_eval_env(condition, eval_config, base_sensors, env_config)
        eval_env: Union[DummyVecEnv, VecNormalize] = base_env

        # Apply normalisation if available
        if vec_normalize_exists:
            eval_env = VecNormalize.load(vec_normalize_path, base_env)
            eval_env.training = False
            eval_env.norm_reward = False

        # Evaluate
        metrics = evaluate_agent(
            model=model,
            env=eval_env,
            n_episodes=n_episodes,
            deterministic=deterministic,
        )

        # Store results: merge metrics dict with condition metadata in one pass
        optional_fields = (
            {"floor_plan": condition["floor_plan"]} if "floor_plan" in condition else {}
        )
        result = {
            **metrics.to_dict(),
            "condition": name,
            "description": description,
            "gnss_noise_multiplier": condition.get("gnss_noise_multiplier", 1.0),
            "imu_noise_multiplier": condition.get("imu_noise_multiplier", 1.0),
            "num_patrol_vehicles": condition.get("num_patrol_vehicles", 0),
            "pedestrian_spawn_probability": condition.get(
                "pedestrian_spawn_probability", 1.0
            ),
            "bay_occupancy_rate": condition.get("bay_occupancy_rate", 0.6),
            **optional_fields,
        }
        results.append(result)

        logger.info(
            "  success_rate=%.1f%%  avg_reward=%.2f  avg_steps=%.0f",
            metrics.success_rate,
            metrics.average_reward,
            metrics.average_steps,
        )

        # Clean up
        eval_env.close()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    return df


def plot_evaluation_results(
    df: pd.DataFrame,
    output_dir: str = "./evaluation_results",
) -> None:
    """
    @brief Create visualisations of evaluation results across conditions.
    @param df: DataFrame with evaluation results.
    @param output_dir: Directory to save plots.
    """
    sns.set_style("whitegrid")

    conditions = df["condition"].tolist()
    x_arr = np.arange(len(conditions))
    x_list = x_arr.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plots 1-3: single-series bar charts, data-driven
    bar_specs = [
        (axes[0, 0], "success_rate",   "steelblue",
         "Success Rate (%)",  "Success Rate vs Condition"),
        (axes[0, 1], "average_reward", "forestgreen",
         "Average Reward",    "Average Reward vs Condition"),
        (axes[1, 0], "average_steps",  "firebrick",
         "Average Steps",     "Average Steps to Termination vs Condition"),
    ]
    for ax, col, colour, ylabel, title in bar_specs:
        ax.bar(x_list, df[col], color=colour, alpha=0.8)
        ax.set_xticks(x_list)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Policy uncertainty estimates (grouped bars)
    if "mean_epistemic_uncertainty" in df.columns:
        bar_width = 0.35
        axes[1, 1].bar(
            x_arr - bar_width / 2, df["mean_epistemic_uncertainty"],
            bar_width, label="Epistemic", alpha=0.8,
        )
        axes[1, 1].bar(
            x_arr + bar_width / 2, df["mean_aleatoric_uncertainty"],
            bar_width, label="Aleatoric", alpha=0.8,
        )
        axes[1, 1].set_xticks(x_list)
        axes[1, 1].set_xticklabels(conditions, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Uncertainty", fontsize=12)
        axes[1, 1].set_title("Policy Uncertainty Estimates", fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Plots saved to %s", plot_path)

    plt.close()


def main() -> None:
    """
    @brief Main entry point for evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained agent across physical conditions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation configuration file",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="configs/deployment/sim/env_config.yaml",
        help="Path to environment config (sensors, parking scenarios)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config (for policy_type used in model loading)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=0,
        help="Episodes per condition (0 = read from eval_config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Configure logging level: DEBUG when debug:true in eval_config, else INFO.
    # This also enables the env's DebugLogger per-step output.
    with open(args.eval_config, "r") as _f:
        _cfg: Dict[str, Any] = yaml.safe_load(_f)
    _log_level = logging.DEBUG if _cfg.get("debug", False) else logging.INFO
    logging.basicConfig(
        level=_log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Run evaluation
    df = evaluate_across_conditions(
        model_path=args.model_path,
        eval_config_path=args.eval_config,
        env_config_path=args.env_config,
        train_config_path=args.train_config,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
    )

    # Create plots
    plot_evaluation_results(df, output_dir=args.output_dir)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
