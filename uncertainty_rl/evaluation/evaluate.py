"""
@file evaluate.py
@brief Evaluation script for trained agents across physical conditions.

This module provides comprehensive evaluation of trained agents under varying
physical conditions (weather, sensor noise, traffic) that produce different
EKF uncertainty levels. Replaces the previous noise-level sweep with a
condition-based sweep driven by eval_config.yaml.
"""

import argparse
import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import CARLAParkingEnv
from uncertainty_rl.networks.sb3_integration import EvidentialPPO


@dataclass
class EvaluationMetrics:
    """
    @class EvaluationMetrics
    @brief Container for evaluation metrics.

    @var success_rate: Percentage of successful parking attempts.
    @var average_reward: Mean episode reward.
    @var average_steps: Mean number of steps to completion.
    @var position_errors: List of final position errors.
    @var orientation_errors: List of final orientation errors.
    @var epistemic_uncertainties: Epistemic uncertainty values during episodes.
    @var aleatoric_uncertainties: Aleatoric uncertainty values during episodes.
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
        return {
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "mean_position_error": (
                float(np.mean(self.position_errors)) if self.position_errors else 0.0
            ),
            "std_position_error": (
                float(np.std(self.position_errors)) if self.position_errors else 0.0
            ),
            "mean_orientation_error": (
                float(np.mean(self.orientation_errors))
                if self.orientation_errors
                else 0.0
            ),
            "std_orientation_error": (
                float(np.std(self.orientation_errors))
                if self.orientation_errors
                else 0.0
            ),
            "mean_epistemic_uncertainty": (
                float(np.mean(self.epistemic_uncertainties))
                if self.epistemic_uncertainties
                else 0.0
            ),
            "mean_aleatoric_uncertainty": (
                float(np.mean(self.aleatoric_uncertainties))
                if self.aleatoric_uncertainties
                else 0.0
            ),
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

    @note Only IMU noise is scaled. Suite A uses 2D LiDAR + IMU only; GNSS is
          excluded (unreliable in covered parking lots).
    """
    scaled = copy.deepcopy(base_sensors)

    imu = scaled.get("imu", {})
    for key in imu:
        if "stddev" in key:
            imu[key] = imu[key] * imu_multiplier
    scaled["imu"] = imu

    return scaled


def make_eval_env(
    condition: Dict[str, Any],
    config: Dict[str, Any],
    base_sensors: Dict[str, Any],
    train_config: Optional[Dict[str, Any]] = None,
) -> DummyVecEnv:
    """
    @brief Create evaluation environment for a specific physical condition.
    @param condition: Condition dict with weather, noise multipliers, traffic.
    @param config: Evaluation configuration dictionary.
    @param base_sensors: Base sensor noise config from training.
    @param train_config: Full training config for parking_scenarios and obs flags.
    @return Vectorised evaluation environment.
    """
    # Scale sensor noise by condition multipliers
    scaled_sensors = _scale_sensor_noise(
        base_sensors,
        imu_multiplier=condition.get("imu_noise_multiplier", 1.0),
    )

    # Build fixed weather conditions config (not randomised, unlike training)
    conditions_config = {
        "weather_presets": [condition.get("weather_preset", "ClearNoon")],
    }

    # Build parking_scenarios_config: NPC counts and lot layout from condition
    # overrides + training defaults. eval_config.yaml uses num_patrol_vehicles
    # (not num_vehicles) to match CARLAParkingEnv's parking_scenarios_config keys.
    base_scenarios: Dict[str, Any] = {}
    if train_config is not None:
        base_scenarios = dict(train_config.get("parking_scenarios", {}))
    parking_config: Dict[str, Any] = {
        "num_patrol_vehicles_max": condition.get("num_patrol_vehicles", 0),
        "pedestrian_spawn_probability": condition.get("pedestrian_spawn_probability", 1.0),
        # In evaluation, occupancy is fixed per condition (min == max).
        # eval_config.yaml uses bay_occupancy_rate (a single value); training uses
        # bay_occupancy_min/max for the per-episode uniform resample range.
        "bay_occupancy_min": condition.get(
            "bay_occupancy_rate",
            base_scenarios.get("bay_occupancy_max", 0.6),
        ),
        "bay_occupancy_max": condition.get(
            "bay_occupancy_rate",
            base_scenarios.get("bay_occupancy_max", 0.6),
        ),
        "floor_plans": base_scenarios.get("floor_plans", {}),
    }
    # Allow per-condition floor plan override (e.g. OOD evaluation)
    if "floor_plan" in condition:
        floor_plan_name: str = condition["floor_plan"]
        floor_plans = base_scenarios.get("floor_plans", {})
        if floor_plan_name in floor_plans:
            parking_config["floor_plans"] = {
                floor_plan_name: floor_plans[floor_plan_name]
            }

    # Observation flags from training config (so baseline-specific obs dims are respected)
    include_covariance: bool = True
    include_obstacle_obs: bool = True
    sensor_suite: str = "suite_a"
    if train_config is not None:
        include_covariance = bool(train_config.get("include_covariance", True))
        include_obstacle_obs = bool(train_config.get("include_obstacle_obs", True))
        sensor_suite = str(train_config.get("sensor_suite", "suite_a"))

    def _init() -> CARLAParkingEnv:
        return CARLAParkingEnv(
            carla_host=config.get("carla_host", "localhost"),
            carla_port=config.get("carla_port", 2000),
            town=config.get("town", "FlatPlane"),
            max_steps=config.get("max_steps", 500),
            ros2_config=config.get("ros2", {}),
            carla_sensors_config=scaled_sensors,
            carla_conditions_config=conditions_config,
            parking_scenarios_config=parking_config,
            include_covariance=include_covariance,
            include_obstacle_obs=include_obstacle_obs,
            sensor_suite=sensor_suite,
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

    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    successes = 0

    # Detect evidential policy to enable uncertainty collection
    is_evidential = isinstance(model, EvidentialPPO) and hasattr(
        model.policy, "get_action_with_uncertainty"
    )

    for episode in range(n_episodes):
        obs = cast(np.ndarray, env.reset())
        done_arr = np.array([False])
        episode_reward = 0.0
        steps = 0
        episode_success = False

        while not done_arr[0]:
            if is_evidential:
                # Use evidential interface to collect uncertainty per step
                obs_tensor = th.as_tensor(obs)
                action_tensor, uncertainty_dict = (
                    model.policy.get_action_with_uncertainty(  # type: ignore[union-attr]
                        obs_tensor, deterministic=deterministic
                    )
                )
                action = action_tensor.cpu().numpy()
                metrics.epistemic_uncertainties.append(
                    float(uncertainty_dict["epistemic"].mean().item())
                )
                metrics.aleatoric_uncertainties.append(
                    float(uncertainty_dict["aleatoric"].mean().item())
                )
            else:
                action, _states = model.predict(obs, deterministic=deterministic)

            step_result = env.step(action)
            obs = cast(np.ndarray, step_result[0])
            reward = cast(np.ndarray, step_result[1])
            done_arr = cast(np.ndarray, step_result[2])
            infos = cast(List[Dict[str, Any]], step_result[4])

            episode_reward += float(reward[0])
            steps += 1

            # Record success from the environment's info dict
            if done_arr[0] and infos[0].get("success", False):
                episode_success = True

            if render:
                env.render()

            if done_arr[0]:
                break

        if episode_success:
            successes += 1

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        # Position/orientation errors are not re-computed here; the env's
        # success flag is the ground-truth termination signal. Placeholder
        # zeros are appended so the metrics dataclass remains consistent.
        metrics.position_errors.append(0.0)
        metrics.orientation_errors.append(0.0)

    # Compute aggregate metrics
    metrics.success_rate = (successes / n_episodes) * 100.0
    metrics.average_reward = float(np.mean(episode_rewards))
    metrics.average_steps = float(np.mean(episode_steps))

    return metrics


def evaluate_across_conditions(
    model_path: str,
    eval_config_path: str,
    train_config_path: str,
    n_episodes: int = 100,
    output_dir: str = "./evaluation_results",
) -> pd.DataFrame:
    """
    @brief Evaluate agent across different physical conditions.
    @param model_path: Path to trained model.
    @param eval_config_path: Path to evaluation configuration file.
    @param train_config_path: Path to training configuration file
        (for base sensor noise).
    @param n_episodes: Number of episodes per condition.
    @param output_dir: Directory to save results.
    @return DataFrame with evaluation results.
    """
    # Load configurations
    with open(eval_config_path, "r") as f:
        eval_config: Dict[str, Any] = yaml.safe_load(f)

    with open(train_config_path, "r") as f:
        train_config: Dict[str, Any] = yaml.safe_load(f)

    base_sensors = train_config.get("carla_sensors", {})
    conditions = eval_config.get("eval_conditions", [])

    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Load normalisation statistics if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    results = []

    for condition in conditions:
        name = condition.get("name", "unknown")
        description = condition.get("description", "")
        print(f"\nEvaluating condition: {name}")
        print(f"  {description}")

        # Create environment for this condition
        base_env = make_eval_env(condition, eval_config, base_sensors, train_config)
        eval_env: Union[DummyVecEnv, VecNormalize] = base_env

        # Apply normalisation if available
        if os.path.exists(vec_normalize_path):
            eval_env = VecNormalize.load(vec_normalize_path, base_env)
            eval_env.training = False
            eval_env.norm_reward = False

        # Evaluate
        metrics = evaluate_agent(
            model=model,
            env=eval_env,
            n_episodes=n_episodes,
            deterministic=eval_config.get("deterministic", True),
        )

        # Store results
        result = metrics.to_dict()
        result["condition"] = name
        result["description"] = description
        result["weather_preset"] = condition.get("weather_preset", "")
        result["imu_noise_multiplier"] = condition.get("imu_noise_multiplier", 1.0)
        result["num_patrol_vehicles"] = condition.get("num_patrol_vehicles", 0)
        result["pedestrian_spawn_probability"] = condition.get(
            "pedestrian_spawn_probability", 1.0
        )
        result["bay_occupancy_rate"] = condition.get("bay_occupancy_rate", 0.6)
        results.append(result)

        print(f"  Success rate: {metrics.success_rate:.1f}%")
        print(f"  Average reward: {metrics.average_reward:.2f}")
        print(f"  Mean position error: " f"{np.mean(metrics.position_errors):.3f} m")

        # Clean up
        eval_env.close()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

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
    x_positions = range(len(conditions))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Success rate vs condition
    axes[0, 0].bar(x_positions, df["success_rate"], color="steelblue", alpha=0.8)
    axes[0, 0].set_xticks(list(x_positions))
    axes[0, 0].set_xticklabels(conditions, rotation=45, ha="right")
    axes[0, 0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0, 0].set_title("Success Rate vs Condition", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Average reward vs condition
    axes[0, 1].bar(x_positions, df["average_reward"], color="forestgreen", alpha=0.8)
    axes[0, 1].set_xticks(list(x_positions))
    axes[0, 1].set_xticklabels(conditions, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Average Reward", fontsize=12)
    axes[0, 1].set_title("Average Reward vs Condition", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Position error vs condition
    axes[1, 0].bar(
        x_positions,
        df["mean_position_error"],
        yerr=df["std_position_error"],
        color="firebrick",
        alpha=0.8,
        capsize=5,
    )
    axes[1, 0].set_xticks(list(x_positions))
    axes[1, 0].set_xticklabels(conditions, rotation=45, ha="right")
    axes[1, 0].set_ylabel("Position Error (m)", fontsize=12)
    axes[1, 0].set_title("Position Error vs Condition", fontsize=14)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Policy uncertainty estimates
    if "mean_epistemic_uncertainty" in df.columns:
        bar_width = 0.35
        x_arr = np.arange(len(conditions))
        axes[1, 1].bar(
            x_arr - bar_width / 2,
            df["mean_epistemic_uncertainty"],
            bar_width,
            label="Epistemic",
            alpha=0.8,
        )
        axes[1, 1].bar(
            x_arr + bar_width / 2,
            df["mean_aleatoric_uncertainty"],
            bar_width,
            label="Aleatoric",
            alpha=0.8,
        )
        axes[1, 1].set_xticks(list(x_arr))
        axes[1, 1].set_xticklabels(conditions, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Uncertainty", fontsize=12)
        axes[1, 1].set_title("Policy Uncertainty Estimates", fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")

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
        "--train-config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file (for base sensor noise)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes per condition",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Run evaluation
    df = evaluate_across_conditions(
        model_path=args.model_path,
        eval_config_path=args.eval_config,
        train_config_path=args.train_config,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
    )

    # Create plots
    plot_evaluation_results(df, output_dir=args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
