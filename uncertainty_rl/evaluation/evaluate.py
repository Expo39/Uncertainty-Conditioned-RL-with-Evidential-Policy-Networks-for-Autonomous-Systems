"""
@file evaluate.py
@brief Evaluation script for trained agents across different uncertainty levels.

This module provides comprehensive evaluation of trained agents under varying
SLAM uncertainty conditions.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from stable_baselines3 import PPO  # Changed from SAC to PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import CARLAParkingEnv
from uncertainty_rl.utils.constants import (
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
)


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
                np.mean(self.position_errors) if self.position_errors else 0.0
            ),
            "std_position_error": (
                np.std(self.position_errors) if self.position_errors else 0.0
            ),
            "mean_orientation_error": (
                np.mean(self.orientation_errors) if self.orientation_errors else 0.0
            ),
            "std_orientation_error": (
                np.std(self.orientation_errors) if self.orientation_errors else 0.0
            ),
            "mean_epistemic_uncertainty": (
                np.mean(self.epistemic_uncertainties)
                if self.epistemic_uncertainties
                else 0.0
            ),
            "mean_aleatoric_uncertainty": (
                np.mean(self.aleatoric_uncertainties)
                if self.aleatoric_uncertainties
                else 0.0
            ),
        }


def make_eval_env(uncertainty_noise_std: float, config: Dict[str, Any]) -> DummyVecEnv:
    """
    @brief Create evaluation environment with specific uncertainty level.
    @param uncertainty_noise_std: Standard deviation of uncertainty noise.
    @param config: Configuration dictionary.
    @return Vectorised evaluation environment.
    """

    def _init():
        return CARLAParkingEnv(
            carla_host=config.get("carla_host", "localhost"),
            carla_port=config.get("carla_port", 2000),
            town=config.get("town", "Town01"),
            uncertainty_noise_std=uncertainty_noise_std,
            max_steps=config.get("max_steps", 500),
        )

    env = DummyVecEnv([_init])
    return env


def evaluate_agent(
    model: PPO,  # Changed from SAC to PPO
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
    """
    metrics = EvaluationMetrics()

    episode_rewards: List[float] = []
    episode_steps: List[int] = []
    successes = 0

    for episode in range(n_episodes):
        obs: np.ndarray = env.reset()  # type: ignore[assignment]
        done_arr = np.array([False])
        episode_reward = 0.0
        steps = 0

        while not done_arr[0]:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, info = env.step(action)  # type: ignore[assignment]

            episode_reward += float(reward[0])
            steps += 1

            if render:
                env.render()

            # Check if done
            if done_arr[0]:
                break

        # Extract final state for error computation
        state = obs[0]
        x, y, yaw = state[0], state[1], state[2]

        # Assume target is at origin (0, 0, 0)
        position_error = np.sqrt(x**2 + y**2)
        orientation_error = np.abs(yaw)

        # Check success using shared constants
        success = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
        )
        if success:
            successes += 1

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        metrics.position_errors.append(float(position_error))
        metrics.orientation_errors.append(float(orientation_error))

    # Compute aggregate metrics
    metrics.success_rate = (successes / n_episodes) * 100.0
    metrics.average_reward = float(np.mean(episode_rewards))
    metrics.average_steps = float(np.mean(episode_steps))

    return metrics


def evaluate_across_noise_levels(
    model_path: str,
    config_path: str,
    noise_levels: List[float],
    n_episodes: int = 100,
    output_dir: str = "./evaluation_results",
) -> pd.DataFrame:
    """
    @brief Evaluate agent across different uncertainty noise levels.
    @param model_path: Path to trained model.
    @param config_path: Path to configuration file.
    @param noise_levels: List of uncertainty noise standard deviations to test.
    @param n_episodes: Number of episodes per noise level.
    @param output_dir: Directory to save results.
    @return DataFrame with evaluation results.
    """
    # Load configuration
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Load normalisation statistics if available
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    results = []

    for noise_std in noise_levels:
        print(f"\nEvaluating with uncertainty noise std = {noise_std:.4f}")

        # Create environment with specific noise level
        base_env = make_eval_env(noise_std, config)
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
            deterministic=True,
        )

        # Store results
        result = metrics.to_dict()
        result["noise_std"] = noise_std
        results.append(result)

        print(f"  Success rate: {metrics.success_rate:.1f}%")
        print(f"  Average reward: {metrics.average_reward:.2f}")
        print(f"  Mean position error: {np.mean(metrics.position_errors):.3f} m")

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
    @brief Create visualisations of evaluation results.
    @param df: DataFrame with evaluation results.
    @param output_dir: Directory to save plots.
    """
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Success rate vs noise level
    axes[0, 0].plot(
        df["noise_std"], df["success_rate"], "o-", linewidth=2, markersize=8
    )
    axes[0, 0].set_xlabel("Uncertainty Noise Std (m)", fontsize=12)
    axes[0, 0].set_ylabel("Success Rate (%)", fontsize=12)
    axes[0, 0].set_title("Success Rate vs Uncertainty Level", fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Average reward vs noise level
    axes[0, 1].plot(
        df["noise_std"],
        df["average_reward"],
        "o-",
        linewidth=2,
        markersize=8,
        color="green",
    )
    axes[0, 1].set_xlabel("Uncertainty Noise Std (m)", fontsize=12)
    axes[0, 1].set_ylabel("Average Reward", fontsize=12)
    axes[0, 1].set_title("Average Reward vs Uncertainty Level", fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Position error vs noise level
    axes[1, 0].errorbar(
        df["noise_std"],
        df["mean_position_error"],
        yerr=df["std_position_error"],
        fmt="o-",
        linewidth=2,
        markersize=8,
        color="red",
        capsize=5,
    )
    axes[1, 0].set_xlabel("Uncertainty Noise Std (m)", fontsize=12)
    axes[1, 0].set_ylabel("Position Error (m)", fontsize=12)
    axes[1, 0].set_title("Position Error vs Uncertainty Level", fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Uncertainty estimates
    if "mean_epistemic_uncertainty" in df.columns:
        axes[1, 1].plot(
            df["noise_std"],
            df["mean_epistemic_uncertainty"],
            "o-",
            linewidth=2,
            markersize=8,
            label="Epistemic",
        )
        axes[1, 1].plot(
            df["noise_std"],
            df["mean_aleatoric_uncertainty"],
            "s-",
            linewidth=2,
            markersize=8,
            label="Aleatoric",
        )
        axes[1, 1].set_xlabel("Uncertainty Noise Std (m)", fontsize=12)
        axes[1, 1].set_ylabel("Uncertainty", fontsize=12)
        axes[1, 1].set_title("Policy Uncertainty Estimates", fontsize=14)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")

    plt.close()


def main() -> None:
    """
    @brief Main entry point for evaluation script.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained agent across uncertainty levels"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        help="List of uncertainty noise levels to evaluate",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of episodes per noise level",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory for output files",
    )

    args = parser.parse_args()

    # Run evaluation
    df = evaluate_across_noise_levels(
        model_path=args.model_path,
        config_path=args.config,
        noise_levels=args.noise_levels,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
    )

    # Create plots
    plot_evaluation_results(df, output_dir=args.output_dir)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
