"""
@file train_ppo.py
@brief Training script for uncertainty-conditioned RL with PPO and evidential policies.

This module provides training functionality using Stable-Baselines3's PPO algorithm
with evidential actor networks for autonomous parking.
"""

import argparse
import os
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from uncertainty_rl.envs import CARLAParkingEnv
from uncertainty_rl.networks import (
    EvidentialActorCriticPolicy,
    EvidentialPPO,
)


def make_env(
    config: Dict[str, Any],
    rank: int = 0,
    carla_sensors_override: Optional[Dict[str, Any]] = None,
    carla_conditions_override: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    @brief Create a callable that returns a new environment instance.
    @param config: Configuration dictionary.
    @param rank: Environment rank for seeding.
    @param carla_sensors_override: Override sensor noise config (for evaluation).
    @param carla_conditions_override: Override conditions config (for evaluation).
    @return Callable that creates environment.
    """

    def _init() -> gym.Env:
        env = CARLAParkingEnv(
            carla_host=config.get("carla_host", "localhost"),
            carla_port=config.get("carla_port", 2000) + rank,
            town=config.get("town", "Town01"),
            max_steps=config.get("max_steps", 500),
            ros2_config=config.get("ros2", {}),
            carla_sensors_config=(
                carla_sensors_override
                if carla_sensors_override is not None
                else config.get("carla_sensors", {})
            ),
            carla_conditions_config=(
                carla_conditions_override
                if carla_conditions_override is not None
                else config.get("carla_conditions", {})
            ),
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


def train(
    config_path: str,
    total_timesteps: int = 1000000,
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    seed: int = 42,
) -> None:
    """
    @brief Train the uncertainty-conditioned RL agent with PPO.
    @param config_path: Path to configuration YAML file.
    @param total_timesteps: Total training timesteps.
    @param log_dir: Directory for TensorBoard logs.
    @param checkpoint_dir: Directory for model checkpoints.
    @param eval_freq: Evaluation frequency (timesteps).
    @param n_eval_episodes: Number of evaluation episodes.
    @param seed: Random seed for reproducibility.
    """
    # Load configuration
    config = load_config(config_path)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create training environment
    print("Creating training environment...")
    train_vec_env = DummyVecEnv([make_env(config)])

    # Normalise observations and rewards
    env = VecNormalize(
        train_vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )

    # Create evaluation environment (same config as training)
    print("Creating evaluation environment...")
    eval_vec_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalise rewards during evaluation
        clip_obs=10.0,
        training=False,  # Important: don't update running statistics
    )

    # Configure evidential PPO policy
    evidential_config = config.get("evidential", {})
    lambda_reg = evidential_config.get("lambda_reg", 0.01)

    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("net_arch", [256, 256]),
            vf=config.get("net_arch", [256, 256]),
        ),
        activation_fn=torch.nn.ReLU,
    )

    # Create EvidentialPPO agent with evidential actor
    print("Initialising EvidentialPPO agent...")
    model = EvidentialPPO(
        policy=EvidentialActorCriticPolicy,
        env=env,
        lambda_reg=lambda_reg,
        learning_rate=config.get("learning_rate", 3e-4),
        n_steps=config.get("n_steps", 2048),
        batch_size=config.get("batch_size", 64),
        n_epochs=config.get("n_epochs", 10),
        gamma=config.get("gamma", 0.99),
        gae_lambda=config.get("gae_lambda", 0.95),
        clip_range=config.get("clip_range", 0.2),
        clip_range_vf=config.get("clip_range_vf", None),
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=config.get("vf_coef", 0.5),
        max_grad_norm=config.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Set up logger
    logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get("checkpoint_freq", 50000),
        save_path=checkpoint_dir,
        name_prefix="ppo_uncertainty_rl",
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )

    callback_list = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=10,
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_model_path)
    env.save(os.path.join(checkpoint_dir, "vec_normalize.pkl"))

    print(f"Training complete! Model saved to {final_model_path}")

    # Clean up
    env.close()
    eval_env.close()


def main() -> None:
    """
    @brief Main entry point for training script.
    """
    parser = argparse.ArgumentParser(
        description="Train uncertainty-conditioned RL agent for autonomous parking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        total_timesteps=args.total_timesteps,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
