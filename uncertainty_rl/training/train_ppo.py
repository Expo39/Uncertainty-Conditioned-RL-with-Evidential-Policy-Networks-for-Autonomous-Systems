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
from stable_baselines3.ppo import PPO

from uncertainty_rl.envs import CARLAParkingEnv
from uncertainty_rl.networks import EvidentialActorCriticPolicy, EvidentialPPO


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
            include_covariance=config.get("include_covariance", True),
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


def train(config: Dict[str, Any]) -> None:
    """
    @brief Train the uncertainty-conditioned RL agent with PPO.
    @param config: Fully-resolved configuration dictionary. All operational
           settings (seed, log_dir, etc.) and hyperparameters are read from
           this dict. CLI arguments override YAML values before this is called.
    """
    # Resolve operational settings from config
    seed = config.get("seed", 42)
    total_timesteps = config.get("total_timesteps", 1000000)
    log_dir = config.get("log_dir", "./logs")
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    eval_freq = config.get("eval_freq", 10000)
    n_eval_episodes = config.get("n_eval_episodes", 10)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create training environment
    print("Creating training environment...")
    train_vec_env = DummyVecEnv([make_env(config)])

    # Normalise observations but not rewards — reward components will be
    # manually scaled via potential-based shaping (see reward TODO in config)
    env = VecNormalize(
        train_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Create evaluation environment (same config as training)
    print("Creating evaluation environment...")
    eval_vec_env = DummyVecEnv([make_env(config)])
    eval_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,  # Don't update running statistics during evaluation
    )

    # Shared policy kwargs for both standard and evidential policies
    policy_kwargs = dict(
        net_arch=dict(
            pi=config.get("net_arch", [256, 256]),
            vf=config.get("net_arch", [256, 256]),
        ),
        activation_fn=torch.nn.ReLU,
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
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
    )

    # Create agent based on policy_type config
    policy_type = config.get("policy_type", "evidential")
    print(f"Initialising {policy_type} PPO agent...")

    model: PPO
    if policy_type == "evidential":
        evidential_config = config.get("evidential", {})
        lambda_reg = evidential_config.get("lambda_reg", 0.01)
        model = EvidentialPPO(
            policy=EvidentialActorCriticPolicy,
            lambda_reg=lambda_reg,
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

    CLI arguments override values from the YAML config file. The config file
    is the single source of truth; CLI args are convenience overrides for
    per-run settings (e.g. --seed 7 for a specific ablation run).
    """
    parser = argparse.ArgumentParser(
        description="Train uncertainty-conditioned RL agent for autonomous parking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file (single source of truth)",
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

    # Load config, then apply CLI overrides
    config = load_config(args.config)
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

    train(config)


if __name__ == "__main__":
    main()
