## @file train_sac.py
#  @brief Training script for uncertainty-conditioned RL with SAC.
#
#  This module provides training functionality using Stable-Baselines3's SAC algorithm
#  with evidential policy networks.
from typing import Optional, Dict, Any, Callable
import os
import yaml
import argparse
from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork


## @brief Create a callable that returns a new environment instance.
#  @param config: Configuration dictionary.
#  @param rank: Environment rank for seeding.
#  @return Callable that creates environment.
def make_env(config: Dict[str, Any], rank: int = 0) -> Callable:
    def _init() -> gym.Env:
        env = CARLAParkingEnv(
            carla_host=config.get("carla_host", "localhost"),
            carla_port=config.get("carla_port", 2000) + rank,
            town=config.get("town", "Town01"),
            uncertainty_noise_std=config.get("uncertainty_noise_std", 0.1),
            max_steps=config.get("max_steps", 500),
        )
        return env
    return _init


## @class UncertaintyLogger
#  @brief Custom callback to log uncertainty metrics during training.
class UncertaintyLogger:
    
    ## @brief Constructor for UncertaintyLogger.
    #  @param verbose: Verbosity level.
    def __init__(self, verbose: int = 0) -> None:
        self.verbose = verbose
        self.epistemic_uncertainties = []
        self.aleatoric_uncertainties = []
        
    ## @brief Called after each environment step.
    #  @return True to continue training.
    def _on_step(self) -> bool:
        # This would log uncertainty from the policy network
        # Implementation depends on integration with SB3
        return True


## @brief Load configuration from YAML file.
#  @param config_path: Path to configuration file.
#  @return Configuration dictionary.
def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


## @brief Train the uncertainty-conditioned RL agent.
#  @param config_path: Path to configuration YAML file.
#  @param total_timesteps: Total training timesteps.
#  @param log_dir: Directory for TensorBoard logs.
#  @param checkpoint_dir: Directory for model checkpoints.
#  @param eval_freq: Evaluation frequency (timesteps).
#  @param n_eval_episodes: Number of evaluation episodes.
#  @param seed: Random seed for reproducibility.
def train(
    config_path: str,
    total_timesteps: int = 1000000,
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    seed: int = 42,
) -> None:
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
    env = DummyVecEnv([make_env(config)])
    
    # Normalise observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_config = config.copy()
    eval_config["uncertainty_noise_std"] = config.get("eval_uncertainty_noise_std", 0.1)
    eval_env = DummyVecEnv([make_env(eval_config)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalise rewards during evaluation
        clip_obs=10.0,
        training=False,  # Important: don't update running statistics
    )
    
    # Configure SAC hyperparameters
    policy_kwargs = dict(
        net_arch=config.get("net_arch", [256, 256]),
        activation_fn=torch.nn.ReLU,
    )
    
    # Create SAC agent
    print("Initialising SAC agent...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.get("learning_rate", 3e-4),
        buffer_size=config.get("buffer_size", 1000000),
        learning_starts=config.get("learning_starts", 10000),
        batch_size=config.get("batch_size", 256),
        tau=config.get("tau", 0.005),
        gamma=config.get("gamma", 0.99),
        train_freq=config.get("train_freq", 1),
        gradient_steps=config.get("gradient_steps", 1),
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
        name_prefix="sac_uncertainty_rl",
        save_replay_buffer=True,
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


## @brief Main entry point for training script.
def main() -> None:
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
