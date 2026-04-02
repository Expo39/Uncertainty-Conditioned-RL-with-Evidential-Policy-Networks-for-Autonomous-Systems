"""
@file __init__.py
@brief RL training with Stable-Baselines3.
"""

from uncertainty_rl.training.train_ppo import load_config, merge_configs, train

__all__ = [
    "train",
    "load_config",
    "merge_configs",
]
