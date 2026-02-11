"""
@file __init__.py
@brief RL training with Stable-Baselines3.
"""

from uncertainty_rl.training.train_sac import load_config, train

__all__ = [
    "train",
    "load_config",
]
