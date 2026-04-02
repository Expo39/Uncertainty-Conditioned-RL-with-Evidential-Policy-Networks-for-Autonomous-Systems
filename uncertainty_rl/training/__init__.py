"""
@file __init__.py
@brief RL training with Stable-Baselines3.
"""

# Lazy imports: train_ppo requires gymnasium/torch which may not be installed
# in CI environments (dependencies are optional). These are only imported when
# actually used (at runtime), not at package load time.

__all__ = [
    "train",
    "load_config",
    "merge_configs",
]


def __getattr__(name):
    """
    @brief Lazy load training functions on first access.
    @param name: Name of the attribute being accessed.
    @return The requested attribute from train_ppo module.
    """
    if name in __all__:
        from uncertainty_rl.training.train_ppo import (  # noqa: E402
            load_config,
            merge_configs,
            train,
        )

        attrs = {
            "train": train,
            "load_config": load_config,
            "merge_configs": merge_configs,
        }
        return attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
