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
    "load_env_config",
    "merge_configs",
    "TrainResult",
]


def __getattr__(name: str) -> object:
    """
    @brief Lazy load training functions on first access.

    After resolving, injects all symbols into module globals so subsequent
    lookups are direct attribute access rather than re-entering __getattr__.

    @param name: Name of the attribute being accessed.
    @return The requested attribute from train_ppo module.
    """
    if name in __all__:
        from uncertainty_rl.training.train_ppo import (  # noqa: E402
            TrainResult,
            load_config,
            load_env_config,
            merge_configs,
            train,
        )

        _resolved = {
            "train": train,
            "load_config": load_config,
            "load_env_config": load_env_config,
            "merge_configs": merge_configs,
            "TrainResult": TrainResult,
        }
        # Inject into module globals so future attribute lookups are direct.
        globals().update(_resolved)
        return _resolved[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
