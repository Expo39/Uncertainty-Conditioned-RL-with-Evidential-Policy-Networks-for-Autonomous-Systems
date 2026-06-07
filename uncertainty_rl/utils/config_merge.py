"""
@file config_merge.py
@brief Single source of truth for the config merge hierarchy.

The project layers YAML configs in one precedence chain:

    sensor_config < agent_config < env_config (+ stage) < train_config (+ baseline)

Every consumer (training, tuning, the demo driver) must merge through the helpers
here so the precedence is defined once.
"""

from typing import Any, Dict, FrozenSet

# Keys a baseline override file (configs/baselines/*.yaml) is permitted to set.
# A baseline is an ablation cell, so it may only change the observation/policy
# configuration and its output dirs - never a hyperparameter or env setting.
# Mirrors the stage `training_overrides` allowlist in train_ppo.py: anything
# outside this set fails loud rather than silently reshaping the run.
BASELINE_KEYS: FrozenSet[str] = frozenset(
    {
        "baseline_name",
        "include_covariance",
        "include_obstacle_obs",
        "policy_type",
    }
)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Recursively merge override into base; override wins on scalar keys.
    @param base: Lower-precedence dict (mutated in place and returned).
    @param override: Higher-precedence dict whose values take priority.
    @return The merged dict.

    @note A nested dict on both sides is merged key-by-key rather than replaced
          wholesale, so a higher-precedence file can override a single key inside
          a shared block (e.g. env_config adding ros2.carla_recovery without
          dropping ros2.covariance_timeout from agent_config). A plain
          {**base, **override} would discard every base key under any block the
          override also defines.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_baseline(config: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Overlay an ablation baseline's allowlisted keys onto a merged config.
    @param config: The merged train+env config (mutated in place and returned).
    @param baseline: Parsed baseline override file (configs/baselines/*.yaml).
    @return The config with the baseline's BASELINE_KEYS applied.
    @warning Raises ValueError on any baseline key outside BASELINE_KEYS, so a
             baseline can never silently change a hyperparameter or env setting.
    """
    unknown = set(baseline) - BASELINE_KEYS
    if unknown:
        raise ValueError(
            f"Baseline override keys {sorted(unknown)} are not permitted. "
            f"A baseline may only set {sorted(BASELINE_KEYS)}."
        )
    for key, value in baseline.items():
        config[key] = value
    return config
