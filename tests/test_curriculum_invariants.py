"""
@file test_curriculum_invariants.py
@brief Structural invariants for the single-phase ADR curriculum stage files.

CPU-only, no CARLA / ROS 2 / GPU - parses YAML only. Encodes the non-negotiable
"all observation channels live in every stage" rule so a future edit that
re-degenerates a channel (the original Stage 1 -> 2 failure mode) fails CI:
  - covariance channel: the GNSS Markov chain is enabled and non-degenerate, so
    mid-episode drift is always on (the process is stage-invariant - not a stage key);
  - LiDAR channel: bay_occupancy_max > 0 and LiDAR noise enabled;
  - target-pose channel: the bay set spans >= 2 approach orientations.
Also checks the per-policy override blocks are allowlisted and start identical.
"""

from pathlib import Path
from typing import Dict, List

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_CURRICULUM = _REPO / "configs" / "deployment" / "sim" / "curriculum"
_LAYOUT = _REPO / "configs" / "layouts" / "rectangle.yaml"
_GNSS_PROFILES = _REPO / "configs" / "deployment" / "sim" / "gnss_noise_profiles.yaml"
_ROS2_CONFIG = _REPO / "configs" / "ros2_config.yaml"
_N_STAGES = 6

# Mirrors _STAGE_TRAINING_OVERRIDE_ALLOWLIST in train_ppo.py (kept in sync so a
# stage can never set an architecture key and break weight loading on resume).
_ALLOWLIST = {
    "stage_timesteps",
    "learning_rate",
    "learning_rate_final",
    "ent_coef",
    "ent_coef_final",
    "clip_range",
    "n_epochs",
    "batch_size",
    "n_steps",
    "target_kl",
}


def _yaw_by_bay() -> Dict[str, float]:
    bays = yaml.safe_load(open(_LAYOUT))["bays"]
    return {
        b["id"]: round(float(b["yaw_deg"]))
        for b in bays
        if not b.get("always_empty", False)
    }


def _stage(n: int) -> Dict:
    return yaml.safe_load(open(_CURRICULUM / f"stage{n}.yaml"))


_STAGES = list(range(1, _N_STAGES + 1))


def test_covariance_channel_live() -> None:
    """The GNSS degradation process is stage-invariant and always on: the Markov
    master switch is enabled and the transition matrix has live off-diagonals, so
    the covariance features are never near-constant in any stage."""
    enabled = (
        yaml.safe_load(open(_ROS2_CONFIG))
        .get("gnss_noise_relay", {})
        .get("enable_markov_transitions")
    )
    assert enabled is True, "enable_markov_transitions must be on globally"
    matrix = yaml.safe_load(open(_GNSS_PROFILES))["transition_matrix"]
    # Every tier must have a non-zero probability of leaving (off-diagonal mass),
    # so the chain genuinely drifts rather than holding the start tier forever.
    tiers = list(matrix)
    for i, tier in enumerate(tiers):
        row = matrix[tier]
        off_diag = sum(row) - row[i]
        assert off_diag > 0.0, f"tier {tier}: no transitions out (off-diag mass 0)"


@pytest.mark.parametrize("n", _STAGES)
def test_lidar_channel_live(n: int) -> None:
    """Occupancy > 0 and LiDAR noise on so the obstacle features carry signal."""
    cfg = _stage(n)
    occ_max = cfg["parking_scenarios"].get("bay_occupancy_max")
    assert (
        isinstance(occ_max, (int, float)) and occ_max > 0.0
    ), f"stage{n}: occ {occ_max}"
    enabled = (
        cfg.get("carla_sensors", {}).get("lidar", {}).get("noise", {}).get("enabled")
    )
    assert enabled is True, f"stage{n}: LiDAR noise not enabled"


@pytest.mark.parametrize("n", _STAGES)
def test_target_pose_channel_live(n: int) -> None:
    """Bay set spans >= 2 orientations (or all 47), so dx/dy/dyaw vary."""
    yaw = _yaw_by_bay()
    allowed: List[str] = _stage(n)["parking_scenarios"].get("allowed_bay_ids") or []
    if not allowed:
        return  # No whitelist -> all eligible bays (all 4 orientations).
    missing = [b for b in allowed if b not in yaw]
    assert not missing, f"stage{n}: unknown bay ids {missing}"
    assert len({yaw[b] for b in allowed}) >= 2, f"stage{n}: < 2 orientations"


@pytest.mark.parametrize("n", _STAGES)
def test_start_tier_and_spawns_fixed(n: int) -> None:
    cfg = _stage(n)
    assert (
        cfg["parking_scenarios"].get("fixed_gnss_tier") == "rtk_fixed"
    ), f"stage{n}: episodes must START in rtk_fixed (drift wanders from there)"
    assert cfg.get("use_extra_spawns") is False, f"stage{n}: extra spawns must be off"


@pytest.mark.parametrize("n", _STAGES)
def test_override_blocks_allowlisted_and_identical(n: int) -> None:
    cfg = _stage(n)
    std = cfg.get("standard_overrides")
    evi = cfg.get("evidential_overrides")
    assert isinstance(std, dict) and isinstance(evi, dict), f"stage{n}: missing blocks"
    for name, blk in (("standard", std), ("evidential", evi)):
        bad = set(blk) - _ALLOWLIST
        assert not bad, f"stage{n}: {name}_overrides has non-allowlisted keys {bad}"
    # Initialised identical (fairness by default; split only on observed instability).
    assert std == evi, f"stage{n}: standard/evidential overrides must start identical"
