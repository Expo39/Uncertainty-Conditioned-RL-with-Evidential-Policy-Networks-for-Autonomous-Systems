"""
@file test_gnss_drift.py
@brief Unit tests for the GNSS mid-episode-drift matrix scaler.

CPU-only, no CARLA / ROS 2 / GPU. Verifies that scale_transition_matrix (used by
the ros2-bridge GnssNoiseRelayNode to apply the per-stage drift_scale) maps
drift_scale = 0 to the identity, drift_scale = 1 to the base matrix, clamps
out-of-range values, and always returns a row-stochastic matrix.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from uncertainty_rl.utils.gnss_drift import scale_transition_matrix

_TIER_ORDER = ["rtk_fixed", "rtk_float", "standalone", "degraded"]


def _base_matrix() -> np.ndarray:
    """Load the real transition matrix from gnss_noise_profiles.yaml."""
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "deployment"
        / "sim"
        / "gnss_noise_profiles.yaml"
    )
    section = yaml.safe_load(open(path))["transition_matrix"]
    return np.array([section[name] for name in _TIER_ORDER], dtype=np.float64)


def test_scale_zero_gives_identity() -> None:
    base = _base_matrix()
    eff = scale_transition_matrix(base, 0.0)
    np.testing.assert_allclose(eff, np.eye(base.shape[0]))


def test_scale_one_reproduces_base() -> None:
    base = _base_matrix()
    eff = scale_transition_matrix(base, 1.0)
    np.testing.assert_allclose(eff, base, atol=1e-12)


@pytest.mark.parametrize("scale", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
def test_rows_remain_stochastic(scale: float) -> None:
    base = _base_matrix()
    eff = scale_transition_matrix(base, scale)
    np.testing.assert_allclose(eff.sum(axis=1), np.ones(base.shape[0]), atol=1e-12)
    assert (eff >= -1e-12).all(), "transition probabilities must be non-negative"


def test_offdiagonals_scale_linearly() -> None:
    base = _base_matrix()
    eff = scale_transition_matrix(base, 0.3)
    n = base.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:
                assert eff[i, j] == pytest.approx(base[i, j] * 0.3)


def test_out_of_range_is_clamped() -> None:
    base = _base_matrix()
    np.testing.assert_allclose(
        scale_transition_matrix(base, 5.0), scale_transition_matrix(base, 1.0)
    )
    np.testing.assert_allclose(
        scale_transition_matrix(base, -2.0), scale_transition_matrix(base, 0.0)
    )


def test_does_not_mutate_base() -> None:
    base = _base_matrix()
    snapshot = base.copy()
    scale_transition_matrix(base, 0.4)
    np.testing.assert_array_equal(base, snapshot)
