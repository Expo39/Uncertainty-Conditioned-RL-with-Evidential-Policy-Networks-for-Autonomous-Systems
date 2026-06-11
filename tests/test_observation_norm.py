"""
@file test_observation_norm.py
@brief Unit tests for the fixed physical-range observation normaliser.

CPU-only, no CARLA / ROS 2 / GPU. Verifies normalise_observation (applied inside
build_observation) scales each observation dimension by its fixed physical range
and clips, so a constant input maps to a constant well-scaled value (no
division-by-near-zero blow-up) and the mapping is deterministic and
layout-independent - the property that lets weights transfer across curriculum
stages and keeps OOD evaluation unconfounded.
"""

import numpy as np

from uncertainty_rl.envs._parking_core import compute_obs_dim, normalise_observation
from uncertainty_rl.utils.constants import (
    OBS_NORM_CLIP,
    OBS_OBSTACLE_DIST_SCALE,
    OBS_SPEED_SCALE,
    OBS_STD_POS_SCALE,
    OBS_TARGET_POS_SCALE,
)


def test_zero_maps_to_zero() -> None:
    """A constant (zero) input must not blow up - it maps to a constant zero."""
    raw = np.zeros(compute_obs_dim(True, True), dtype=np.float32)
    out = normalise_observation(raw, True, True)
    np.testing.assert_array_equal(out, np.zeros_like(raw))


def test_known_components_scale_by_physical_range() -> None:
    """Each component is divided by its fixed physical scale (full 13-dim layout)."""
    raw = np.zeros(13, dtype=np.float32)
    raw[0] = OBS_SPEED_SCALE  # speed -> 1.0
    raw[2] = OBS_STD_POS_SCALE  # std_x -> 1.0
    raw[5] = OBS_TARGET_POS_SCALE  # dx -> 1.0
    raw[8] = OBS_OBSTACLE_DIST_SCALE  # left_dist -> 1.0
    out = normalise_observation(raw, True, True)
    assert out[0] == np.float32(1.0)
    assert out[2] == np.float32(1.0)
    assert out[5] == np.float32(1.0)
    assert out[8] == np.float32(1.0)


def test_clip_bounds_outliers() -> None:
    """Extreme raw values are clipped to +/-OBS_NORM_CLIP, never beyond."""
    raw = np.full(13, 1e6, dtype=np.float32)
    out = normalise_observation(raw, True, True)
    assert out.max() <= OBS_NORM_CLIP and out.min() >= -OBS_NORM_CLIP
    raw[:] = -1e6
    out = normalise_observation(raw, True, True)
    assert out.min() >= -OBS_NORM_CLIP


def test_deterministic_and_layout_independent() -> None:
    """No running statistics: the same raw obs always maps to the same output."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal(13).astype(np.float32)
    a = normalise_observation(raw, True, True)
    b = normalise_observation(raw.copy(), True, True)
    np.testing.assert_array_equal(a, b)


def test_does_not_mutate_input() -> None:
    raw = np.full(13, 2.0, dtype=np.float32)
    snapshot = raw.copy()
    normalise_observation(raw, True, True)
    np.testing.assert_array_equal(raw, snapshot)


def test_scale_vector_length_matches_each_ablation() -> None:
    """The scale layout matches compute_obs_dim for every ablation flag pair."""
    for cov in (True, False):
        for obs in (True, False):
            dim = compute_obs_dim(cov, obs)
            out = normalise_observation(np.zeros(dim, dtype=np.float32), cov, obs)
            assert out.shape == (dim,)
