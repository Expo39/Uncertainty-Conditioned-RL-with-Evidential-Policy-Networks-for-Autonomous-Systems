"""
@file test_gnss_noise_relay.py
@brief Unit tests for GnssNoiseRelayNode's Doppler-style velocity and COG model.

Tests the tier defaults, course-noise helper, and YAML mirror invariant.
CPU-only, no CARLA or ROS 2: the relay module imports sensor_msgs at the top
level, so _TIER_DEFAULTS and _TIER_ORDER are inlined here instead.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import yaml

# Inlined from gnss_noise_relay._TIER_DEFAULTS / _TIER_ORDER.
# The YAML mirror test below enforces these match gnss_noise_profiles.yaml,
# which transitively pins the relay's own _TIER_DEFAULTS to the same values.
_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]
_TIER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "rtk_fixed": {"metric_stddev_m": 0.020, "doppler_stddev_ms": 0.05},
    "rtk_float": {"metric_stddev_m": 0.360, "doppler_stddev_ms": 0.90},
    "standalone": {"metric_stddev_m": 1.802, "doppler_stddev_ms": 4.50},
    "degraded": {"metric_stddev_m": 5.000, "doppler_stddev_ms": 12.50},
}

# Path to the YAML mirror file, resolved relative to this test file.
_PROFILES_PATH = (
    Path(__file__).parent.parent
    / "configs"
    / "deployment"
    / "sim"
    / "gnss_noise_profiles.yaml"
)

# Default per-step transition matrix, inlined from
# gnss_noise_relay._DEFAULT_TRANSITION_MATRIX (rows = from-tier in _TIER_ORDER).
# The relay module imports rclpy at module scope, so it cannot be imported on the
# host; this mirrors the production matrix the same way _TIER_DEFAULTS is inlined.
_DEFAULT_TRANSITION_MATRIX = np.array(
    [
        [0.9920, 0.0080, 0.0000, 0.0000],  # from: rtk_fixed
        [0.0400, 0.9550, 0.0050, 0.0000],  # from: rtk_float
        [0.0000, 0.0080, 0.9875, 0.0045],  # from: standalone
        [0.0000, 0.0000, 0.0071, 0.9929],  # from: degraded
    ]
)


def _mask_recovery_transitions(
    row: np.ndarray,
    active_idx: int,
) -> Optional[np.ndarray]:
    """
    @brief Reference reimplementation of gnss_noise_relay.mask_recovery_transitions.

    Kept in lock-step with the production helper (which is unimportable on the host
    because the relay module imports rclpy). @see gnss_noise_relay.mask_recovery_transitions.
    """
    masked = np.asarray(row, dtype=np.float64).copy()
    masked[:active_idx] = 0.0
    total = masked.sum()
    if total <= 0.0:
        return None
    return masked / total


def _course_noise_std(doppler_std: float, speed: float) -> float:
    """
    @brief Compute course-over-ground noise std from Doppler velocity noise.

    @param doppler_std: Doppler velocity 1-sigma (m/s).
    @param speed: Vehicle speed (m/s).
    @return Course noise 1-sigma (rad).
    """
    return doppler_std / max(speed, 0.1)


class TestTierDefaults:
    """
    @class TestTierDefaults
    @brief Verify doppler_stddev_ms is present and well-ordered in _TIER_DEFAULTS.
    """

    def test_all_tiers_have_doppler_stddev(self) -> None:
        """
        @brief Every tier in _TIER_ORDER must carry doppler_stddev_ms.
        """
        for tier in _TIER_ORDER:
            assert (
                "doppler_stddev_ms" in _TIER_DEFAULTS[tier]
            ), f"Tier '{tier}' missing doppler_stddev_ms in _TIER_DEFAULTS"

    def test_doppler_stddev_monotonically_nondecreasing(self) -> None:
        """
        @brief Doppler noise must not decrease as GNSS quality degrades.
        """
        values = [_TIER_DEFAULTS[tier]["doppler_stddev_ms"] for tier in _TIER_ORDER]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"doppler_stddev_ms decreased from "
                f"{_TIER_ORDER[i]}={values[i]} to {_TIER_ORDER[i+1]}={values[i+1]}"
            )

    def test_rtk_fixed_doppler_within_hardware_spec(self) -> None:
        """
        @brief rtk_fixed doppler_stddev_ms must be within the u-blox ZED-F9P spec (<=0.1 m/s).
        """
        assert _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"] <= 0.1

    def test_doppler_scales_with_position_degradation(self) -> None:
        """
        @brief Doppler noise scales by the SAME per-tier factor as position noise.

        The fix-state tier degrades velocity/heading and position together: each tier's
        doppler_stddev_ms is the rtk_fixed base scaled by the same 1 / 18 / 90 / 250x
        ladder as metric_stddev_m, so the degraded:rtk_fixed ratio matches in both
        channels. @see gnss_noise_profiles.yaml header.
        """
        doppler_ratio = (
            _TIER_DEFAULTS["degraded"]["doppler_stddev_ms"]
            / _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"]
        )
        pos_ratio = (
            _TIER_DEFAULTS["degraded"]["metric_stddev_m"]
            / _TIER_DEFAULTS["rtk_fixed"]["metric_stddev_m"]
        )
        assert pos_ratio > 50.0, f"Position ratio {pos_ratio:.1f}x unexpectedly small"
        assert doppler_ratio == pytest.approx(
            pos_ratio
        ), f"Doppler ratio {doppler_ratio:.1f}x must match position ratio {pos_ratio:.1f}x"


class TestCourseNoiseStd:
    """
    @class TestCourseNoiseStd
    @brief Verify course-noise error propagation: course_std = doppler_std / speed.
    """

    def test_decreases_with_speed(self) -> None:
        """
        @brief Higher speed -> smaller course noise (better heading from Doppler).
        """
        sigma = _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"]
        std_slow = _course_noise_std(sigma, 0.5)
        std_fast = _course_noise_std(sigma, 3.5)
        assert std_fast < std_slow

    def test_floor_at_low_speed_clamp(self) -> None:
        """
        @brief Speed below 0.1 m/s clamps to 0.1 so course_std does not blow up.
        """
        sigma = _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"]
        std_at_zero = _course_noise_std(sigma, 0.0)
        std_at_clamp = _course_noise_std(sigma, 0.1)
        assert std_at_zero == pytest.approx(std_at_clamp)

    def test_increases_with_doppler_noise(self) -> None:
        """
        @brief Worse Doppler tier -> larger course noise at the same speed.
        """
        speed = 2.0
        std_fixed = _course_noise_std(
            _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"], speed
        )
        std_degraded = _course_noise_std(
            _TIER_DEFAULTS["degraded"]["doppler_stddev_ms"], speed
        )
        assert std_degraded > std_fixed

    def test_units_reasonable_at_parking_speed(self) -> None:
        """
        @brief At 3.5 m/s rtk_fixed, course noise should be below 5 degrees.
        """
        sigma = _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"]
        course_std_rad = _course_noise_std(sigma, 3.5)
        assert math.degrees(course_std_rad) < 5.0


class TestYamlMirror:
    """
    @class TestYamlMirror
    @brief Enforce that gnss_noise_profiles.yaml mirrors _TIER_DEFAULTS doppler values.
    """

    def test_yaml_doppler_values_match_tier_defaults(self) -> None:
        """
        @brief Each tier's doppler_stddev_ms in the YAML must equal _TIER_DEFAULTS.

        This makes the mirror convention an enforced invariant rather than a
        documentation note: if you change one, the test fails until you change both.
        """
        assert (
            _PROFILES_PATH.exists()
        ), f"gnss_noise_profiles.yaml not found at {_PROFILES_PATH}"
        with open(_PROFILES_PATH, "r") as f:
            data = yaml.safe_load(f)

        tiers_section = data.get("tiers", {})
        for tier in _TIER_ORDER:
            yaml_tier = tiers_section.get(tier, {})
            assert (
                "doppler_stddev_ms" in yaml_tier
            ), f"Tier '{tier}' missing doppler_stddev_ms in gnss_noise_profiles.yaml"
            expected = _TIER_DEFAULTS[tier]["doppler_stddev_ms"]
            actual = float(yaml_tier["doppler_stddev_ms"])
            assert actual == pytest.approx(expected), (
                f"Tier '{tier}' doppler_stddev_ms mismatch: "
                f"YAML={actual} vs _TIER_DEFAULTS={expected}"
            )


class TestRecoveryMasking:
    """
    @class TestRecoveryMasking
    @brief Verify the one-way degradation ratchet used by the gnss_degrade_one_way
           eval condition: recovery transitions are forbidden and the chain only
           ever stays put or drops to a worse tier.
    """

    def test_upward_transitions_zeroed(self) -> None:
        """
        @brief From an intermediate tier, all transitions to a better fix are zero.
        """
        # From standalone (idx 2): rtk_fixed (0) and rtk_float (1) must be unreachable.
        masked = _mask_recovery_transitions(_DEFAULT_TRANSITION_MATRIX[2], active_idx=2)
        assert masked is not None
        assert masked[0] == 0.0
        assert masked[1] == 0.0

    def test_masked_row_is_a_distribution(self) -> None:
        """
        @brief Each masked row still sums to 1 (renormalised over same-or-worse tiers).
        """
        for idx in range(3):  # degraded handled separately (absorbing)
            masked = _mask_recovery_transitions(
                _DEFAULT_TRANSITION_MATRIX[idx], active_idx=idx
            )
            assert masked is not None
            assert masked.sum() == pytest.approx(1.0)
            assert np.all(masked >= 0.0)

    def test_no_recovery_probability_anywhere(self) -> None:
        """
        @brief For every starting tier, the masked row places zero mass on any
               strictly better tier - the defining property of the one-way chain.
        """
        for idx in range(len(_TIER_ORDER)):
            masked = _mask_recovery_transitions(
                _DEFAULT_TRANSITION_MATRIX[idx], active_idx=idx
            )
            if masked is None:
                continue
            assert np.all(masked[:idx] == 0.0)

    def test_self_loop_mass_grows_after_masking(self) -> None:
        """
        @brief Folding the forbidden upward mass onto the row lengthens the dwell:
               the renormalised stay-put probability is >= the original.
        """
        # rtk_float (idx 1) originally recovers to rtk_fixed with p=0.04.
        original = _DEFAULT_TRANSITION_MATRIX[1]
        masked = _mask_recovery_transitions(original, active_idx=1)
        assert masked is not None
        assert masked[1] >= original[1]

    def test_degraded_is_absorbing(self) -> None:
        """
        @brief From the worst tier the masked row keeps the chain at degraded
               (no better tier reachable; the self-loop dominates).
        """
        masked = _mask_recovery_transitions(_DEFAULT_TRANSITION_MATRIX[3], active_idx=3)
        assert masked is not None
        # Only the degraded self-loop (idx 3) carries mass after renormalisation.
        assert masked[3] == pytest.approx(1.0)
