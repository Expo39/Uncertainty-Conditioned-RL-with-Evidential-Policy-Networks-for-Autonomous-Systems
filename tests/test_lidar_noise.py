"""
@file test_lidar_noise.py
@brief Unit tests for the SICK TiM571 LiDAR noise model in SensorManager.

All tests are CPU-only and require no CARLA server, ROS 2, or GPU.
Noise parameters are constructed inline per test class to avoid conftest coupling.
"""

import math
from typing import Any, Dict

import numpy as np

from uncertainty_rl.envs.sim.helpers._sensor_manager import SensorManager


def _make_manager(noise_cfg: Dict[str, Any]) -> SensorManager:
    """Build a SensorManager with a minimal sensors_config dict."""
    return SensorManager({"lidar": {"noise": noise_cfg}})


def _enabled_cfg(
    range_random_stddev_m: float = 0.020,
    range_bias_limit_m: float = 0.060,
    dropout_rate: float = 0.0,
    min_range_m: float = 0.05,
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "range_random_stddev_m": range_random_stddev_m,
        "range_bias_limit_m": range_bias_limit_m,
        "dropout_rate": dropout_rate,
        "min_range_m": min_range_m,
    }


def _point(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Return a single-point (1, 3) float32 array."""
    return np.array([[x, y, z]], dtype=np.float32)


def _points(n: int, x: float = 5.0) -> np.ndarray:
    """Return n identical points along the x-axis."""
    return np.tile(_point(x), (n, 1))


class TestLidarNoiseDisabled:
    """Noise is disabled - input must pass through unchanged."""

    def test_disabled_returns_input_unchanged(self) -> None:
        mgr = _make_manager({"enabled": False})
        pts = _points(5)
        result = mgr._apply_lidar_noise(pts)
        np.testing.assert_array_equal(result, pts)

    def test_bias_stays_zero_when_disabled(self) -> None:
        mgr = _make_manager({"enabled": False, "range_bias_limit_m": 0.060})
        rng = np.random.default_rng(0)
        mgr.sample_lidar_noise_bias(rng)
        assert mgr._lidar_range_bias_m == 0.0


class TestLidarNoiseBiasSampling:
    """Per-run systematic bias sampling behaviour."""

    def test_bias_within_limit(self) -> None:
        limit = 0.060
        mgr = _make_manager(_enabled_cfg(range_bias_limit_m=limit))
        for seed in range(200):
            mgr._lidar_range_bias_m = float("nan")
            mgr.sample_lidar_noise_bias(np.random.default_rng(seed))
            assert abs(mgr._lidar_range_bias_m) <= limit + 1e-9

    def test_bias_is_scalar_float(self) -> None:
        mgr = _make_manager(_enabled_cfg())
        mgr.sample_lidar_noise_bias(np.random.default_rng(1))
        assert isinstance(mgr._lidar_range_bias_m, float)

    def test_bias_zero_when_limit_is_zero(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_bias_limit_m=0.0))
        mgr.sample_lidar_noise_bias(np.random.default_rng(42))
        assert mgr._lidar_range_bias_m == 0.0

    def test_bias_varies_across_runs(self) -> None:
        values = []
        for seed in range(50):
            mgr = _make_manager(_enabled_cfg())
            mgr.sample_lidar_noise_bias(np.random.default_rng(seed))
            values.append(mgr._lidar_range_bias_m)
        assert len(set(values)) > 1

    def test_resamples_on_each_call(self) -> None:
        mgr = _make_manager(_enabled_cfg())
        values = []
        for seed in range(20):
            mgr.sample_lidar_noise_bias(np.random.default_rng(seed))
            values.append(mgr._lidar_range_bias_m)
        assert len(set(values)) > 1


class TestLidarNoiseRangeEffect:
    """Bias and random noise alter range in the expected direction."""

    def test_positive_bias_increases_range(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.0))
        mgr._lidar_range_bias_m = 0.060
        result = mgr._apply_lidar_noise(_point(5.0))
        assert result[0, 0] > 5.0

    def test_negative_bias_decreases_range(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.0))
        mgr._lidar_range_bias_m = -0.060
        result = mgr._apply_lidar_noise(_point(5.0))
        assert result[0, 0] < 5.0

    def test_zero_bias_mean_range_near_true(self) -> None:
        # With zero bias and Gaussian random noise, the sample mean should be
        # near 5.0 within 3 sigma of the sampling distribution.
        n = 10_000
        stddev = 0.020
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=stddev))
        mgr._lidar_range_bias_m = 0.0
        pts = _points(n, x=5.0)
        result = mgr._apply_lidar_noise(pts)
        mean_x = float(result[:, 0].mean())
        tolerance = 3.0 * stddev / math.sqrt(n)
        assert abs(mean_x - 5.0) < tolerance

    def test_random_noise_produces_different_outputs(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.020))
        mgr._lidar_range_bias_m = 0.0
        pts = _points(100)
        out1 = mgr._apply_lidar_noise(pts.copy())
        out2 = mgr._apply_lidar_noise(pts.copy())
        assert not np.array_equal(out1, out2)


class TestLidarNoiseMinRangeClipping:
    """Returns below min_range_m are discarded after noise."""

    def test_sub_minimum_return_discarded(self) -> None:
        # Point at 0.06 m with negative bias -0.060 puts the noisy range
        # below the 0.05 m min_range_m floor.
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.0))
        mgr._lidar_range_bias_m = -0.060
        result = mgr._apply_lidar_noise(_point(0.06))
        assert result.shape[0] == 0

    def test_valid_returns_survive_clipping(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.0))
        mgr._lidar_range_bias_m = 0.0
        pts = _points(10, x=5.0)
        result = mgr._apply_lidar_noise(pts)
        assert result.shape[0] == 10


class TestLidarNoiseDropout:
    """Point dropout removes the expected fraction of returns."""

    def test_zero_dropout_keeps_all(self) -> None:
        mgr = _make_manager(_enabled_cfg(dropout_rate=0.0))
        mgr._lidar_range_bias_m = 0.0
        result = mgr._apply_lidar_noise(_points(50))
        assert result.shape[0] == 50

    def test_full_dropout_removes_all(self) -> None:
        mgr = _make_manager(_enabled_cfg(dropout_rate=1.0))
        mgr._lidar_range_bias_m = 0.0
        result = mgr._apply_lidar_noise(_points(50))
        assert result.shape[0] == 0

    def test_half_dropout_removes_approximately_half(self) -> None:
        n = 10_000
        mgr = _make_manager(_enabled_cfg(dropout_rate=0.5))
        mgr._lidar_range_bias_m = 0.0
        result = mgr._apply_lidar_noise(_points(n))
        # 3-sigma bounds for Binomial(n=10000, p=0.5): mean=5000, std=50
        assert 4_750 <= result.shape[0] <= 5_250


class TestLidarNoiseShapePreservation:
    """Output array dtype, shape, and z-column are correct."""

    def test_output_dtype_is_float32(self) -> None:
        mgr = _make_manager(_enabled_cfg())
        mgr._lidar_range_bias_m = 0.0
        result = mgr._apply_lidar_noise(_points(10))
        assert result.dtype == np.float32

    def test_output_shape_n_by_3(self) -> None:
        mgr = _make_manager(_enabled_cfg())
        mgr._lidar_range_bias_m = 0.0
        result = mgr._apply_lidar_noise(_points(10))
        assert result.ndim == 2 and result.shape[1] == 3

    def test_empty_input_returns_empty(self) -> None:
        mgr = _make_manager(_enabled_cfg())
        mgr._lidar_range_bias_m = 0.0
        empty = np.empty((0, 3), dtype=np.float32)
        result = mgr._apply_lidar_noise(empty)
        assert result.shape == (0, 3)

    def test_z_coordinate_preserved(self) -> None:
        mgr = _make_manager(_enabled_cfg(range_random_stddev_m=0.0, dropout_rate=0.0))
        mgr._lidar_range_bias_m = 0.0
        pts = np.array([[5.0, 0.0, 1.23], [4.0, 1.0, -0.5]], dtype=np.float32)
        result = mgr._apply_lidar_noise(pts)
        assert result.shape[0] == 2
        np.testing.assert_allclose(result[:, 2], pts[:, 2], atol=1e-6)


class TestLidarNoiseIntegrationWithExtractFeatures:
    """Noisy scans must remain compatible with extract_obstacle_features."""

    def test_noisy_scan_produces_finite_features(self) -> None:
        from uncertainty_rl.envs._parking_core import extract_obstacle_features
        from uncertainty_rl.utils.constants import OBSTACLE_FEATURES_DIM

        mgr = _make_manager(_enabled_cfg())
        mgr._lidar_range_bias_m = 0.010

        # Points in all three sectors (left, forward, right), beyond self-return threshold.
        pts = np.array(
            [
                [3.0, 2.0, 0.0],  # left sector  (bearing > +15 deg)
                [3.0, 0.0, 0.0],  # forward sector
                [3.0, -2.0, 0.0],  # right sector  (bearing < -15 deg)
            ],
            dtype=np.float32,
        )

        noisy = mgr._apply_lidar_noise(pts)
        out = np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)
        result = extract_obstacle_features(noisy, out)
        assert np.all(np.isfinite(result))

    def test_max_datasheet_noise_preserves_sector_assignment(self) -> None:
        """
        @brief Worst-case TiM571 range noise never moves a point across a
               sector boundary, since range noise does not alter bearing.
        """
        from uncertainty_rl.envs._parking_core import extract_obstacle_features
        from uncertainty_rl.utils.constants import OBSTACLE_FEATURES_DIM

        bearing_deg = 30.0
        r = 3.0
        x = r * math.cos(math.radians(bearing_deg))
        y = r * math.sin(math.radians(bearing_deg))
        pts = np.array([[x, y, 0.0]], dtype=np.float32)

        hits_left = 0
        for _ in range(100):
            mgr = _make_manager(
                _enabled_cfg(
                    range_random_stddev_m=0.020,
                    range_bias_limit_m=0.060,
                )
            )
            mgr._lidar_range_bias_m = 0.060  # worst-case positive bias
            noisy = mgr._apply_lidar_noise(pts.copy())
            if len(noisy) == 0:
                continue
            out = np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)
            extract_obstacle_features(noisy, out)
            if out[0] > 0.0:  # left_dist populated
                hits_left += 1

        # All 100 trials must keep the point in the left sector (bearing unchanged).
        assert hits_left == 100
