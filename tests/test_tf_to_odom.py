"""
@file test_tf_to_odom.py
@brief Unit tests for the TF-to-Odometry bridge pure functions.

Tests cover the three free functions and the covariance scaling logic in
TfToOdomNode._compute_covariance_scale(). All tests are CPU-only and do not
require ROS 2, rclpy, or CARLA. TfToOdomNode is constructed with a mocked
rclpy so its __init__ can be called without a live ROS context.
"""

import math
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers -- import the free functions directly (no Node needed)
# ---------------------------------------------------------------------------


def _import_free_functions():
    """
    @brief Import _yaw_from_quaternion and _make_diagonal_covariance without
           triggering the rclpy import at module level.
    @return Tuple of (_yaw_from_quaternion, _make_diagonal_covariance).
    """
    # tf_to_odom imports rclpy at module level; patch it before import.
    mock_rclpy = MagicMock()
    mock_rclpy.node.Node = object  # Node base becomes plain object
    with patch.dict("sys.modules", {
        "rclpy": mock_rclpy,
        "rclpy.node": mock_rclpy.node,
        "rclpy.qos": MagicMock(),
        "rclpy.time": MagicMock(),
        "tf2_ros": MagicMock(),
        "nav_msgs": MagicMock(),
        "nav_msgs.msg": MagicMock(),
        "geometry_msgs": MagicMock(),
        "geometry_msgs.msg": MagicMock(),
    }):
        import sys
        # Force reload so the patched sys.modules takes effect
        mod_name = "uncertainty_rl.ros2.uncertainty_rl_ros2.tf_to_odom"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import uncertainty_rl.ros2.uncertainty_rl_ros2.tf_to_odom as m
        return m._yaw_from_quaternion, m._make_diagonal_covariance, m


# ---------------------------------------------------------------------------
# _yaw_from_quaternion
# ---------------------------------------------------------------------------


class TestYawFromQuaternion:
    """
    @class TestYawFromQuaternion
    @brief Tests for the quaternion-to-yaw helper.
    """

    def setup_method(self) -> None:
        """
        @brief Import free functions once per test class.
        """
        self._yaw, self._cov, self._mod = _import_free_functions()

    def test_identity_quaternion_gives_zero_yaw(self) -> None:
        """
        @brief Identity quaternion (0, 0, 0, 1) -> yaw = 0.
        """
        yaw = self._yaw(0.0, 0.0, 0.0, 1.0)
        assert abs(yaw) < 1e-9

    def test_90_deg_rotation_about_z(self) -> None:
        """
        @brief 90 deg rotation about z: qz = sin(pi/4), qw = cos(pi/4) -> yaw = pi/2.
        """
        angle = math.pi / 2.0
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)
        yaw = self._yaw(0.0, 0.0, qz, qw)
        assert abs(yaw - math.pi / 2.0) < 1e-6

    def test_minus_90_deg_rotation(self) -> None:
        """
        @brief -90 deg rotation about z -> yaw = -pi/2.
        """
        angle = -math.pi / 2.0
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)
        yaw = self._yaw(0.0, 0.0, qz, qw)
        assert abs(yaw - (-math.pi / 2.0)) < 1e-6

    def test_180_deg_rotation(self) -> None:
        """
        @brief 180 deg rotation -> yaw = +/- pi (both are equivalent).
        """
        angle = math.pi
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)
        yaw = self._yaw(0.0, 0.0, qz, qw)
        assert abs(abs(yaw) - math.pi) < 1e-6

    def test_output_in_minus_pi_to_pi(self) -> None:
        """
        @brief Output is always in (-pi, pi] for arbitrary unit quaternions.
        """
        import random
        random.seed(42)
        for _ in range(50):
            yaw_in = random.uniform(-math.pi, math.pi)
            # Build quaternion from yaw only (2D case)
            qz = math.sin(yaw_in / 2.0)
            qw = math.cos(yaw_in / 2.0)
            yaw_out = self._yaw(0.0, 0.0, qz, qw)
            assert -math.pi <= yaw_out <= math.pi, (
                f"yaw_out={yaw_out:.4f} out of range for yaw_in={yaw_in:.4f}"
            )

    def test_45_deg_rotation(self) -> None:
        """
        @brief 45 deg rotation -> yaw = pi/4 within tolerance.
        """
        angle = math.pi / 4.0
        qz = math.sin(angle / 2.0)
        qw = math.cos(angle / 2.0)
        yaw = self._yaw(0.0, 0.0, qz, qw)
        assert abs(yaw - math.pi / 4.0) < 1e-6


# ---------------------------------------------------------------------------
# _make_diagonal_covariance
# ---------------------------------------------------------------------------


class TestMakeDiagonalCovariance:
    """
    @class TestMakeDiagonalCovariance
    @brief Tests for the 6x6 diagonal covariance builder.
    """

    def setup_method(self) -> None:
        """
        @brief Import free functions once per test class.
        """
        _, self._cov, self._mod = _import_free_functions()

    def test_output_length_36(self) -> None:
        """
        @brief Output must have exactly 36 elements (6x6 flat).
        """
        result = self._cov([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert len(result) == 36

    def test_diagonal_values_set_correctly(self) -> None:
        """
        @brief Diagonal elements are at indices 0, 7, 14, 21, 28, 35.
        """
        diag = [0.1, 0.2, 1e6, 1e6, 1e6, 0.05]
        result = self._cov(diag)
        for i, expected in enumerate(diag):
            assert abs(result[i * 7] - expected) < 1e-12, (
                f"Diagonal[{i}] expected {expected}, got {result[i * 7]}"
            )

    def test_off_diagonal_elements_are_zero(self) -> None:
        """
        @brief All off-diagonal elements must be exactly 0.0.
        """
        result = self._cov([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        diagonal_indices = {i * 7 for i in range(6)}
        for idx, val in enumerate(result):
            if idx not in diagonal_indices:
                assert val == 0.0, f"Off-diagonal index {idx} is {val}, expected 0.0"

    def test_zero_diagonal(self) -> None:
        """
        @brief All-zero diagonal produces an all-zero covariance.
        """
        result = self._cov([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert all(v == 0.0 for v in result)

    def test_large_off_diagonal_penalty_values(self) -> None:
        """
        @brief Standard usage: z/roll/pitch locked out with 1e6 penalty.
        """
        diag = [0.05, 0.05, 1e6, 1e6, 1e6, 0.05]
        result = self._cov(diag)
        assert abs(result[14] - 1e6) < 1.0   # z variance at index 14
        assert abs(result[21] - 1e6) < 1.0   # roll variance at index 21
        assert abs(result[28] - 1e6) < 1.0   # pitch variance at index 28


# ---------------------------------------------------------------------------
# _compute_covariance_scale (TfToOdomNode method)
# ---------------------------------------------------------------------------


def _make_node_no_ros(base_xy: float = 0.05,
                      stale_threshold: float = 0.15,
                      staleness_scale: float = 100.0,
                      stale_max_sec: float = 2.0,
                      jump_threshold: float = 1.0,
                      jump_scale: float = 50.0,
                      jump_decay_steps: int = 10) -> Any:
    """
    @brief Construct a TfToOdomNode-like object without ROS for testing
           _compute_covariance_scale() in isolation.
    @return Object with _compute_covariance_scale bound and state fields set.
    """
    import types

    # Import the method source via the patched module
    _, _, mod = _import_free_functions()

    # Create a minimal object that mimics TfToOdomNode's fields accessed by
    # _compute_covariance_scale, without calling __init__
    node = types.SimpleNamespace()
    node._stale_threshold = stale_threshold
    node._staleness_scale = staleness_scale
    node._stale_max_sec = stale_max_sec
    node._jump_threshold = jump_threshold
    node._jump_scale = jump_scale
    node._jump_decay_steps = jump_decay_steps
    node._jump_inflation_steps = 0
    node._last_tf_wall_sec = None

    # Bind the method
    node._compute_covariance_scale = (
        mod.TfToOdomNode._compute_covariance_scale.__get__(node)
    )

    # Mock get_clock() to return a controllable time
    clock = MagicMock()
    clock.now.return_value.nanoseconds = 0
    node.get_clock = lambda: clock
    node._clock = clock

    # Mock get_logger() to suppress output
    node.get_logger = lambda: MagicMock()

    return node


class TestComputeCovarianceScale:
    """
    @class TestComputeCovarianceScale
    @brief Tests for TfToOdomNode._compute_covariance_scale().
    """

    def test_returns_one_when_no_staleness_no_jump(self) -> None:
        """
        @brief Scale = 1.0 when TF was just received and no jump detected.
        """
        node = _make_node_no_ros()
        now_ns = int(1000.0 * 1e9)  # 1000 seconds
        node.get_clock().now.return_value.nanoseconds = now_ns
        # Mark TF as just received (0.05 s ago -- within stale_threshold of 0.15)
        node._last_tf_wall_sec = now_ns * 1e-9 - 0.05

        scale = node._compute_covariance_scale(position_delta_m=0.0)
        assert abs(scale - 1.0) < 1e-9

    def test_scale_one_when_no_previous_tf(self) -> None:
        """
        @brief Scale = 1.0 when _last_tf_wall_sec is None (no TF seen yet,
               staleness check is skipped).
        """
        node = _make_node_no_ros()
        # _last_tf_wall_sec remains None
        scale = node._compute_covariance_scale(position_delta_m=0.0)
        assert abs(scale - 1.0) < 1e-9

    def test_scale_increases_when_tf_stale(self) -> None:
        """
        @brief Scale > 1.0 after the TF has been stale longer than stale_threshold.
        """
        node = _make_node_no_ros(
            stale_threshold=0.15, staleness_scale=100.0, stale_max_sec=2.0
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        # Last TF 1 second ago -- well past stale_threshold
        node._last_tf_wall_sec = now_ns * 1e-9 - 1.0

        scale = node._compute_covariance_scale(position_delta_m=0.0)
        assert scale > 1.0

    def test_scale_reaches_max_at_stale_max_sec(self) -> None:
        """
        @brief Scale reaches staleness_scale when age >= stale_max_sec.
        """
        node = _make_node_no_ros(
            stale_threshold=0.15, staleness_scale=100.0, stale_max_sec=2.0
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        # Age = stale_max_sec + large buffer -> t clamped to 1.0
        node._last_tf_wall_sec = now_ns * 1e-9 - 10.0

        scale = node._compute_covariance_scale(position_delta_m=0.0)
        assert abs(scale - 100.0) < 1e-6

    def test_jump_inflates_scale(self) -> None:
        """
        @brief A position delta > jump_threshold triggers jump inflation.
        """
        node = _make_node_no_ros(
            jump_threshold=1.0, jump_scale=50.0, jump_decay_steps=10
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        # TF is fresh, so staleness factor = 1.0
        node._last_tf_wall_sec = now_ns * 1e-9 - 0.01

        scale = node._compute_covariance_scale(position_delta_m=2.0)
        assert scale > 1.0

    def test_jump_scale_initial_value(self) -> None:
        """
        @brief Immediately after a jump, scale should be close to jump_scale.
        """
        node = _make_node_no_ros(
            jump_threshold=1.0, jump_scale=50.0, jump_decay_steps=10
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        node._last_tf_wall_sec = now_ns * 1e-9 - 0.01

        scale = node._compute_covariance_scale(position_delta_m=5.0)
        # After triggering jump, jump_inflation_steps = 10 -> t = 10/10 = 1.0
        # scale = 1 + 1.0 * (50 - 1) = 50
        assert abs(scale - 50.0) < 1e-6

    def test_jump_decays_over_steps(self) -> None:
        """
        @brief Scale decreases monotonically after a jump as steps are consumed.
        """
        node = _make_node_no_ros(
            jump_threshold=1.0, jump_scale=50.0,
            jump_decay_steps=5, stale_threshold=9999.0
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        node._last_tf_wall_sec = now_ns * 1e-9 - 0.01

        # Trigger jump
        node._compute_covariance_scale(position_delta_m=2.0)

        scales = []
        for _ in range(5):
            s = node._compute_covariance_scale(position_delta_m=0.0)
            scales.append(s)

        # Each call should produce a scale <= the previous one
        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1] + 1e-9, (
                f"Scale not monotonically decreasing: {scales}"
            )

    def test_scale_returns_to_one_after_decay(self) -> None:
        """
        @brief After jump_decay_steps calls, scale returns to 1.0 (no staleness).
        """
        decay_steps = 4
        node = _make_node_no_ros(
            jump_threshold=1.0, jump_scale=50.0,
            jump_decay_steps=decay_steps, stale_threshold=9999.0
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        node._last_tf_wall_sec = now_ns * 1e-9 - 0.01

        # Trigger jump then consume all decay steps
        node._compute_covariance_scale(position_delta_m=2.0)
        for _ in range(decay_steps):
            node._compute_covariance_scale(position_delta_m=0.0)

        # One more call: _jump_inflation_steps should be 0 now
        scale = node._compute_covariance_scale(position_delta_m=0.0)
        assert abs(scale - 1.0) < 1e-9

    def test_staleness_dominates_over_small_jump(self) -> None:
        """
        @brief When staleness > jump inflation, the staleness factor is returned.
        """
        node = _make_node_no_ros(
            stale_threshold=0.15, staleness_scale=200.0, stale_max_sec=2.0,
            jump_threshold=1.0, jump_scale=10.0, jump_decay_steps=10
        )
        now_ns = int(500.0 * 1e9)
        node.get_clock().now.return_value.nanoseconds = now_ns
        # Age = stale_max_sec: staleness_factor = 200
        node._last_tf_wall_sec = now_ns * 1e-9 - 10.0

        # Trigger a small jump (jump_scale=10 < staleness_scale=200)
        scale = node._compute_covariance_scale(position_delta_m=2.0)
        # max(200, 10) = 200
        assert scale >= 100.0  # staleness dominates
