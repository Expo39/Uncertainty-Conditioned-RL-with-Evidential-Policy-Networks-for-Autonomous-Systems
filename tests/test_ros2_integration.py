"""
@file test_ros2_integration.py
@brief Integration tests for ROS 2 covariance pipeline.

These tests require the full Docker stack (CARLA + ros2-bridge + training)
to be running. They are marked with @pytest.mark.integration and skipped
in CI where Docker containers are not available.

Run with: pytest -m integration tests/test_ros2_integration.py
"""

import numpy as np
import pytest

from uncertainty_rl.utils.constants import COVARIANCE_FEATURES_DIM, TOTAL_OBS_DIM

try:
    import rclpy

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not _ROS2_AVAILABLE, reason="rclpy not available")
class TestROS2CovariancePipeline:
    """
    @class TestROS2CovariancePipeline
    @brief Integration tests verifying real EKF covariance flows into the env.

    @note Requires Docker stack running: make docker-up
    """

    def test_covariance_received_within_timeout(self) -> None:
        """
        @brief Verify EKF covariance message arrives within the configured timeout.

        Creates a CARLAParkingEnv, resets it (which calls _wait_for_covariance),
        and checks that the reset completes without RuntimeError.
        """
        from uncertainty_rl.envs import CARLAParkingEnv

        env = CARLAParkingEnv(
            max_steps=10,
            ros2_config={
                "covariance_topic": "/ekf_uncertainty/covariance",
                "covariance_timeout": 30.0,
            },
        )
        try:
            obs, info = env.reset()
            # If we get here, covariance was received within timeout
            assert obs.shape == (TOTAL_OBS_DIM,)
        finally:
            env.close()

    def test_covariance_dimensions_match_constant(self) -> None:
        """
        @brief Verify covariance subscriber returns correct elements.
        """
        from uncertainty_rl.envs.carla_parking import _CovarianceSubscriber

        if not rclpy.ok():
            rclpy.init()

        sub = _CovarianceSubscriber(
            covariance_topic="/ekf_uncertainty/covariance",
            node_name="test_cov_dim_check",
        )

        # Spin briefly to receive a message
        import time

        start = time.monotonic()
        while not sub.has_data and (time.monotonic() - start) < 15.0:
            rclpy.spin_once(sub, timeout_sec=0.1)

        uncertainty = sub.get_latest_uncertainty()
        sub.destroy_node()

        assert uncertainty is not None, "No covariance message received within 15s"
        assert uncertainty.shape == (COVARIANCE_FEATURES_DIM,)

    def test_state_vector_contains_real_uncertainty(self) -> None:
        """
        @brief Verify that _get_state() returns non-zero uncertainty features
               after reset (indices 6-14).
        """
        from uncertainty_rl.envs import CARLAParkingEnv

        env = CARLAParkingEnv(
            max_steps=10,
            ros2_config={
                "covariance_topic": "/ekf_uncertainty/covariance",
                "covariance_timeout": 30.0,
            },
        )
        try:
            obs, _ = env.reset()
            # Uncertainty features are indices 6:15
            uncertainty_features = obs[6:15]
            # At least some should be non-zero after EKF has run
            assert np.any(
                uncertainty_features != 0.0
            ), "Uncertainty features should be non-zero with real EKF data"
        finally:
            env.close()
