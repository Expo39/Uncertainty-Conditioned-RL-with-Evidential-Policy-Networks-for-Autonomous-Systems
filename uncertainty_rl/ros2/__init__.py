"""
@file __init__.py
@brief ROS 2 nodes for EKF covariance extraction.

The implementations live in uncertainty_rl_ros2/, the one copy shared by
colcon and this subpackage. The import is conditional because rclpy exists
only in containers with ROS 2 installed.
"""

try:
    from uncertainty_rl_ros2.covariance_extractor import (  # noqa: F401
        CovarianceExtractorNode,
        CovarianceMonitorNode,
    )

    __all__ = [
        "CovarianceExtractorNode",
        "CovarianceMonitorNode",
    ]
except ImportError:
    # rclpy / uncertainty_rl_ros2 not available (e.g. host machine, CI)
    __all__ = []
