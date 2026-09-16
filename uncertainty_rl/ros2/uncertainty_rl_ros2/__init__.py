"""
@file __init__.py
@brief ROS 2 package for EKF covariance extraction.

The ament_python package directory colcon builds inside the ros2-bridge
container.
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
    # Outside the container (CI, host dev): rclpy and the built package exist
    # only after a colcon build inside it.
    __all__ = []
