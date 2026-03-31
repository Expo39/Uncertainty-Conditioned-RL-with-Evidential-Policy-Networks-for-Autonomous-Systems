"""
@file __init__.py
@brief ROS 2 package for EKF covariance extraction.

This is the ament_python package directory used by colcon inside the
ros2-bridge container. The actual node implementations live in
covariance_extractor.py within this directory.
"""

try:
    from uncertainty_rl_ros2.covariance_extractor import (
        CovarianceExtractorNode,
        CovarianceMonitorNode,
    )

    __all__ = [
        "CovarianceExtractorNode",
        "CovarianceMonitorNode",
    ]
except ModuleNotFoundError:
    # Outside the ros2-bridge container (CI, unit tests, host dev).
    # uncertainty_rl_ros2 is an ament_python package only available
    # after colcon build inside the container.
    __all__ = []
