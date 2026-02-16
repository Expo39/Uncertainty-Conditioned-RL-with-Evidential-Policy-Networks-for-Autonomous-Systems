"""
@file __init__.py
@brief ROS 2 nodes for SLAM covariance extraction.

The actual node implementations live in the uncertainty_rl_ros2/ ament package
directory, which is the single source of truth used by both colcon (ROS 2
container) and this subpackage (training container). Imports are conditional
because rclpy is only available in containers with ROS 2 installed.
"""

try:
    pass

    __all__ = [
        "CovarianceExtractorNode",
        "CovarianceMonitorNode",
    ]
except ImportError:
    # rclpy / uncertainty_rl_ros2 not available (e.g. host machine, CI)
    __all__ = []
