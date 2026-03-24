"""
@file covariance_subscriber.py
@brief File-based covariance reader for EKF uncertainty features.

Reads the latest EKF state from a shared JSON file written by the
CovarianceExtractorNode in the ros2-bridge container. This avoids DDS
cross-distro serialisation issues between ROS 2 Humble (training container)
and Jazzy (ros2-bridge container).

The file is written atomically (via rename) by the extractor node at the
EKF publish rate (~20 Hz) to /workspace/outputs/ekf_state.json, which is
on a Docker shared volume visible to both containers.

Also publishes /initialpose via rclpy for Cartographer pure localisation
convergence at episode reset.
"""

import json
import math
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import numpy as np

try:
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

from uncertainty_rl.utils.covariance_utils import extract_2d_covariance_features

# Shared file path (Docker volume mount: outputs/ is rw in both containers)
_EKF_STATE_PATH = Path("/workspace/outputs/ekf_state.json")

if TYPE_CHECKING:
    from rclpy.node import Node as _NodeBase
else:
    _NodeBase = Node if _ROS2_AVAILABLE else object


class _CovarianceSubscriber(_NodeBase):
    """
    @class _CovarianceSubscriber
    @brief Reads EKF state from a shared JSON file + publishes /initialpose.

    The CovarianceExtractorNode (ros2-bridge, Jazzy) writes the latest EKF
    pose, velocity, and 3x3 covariance to a shared file. This class reads
    that file on demand -- no DDS subscription needed.

    Inherits from rclpy.Node only for the /initialpose publisher (needed
    for Cartographer pure localisation convergence at episode reset).
    """

    def __init__(
        self,
        covariance_topic: str = "/odometry/filtered",
        node_name: str = "covariance_subscriber",
        ros2_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        @brief Initialise the covariance reader.
        @param covariance_topic: Unused (kept for API compatibility).
        @param node_name: Unique node name for the rclpy publisher node.
        @param ros2_config: Optional ROS 2 config dict from train_config.yaml.
        """
        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None

        # Initialise rclpy Node for /initialpose publisher only
        if _ROS2_AVAILABLE:
            super().__init__(node_name)
            config = ros2_config or {}
            initial_pose_topic = config.get(
                "initial_pose_topic", "/initialpose"
            )
            self._initial_pose_pub = self.create_publisher(
                PoseWithCovarianceStamped,
                initial_pose_topic,
                10,
            )
            self.get_logger().info(
                f"Covariance reader: file={_EKF_STATE_PATH}, "
                f"initialpose={initial_pose_topic}"
            )

    def _read_file(self) -> bool:
        """
        @brief Read the latest EKF state from the shared JSON file.
        @return True if new data was read successfully.
        """
        try:
            if not _EKF_STATE_PATH.exists():
                return False
            data = json.loads(_EKF_STATE_PATH.read_text())
            cov_3x3 = np.array(data["covariance"]).reshape(3, 3)
            features = extract_2d_covariance_features(cov_3x3)
            with self._lock:
                self._latest_uncertainty = features
                self._latest_pose = np.array(
                    [
                        data["x"], data["y"], data["yaw"],
                        data["vx"], data["vy"], data["vyaw"],
                    ],
                    dtype=np.float64,
                )
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def get_latest_uncertainty(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent 9-element uncertainty feature vector.
        @return Array of shape (9,) or None if no data available.
        """
        self._read_file()
        with self._lock:
            if self._latest_uncertainty is not None:
                return cast(np.ndarray, self._latest_uncertainty.copy())
            return None

    def get_latest_pose(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent EKF pose and velocity estimate.
        @return Array of shape (6,) = [x, y, yaw, vx, vy, vyaw] or None.
        """
        self._read_file()
        with self._lock:
            if self._latest_pose is not None:
                return cast(np.ndarray, self._latest_pose.copy())
            return None

    def publish_initial_pose(self, x: float, y: float, yaw: float) -> None:
        """
        @brief Publish the vehicle spawn pose to /initialpose for Cartographer
               pure localisation mode.

        @param x: Spawn X in world frame (metres).
        @param y: Spawn Y in world frame (metres).
        @param yaw: Spawn heading in radians.
        """
        if not _ROS2_AVAILABLE:
            return

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0] = 0.1   # xx
        msg.pose.covariance[7] = 0.1   # yy
        msg.pose.covariance[35] = 0.05  # yaw-yaw
        self._initial_pose_pub.publish(msg)

    @property
    def has_data(self) -> bool:
        """
        @brief Check whether EKF state data is available.
        @return True if data file exists and was read successfully.
        """
        self._read_file()
        with self._lock:
            return self._latest_uncertainty is not None
