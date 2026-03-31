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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

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
    # For static analysis: always treat the base class as rclpy.Node so mypy
    # can resolve all Node attributes (get_logger, create_publisher, etc.).
    from rclpy.node import Node as _NodeBase
else:
    # At runtime: inherit from Node when available, plain object otherwise.
    # plain object is only used in CI / unit tests where rclpy is absent;
    # the Node-specific methods (create_publisher, get_clock) are never
    # called in that context.
    _NodeBase = Node if _ROS2_AVAILABLE else object


class _CovarianceSubscriber(_NodeBase):  # type: ignore[misc]
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
        @param covariance_topic: Unused -- EKF state is read from the shared
               JSON file, not via DDS. Accepted for call-site compatibility.
        @param node_name: Unique node name for the rclpy publisher node.
        @param ros2_config: Optional ROS 2 config dict from train_config.yaml.
        """
        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        # Monotonic time after which the file mtime must fall to be considered
        # fresh.  Set by invalidate() at each episode reset so stale pre-reset
        # data is never served as a valid current reading.
        self._valid_after: float = 0.0

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

    def invalidate(self) -> None:
        """
        @brief Mark cached data as stale at the current wall-clock time.

        Call this at episode reset (before publish_initial_pose and before
        _wait_for_covariance) so that any JSON written before this moment is
        rejected.  _read_file() will only accept a file whose mtime is strictly
        after the invalidation timestamp.
        """
        with self._lock:
            self._valid_after = time.time()
            self._latest_uncertainty = None
            self._latest_pose = None

    def _read_file(self) -> bool:
        """
        @brief Read the latest EKF state from the shared JSON file.

        Only accepts the file if its modification time is strictly after the
        last invalidate() call, preventing stale pre-reset data from being
        returned during the wait period at episode start.

        Updates both _latest_pose and _latest_uncertainty atomically under
        the lock so callers never see a partially-updated state.

        @return True if fresh data was read successfully.
        """
        try:
            if not _EKF_STATE_PATH.exists():
                return False
            # Reject files written before the last invalidation.
            with self._lock:
                valid_after = self._valid_after
            if _EKF_STATE_PATH.stat().st_mtime <= valid_after:
                return False
            data = json.loads(_EKF_STATE_PATH.read_text())
            cov_3x3 = np.array(data["covariance"]).reshape(3, 3)
            features = extract_2d_covariance_features(cov_3x3)
            pose = np.array(
                [
                    data["x"], data["y"], data["yaw"],
                    data["vx"], data["vy"], data["vyaw"],
                ],
                dtype=np.float64,
            )
            with self._lock:
                self._latest_pose = pose
                self._latest_uncertainty = features
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def get_latest_state(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        @brief Read the shared file once and return both pose and uncertainty.

        Reads the file exactly once per call, avoiding a
        redundant stat + JSON parse when both values are needed (e.g. _get_state).

        @return Tuple of (pose, uncertainty) where:
                pose: shape (6,) = [x, y, yaw, vx, vy, vyaw], or None.
                uncertainty: shape (9,) uncertainty feature vector, or None.
        """
        self._read_file()
        with self._lock:
            pose = (
                cast(np.ndarray, self._latest_pose.copy())
                if self._latest_pose is not None
                else None
            )
            uncertainty = (
                cast(np.ndarray, self._latest_uncertainty.copy())
                if self._latest_uncertainty is not None
                else None
            )
        return pose, uncertainty

    def get_latest_uncertainty(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent 9-element uncertainty feature vector.

        @note Use get_latest_state() when pose is also needed to avoid a
              second file read.
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

        @note Use get_latest_state() when uncertainty is also needed to avoid
              a second file read.
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

        Converts from CARLA world frame (left-handed, y increases rightward)
        to ROS/Cartographer map frame (right-handed, y increases leftward) by
        negating y and yaw before publishing.

        @param x: Spawn X in CARLA world frame (metres).
        @param y: Spawn Y in CARLA world frame (metres).
        @param yaw: Spawn heading in radians (CARLA convention).
        """
        if not _ROS2_AVAILABLE:
            return

        # CARLA -> ROS frame: negate y and yaw (left-hand to right-hand mirror)
        ros_y = -y
        ros_yaw = -yaw

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = ros_y
        msg.pose.pose.orientation.z = math.sin(ros_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(ros_yaw / 2.0)
        msg.pose.covariance[0] = 0.1   # xx
        msg.pose.covariance[7] = 0.1   # yy
        msg.pose.covariance[35] = 0.05  # yaw-yaw
        self._initial_pose_pub.publish(msg)

    @property
    def has_data(self) -> bool:
        """
        @brief Check whether fresh EKF state data is available.

        Attempts a file read if no data is cached yet. Returns True if a
        valid (post-invalidation) reading is held in memory.

        @return True if data file exists and was read successfully.
        """
        with self._lock:
            if self._latest_uncertainty is not None:
                return True
        # Nothing cached -- attempt a read.
        self._read_file()
        with self._lock:
            return self._latest_uncertainty is not None
