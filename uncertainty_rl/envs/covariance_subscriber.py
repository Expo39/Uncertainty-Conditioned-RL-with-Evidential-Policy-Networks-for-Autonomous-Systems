"""
@file covariance_subscriber.py
@brief ROS 2 covariance subscriber node for EKF uncertainty features.

Runs via rclpy.spin() in a daemon thread so it does not block Gymnasium step().
Consumed by CARLAParkingEnv when include_covariance=True.
"""

import threading
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

try:
    import rclpy  # noqa: F401
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from uncertainty_rl_msgs.msg import CovarianceEstimate

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

from uncertainty_rl.utils.covariance_utils import extract_2d_covariance_features

if TYPE_CHECKING:
    from rclpy.node import Node as _NodeBase
else:
    _NodeBase = Node if _ROS2_AVAILABLE else object


class _CovarianceSubscriber(_NodeBase):
    """
    @class _CovarianceSubscriber
    @brief Lightweight rclpy Node that subscribes to EKF covariance.

    Caches the latest 9-element uncertainty feature vector and 6-element pose
    + velocity vector in a thread-safe manner. Runs via rclpy.spin() in a
    daemon thread so it does not block Gymnasium step().

    The CovarianceExtractorNode publishes a CovarianceEstimate message with
    semantic fields (x, y, yaw, vx, vy, vyaw, covariance[9]). We reshape the
    covariance field into a 3x3 matrix and call extract_2d_covariance_features().
    The pose + velocity fields are cached separately so _get_state() can use
    EKF estimates for obs indices 0-5 rather than CARLA ground truth.
    """

    def __init__(
        self,
        covariance_topic: str = "/ekf_uncertainty/covariance",
        node_name: str = "covariance_subscriber",
    ) -> None:
        """
        @brief Initialise the covariance subscriber node.
        @param covariance_topic: ROS 2 topic to subscribe to.
        @param node_name: Unique node name (important when multiple envs exist).
        """
        if not _ROS2_AVAILABLE:
            return

        super().__init__(node_name)

        self._lock = threading.Lock()
        self._latest_uncertainty: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._message_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            depth=10,
        )

        self._subscription = self.create_subscription(
            CovarianceEstimate,
            covariance_topic,
            self._covariance_callback,
            qos,
        )
        self.get_logger().info(f"Subscribed to covariance topic: {covariance_topic}")

    def _covariance_callback(self, msg: "CovarianceEstimate") -> None:
        """
        @brief Callback for incoming covariance messages.
        @param msg: CovarianceEstimate with semantic fields
                    (x, y, yaw, vx, vy, vyaw, covariance[9]).
        """
        cov_matrix = np.array(msg.covariance).reshape(3, 3)
        features = extract_2d_covariance_features(cov_matrix)

        with self._lock:
            self._latest_uncertainty = features
            self._latest_pose = np.array(
                [msg.x, msg.y, msg.yaw, msg.vx, msg.vy, msg.vyaw], dtype=np.float64
            )
            self._message_count += 1

    def get_latest_uncertainty(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent 9-element uncertainty feature vector.
        @return Array of shape (9,) or None if no message received yet.
        """
        with self._lock:
            if self._latest_uncertainty is not None:
                return cast(np.ndarray, self._latest_uncertainty.copy())
            return None

    def get_latest_pose(self) -> Optional[np.ndarray]:
        """
        @brief Get the most recent EKF pose and velocity estimate.
        @return Array of shape (6,) = [x, y, yaw, vx, vy, vyaw] or None if no
                message has been received yet.
        """
        with self._lock:
            if self._latest_pose is not None:
                return cast(np.ndarray, self._latest_pose.copy())
            return None

    @property
    def has_data(self) -> bool:
        """
        @brief Check whether at least one covariance message has been received.
        @return True if data is available.
        """
        with self._lock:
            return self._latest_uncertainty is not None
