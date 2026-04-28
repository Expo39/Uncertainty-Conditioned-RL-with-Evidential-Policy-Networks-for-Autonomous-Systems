"""
@file covariance_extractor.py
@brief ROS 2 node for extracting covariance from robot_localization.

This module implements a ROS 2 node that subscribes to odometry messages from
robot_localization and extracts the covariance matrix for use in RL training.
Publishes a custom CovarianceEstimate message with semantic fields per Henki
ROS 2 best practices.

@author Antonio Galdes
"""

import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from uncertainty_rl_msgs.msg import CovarianceEstimate


class CovarianceExtractorNode(Node):
    """
    @class CovarianceExtractorNode
    @brief ROS 2 node for extracting localisation covariance.

    Subscribes to odometry messages from robot_localization and publishes
    a CovarianceEstimate message for consumption by the RL agent.
    """

    # Indices into the 6x6 EKF pose covariance for [x, y, yaw].
    # Full order: [x, y, z, roll, pitch, yaw] -> indices 0, 1, 5.
    _COV_INDICES: List[int] = [0, 1, 5]

    # Default shared file paths. Overridden at runtime by the EKF_STATE_FILE
    # environment variable so that multiple ros2-bridge instances (parallel
    # CARLA workers) each write to a separate file without race conditions.
    # Worker 0: ekf_state.json (default), Worker 1: ekf_state_1.json, etc.
    _DEFAULT_SHARED_PATH: str = "/workspace/outputs/ekf_state.json"

    # File-based /set_pose signal written by the training container.
    # The training container writes {seq, x, y, yaw} in CARLA world frame.
    # This node watches the file and publishes /set_pose locally so
    # the EKF (same Jazzy DDS domain) receives it.
    _DEFAULT_INITIAL_POSE_PATH: str = "/workspace/outputs/initial_pose.json"

    # Log every N odometry callbacks (~100 at 20 Hz = every 5 s).
    _LOG_EVERY_N: int = 100

    def __init__(self, node_name: str = "covariance_extractor") -> None:
        """
        @brief Constructor for CovarianceExtractorNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("covariance_topic", "/ekf_uncertainty/covariance")
        self.declare_parameter("publish_rate", 10.0)  # Hz
        # Kept for launch file compatibility; no longer used.
        self.declare_parameter("twist_in_odom_frame", False)
        # Per-instance EKF state file path. Defaults to EKF_STATE_FILE env var
        # (set by docker-compose.parallel.yml for worker 1+), then falls back to
        # _DEFAULT_SHARED_PATH for worker 0 / single-instance deployment.
        self.declare_parameter(
            "ekf_state_file",
            os.environ.get("EKF_STATE_FILE", self._DEFAULT_SHARED_PATH),
        )

        # Get parameters
        odom_topic: str = str(self.get_parameter("odom_topic").value)
        covariance_topic: str = str(self.get_parameter("covariance_topic").value)
        publish_rate: float = float(self.get_parameter("publish_rate").value)
        # Instance-specific file paths derived from the ekf_state_file parameter.
        ekf_path: str = str(self.get_parameter("ekf_state_file").value)
        self._SHARED_PATH: str = ekf_path
        self._TMP_PATH: str = ekf_path + ".tmp"

        # Set up QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Create subscriber for odometry
        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile
        )

        # Create publisher for covariance
        self.covariance_publisher = self.create_publisher(
            CovarianceEstimate, covariance_topic, qos_profile
        )

        # Latest extracted state: pose tuple + pre-serialised covariance list.
        # Both updated atomically at the end of odom_callback so publish_covariance
        # never sees a partially-updated state.
        self._latest_pose: Optional[Tuple[float, float, float, float]] = None
        self._latest_cov_flat: Optional[List[float]] = None
        self._log_counter: int = 0
        # Monotonically increasing counter written into ekf_state.json so the
        # training container can detect genuinely new writes without relying on
        # file mtime (which is unreliable across Docker container clocks).
        self._write_seq: int = 0

        # Ensure the outputs directory exists before the first file write.
        # The container may not have /workspace/outputs on first run;
        # this guard prevents a FileNotFoundError on the first odom_callback.
        # Uses the instance-specific path so parallel workers each create their
        # own output directory if it differs from the default.
        os.makedirs(os.path.dirname(os.path.abspath(self._SHARED_PATH)), exist_ok=True)

        self.get_logger().info(
            f"CovarianceExtractor: writing EKF state to {self._SHARED_PATH}"
        )

        # Create timer for publishing
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_covariance)

        # --- /set_pose file watcher -------------------------------------------
        # The training container (Humble) writes initial_pose.json at episode
        # reset. This node watches the file and publishes on /set_pose so the
        # EKF resets its state to the vehicle spawn pose.
        # NOTE: robot_localization 3.8.3 (Jazzy) subscribes to "set_pose",
        # NOT "/initialpose". The /initialpose topic is a ROS 1 convention.
        initial_pose_path: str = os.environ.get(
            "INITIAL_POSE_FILE", self._DEFAULT_INITIAL_POSE_PATH
        )
        self._initial_pose_path: str = initial_pose_path
        self._initial_pose_last_seq: int = 0
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/set_pose", 10
        )
        # Poll at 10 Hz -- fast enough to catch the file within 0.1 s of write.
        self._initial_pose_timer = self.create_timer(
            0.1, self._check_initial_pose_file
        )

        self.get_logger().info(
            f"CovarianceExtractor: {odom_topic} -> {covariance_topic} "
            f"at {publish_rate} Hz, "
            f"initial_pose_file={self._initial_pose_path}"
        )

    def odom_callback(self, msg: Odometry) -> None:
        """
        @brief Callback for odometry messages.

        Extracts pose, velocity, and covariance from the odometry message and
        stores them. Velocity comes from msg.twist.twist which robot_localization
        populates with EKF-filtered linear and angular velocity estimates.
        @param msg: Odometry message from robot_localization.
        """
        # -- Pose ---------------------------------------------------------------
        x = msg.pose.pose.position.x
        # Negate y: CARLA uses a left-handed coordinate system (y increases
        # rightward / southward) while ROS uses right-handed
        # (y increases leftward / northward). Negating here ensures all
        # downstream consumers (training container, calibration, _get_state)
        # work in CARLA world-frame convention consistently.
        y = -msg.pose.pose.position.y

        # Convert quaternion to yaw. Negate because the y-axis flip mirrors
        # the rotation direction (left-hand vs right-hand convention).
        # Wrap explicitly to [-pi, pi]: robot_localization's EKF yaw state
        # can drift past +/-pi when two_d_mode=true accumulates yaw without
        # normalisation, producing values like -270 deg = -4.71 rad.
        # Re-deriving yaw from the published quaternion (which IS normalised)
        # rather than reading the EKF state directly avoids this.
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = -math.atan2(siny_cosp, cosy_cosp)

        # -- Velocity -----------------------------------------------------------
        # vyaw negated for y-axis flip (left-hand to right-hand convention).
        vyaw = -msg.twist.twist.angular.z

        # -- Covariance ---------------------------------------------------------
        # Extract 3x3 [x, y, yaw] submatrix from the 6x6 pose covariance.
        # Full order: [x, y, z, roll, pitch, yaw] -> indices _COV_INDICES = [0,1,5].
        covariance_3x3 = np.array(msg.pose.covariance).reshape(6, 6)[
            np.ix_(self._COV_INDICES, self._COV_INDICES)
        ]
        cov_flat: List[float] = covariance_3x3.flatten().tolist()

        # -- Atomic shared-file write ------------------------------------------
        # Cross-distro access: training container (Humble) reads this file since
        # DDS wire protocol is incompatible between Jazzy and Humble containers.
        # Atomic rename prevents partial reads by the training container.
        # `seq` is a monotonically increasing counter; the training container
        # tracks the last-seen seq and only accepts a read whose seq is strictly
        # greater than the seq at the time of the last invalidate() call.
        # This is clock-skew-proof -- mtime comparisons across Docker container
        # clocks are unreliable on some host configurations.
        self._write_seq += 1
        data = {
            "seq": self._write_seq,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "vyaw": float(vyaw),
            "covariance": cov_flat,
        }
        # Log a one-shot warning when the EKF first produces NaN so the
        # container log shows exactly when and what the EKF published.
        # robot_localization logs "Critical Error, NaNs were detected" itself,
        # but it can be buried -- this warning is searchable in docker logs.
        if math.isnan(x) or math.isnan(y) or math.isnan(yaw):
            self.get_logger().warn(
                f"EKF output contains NaN (seq={self._write_seq}): "
                f"x={x} y={y} yaw={yaw} vyaw={vyaw} "
                f"cov_diag=[{cov_flat[0]:.4f},{cov_flat[4]:.4f},{cov_flat[8]:.4f}]. "
                "Check for: (1) sensor with zero covariance reaching the EKF on an "
                "enabled axis, (2) huge time delta on first predict() call."
            )

        with open(self._TMP_PATH, "w") as f:
            json.dump(data, f)
        os.replace(self._TMP_PATH, self._SHARED_PATH)

        # -- Update state atomically -------------------------------------------
        # Both attributes are written here; publish_covariance only reads them.
        self._latest_pose = (x, y, yaw, vyaw)
        self._latest_cov_flat = cov_flat

        # -- Periodic log -------------------------------------------------------
        self._log_counter += 1
        if self._log_counter >= self._LOG_EVERY_N:
            self._log_counter = 0
            std_x = math.sqrt(covariance_3x3[0, 0])
            std_y = math.sqrt(covariance_3x3[1, 1])
            std_yaw = math.sqrt(covariance_3x3[2, 2])
            self.get_logger().info(
                f"Uncertainty - X: {std_x:.4f}m, "
                f"Y: {std_y:.4f}m, Yaw: {math.degrees(std_yaw):.2f}deg"
            )

    def publish_covariance(self) -> None:
        """
        @brief Publish the latest covariance as a CovarianceEstimate message.

        Uses semantic fields (x, y, yaw, vyaw, covariance) instead of
        a flat array. Includes a timestamped header for latency measurement and
        ordering. The covariance flat list is pre-computed in odom_callback to
        avoid redundant numpy serialisation on every publish tick.
        """
        if self._latest_pose is None or self._latest_cov_flat is None:
            return

        x, y, yaw, vyaw = self._latest_pose

        msg = CovarianceEstimate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.x = x
        msg.y = y
        msg.yaw = yaw
        msg.vyaw = vyaw
        msg.covariance = self._latest_cov_flat

        self.covariance_publisher.publish(msg)

    def _check_initial_pose_file(self) -> None:
        """
        @brief Poll initial_pose.json and publish /set_pose to reset the EKF.

        At each episode reset the training container writes initial_pose.json
        with the vehicle spawn pose. This method publishes on /set_pose so the
        robot_localisation EKF resets its state estimate. No datum reset is
        needed since navsat_transform has been replaced by the flat-earth
        projection in GnssNoiseRelayNode.
        """
        try:
            with open(self._initial_pose_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        seq = int(data.get("seq", 0))
        if seq <= self._initial_pose_last_seq:
            return
        self._initial_pose_last_seq = seq

        # CARLA -> ROS frame: negate y and yaw (left-hand to right-hand).
        x = float(data["x"])
        ros_y = -float(data["y"])
        ros_yaw = -float(data["yaw"])

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = ros_y
        msg.pose.pose.orientation.z = math.sin(ros_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(ros_yaw / 2.0)
        # Tight covariance forces the EKF to snap to the given spawn pose
        # rather than treating it as a soft hint. Without tight covariance,
        # the EKF blends the reset pose with its prior state, leaving a
        # residual yaw error (~5 deg) that persists through calibration.
        #   xx/yy: 0.01 m^2 = 10 cm stddev (RTK-fixed quality)
        #   yaw-yaw: 1e-4 rad^2 = 0.57 deg stddev (effectively known heading)
        msg.pose.covariance[0] = 0.01  # xx
        msg.pose.covariance[7] = 0.01  # yy
        msg.pose.covariance[35] = 1.0e-4  # yaw-yaw
        self._initial_pose_pub.publish(msg)
        self.get_logger().info(
            f"Published /set_pose: x={x:.2f} y={ros_y:.2f} "
            f"yaw={math.degrees(ros_yaw):.1f}deg (seq={seq})"
        )


class CovarianceMonitorNode(Node):
    """
    @class CovarianceMonitorNode
    @brief ROS 2 node for monitoring and visualising covariance.

    Subscribes to CovarianceEstimate and logs pose, velocity, and uncertainty
    statistics. Rate-limited to avoid flooding the terminal at 10 Hz.
    """

    # Log every N messages (~10 Hz publish rate -> every 2 s at N=20).
    _LOG_EVERY_N: int = 20

    def __init__(self, node_name: str = "covariance_monitor") -> None:
        """
        @brief Constructor for CovarianceMonitorNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        self.declare_parameter("covariance_topic", "/ekf_uncertainty/covariance")
        covariance_topic = str(self.get_parameter("covariance_topic").value)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.covariance_subscriber = self.create_subscription(
            CovarianceEstimate,
            covariance_topic,
            self._covariance_callback,
            qos_profile,
        )

        self._log_counter: int = 0
        self.get_logger().info(f"Covariance monitor initialised on {covariance_topic}")

    def _covariance_callback(self, msg: CovarianceEstimate) -> None:
        """
        @brief Callback for covariance messages.
        @param msg: CovarianceEstimate containing pose and covariance data.
        """
        self._log_counter += 1
        if self._log_counter < self._LOG_EVERY_N:
            return
        self._log_counter = 0

        # Index directly to avoid a full reshape for just the diagonal.
        std_x = math.sqrt(msg.covariance[0])
        std_y = math.sqrt(msg.covariance[4])
        std_yaw = math.sqrt(msg.covariance[8])

        self.get_logger().info(
            f"Pose: ({msg.x:.2f}, {msg.y:.2f}, {math.degrees(msg.yaw):.1f}deg) | "
            f"vyaw={math.degrees(msg.vyaw):.2f}deg/s | "
            f"Uncertainty: std_x={std_x:.4f}m, "
            f"std_y={std_y:.4f}m, std_yaw={math.degrees(std_yaw):.2f}deg"
        )


def main(args=None) -> None:
    """
    @brief Main entry point for the ROS 2 node.
    """
    rclpy.init(args=args)

    node = CovarianceExtractorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_monitor(args=None) -> None:
    """
    @brief Main entry point for the monitor node.
    """
    rclpy.init(args=args)

    node = CovarianceMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
