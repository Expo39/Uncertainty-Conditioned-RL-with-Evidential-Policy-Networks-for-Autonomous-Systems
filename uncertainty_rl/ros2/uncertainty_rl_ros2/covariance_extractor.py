"""
@file covariance_extractor.py
@brief ROS 2 node for extracting covariance from robot_localization.

This module implements a ROS 2 node that subscribes to odometry messages from
robot_localization and extracts the covariance matrix for use in RL training.
"""

import json
import math
import os
from typing import List, Optional, Tuple

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

    # Flat indices into the 36-element pose covariance for the 3x3 [x, y, yaw]
    # submatrix. Full 6x6 order: [x, y, z, roll, pitch, yaw].
    # Row/col positions of [0,1,5] x [0,1,5] in row-major 6x6:
    #   (0,0)=0  (0,1)=1  (0,5)=5
    #   (1,0)=6  (1,1)=7  (1,5)=11
    #   (5,0)=30 (5,1)=31 (5,5)=35
    _COV_FLAT_IDX: Tuple[int, ...] = (0, 1, 5, 6, 7, 11, 30, 31, 35)

    # Default shared file paths. Overridden at runtime by the EKF_STATE_FILE
    # environment variable.
    _DEFAULT_SHARED_PATH: str = "/workspace/outputs/ekf_state.json"

    # File-based /set_pose signal written by the training container.
    # The training container writes {seq, x, y, yaw} in CARLA world frame.
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
        # Per-instance EKF state file path.
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
        self._latest_pose: Optional[Tuple[float, float, float, float, float]] = None
        self._latest_cov_flat: Optional[List[float]] = None
        self._log_counter: int = 0
        # Monotonically increasing counter written into ekf_state.json so the
        # training container can detect genuinely new writes without relying on
        # file mtime (which is unreliable across Docker container clocks).
        self._write_seq: int = 0

        # Ensure the outputs directory exists before the first file write.
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
        initial_pose_path: str = os.environ.get(
            "INITIAL_POSE_FILE", self._DEFAULT_INITIAL_POSE_PATH
        )
        self._initial_pose_path: str = initial_pose_path
        self._initial_pose_last_seq: int = 0
        self._initial_pose_mtime_ns: int = 0
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/set_pose", 10
        )
        # Poll at 10 Hz - fast enough to catch the file within 0.1 s of write.
        self._initial_pose_timer = self.create_timer(0.1, self._check_initial_pose_file)

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
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = -math.atan2(siny_cosp, cosy_cosp)

        # -- Velocity -----------------------------------------------------------
        # vyaw stays in REP-103 (left turn positive) to match the rest of the
        # codebase (LiDAR bearing, target dyaw, reward signs). Position y and
        # yaw above are negated to recover CARLA's left-handed world frame.
        vyaw = msg.twist.twist.angular.z
        # Body-frame longitudinal velocity. The EKF publishes twist in
        # base_link, so no rotation is needed.
        vx = msg.twist.twist.linear.x

        # -- Covariance ---------------------------------------------------------
        # Extract the 3x3 [x, y, yaw] submatrix directly from the flat 36-element
        # pose covariance using pre-computed indices.
        raw_cov = msg.pose.covariance
        cov_flat: List[float] = [raw_cov[i] for i in self._COV_FLAT_IDX]

        # -- Atomic shared-file write ------------------------------------------
        # Cross-distro access: training container (Humble) reads this file since
        # DDS wire protocol is incompatible between Jazzy and Humble containers.
        self._write_seq += 1
        data = {
            "seq": self._write_seq,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "vyaw": float(vyaw),
            "vx": float(vx),
            "covariance": cov_flat,
        }
        # Log a one-shot warning when the EKF first produces NaN so the
        # container log shows exactly when and what the EKF published.
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
        self._latest_pose = (x, y, yaw, vyaw, vx)
        self._latest_cov_flat = cov_flat

        # -- Periodic log -------------------------------------------------------
        self._log_counter += 1
        if self._log_counter >= self._LOG_EVERY_N:
            self._log_counter = 0
            # Diagonal elements of the 3x3: indices 0 (xx), 4 (yy), 8 (yaw-yaw).
            std_x = math.sqrt(cov_flat[0])
            std_y = math.sqrt(cov_flat[4])
            std_yaw = math.sqrt(cov_flat[8])
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

        # vx is written to ekf_state.json only; the DDS message carries
        # pose + vyaw + covariance.
        x, y, yaw, vyaw, _vx = self._latest_pose

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
            mtime_ns = os.stat(self._initial_pose_path).st_mtime_ns
        except OSError:
            return
        if mtime_ns == self._initial_pose_mtime_ns:
            return
        self._initial_pose_mtime_ns = mtime_ns

        try:
            with open(self._initial_pose_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        seq = int(data.get("seq", 0))
        # A regression in seq indicates the training container's subscriber
        # was reconstructed (e.g. new Optuna trial). Reset the counter so
        # the first reset of the new session is honoured rather than dropped.
        if seq < self._initial_pose_last_seq:
            self.get_logger().info(
                f"initial_pose seq regression ({seq} < "
                f"{self._initial_pose_last_seq}): new training session, "
                "resetting counter."
            )
            self._initial_pose_last_seq = 0
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
        # rather than treating it as a soft hint.
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
