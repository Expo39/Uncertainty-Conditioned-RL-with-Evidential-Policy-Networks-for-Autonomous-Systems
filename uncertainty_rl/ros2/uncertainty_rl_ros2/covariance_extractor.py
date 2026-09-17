"""
@file covariance_extractor.py
@brief ROS 2 node for extracting covariance from robot_localization.
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

    # Flat row-major indices of the [x, y, yaw] submatrix within the 36-element
    # pose covariance, whose 6x6 axis order is [x, y, z, roll, pitch, yaw].
    _COV_FLAT_IDX: Tuple[int, ...] = (0, 1, 5, 6, 7, 11, 30, 31, 35)

    # Overridden by the EKF_STATE_FILE environment variable.
    _DEFAULT_SHARED_PATH: str = "/workspace/outputs/ekf_state.json"

    # File-based /set_pose signal: the training container writes
    # {seq, x, y, yaw} here in CARLA world frame.
    _DEFAULT_INITIAL_POSE_PATH: str = "/workspace/outputs/initial_pose.json"

    # Every 5 s at the EKF's 20 Hz odometry rate.
    _LOG_EVERY_N: int = 100

    def __init__(self, node_name: str = "covariance_extractor") -> None:
        """
        @brief Constructor for CovarianceExtractorNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("covariance_topic", "/ekf_uncertainty/covariance")
        self.declare_parameter("publish_rate", 10.0)  # Hz
        # Both launch files still pass this, so it must be declared, but the
        # node never reads it: the EKF already publishes twist in the body frame.
        self.declare_parameter("twist_in_odom_frame", False)
        self.declare_parameter(
            "ekf_state_file",
            os.environ.get("EKF_STATE_FILE", self._DEFAULT_SHARED_PATH),
        )

        odom_topic: str = str(self.get_parameter("odom_topic").value)
        covariance_topic: str = str(self.get_parameter("covariance_topic").value)
        publish_rate: float = float(self.get_parameter("publish_rate").value)
        ekf_path: str = str(self.get_parameter("ekf_state_file").value)
        self._SHARED_PATH: str = ekf_path
        self._TMP_PATH: str = ekf_path + ".tmp"

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_subscriber = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile
        )

        self.covariance_publisher = self.create_publisher(
            CovarianceEstimate, covariance_topic, qos_profile
        )

        # Both are written together at the end of odom_callback so
        # publish_covariance never sees a half-updated state.
        self._latest_pose: Optional[Tuple[float, float, float, float, float]] = None
        self._latest_cov_flat: Optional[List[float]] = None
        self._log_counter: int = 0
        # Lets the training container detect genuinely new writes without file
        # mtime, which is unreliable across Docker container clocks.
        self._write_seq: int = 0

        os.makedirs(os.path.dirname(os.path.abspath(self._SHARED_PATH)), exist_ok=True)

        self.get_logger().info(
            f"CovarianceExtractor: writing EKF state to {self._SHARED_PATH}"
        )

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_covariance)

        # The training container (Humble) writes initial_pose.json at episode
        # reset; watching it here drives the /set_pose EKF reset.
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
        @brief Extract pose, velocity and covariance from EKF odometry.
        @param msg: Odometry message from robot_localization.
        """
        x = msg.pose.pose.position.x
        # ROS is right-handed (+y north), CARLA left-handed (+y south).
        # Negating here keeps every downstream consumer in the CARLA frame.
        y = -msg.pose.pose.position.y

        # Yaw is negated for the same handedness reason: flipping y mirrors
        # the direction of rotation.
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = -math.atan2(siny_cosp, cosy_cosp)

        # Deliberately NOT negated: vyaw stays REP-103 (left turn positive) to
        # match LiDAR bearing, target dyaw and the reward signs.
        vyaw = msg.twist.twist.angular.z
        # Already body-frame: the EKF publishes twist in base_link.
        vx = msg.twist.twist.linear.x

        raw_cov = msg.pose.covariance
        cov_flat: List[float] = [raw_cov[i] for i in self._COV_FLAT_IDX]

        # The file, not DDS, is the transport to the training container: the
        # DDS wire protocol is incompatible between Jazzy and Humble.
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

        self._latest_pose = (x, y, yaw, vyaw, vx)
        self._latest_cov_flat = cov_flat

        self._log_counter += 1
        if self._log_counter >= self._LOG_EVERY_N:
            self._log_counter = 0
            # Diagonal of the 3x3: xx, yy, yaw-yaw.
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
        """
        if self._latest_pose is None or self._latest_cov_flat is None:
            return

        # vx goes to ekf_state.json only; the message carries pose + vyaw + cov.
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

        @note No datum reset accompanies this: the flat-earth projection in
              GnssNoiseRelayNode stands in for navsat_transform.
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
        # A seq regression means the training container's subscriber has been
        # reconstructed (e.g. a new Optuna trial), so the counter restarts;
        # clearing it here keeps the new session's first reset from being dropped.
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
    @brief Debug node logging pose and uncertainty from CovarianceEstimate.
    """

    # Every 2 s at the 10 Hz publish rate; unthrottled would flood the terminal.
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
