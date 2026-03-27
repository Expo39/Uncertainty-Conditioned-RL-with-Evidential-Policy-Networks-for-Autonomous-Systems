"""
@file tf_to_odom.py
@brief Bridge node that converts Cartographer TF to nav_msgs/Odometry.

Cartographer publishes its scan-matched pose estimate purely via TF
(odom -> tracking_frame), not as an Odometry message. The robot_localisation
EKF requires an Odometry topic (odom0) for its correction step.

This node periodically looks up the TF from odom -> ego_vehicle/imu
(Cartographer's tracking frame) and publishes it as nav_msgs/Odometry on
/scan_matched_odometry with a fixed covariance. The EKF fuses this with
IMU predictions, and its own output covariance varies naturally based on
the update frequency and consistency of the Cartographer TF signal.

@note The covariance in the published Odometry is a fixed baseline. When
      Cartographer scan matching degrades (sparse environment, occluded
      cones), the TF updates become less frequent or jittery, causing the
      EKF to rely more on IMU prediction and produce higher output
      covariance -- which is exactly the uncertainty signal the RL policy
      needs.

@author Antonio Galdes
"""

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


def _yaw_from_quaternion(q_x: float, q_y: float, q_z: float, q_w: float) -> float:
    """
    @brief Extract yaw angle from a quaternion (2D mode).
    @param q_x: Quaternion x component.
    @param q_y: Quaternion y component.
    @param q_z: Quaternion z component.
    @param q_w: Quaternion w component.
    @return Yaw angle in radians.
    """
    siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    return math.atan2(siny_cosp, cosy_cosp)


class TfToOdomNode(Node):
    """
    @class TfToOdomNode
    @brief Converts Cartographer TF to nav_msgs/Odometry for the EKF.

    Looks up odom -> tracking_frame TF at a configurable rate and publishes
    as an Odometry message on /scan_matched_odometry.
    """

    def __init__(self, node_name: str = "tf_to_odom") -> None:
        """
        @brief Constructor for TfToOdomNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("tracking_frame", "ego_vehicle/imu")
        self.declare_parameter("publish_topic", "/scan_matched_odometry")
        self.declare_parameter("publish_rate", 20.0)  # Hz
        # Fixed pose covariance diagonal [x, y, z, roll, pitch, yaw]
        # These are baseline values; the EKF output covariance will still
        # vary based on how often this node can successfully look up the TF.
        self.declare_parameter("pose_covariance_diagonal",
                               [0.05, 0.05, 1e6, 1e6, 1e6, 0.1])
        self.declare_parameter("twist_covariance_diagonal",
                               [0.1, 0.1, 1e6, 1e6, 1e6, 0.2])

        # Get parameters
        self._odom_frame = str(
            self.get_parameter("odom_frame").value
        )
        self._tracking_frame = str(
            self.get_parameter("tracking_frame").value
        )
        publish_topic = str(
            self.get_parameter("publish_topic").value
        )
        publish_rate = float(
            self.get_parameter("publish_rate").value or 20.0
        )
        pose_cov_diag = self.get_parameter("pose_covariance_diagonal").value
        twist_cov_diag = self.get_parameter("twist_covariance_diagonal").value

        # Pre-compute fixed 36-element covariance arrays (6x6 diagonal)
        # so the timer callback avoids per-call loops.
        self._pose_cov = [0.0] * 36
        self._twist_cov = [0.0] * 36
        for i in range(6):
            self._pose_cov[i * 7] = pose_cov_diag[i]
            self._twist_cov[i * 7] = twist_cov_diag[i]

        # TF listener
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._odom_pub = self.create_publisher(
            Odometry, publish_topic, qos_profile
        )

        # State for velocity estimation via finite differences
        self._prev_transform: Optional[TransformStamped] = None
        self._prev_time_sec: Optional[float] = None

        # Timer
        period = 1.0 / publish_rate
        self._timer = self.create_timer(period, self._timer_callback)

        self.get_logger().info(
            f"TfToOdom: {self._odom_frame} -> {self._tracking_frame} "
            f"at {publish_rate} Hz on {publish_topic}"
        )

    def _timer_callback(self) -> None:
        """
        @brief Timer callback: look up TF and publish Odometry.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                self._odom_frame,
                self._tracking_frame,
                rclpy.time.Time(),  # Latest available
            )
        except Exception:
            # TF not available yet (Cartographer hasn't published)
            return

        # Build Odometry message
        odom = Odometry()
        odom.header.stamp = tf.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._tracking_frame

        # Pose from TF
        odom.pose.pose.position.x = tf.transform.translation.x
        odom.pose.pose.position.y = tf.transform.translation.y
        odom.pose.pose.position.z = tf.transform.translation.z
        odom.pose.pose.orientation = tf.transform.rotation

        # Pre-computed fixed pose covariance
        odom.pose.covariance = self._pose_cov

        # Estimate twist via finite differences
        curr_time = (
            tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        )
        if self._prev_transform is not None and self._prev_time_sec is not None:
            dt = curr_time - self._prev_time_sec
            if dt > 1e-6:
                prev = self._prev_transform
                dx = tf.transform.translation.x - prev.transform.translation.x
                dy = tf.transform.translation.y - prev.transform.translation.y

                curr_yaw = _yaw_from_quaternion(
                    tf.transform.rotation.x,
                    tf.transform.rotation.y,
                    tf.transform.rotation.z,
                    tf.transform.rotation.w,
                )
                prev_yaw = _yaw_from_quaternion(
                    prev.transform.rotation.x,
                    prev.transform.rotation.y,
                    prev.transform.rotation.z,
                    prev.transform.rotation.w,
                )
                dyaw = math.atan2(
                    math.sin(curr_yaw - prev_yaw),
                    math.cos(curr_yaw - prev_yaw),
                )

                # Rotate odom-frame displacement into vehicle body frame.
                # dx/dy are in the odom (world-aligned) frame; the EKF
                # odom0_config expects twist in the child frame (body frame).
                cos_yaw = math.cos(curr_yaw)
                sin_yaw = math.sin(curr_yaw)
                vx_body = (cos_yaw * dx + sin_yaw * dy) / dt
                vy_body = (-sin_yaw * dx + cos_yaw * dy) / dt
                odom.twist.twist.linear.x = vx_body
                odom.twist.twist.linear.y = vy_body
                odom.twist.twist.angular.z = dyaw / dt

        # Pre-computed fixed twist covariance
        odom.twist.covariance = self._twist_cov

        self._odom_pub.publish(odom)

        # Store for next iteration
        self._prev_transform = tf
        self._prev_time_sec = curr_time


def main() -> None:
    """
    @brief Entry point for the tf_to_odom node.
    """
    rclpy.init()
    node = TfToOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
