"""
@file covariance_extractor.py
@brief ROS 2 node for extracting covariance from robot_localization.

This module implements a ROS 2 node that subscribes to odometry messages from
robot_localization and extracts the covariance matrix for use in RL training.
Publishes a custom CovarianceEstimate message with semantic fields per Henki
ROS 2 best practices.
"""

from typing import Optional, Tuple

import numpy as np
import rclpy
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

        # Get parameters
        odom_topic = self.get_parameter("odom_topic").value
        covariance_topic = self.get_parameter("covariance_topic").value
        publish_rate = self.get_parameter("publish_rate").value

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

        # Store latest covariance
        self.latest_covariance: Optional[np.ndarray] = None
        self.latest_pose: Optional[Tuple[float, float, float, float, float, float]] = (
            None
        )
        self._log_counter: int = 0

        # Create timer for publishing
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_covariance)

        self.get_logger().info("Covariance extractor node initialised")
        self.get_logger().info(f"  Subscribing to: {odom_topic}")
        self.get_logger().info(f"  Publishing to: {covariance_topic}")

    def odom_callback(self, msg: Odometry) -> None:
        """
        @brief Callback for odometry messages.

        Extracts pose, velocity, and covariance from the odometry message and
        stores them. Velocity comes from msg.twist.twist which robot_localization
        populates with EKF-filtered linear and angular velocity estimates.
        @param msg: Odometry message from robot_localization.
        """
        # Extract pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Extract yaw from quaternion
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # Convert quaternion to yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Extract EKF-filtered velocity from twist
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vyaw = msg.twist.twist.angular.z

        self.latest_pose = (x, y, yaw, vx, vy, vyaw)

        # Extract covariance matrix (6x6) for pose
        # Format: [x, y, z, roll, pitch, yaw]
        # We extract: [x, y, yaw] which are at indices [0, 1, 5]
        covariance_full = np.array(msg.pose.covariance).reshape(6, 6)

        # Extract 3x3 submatrix for [x, y, yaw]
        indices = [0, 1, 5]
        covariance_3x3 = covariance_full[np.ix_(indices, indices)]

        self.latest_covariance = covariance_3x3

        # Log uncertainty statistics periodically
        self._log_counter += 1

        if self._log_counter % 100 == 0:
            std_x = np.sqrt(covariance_3x3[0, 0])
            std_y = np.sqrt(covariance_3x3[1, 1])
            std_yaw = np.sqrt(covariance_3x3[2, 2])
            self.get_logger().info(
                f"Uncertainty - X: {std_x:.4f}m, "
                f"Y: {std_y:.4f}m, Yaw: {np.rad2deg(std_yaw):.2f}deg"
            )

    def publish_covariance(self) -> None:
        """
        @brief Publish the latest covariance as a CovarianceEstimate message.

        Uses semantic fields (x, y, yaw, vx, vy, vyaw, covariance) instead of
        a flat array. Includes a timestamped header for latency measurement and
        ordering.
        """
        if self.latest_covariance is None or self.latest_pose is None:
            return

        msg = CovarianceEstimate()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.x = self.latest_pose[0]
        msg.y = self.latest_pose[1]
        msg.yaw = self.latest_pose[2]
        msg.vx = self.latest_pose[3]
        msg.vy = self.latest_pose[4]
        msg.vyaw = self.latest_pose[5]
        msg.covariance = self.latest_covariance.flatten().tolist()

        self.covariance_publisher.publish(msg)


class CovarianceMonitorNode(Node):
    """
    @class CovarianceMonitorNode
    @brief ROS 2 node for monitoring and visualising covariance.

    Provides additional monitoring capabilities for debugging and analysis.
    """

    def __init__(self, node_name: str = "covariance_monitor") -> None:
        """
        @brief Constructor for CovarianceMonitorNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("covariance_topic", "/ekf_uncertainty/covariance")

        covariance_topic = self.get_parameter("covariance_topic").value

        # Subscribe to covariance
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.covariance_subscriber = self.create_subscription(
            CovarianceEstimate,
            covariance_topic,
            self.covariance_callback,
            qos_profile,
        )

        self.get_logger().info("Covariance monitor initialised")

    def covariance_callback(self, msg: CovarianceEstimate) -> None:
        """
        @brief Callback for covariance messages.
        @param msg: CovarianceEstimate containing pose and covariance data.
        """
        covariance = np.array(msg.covariance).reshape(3, 3)

        std_x = np.sqrt(covariance[0, 0])
        std_y = np.sqrt(covariance[1, 1])
        std_yaw = np.sqrt(covariance[2, 2])

        self.get_logger().info(
            f"Pose: ({msg.x:.2f}, {msg.y:.2f}, {np.rad2deg(msg.yaw):.1f}deg) | "
            f"Vel: ({msg.vx:.2f}, {msg.vy:.2f}, {np.rad2deg(msg.vyaw):.2f}deg/s) | "
            f"Uncertainty: std_x={std_x:.4f}m, "
            f"std_y={std_y:.4f}m, std_yaw={np.rad2deg(std_yaw):.2f}deg"
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
