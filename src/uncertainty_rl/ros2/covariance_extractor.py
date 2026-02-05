## @file covariance_extractor.py
#  @brief ROS 2 node for extracting covariance from robot_localization.
#
#  This module implements a ROS 2 node that subscribes to odometry messages from
#  robot_localization and extracts the covariance matrix for use in RL training.
from typing import Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray


## @class CovarianceExtractorNode
#  @brief ROS 2 node for extracting localisation covariance.
#
#  Subscribes to odometry messages from robot_localization and publishes
#  the covariance matrix elements for consumption by the RL agent.
class CovarianceExtractorNode(Node):
    
    ## @brief Constructor for CovarianceExtractorNode.
    #  @param node_name: Name of the ROS node.
    def __init__(self, node_name: str = "covariance_extractor") -> None:
        super().__init__(node_name)
        
        # Declare parameters
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("covariance_topic", "/slam_uncertainty/covariance")
        self.declare_parameter("publish_rate", 10.0)  # Hz
        
        # Get parameters
        odom_topic = self.get_parameter("odom_topic").value
        covariance_topic = self.get_parameter("covariance_topic").value
        publish_rate = self.get_parameter("publish_rate").value
        
        # Set up QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscriber for odometry
        self.odom_subscriber = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            qos_profile
        )
        
        # Create publisher for covariance
        self.covariance_publisher = self.create_publisher(
            Float64MultiArray,
            covariance_topic,
            qos_profile
        )
        
        # Store latest covariance
        self.latest_covariance: Optional[np.ndarray] = None
        self.latest_pose: Optional[Tuple[float, float, float]] = None
        
        # Create timer for publishing
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.publish_covariance)
        
        self.get_logger().info(f"Covariance extractor node initialised")
        self.get_logger().info(f"  Subscribing to: {odom_topic}")
        self.get_logger().info(f"  Publishing to: {covariance_topic}")
        
    ## @brief Callback for odometry messages.
    #
    #  Extracts pose covariance from the odometry message and stores it.
    #  @param msg: Odometry message from robot_localization.
    def odom_callback(self, msg: Odometry) -> None:
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
        
        self.latest_pose = (x, y, yaw)
        
        # Extract covariance matrix (6x6) for pose
        # Format: [x, y, z, roll, pitch, yaw]
        # We extract: [x, y, yaw] which are at indices [0, 1, 5]
        covariance_full = np.array(msg.pose.covariance).reshape(6, 6)
        
        # Extract 3x3 submatrix for [x, y, yaw]
        indices = [0, 1, 5]
        covariance_3x3 = covariance_full[np.ix_(indices, indices)]
        
        self.latest_covariance = covariance_3x3
        
        # Log uncertainty statistics periodically
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        if self._log_counter % 100 == 0:
            std_x = np.sqrt(covariance_3x3[0, 0])
            std_y = np.sqrt(covariance_3x3[1, 1])
            std_yaw = np.sqrt(covariance_3x3[2, 2])
            self.get_logger().info(
                f"Uncertainty - X: {std_x:.4f}m, Y: {std_y:.4f}m, Yaw: {np.rad2deg(std_yaw):.2f}°"
            )
            
    ## @brief Publish the latest covariance matrix.
    #
    #  Publishes the covariance matrix as a flattened array.
    def publish_covariance(self) -> None:
        if self.latest_covariance is None:
            return
            
        # Create message
        msg = Float64MultiArray()
        
        # Flatten covariance matrix (3x3 -> 9 elements)
        msg.data = self.latest_covariance.flatten().tolist()
        
        # Add pose information as well (optional, first 3 elements)
        if self.latest_pose is not None:
            full_data = list(self.latest_pose) + msg.data
            msg.data = full_data
            
        # Publish
        self.covariance_publisher.publish(msg)
        
    ## @brief Get the current uncertainty state vector.
    #  @return Uncertainty state vector: [std_x, std_y, std_yaw, 
    #                                    cov_xx, cov_yy, cov_yawyaw,
    #                                    cov_xy, cov_xyaw, cov_yyaw]
    #          or None if no covariance data is available.
    def get_uncertainty_state(self) -> Optional[np.ndarray]:
        if self.latest_covariance is None:
            return None
            
        # Extract standard deviations
        std_x = np.sqrt(self.latest_covariance[0, 0])
        std_y = np.sqrt(self.latest_covariance[1, 1])
        std_yaw = np.sqrt(self.latest_covariance[2, 2])
        
        # Extract covariance elements
        cov_xx = self.latest_covariance[0, 0]
        cov_yy = self.latest_covariance[1, 1]
        cov_yawyaw = self.latest_covariance[2, 2]
        cov_xy = self.latest_covariance[0, 1]
        cov_xyaw = self.latest_covariance[0, 2]
        cov_yyaw = self.latest_covariance[1, 2]
        
        # Construct uncertainty state vector
        uncertainty_state = np.array([
            std_x, std_y, std_yaw,
            cov_xx, cov_yy, cov_yawyaw,
            cov_xy, cov_xyaw, cov_yyaw
        ])
        
        return uncertainty_state


## @class CovarianceMonitorNode
#  @brief ROS 2 node for monitoring and visualising covariance.
#
#  Provides additional monitoring capabilities for debugging and analysis.
class CovarianceMonitorNode(Node):
    
    ## @brief Constructor for CovarianceMonitorNode.
    #  @param node_name: Name of the ROS node.
    def __init__(self, node_name: str = "covariance_monitor") -> None:
        super().__init__(node_name)
        
        # Declare parameters
        self.declare_parameter("covariance_topic", "/slam_uncertainty/covariance")
        
        covariance_topic = self.get_parameter("covariance_topic").value
        
        # Subscribe to covariance
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.covariance_subscriber = self.create_subscription(
            Float64MultiArray,
            covariance_topic,
            self.covariance_callback,
            qos_profile
        )
        
        self.get_logger().info(f"Covariance monitor initialised")
        
    ## @brief Callback for covariance messages.
    #  @param msg: Float64MultiArray containing covariance data.
    def covariance_callback(self, msg: Float64MultiArray) -> None:
        data = np.array(msg.data)
        
        # Parse data (first 3: pose, remaining 9: covariance)
        if len(data) >= 12:
            x, y, yaw = data[0], data[1], data[2]
            covariance = data[3:].reshape(3, 3)
            
            std_x = np.sqrt(covariance[0, 0])
            std_y = np.sqrt(covariance[1, 1])
            std_yaw = np.sqrt(covariance[2, 2])
            
            self.get_logger().info(
                f"Pose: ({x:.2f}, {y:.2f}, {np.rad2deg(yaw):.1f}°) | "
                f"Uncertainty: σ_x={std_x:.4f}m, σ_y={std_y:.4f}m, σ_yaw={np.rad2deg(std_yaw):.2f}°"
            )


## @brief Main entry point for the ROS 2 node.
def main(args=None) -> None:
    rclpy.init(args=args)
    
    node = CovarianceExtractorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


## @brief Main entry point for the monitor node.
def main_monitor(args=None) -> None:
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
