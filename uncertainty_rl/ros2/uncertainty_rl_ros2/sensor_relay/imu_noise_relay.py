"""
@file imu_noise_relay.py
@brief ROS 2 node that stamps realistic covariance onto the CARLA IMU topic.

The CARLA ROS bridge publishes sensor_msgs/Imu with zero covariance on all
fields. robot_localization interprets zero covariance as infinite sensor
reliability, which pins the EKF state to the IMU measurement with no
uncertainty growth between GNSS fixes. This node relays the IMU message with
a diagonal angular_velocity_covariance matching the VectorNav VN-100 gyro
specification so the EKF weights vyaw correctly.

The CARLA bridge already converts angular_velocity from CARLA left-handed
(z-down, CW positive) to ROS right-handed (z-up, CCW positive) convention
before publishing. No further sign conversion is applied here.

On the real vehicle the IMU driver populates the sensor_msgs/Imu covariance
fields directly from hardware; robot_localization uses message covariance in
preference to process_noise_covariance when it is non-zero, so no change to
this file or ros2_config.yaml is needed at deployment.

@author Antonio Galdes
"""

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class ImuNoiseRelayNode(Node):
    """
    @class ImuNoiseRelayNode
    @brief Stamps realistic angular_velocity_covariance onto CARLA IMU messages.

    Subscribes to the raw CARLA IMU topic (zero covariance) and republishes
    with a diagonal angular_velocity_covariance derived from the VectorNav
    VN-100 gyro noise density specification. Orientation and
    linear_acceleration covariance sentinel values are also set so
    robot_localization does not attempt Mahalanobis gating on those fields.
    """

    def __init__(self, node_name: str = "imu_noise_relay") -> None:
        """
        @brief Constructor for ImuNoiseRelayNode.
        @param node_name: Name of the ROS 2 node.
        """
        super().__init__(node_name)

        self.declare_parameter("imu_input_topic", "/carla/ego_vehicle/imu")
        self.declare_parameter("imu_output_topic", "/carla/ego_vehicle/imu/stamped")
        # Gyro noise density from VectorNav VN-100 datasheet (Table 3):
        #   0.0035 deg/s/sqrt(Hz) = 6.11e-5 rad/s/sqrt(Hz)
        #   Per-sample variance at 20 Hz: (6.11e-5 * sqrt(20))^2 = 7.46e-8 (rad/s)^2
        # Launch file overrides this from ros2_config.yaml gnss_noise_relay.imu_gyro_variance.
        self.declare_parameter("imu_gyro_variance", 1.0e-7)
        # ZUPT threshold (rad/s): angular_velocity.z below this magnitude is
        # clamped to zero. CARLA's physics engine produces ~0.012 rad/s gyro
        # bias when the vehicle is stationary; the VN-100 bias is ~0.003 rad/s.
        # 0.01 rad/s (0.57 deg/s) suppresses CARLA simulation noise without
        # affecting real yaw rates (parking turns are typically > 0.1 rad/s).
        # At deployment the real VN-100 bias (~0.003 rad/s) is well below this
        # threshold, so ZUPT behaves identically on real hardware.
        self.declare_parameter("zupt_threshold_rad_s", 0.01)

        imu_input_topic: str = str(
            self.get_parameter("imu_input_topic").get_parameter_value().string_value
        )
        imu_output_topic: str = str(
            self.get_parameter("imu_output_topic").get_parameter_value().string_value
        )
        self._imu_gyro_variance: float = float(
            self.get_parameter("imu_gyro_variance").get_parameter_value().double_value
        )
        self._zupt_threshold: float = float(
            self.get_parameter("zupt_threshold_rad_s").get_parameter_value().double_value
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            Imu, imu_input_topic, self._imu_callback, qos
        )
        self._pub = self.create_publisher(Imu, imu_output_topic, qos)

        self.get_logger().info(
            f"ImuNoiseRelay: {imu_input_topic} -> {imu_output_topic} "
            f"(gyro_variance: {self._imu_gyro_variance:.2e} (rad/s)^2, "
            f"1-sigma: {math.sqrt(self._imu_gyro_variance)*1000:.3f} mrad/s, "
            f"zupt_threshold: {self._zupt_threshold:.4f} rad/s)"
        )

    def _imu_callback(self, msg: Imu) -> None:
        """
        @brief Relay IMU with realistic angular_velocity_covariance stamped.

        The CARLA bridge publishes zero covariance on all Imu fields.
        robot_localization interprets zero covariance as infinite sensor
        reliability, which pins the EKF state to the IMU measurement with no
        uncertainty growth. This callback stamps a diagonal
        angular_velocity_covariance matching the VN-100 gyro so the EKF
        correctly inflates covariance between GNSS fixes.

        The CARLA bridge already applies the CARLA left-handed to ROS
        right-handed sign conversion on angular_velocity.z before publishing,
        so angular_velocity is passed through unchanged.

        @param msg: Raw sensor_msgs/Imu from CARLA bridge.
        """
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.linear_acceleration = msg.linear_acceleration

        # ZUPT: clamp angular_velocity.z to zero when below threshold.
        # CARLA's physics engine produces ~0.012 rad/s gyro bias when stationary;
        # without this clamp the EKF integrates the bias as real rotation and
        # yaw drifts ~0.7 deg/s even when the vehicle is not turning.
        out.angular_velocity.x = msg.angular_velocity.x
        out.angular_velocity.y = msg.angular_velocity.y
        raw_vyaw = msg.angular_velocity.z
        out.angular_velocity.z = (
            raw_vyaw if abs(raw_vyaw) >= self._zupt_threshold else 0.0
        )

        # Set orientation_covariance[0] = -1: ROS convention for
        # "this sensor does not provide orientation". Zero means infinite
        # precision, which causes robot_localization to attempt a Mahalanobis
        # check with a singular covariance and NaN the entire filter -- even
        # when orientation is disabled in imu0_config.
        out.orientation_covariance = [-1.0] + [0.0] * 8

        # Stamp diagonal angular_velocity_covariance. Only the z/z element
        # (index 8, vyaw) is active in imu0_config; all diagonal elements are
        # set to the gyro variance for correctness.
        v = self._imu_gyro_variance
        out.angular_velocity_covariance = [
            v,   0.0, 0.0,
            0.0, v,   0.0,
            0.0, 0.0, v,
        ]

        # Set linear_acceleration_covariance[0] = -1: same sentinel as above.
        # Linear acceleration is disabled in imu0_config but zero covariance
        # still triggers the singular-matrix NaN path in robot_localization.
        out.linear_acceleration_covariance = [-1.0] + [0.0] * 8

        self._pub.publish(out)


def main(args: Optional[object] = None) -> None:
    """@brief Entry point for the imu_noise_relay node."""
    rclpy.init(args=args)
    node = ImuNoiseRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
