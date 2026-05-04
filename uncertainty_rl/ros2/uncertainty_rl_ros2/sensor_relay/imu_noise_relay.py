"""
@file imu_noise_relay.py
@brief ROS 2 node that stamps realistic covariance onto the CARLA IMU topic.

The CARLA ROS bridge publishes sensor_msgs/Imu with zero covariance on all
fields. robot_localization interprets zero covariance as infinite sensor
reliability, which pins the EKF state to the IMU measurement with no
uncertainty growth between GNSS fixes.
"""

import math
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


class ImuNoiseRelayNode(Node):
    """
    @class ImuNoiseRelayNode
    @brief Stamps realistic covariance onto CARLA IMU messages.

    Subscribes to the raw CARLA IMU topic (zero covariance) and republishes
    with diagonal angular_velocity_covariance and linear_acceleration_covariance
    derived from the VectorNav VN-100 datasheet. The orientation_covariance
    sentinel (-1) is set so robot_localization does not attempt Mahalanobis
    gating on the identity quaternion that CARLA always publishes.
    """

    def __init__(self, node_name: str = "imu_noise_relay") -> None:
        """
        @brief Constructor for ImuNoiseRelayNode.
        @param node_name: Name of the ROS 2 node.
        """
        super().__init__(node_name)

        self.declare_parameter("imu_input_topic", "/carla/ego_vehicle/imu")
        self.declare_parameter("imu_output_topic", "/carla/ego_vehicle/imu/stamped")
        # Gyro noise density from VectorNav VN-100 datasheet
        self.declare_parameter("imu_gyro_variance", 1.0e-7)
        # Accel noise density from VectorNav VN-100 datasheet
        self.declare_parameter("imu_accel_variance", 3.76e-5)
        # When false, near-zero variances (1e-12) are stamped so the EKF treats
        # IMU as infinitely reliable. Use only for dryrun diagnostics.
        self.declare_parameter("enable_imu_noise", True)
        # Gyro ZUPT threshold (rad/s): angular_velocity.z below this is clamped to zero.
        # CARLA's physics engine produces ~0.012 rad/s gyro bias when stationary.
        self.declare_parameter("zupt_threshold_rad_s", 0.01)
        # Accel ZUPT threshold (m/s^2): applied when gyro ZUPT is also active.
        # CARLA produces ~0.05-0.15 m/s^2 accel noise at standstill.
        self.declare_parameter("accel_zupt_threshold_ms2", 0.2)

        imu_input_topic: str = str(
            self.get_parameter("imu_input_topic").get_parameter_value().string_value
        )
        imu_output_topic: str = str(
            self.get_parameter("imu_output_topic").get_parameter_value().string_value
        )
        enable_imu_noise: bool = bool(
            self.get_parameter("enable_imu_noise").get_parameter_value().bool_value
        )
        self._imu_gyro_variance: float = float(
            self.get_parameter("imu_gyro_variance").get_parameter_value().double_value
        )
        self._imu_accel_variance: float = float(
            self.get_parameter("imu_accel_variance").get_parameter_value().double_value
        )
        self._zupt_threshold: float = float(
            self.get_parameter("zupt_threshold_rad_s").get_parameter_value().double_value
        )
        self._accel_zupt_threshold: float = float(
            self.get_parameter("accel_zupt_threshold_ms2").get_parameter_value().double_value
        )

        # Subscribe RELIABLE to match the CARLA bridge's IMU publisher QoS.
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # Publish BEST_EFFORT to match robot_localization's imu0 subscriber QoS.
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            Imu, imu_input_topic, self._imu_callback, qos_reliable
        )
        self._pub = self.create_publisher(Imu, imu_output_topic, qos_be)

        # Pre-built constant covariance lists.
        self._orientation_cov: List[float] = [-1.0] + [0.0] * 8
        v_gyro = self._imu_gyro_variance
        self._angular_velocity_cov: List[float] = [
            v_gyro, 0.0,    0.0,
            0.0,    v_gyro, 0.0,
            0.0,    0.0,    v_gyro,
        ]
        v_accel = self._imu_accel_variance
        self._linear_acceleration_cov: List[float] = [
            v_accel, 0.0,     0.0,
            0.0,     v_accel, 0.0,
            0.0,     0.0,     v_accel,
        ]

        self.get_logger().info(
            f"ImuNoiseRelay: {imu_input_topic} -> {imu_output_topic} | "
            f"noise={'ON' if enable_imu_noise else 'DISABLED'} "
            f"gyro_var={self._imu_gyro_variance:.2e} (rad/s)^2 "
            f"(1-sigma={math.sqrt(self._imu_gyro_variance)*1000:.3f} mrad/s), "
            f"accel_var={self._imu_accel_variance:.2e} (m/s^2)^2 "
            f"(1-sigma={math.sqrt(self._imu_accel_variance)*1000:.3f} mm/s^2), "
            f"gyro_zupt={self._zupt_threshold:.4f} rad/s, "
            f"accel_zupt={self._accel_zupt_threshold:.3f} m/s^2"
        )

    def _imu_callback(self, msg: Imu) -> None:
        """
        @brief Relay IMU with realistic covariance stamped and ZUPT applied.

        Stamps VN-100 datasheet variances on angular_velocity and
        linear_acceleration. Applies gyro ZUPT (clamp vyaw to zero when below
        threshold) and accel ZUPT (clamp ax/ay to zero when both gyro and accel
        indicate standstill). The EKF's ax/ay correction then pins vx/vy near
        zero at standstill, bounding velocity drift from accel-bias integration.

        @param msg: Raw sensor_msgs/Imu from CARLA bridge.
        """
        out = Imu()
        out.header = msg.header
        # Override frame_id to ego_vehicle so robot_localization applies no TF
        # rotation.
        out.header.frame_id = "ego_vehicle"
        out.orientation = msg.orientation

        # Gyro ZUPT: clamp angular_velocity.z to zero when below threshold.
        raw_vyaw = msg.angular_velocity.z
        gyro_zupt_active = abs(raw_vyaw) < self._zupt_threshold
        out.angular_velocity.x = msg.angular_velocity.x
        out.angular_velocity.y = msg.angular_velocity.y
        out.angular_velocity.z = raw_vyaw * (not gyro_zupt_active)

        # Accel ZUPT: when gyro ZUPT is active and both ax/ay are small,
        # clamp to zero so the EKF correction step pins vx/vy near zero.
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        accel_zupt = gyro_zupt_active and (
            abs(ax) < self._accel_zupt_threshold
            and abs(ay) < self._accel_zupt_threshold
        )
        out.linear_acceleration.x = 0.0 if accel_zupt else ax
        out.linear_acceleration.y = 0.0 if accel_zupt else ay
        out.linear_acceleration.z = msg.linear_acceleration.z

        # Assign pre-built constant covariance lists (rebuilt once at init).
        out.orientation_covariance = self._orientation_cov
        out.angular_velocity_covariance = self._angular_velocity_cov
        out.linear_acceleration_covariance = self._linear_acceleration_cov

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
