"""
@file imu_noise_relay.py
@brief ROS 2 node that stamps realistic covariance onto the CARLA IMU topic.

The CARLA ROS bridge publishes sensor_msgs/Imu with zero covariance on all
fields. robot_localization interprets zero covariance as infinite sensor
reliability, which pins the EKF state to the IMU measurement with no
uncertainty growth between GNSS fixes. This node relays the IMU message with:
  - diagonal angular_velocity_covariance from the VN-100 gyro specification
  - diagonal linear_acceleration_covariance from the VN-100 accel specification
  - angular ZUPT: angular_velocity.z clamped to zero when below threshold
  - accel ZUPT: linear_acceleration.x/y clamped to zero when stationary

The CARLA bridge converts angular_velocity from CARLA left-handed
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
    @brief Stamps realistic covariance onto CARLA IMU messages.

    Subscribes to the raw CARLA IMU topic (zero covariance) and republishes
    with diagonal angular_velocity_covariance and linear_acceleration_covariance
    derived from the VectorNav VN-100 datasheet. The orientation_covariance
    sentinel (-1) is set so robot_localization does not attempt Mahalanobis
    gating on the identity quaternion that CARLA always publishes.

    Gyro ZUPT clamps angular_velocity.z to zero when below threshold, suppressing
    CARLA simulation bias (~0.012 rad/s) at standstill. Accel ZUPT clamps
    linear_acceleration.x/y to zero when both gyro ZUPT and accel magnitude
    are below their thresholds, pinning vx/vy to zero while stationary.
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
        # Accel noise density from VectorNav VN-100 datasheet (Table 3):
        #   0.14 mg/sqrt(Hz) = 1.37e-3 m/s^2/sqrt(Hz)
        #   Per-sample variance at 20 Hz: (1.37e-3 * sqrt(20))^2 = 3.76e-5 (m/s^2)^2
        # Post-calibration optimistic value: 1.5e-5. Datasheet-conservative: 3.76e-5.
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
        # Use real datasheet variances even when noise injection is disabled.
        # 1e-12 collapses EKF covariance to near-zero, killing the Kalman gain and
        # preventing yaw corrections from GNSS heading.
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
        # A RELIABLE publisher paired with a BEST_EFFORT subscriber delivers no
        # messages in ROS 2.
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            Imu, imu_input_topic, self._imu_callback, qos_reliable
        )
        self._pub = self.create_publisher(Imu, imu_output_topic, qos_be)

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

        CARLA IMU publishes body-frame acceleration. On the FlatPlane (roll=0,
        pitch=0) gravity is purely along body z, so body x/y are gravity-free.
        two_d_mode: true in the EKF ignores z entirely; no gravity subtraction
        is needed for x/y accel fusion.

        @param msg: Raw sensor_msgs/Imu from CARLA bridge.
        """
        out = Imu()
        out.header = msg.header
        # Override frame_id to ego_vehicle so robot_localization applies no TF
        # rotation. The CARLA bridge publishes map->ego_vehicle/imu with a
        # spurious rotation that corrupts vyaw when used as the TF lookup path.
        # The IMU is rigidly mounted to the vehicle body with no rotation offset.
        out.header.frame_id = "ego_vehicle"
        out.orientation = msg.orientation

        # Gyro ZUPT: clamp angular_velocity.z to zero when below threshold.
        # CARLA's physics engine produces ~0.012 rad/s bias when stationary;
        # without this the EKF integrates the bias as real rotation (~0.7 deg/s).
        out.angular_velocity.x = msg.angular_velocity.x
        out.angular_velocity.y = msg.angular_velocity.y
        raw_vyaw = msg.angular_velocity.z
        gyro_zupt_active = abs(raw_vyaw) < self._zupt_threshold
        out.angular_velocity.z = 0.0 if gyro_zupt_active else raw_vyaw

        # Accel ZUPT: when gyro ZUPT is active and both ax/ay are small,
        # clamp to zero so the EKF correction step pins vx/vy near zero.
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        if gyro_zupt_active and (
            abs(ax) < self._accel_zupt_threshold
            and abs(ay) < self._accel_zupt_threshold
        ):
            ax = 0.0
            ay = 0.0
        out.linear_acceleration.x = ax
        out.linear_acceleration.y = ay
        out.linear_acceleration.z = msg.linear_acceleration.z

        # orientation_covariance[0] = -1: ROS sentinel for "not provided".
        # Zero would cause robot_localization to treat CARLA's identity
        # quaternion as an infinitely precise yaw=0 measurement.
        out.orientation_covariance = [-1.0] + [0.0] * 8

        # Diagonal angular_velocity_covariance: VN-100 gyro spec.
        # Only z/z (index 8, vyaw) is active in imu0_config.
        v_gyro = self._imu_gyro_variance
        out.angular_velocity_covariance = [
            v_gyro, 0.0,    0.0,
            0.0,    v_gyro, 0.0,
            0.0,    0.0,    v_gyro,
        ]

        # Diagonal linear_acceleration_covariance: VN-100 accel spec.
        # ax (index 0) and ay (index 4) are active in imu0_config.
        v_accel = self._imu_accel_variance
        out.linear_acceleration_covariance = [
            v_accel, 0.0,     0.0,
            0.0,     v_accel, 0.0,
            0.0,     0.0,     v_accel,
        ]

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
