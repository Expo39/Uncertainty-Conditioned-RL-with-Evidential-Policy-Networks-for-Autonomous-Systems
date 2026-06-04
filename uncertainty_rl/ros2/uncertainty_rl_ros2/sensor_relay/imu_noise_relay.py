"""
@file imu_noise_relay.py
@brief ROS 2 node that stamps realistic covariance and injects Gaussian noise
       onto the CARLA IMU topic.

The CARLA ROS bridge publishes sensor_msgs/Imu with zero covariance on all
fields. robot_localization interprets zero covariance as infinite sensor
reliability, which pins the EKF state to the IMU measurement with no
uncertainty growth between GNSS fixes.

See documentation/detailed_notes/sensor_noise_models.md for full derivation.
"""

import json
import math
import os
import random
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
    with diagonal angular_velocity_covariance and linear_acceleration_covariance.
    The orientation_covariance
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
        # Defaults computed from datasheet figures - see sensor_noise_models.md.
        _gyro_noise_density_rad = 0.0035 * math.pi / 180.0
        _gyro_variance_default: float = _gyro_noise_density_rad**2 * 20.0
        self.declare_parameter("imu_gyro_variance", _gyro_variance_default)

        _accel_noise_density_ms2 = 0.14e-3 * 9.81
        _accel_variance_default: float = _accel_noise_density_ms2**2 * 20.0
        self.declare_parameter("imu_accel_variance", _accel_variance_default)

        self.declare_parameter("enable_imu_noise", True)
        self.declare_parameter("zupt_threshold_rad_s", 0.01)
        self.declare_parameter("accel_zupt_threshold_ms2", 0.2)

        _gyro_bias_limit_default: float = 5.0 * math.pi / 180.0 / 3600.0
        self.declare_parameter("imu_gyro_bias_limit_rad_s", _gyro_bias_limit_default)

        _accel_bias_limit_default: float = 0.04e-3 * 9.81
        self.declare_parameter("imu_accel_bias_limit_ms2", _accel_bias_limit_default)

        # Per-episode multiplicative scale-factor error on the gyroscope,
        # constant per power-cycle.
        self.declare_parameter("imu_gyro_scale_factor_limit", 0.005)
        # Per-episode accelerometer scale-factor error, per axis. Each axis
        # has independent residual scale after factory calibration.
        self.declare_parameter("imu_accel_scale_factor_limit", 0.005)
        # Seed for the noise RNG. This node runs in the ros2-bridge container,
        # a separate process from training, so it cannot inherit the training
        # seed; it must be seeded independently. Matches the training seed (42)
        # by default so a fixed-seed training run sees reproducible IMU noise.
        self.declare_parameter("seed", 42)

        imu_input_topic: str = str(
            self.get_parameter("imu_input_topic").get_parameter_value().string_value
        )
        imu_output_topic: str = str(
            self.get_parameter("imu_output_topic").get_parameter_value().string_value
        )
        self._enable_imu_noise: bool = bool(
            self.get_parameter("enable_imu_noise").get_parameter_value().bool_value
        )
        self._imu_gyro_variance: float = float(
            self.get_parameter("imu_gyro_variance").get_parameter_value().double_value
        )
        self._imu_accel_variance: float = float(
            self.get_parameter("imu_accel_variance").get_parameter_value().double_value
        )
        self._zupt_threshold: float = float(
            self.get_parameter("zupt_threshold_rad_s")
            .get_parameter_value()
            .double_value
        )
        self._accel_zupt_threshold: float = float(
            self.get_parameter("accel_zupt_threshold_ms2")
            .get_parameter_value()
            .double_value
        )
        gyro_bias_limit: float = float(
            self.get_parameter("imu_gyro_bias_limit_rad_s")
            .get_parameter_value()
            .double_value
        )
        accel_bias_limit: float = float(
            self.get_parameter("imu_accel_bias_limit_ms2")
            .get_parameter_value()
            .double_value
        )
        gyro_scale_limit: float = float(
            self.get_parameter("imu_gyro_scale_factor_limit")
            .get_parameter_value()
            .double_value
        )
        accel_scale_limit: float = float(
            self.get_parameter("imu_accel_scale_factor_limit")
            .get_parameter_value()
            .double_value
        )
        # Pre-compute per-sample noise stddevs (sqrt taken once at init).
        self._gyro_noise_stddev: float = math.sqrt(self._imu_gyro_variance)
        self._accel_noise_stddev: float = math.sqrt(self._imu_accel_variance)
        # Per-episode bias limits (stored for resampling on each episode reset).
        self._gyro_bias_limit: float = gyro_bias_limit
        self._accel_bias_limit: float = accel_bias_limit
        self._gyro_scale_limit: float = gyro_scale_limit
        self._accel_scale_limit: float = accel_scale_limit
        # Dedicated seeded RNG instance, isolated from the global random module.
        seed = int(self.get_parameter("seed").get_parameter_value().integer_value)
        self._rng: random.Random = random.Random(seed)
        # Per-episode in-run bias offsets. Resampled when episode_config.json seq increments.
        self._gyro_bias: float = self._rng.uniform(-gyro_bias_limit, gyro_bias_limit)
        self._accel_bias_x: float = self._rng.uniform(
            -accel_bias_limit, accel_bias_limit
        )
        self._accel_bias_y: float = self._rng.uniform(
            -accel_bias_limit, accel_bias_limit
        )
        # Per-episode multiplicative scale factors. 1.0 == perfect.
        self._gyro_scale_factor: float = 1.0 + self._rng.uniform(
            -gyro_scale_limit, gyro_scale_limit
        )
        self._accel_scale_factor_x: float = 1.0 + self._rng.uniform(
            -accel_scale_limit, accel_scale_limit
        )
        self._accel_scale_factor_y: float = 1.0 + self._rng.uniform(
            -accel_scale_limit, accel_scale_limit
        )
        # Episode config file watching - same file written by _CovarianceSubscriber at reset.
        self._episode_config_path: str = os.environ.get(
            "EPISODE_CONFIG_FILE", "/workspace/outputs/episode_config.json"
        )
        self._episode_config_seq: int = -1
        self._episode_config_mtime_ns: int = 0

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
            v_gyro,
            0.0,
            0.0,
            0.0,
            v_gyro,
            0.0,
            0.0,
            0.0,
            v_gyro,
        ]
        v_accel = self._imu_accel_variance
        self._linear_acceleration_cov: List[float] = [
            v_accel,
            0.0,
            0.0,
            0.0,
            v_accel,
            0.0,
            0.0,
            0.0,
            v_accel,
        ]

        self.get_logger().info(
            f"ImuNoiseRelay: {imu_input_topic} -> {imu_output_topic} | "
            f"noise={'ON' if self._enable_imu_noise else 'DISABLED'} | "
            f"gyro_var={self._imu_gyro_variance:.2e} (rad/s)^2 "
            f"(1-sigma={self._gyro_noise_stddev*1000:.3f} mrad/s) "
            f"gyro_bias={self._gyro_bias*1e6:.2f} urad/s | "
            f"accel_var={self._imu_accel_variance:.2e} (m/s^2)^2 "
            f"(1-sigma={self._accel_noise_stddev*1000:.3f} mm/s^2) "
            f"accel_bias_x={self._accel_bias_x*1e6:.2f} um/s^2 "
            f"accel_bias_y={self._accel_bias_y*1e6:.2f} um/s^2 | "
            f"gyro_zupt={self._zupt_threshold:.4f} rad/s "
            f"accel_zupt={self._accel_zupt_threshold:.3f} m/s^2"
        )

    def _check_episode_config(self) -> None:
        """
        @brief Resample per-episode IMU bias when episode_config.json seq increments.

        Watches the shared episode_config.json file written by _CovarianceSubscriber
        at each CARLAParkingEnv.reset(). When the seq field increments, new bias
        offsets are drawn from Uniform(+-limit) for gyro and accel axes.
        """
        try:
            mtime_ns = os.stat(self._episode_config_path).st_mtime_ns
        except OSError:
            return
        if mtime_ns == self._episode_config_mtime_ns:
            return
        self._episode_config_mtime_ns = mtime_ns
        try:
            with open(self._episode_config_path, "r") as f:
                data = json.load(f)
            seq = int(data.get("seq", -1))
            if seq <= self._episode_config_seq:
                return
            self._episode_config_seq = seq
            self._gyro_bias = self._rng.uniform(
                -self._gyro_bias_limit, self._gyro_bias_limit
            )
            self._accel_bias_x = self._rng.uniform(
                -self._accel_bias_limit, self._accel_bias_limit
            )
            self._accel_bias_y = self._rng.uniform(
                -self._accel_bias_limit, self._accel_bias_limit
            )
            self._gyro_scale_factor = 1.0 + self._rng.uniform(
                -self._gyro_scale_limit, self._gyro_scale_limit
            )
            self._accel_scale_factor_x = 1.0 + self._rng.uniform(
                -self._accel_scale_limit, self._accel_scale_limit
            )
            self._accel_scale_factor_y = 1.0 + self._rng.uniform(
                -self._accel_scale_limit, self._accel_scale_limit
            )
            self.get_logger().debug(
                f"IMU bias resampled (seq={seq}): "
                f"gyro={self._gyro_bias*1e6:.2f} urad/s "
                f"accel_x={self._accel_bias_x*1e6:.2f} um/s^2 "
                f"accel_y={self._accel_bias_y*1e6:.2f} um/s^2"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to read episode_config.json: {exc}")

    def _imu_callback(self, msg: Imu) -> None:
        """
        @brief Relay IMU with realistic covariance stamped and ZUPT applied.

        Stamps variances on angular_velocity and
        linear_acceleration. Applies gyro ZUPT (clamp vyaw to zero when below
        threshold) and accel ZUPT (clamp ax/ay to zero when both gyro and accel
        indicate standstill). The EKF's ax/ay correction then pins vx/vy near
        zero at standstill, bounding velocity drift from accel-bias integration.

        @param msg: Raw sensor_msgs/Imu from CARLA bridge.
        """
        self._check_episode_config()
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

        # Inject Gaussian noise + per-episode bias into measurement values so that the EKF
        # prediction step is realistically noisier between GNSS fixes. Noise is applied
        # only to axes that are not already zeroed by ZUPT, preserving standstill clamping.
        # Always stamp covariance so robot_localization receives valid sensor
        # noise characteristics regardless of whether value noise is enabled.
        out.orientation_covariance = self._orientation_cov
        out.angular_velocity_covariance = self._angular_velocity_cov
        out.linear_acceleration_covariance = self._linear_acceleration_cov

        if self._enable_imu_noise:
            if not gyro_zupt_active:
                # Apply multiplicative scale factor first (acts on signal), then
                # additive noise and bias on top. Real gyro errors compose this
                # way: scale * truth + bias + noise.
                out.angular_velocity.z = (
                    out.angular_velocity.z * self._gyro_scale_factor
                    + self._rng.gauss(0.0, self._gyro_noise_stddev)
                    + self._gyro_bias
                )
            if not accel_zupt:
                out.linear_acceleration.x = (
                    out.linear_acceleration.x * self._accel_scale_factor_x
                    + self._rng.gauss(0.0, self._accel_noise_stddev)
                    + self._accel_bias_x
                )
                out.linear_acceleration.y = (
                    out.linear_acceleration.y * self._accel_scale_factor_y
                    + self._rng.gauss(0.0, self._accel_noise_stddev)
                    + self._accel_bias_y
                )

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
