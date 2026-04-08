"""
@file gnss_noise_relay.py
@brief ROS 2 node that adds dynamic noise to CARLA GNSS and stamps covariance.

This node subscribes to the raw NavSatFix from the CARLA GNSS sensor, adds
Gaussian noise to lat/lon/alt, stamps the position_covariance diagonal with
metric variance matching the current episode's RTK fix-state tier, and
republishes on /gnss/noisy.

The noise multiplier is signalled per-episode via a shared JSON file
(gnss_noise_config.json) written by the training container at each reset().
This follows the same file-based signalling pattern as initial_pose.json.

@author Antonio Galdes
"""

import json
import math
import os
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix, NavSatStatus


class GnssNoiseRelayNode(Node):
    """
    @class GnssNoiseRelayNode
    @brief Adds per-episode noise to CARLA GNSS and stamps position_covariance.

    The CARLA GNSS sensor already has base noise configured at spawn time
    (matching RTK-fixed ~0.02 m). This node adds EXTRA Gaussian noise on
    top of that base to model degraded RTK states (float, standalone, degraded).

    The noise magnitude is controlled by a JSON config file written by the
    training container at each episode reset. The file contains:
      - lat_stddev_deg: extra latitude noise stddev (degrees)
      - lon_stddev_deg: extra longitude noise stddev (degrees)
      - alt_stddev_m: extra altitude noise stddev (metres)
      - metric_stddev_m: metric position stddev for covariance stamping (metres)

    The position_covariance field is stamped with the TOTAL metric variance
    (base + extra) so navsat_transform_node and the downstream EKF receive
    correctly scaled measurement noise.
    """

    # Default shared file path for GNSS noise config. Overridden at runtime
    # by the GNSS_NOISE_CONFIG_FILE environment variable for parallel workers.
    _DEFAULT_CONFIG_PATH: str = "/workspace/outputs/gnss_noise_config.json"

    # Check config file every N callbacks (~20 at 20 Hz = every 1 s).
    _CONFIG_CHECK_INTERVAL: int = 20

    def __init__(self, node_name: str = "gnss_noise_relay") -> None:
        """
        @brief Constructor for GnssNoiseRelayNode.
        @param node_name: Name of the ROS 2 node.
        """
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("input_topic", "/carla/ego_vehicle/gnss")
        self.declare_parameter("output_topic", "/gnss/noisy")
        self.declare_parameter(
            "config_file",
            os.environ.get("GNSS_NOISE_CONFIG_FILE", self._DEFAULT_CONFIG_PATH),
        )
        # Base metric stddev (metres) corresponding to RTK-fixed conditions.
        # Used to compute total covariance when no extra noise is applied.
        self.declare_parameter("base_metric_stddev_m", 0.02)

        input_topic = str(
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        output_topic = str(
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self._config_path = str(
            self.get_parameter("config_file").get_parameter_value().string_value
        )
        self._base_metric_stddev: float = float(
            self.get_parameter("base_metric_stddev_m")
            .get_parameter_value()
            .double_value
        )

        # Current noise state (updated from config file).
        self._extra_lat_stddev_deg: float = 0.0
        self._extra_lon_stddev_deg: float = 0.0
        self._extra_alt_stddev_m: float = 0.0
        self._metric_stddev_m: float = self._base_metric_stddev
        self._config_seq: int = -1  # Sequence number from config file
        self._callback_count: int = 0

        # RNG for reproducible noise injection.
        self._rng = np.random.default_rng()

        # QoS: match CARLA bridge default (RELIABLE, keep_last=10).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(
            NavSatFix, input_topic, self._gnss_callback, qos
        )
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)

        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"(config: {self._config_path})"
        )

    def _check_config_file(self) -> None:
        """
        @brief Read GNSS noise config from shared JSON file if updated.

        The training container writes this file at each episode reset with
        the noise parameters for the sampled RTK fix-state tier.
        """
        try:
            if not os.path.exists(self._config_path):
                return

            with open(self._config_path, "r") as f:
                data = json.load(f)

            seq = int(data.get("seq", -1))
            if seq <= self._config_seq:
                return  # Already processed this config

            self._config_seq = seq
            self._extra_lat_stddev_deg = float(data.get("lat_stddev_deg", 0.0))
            self._extra_lon_stddev_deg = float(data.get("lon_stddev_deg", 0.0))
            self._extra_alt_stddev_m = float(data.get("alt_stddev_m", 0.0))
            self._metric_stddev_m = float(
                data.get("metric_stddev_m", self._base_metric_stddev)
            )

            self.get_logger().info(
                f"GNSS noise config updated (seq={seq}): "
                f"lat_stddev={self._extra_lat_stddev_deg:.10f}deg, "
                f"metric_stddev={self._metric_stddev_m:.3f}m"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to read GNSS noise config: {exc}")

    def _gnss_callback(self, msg: NavSatFix) -> None:
        """
        @brief Process incoming GNSS fix: add noise and stamp covariance.
        @param msg: Raw NavSatFix from CARLA GNSS sensor.
        """
        self._callback_count += 1

        # Periodically check for config file updates.
        if self._callback_count % self._CONFIG_CHECK_INTERVAL == 0:
            self._check_config_file()

        # Build output message (copy header and status).
        out = NavSatFix()
        out.header = msg.header
        out.status = msg.status

        # Add extra Gaussian noise to lat/lon/alt.
        out.latitude = msg.latitude + float(
            self._rng.normal(0.0, self._extra_lat_stddev_deg)
            if self._extra_lat_stddev_deg > 0.0
            else 0.0
        )
        out.longitude = msg.longitude + float(
            self._rng.normal(0.0, self._extra_lon_stddev_deg)
            if self._extra_lon_stddev_deg > 0.0
            else 0.0
        )
        out.altitude = msg.altitude + float(
            self._rng.normal(0.0, self._extra_alt_stddev_m)
            if self._extra_alt_stddev_m > 0.0
            else 0.0
        )

        # Stamp position_covariance with metric variance (metres^2).
        # navsat_transform_node reads this to weight the GNSS measurement
        # in the EKF. Diagonal: [lat_var, 0, 0, 0, lon_var, 0, 0, 0, alt_var].
        # NavSatFix covariance is in ENU frame (metres^2).
        metric_var = self._metric_stddev_m ** 2
        alt_var = (self._extra_alt_stddev_m + 0.05) ** 2  # Base alt noise = 0.05 m
        out.position_covariance = [
            metric_var, 0.0, 0.0,
            0.0, metric_var, 0.0,
            0.0, 0.0, alt_var,
        ]
        # COVARIANCE_TYPE_DIAGONAL_KNOWN = 2
        out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self._pub.publish(out)


def main() -> None:
    """@brief Entry point for the gnss_noise_relay node."""
    rclpy.init()
    node = GnssNoiseRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
