"""
@file gnss_noise_relay.py
@brief ROS 2 node that injects per-episode GNSS noise, projects to local XY,
       and publishes Odometry + COG heading for the robot_localisation EKF.

Subscribes to CARLA NavSatFix, adds tier-appropriate Gaussian noise, converts
to metric Odometry via flat-earth projection (/odometry/gps), and derives a
Course Over Ground heading from successive noisy fixes (/gnss/heading).

@see documentation/detailed_notes/ros2_architecture.md for the COG heading
     variance formula and Markov tier model.
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix

# Ordered tier names - index position defines row/column in transition matrix.
_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]
# reverse lookup: tier name -> index in _TIER_ORDER / transition matrix.
_TIER_INDEX: Dict[str, int] = {name: i for i, name in enumerate(_TIER_ORDER)}

# Noise parameters for each tier (fallback if config file is absent).
# See documentation/detailed_notes/sensor_noise_models.md for full derivation.
_TIER_DEFAULTS: Dict[str, Dict[str, float]] = {
    # RTK fixed:
    # Conservative 0.020 m used to cover antenna phase-centre offset 
    # 0.020 / 111320 = 1.797e-7 deg.
    "rtk_fixed":  {"lat_stddev_deg": 0.0000002, "lon_stddev_deg": 0.0000002,
                   "alt_stddev_m": 0.05,  "metric_stddev_m": 0.020},
    # RTK float: 
    # 0.360 / 111320 = 3.233e-6 deg.
    "rtk_float":  {"lat_stddev_deg": 0.0000032, "lon_stddev_deg": 0.0000032,
                   "alt_stddev_m": 0.5,   "metric_stddev_m": 0.360},
    # Standalone PVT:
    # 1.802 / 111320 = 1.619e-5 deg.
    "standalone": {"lat_stddev_deg": 0.0000162, "lon_stddev_deg": 0.0000162,
                   "alt_stddev_m": 5.0,   "metric_stddev_m": 1.802},
    # Degraded: 
    # 5.0 / 111320 = 4.492e-5 deg.
    "degraded":   {"lat_stddev_deg": 0.0000449, "lon_stddev_deg": 0.0000449,
                   "alt_stddev_m": 10.0,  "metric_stddev_m": 5.0},
}

# Default per-step transition matrix (rows = from, cols = to).
_DEFAULT_TRANSITION_MATRIX: List[List[float]] = [
    # to:  fixed   float   standalone  degraded
    [0.9950, 0.0050, 0.0000, 0.0000],  # from: rtk_fixed
    [0.0030, 0.9920, 0.0050, 0.0000],  # from: rtk_float
    [0.0000, 0.0040, 0.9930, 0.0030],  # from: standalone
    [0.0000, 0.0000, 0.0050, 0.9950],  # from: degraded
]


class GnssNoiseRelayNode(Node):
    """
    @class GnssNoiseRelayNode
    @brief Adds per-episode noise to CARLA GNSS, publishes Odometry + COG heading.

    @see documentation/design/gnss_markov_transitions.md
    """

    _DEFAULT_CONFIG_PATH: str = "/workspace/outputs/episode_config.json"

    def __init__(self, node_name: str = "gnss_noise_relay") -> None:
        """
        @brief Constructor for GnssNoiseRelayNode.
        @param node_name: Name of the ROS 2 node.
        """
        super().__init__(node_name)

        self.declare_parameter("input_topic", "/carla/ego_vehicle/gnss")
        self.declare_parameter("output_topic", "/gnss/noisy")
        self.declare_parameter(
            "config_file",
            os.environ.get("EPISODE_CONFIG_FILE", self._DEFAULT_CONFIG_PATH),
        )
        self.declare_parameter("base_metric_stddev_m", 0.02)
        self.declare_parameter("enable_markov_transitions", True)
        self.declare_parameter("enable_gnss_noise", True)
        self.declare_parameter("datum_lat", 0.0)
        self.declare_parameter("datum_lon", 0.0)
        self.declare_parameter("odom_output_topic", "/odometry/gps")
        self.declare_parameter("heading_output_topic", "/gnss/heading")
        # Minimum GNSS-derived speed (m/s) to update the COG estimate.
        # Below this the last valid heading is re-published unchanged.
        self.declare_parameter("cog_min_speed_ms", 0.3)

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
            self.get_parameter("base_metric_stddev_m").get_parameter_value().double_value
        )
        self._markov_enabled: bool = bool(
            self.get_parameter("enable_markov_transitions").get_parameter_value().bool_value
        )
        self._gnss_noise_enabled: bool = bool(
            self.get_parameter("enable_gnss_noise").get_parameter_value().bool_value
        )
        self._datum_lat: float = float(
            self.get_parameter("datum_lat").get_parameter_value().double_value
        )
        self._datum_lon: float = float(
            self.get_parameter("datum_lon").get_parameter_value().double_value
        )
        odom_output_topic = str(
            self.get_parameter("odom_output_topic").get_parameter_value().string_value
        )
        heading_output_topic = str(
            self.get_parameter("heading_output_topic").get_parameter_value().string_value
        )
        self._cog_min_speed_ms: float = float(
            self.get_parameter("cog_min_speed_ms").get_parameter_value().double_value
        )

        # Precompute flat-earth scale factors for the datum latitude.
        self._metres_per_deg_lat: float = 111320.0
        self._metres_per_deg_lon: float = (
            111320.0 * math.cos(math.radians(self._datum_lat))
        )

        self._tier_params: Dict[str, Dict[str, float]] = dict(_TIER_DEFAULTS)
        self._transition_matrix: np.ndarray = np.array(
            _DEFAULT_TRANSITION_MATRIX, dtype=np.float64
        )
        self._active_tier_idx: int = 0

        self._extra_alt_stddev_m: float = 0.0
        self._metric_stddev_m: float = self._base_metric_stddev

        self._config_seq: int = -1
        self._config_mtime_ns: int = 0
        self._callback_count: int = 0

        self._auto_datum: bool = (
            self._datum_lat == 0.0 and self._datum_lon == 0.0
        )
        self._datum_latched: bool = not self._auto_datum

        self._rng = np.random.default_rng()

        # Pre-allocated 36-element zeroed lists reused at each callback.
        self._odom_cov_template: List[float] = [0.0] * 36
        self._heading_cov_template: List[float] = [0.0] * 36
        # Fixed non-zero slots for heading covariance (all position/vel dims
        # have infinite variance; only yaw slot carries signal).
        self._heading_cov_template[0] = 1.0e6
        self._heading_cov_template[7] = 1.0e6
        self._heading_cov_template[14] = 1.0e6
        self._heading_cov_template[21] = 1.0e6
        self._heading_cov_template[28] = 1.0e6
        # yaw slot (35) is updated per-callback from self._last_heading_var.
        # odom cov slot 35 (yaw) always 1e6.
        self._odom_cov_template[35] = 1.0e6

        # Cached RTK-fixed baseline variance (used as floor in odom and COG).
        self._rtk_fixed_var: float = 0.02 ** 2

        # Previous noisy fix in local XY (metres) and its timestamp (seconds).
        # Used to compute COG heading via displacement / dt.
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_stamp_sec: Optional[float] = None

        # COG heading state.
        self._cog_initialised: bool = False
        self._last_heading_rad: float = 0.0
        self._last_heading_var: float = (math.pi ** 2) / 3.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # robot_localization subscribes to odom0/pose0 with BEST_EFFORT.
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._sub = self.create_subscription(NavSatFix, input_topic, self._gnss_callback, qos)
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)
        self._odom_pub = self.create_publisher(Odometry, odom_output_topic, qos_be)
        self._heading_pub = self.create_publisher(
            PoseWithCovarianceStamped, heading_output_topic, qos_be
        )

        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"-> {odom_output_topic} (flat-earth, "
            f"COG heading -> {heading_output_topic}, "
            f"cog_min_speed={self._cog_min_speed_ms:.2f} m/s, "
            f"markov_transitions={self._markov_enabled})"
        )

    # ------------------------------------------------------------------
    # Config and tier helpers
    # ------------------------------------------------------------------

    def _apply_tier(self, tier_name: str) -> None:
        """
        @brief Update active noise state from tier name.
        @param tier_name: One of 'rtk_fixed', 'rtk_float', 'standalone', 'degraded'.
        """
        if tier_name not in self._tier_params:
            self.get_logger().warn(
                f"Unknown GNSS tier '{tier_name}', keeping current tier."
            )
            return

        params = self._tier_params[tier_name]
        extra_alt = max(0.0, params["alt_stddev_m"] - 0.05)
        self._extra_alt_stddev_m = extra_alt
        self._metric_stddev_m = params["metric_stddev_m"]

        if tier_name in _TIER_INDEX:
            self._active_tier_idx = _TIER_INDEX[tier_name]

    def _check_config_file(self) -> None:
        """
        @brief Read GNSS noise config from shared JSON file if updated.

        The training container writes this file at each episode reset with
        the initial tier for the episode.
        """
        try:
            mtime_ns = os.stat(self._config_path).st_mtime_ns
        except OSError:
            return
        if mtime_ns == self._config_mtime_ns:
            return
        self._config_mtime_ns = mtime_ns

        try:
            with open(self._config_path, "r") as f:
                data = json.load(f)

            seq = int(data.get("seq", -1))
            if seq <= self._config_seq:
                return

            self._config_seq = seq

            tier_name: Optional[str] = data.get("tier_name")
            if tier_name and tier_name in _TIER_ORDER:
                self._apply_tier(tier_name)
            else:
                self.get_logger().warn(
                    f"episode_config.json has unknown or missing tier_name"
                    f" '{tier_name}', keeping current tier."
                )

            datum_lat = data.get("datum_lat")
            datum_lon = data.get("datum_lon")
            if datum_lat is not None and datum_lon is not None:
                self._datum_lat = float(datum_lat)
                self._datum_lon = float(datum_lon)
                self._metres_per_deg_lon = (
                    111320.0 * math.cos(math.radians(self._datum_lat))
                )
                self._datum_latched = True
                # Invalidate previous fix so stale positions from the old datum
                # are not used for COG on the first callback after reset.
                self._prev_x = None
                self._prev_y = None
                self._prev_stamp_sec = None
                self._cog_initialised = False
                self.get_logger().info(
                    f"GNSS datum re-latched: lat={self._datum_lat:.7f} "
                    f"lon={self._datum_lon:.7f}"
                )

            # Seed COG heading from the known spawn yaw (layout geometry).
            spawn_yaw = data.get("spawn_yaw")
            if spawn_yaw is not None:
                self._last_heading_rad = float(spawn_yaw)
                # Use a moderately tight variance: the spawn heading is known
                # from the layout but the vehicle may not be perfectly aligned.
                self._last_heading_var = (math.radians(5.0)) ** 2
                self._cog_initialised = True
                self.get_logger().info(
                    f"COG heading seeded from spawn_yaw: "
                    f"{math.degrees(self._last_heading_rad):.1f} deg"
                )
            else:
                self._datum_latched = False
                self.get_logger().warn(
                    "No datum_lat/datum_lon in GNSS noise config - "
                    "auto-latching datum on next GNSS callback. "
                    "EKF may drift briefly at episode start."
                )

            self.get_logger().info(
                f"GNSS noise config updated (seq={seq}, tier={tier_name}): "
                f"metric_stddev={self._metric_stddev_m:.3f}m"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to read GNSS noise config: {exc}")

    def _write_active_tier(self, tier_name: str) -> None:
        """
        @brief Write the current active tier back to episode_config.json.
        @param tier_name: Active RTK fix-state tier name.
        """
        tmp_path = self._config_path + ".markov.tmp"
        try:
            data = {"seq": self._config_seq, "tier_name": tier_name}
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, self._config_path)
        except OSError as exc:
            self.get_logger().warn(f"Failed to write active tier to config: {exc}")

    def _step_markov(self) -> None:
        """@brief Advance the Markov chain by one step."""
        row = self._transition_matrix[self._active_tier_idx]
        next_idx = int(self._rng.choice(len(_TIER_ORDER), p=row))
        if next_idx != self._active_tier_idx:
            from_name = _TIER_ORDER[self._active_tier_idx]
            to_name = _TIER_ORDER[next_idx]
            self._apply_tier(to_name)
            self._write_active_tier(to_name)
            self.get_logger().info(
                f"GNSS tier transition: {from_name} -> {to_name} "
                f"(metric_stddev={self._metric_stddev_m:.3f}m)"
            )

    # ------------------------------------------------------------------
    # COG heading helpers
    # ------------------------------------------------------------------

    def _cog_heading_variance(self, speed_ms: float) -> float:
        """
        @brief Compute COG heading variance from GNSS position noise and speed.

        @param speed_ms: GNSS-derived speed (displacement / dt) in m/s.
        @return Heading variance in rad^2.
        """
        sigma = self._metric_stddev_m if self._gnss_noise_enabled else 0.02
        # Floor speed at cog_min_speed_ms to avoid division by near-zero.
        eff_speed_sq = max(speed_ms, self._cog_min_speed_ms) ** 2
        raw_var = 2.0 * (sigma ** 2) / eff_speed_sq
        min_var = 2.0 * self._rtk_fixed_var / eff_speed_sq
        return max(raw_var, min_var)

    # ------------------------------------------------------------------
    # Main callback
    # ------------------------------------------------------------------

    def _gnss_callback(self, msg: NavSatFix) -> None:
        """
        @brief Process a single GNSS fix: inject noise, project to XY, publish odom + COG.

        @param msg: Raw NavSatFix from the CARLA GNSS sensor.
        """
        self._callback_count += 1
        self._check_config_file()

        if self._markov_enabled:
            self._step_markov()

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # -- Auto-datum latch --------------------------------------------------
        if not self._datum_latched:
            self._datum_lat = msg.latitude
            self._datum_lon = msg.longitude
            self._metres_per_deg_lon = (
                111320.0 * math.cos(math.radians(self._datum_lat))
            )
            self._datum_latched = True
            self._prev_x = None
            self._prev_y = None
            self._prev_stamp_sec = None
            self._cog_initialised = False
            self.get_logger().info(
                f"GnssNoiseRelay: auto-latched datum "
                f"lat={self._datum_lat:.6f} lon={self._datum_lon:.6f}"
            )

        # -- Inject noise to lat/lon -------------------------------------------
        out = NavSatFix()
        out.header = msg.header
        out.status = msg.status

        if self._gnss_noise_enabled:
            sigma = self._metric_stddev_m
            # Single PRNG draw for all three noise terms then scale.
            n_xy, n_xy2, n_alt = self._rng.standard_normal(3)
            out.longitude = msg.longitude + (
                n_xy * sigma / self._metres_per_deg_lon if sigma > 0.0 else 0.0
            )
            out.latitude = msg.latitude + (
                n_xy2 * sigma / self._metres_per_deg_lat if sigma > 0.0 else 0.0
            )
            out.altitude = msg.altitude + (
                n_alt * self._extra_alt_stddev_m if self._extra_alt_stddev_m > 0.0 else 0.0
            )
        else:
            out.longitude = msg.longitude
            out.latitude = msg.latitude
            out.altitude = msg.altitude

        metric_var = self._metric_stddev_m ** 2
        alt_var = (self._extra_alt_stddev_m + 0.05) ** 2
        out.position_covariance = [
            metric_var, 0.0, 0.0,
            0.0, metric_var, 0.0,
            0.0, 0.0, alt_var,
        ]
        out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._pub.publish(out)

        # -- Flat-earth projection: noisy lat/lon -> local XY ------------------
        local_x = (out.longitude - self._datum_lon) * self._metres_per_deg_lon
        local_y = (out.latitude - self._datum_lat) * self._metres_per_deg_lat

        # -- Publish GNSS odometry (EKF odom0: x, y correction) ---------------
        odom_msg = Odometry()
        odom_msg.header = out.header
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "ego_vehicle"
        odom_msg.pose.pose.position.x = local_x
        odom_msg.pose.pose.position.y = local_y

        xy_var = metric_var if self._gnss_noise_enabled else self._rtk_fixed_var
        pose_cov = list(self._odom_cov_template)
        pose_cov[0] = xy_var
        pose_cov[7] = xy_var
        odom_msg.pose.covariance = pose_cov
        self._odom_pub.publish(odom_msg)

        # -- COG heading (EKF pose0: yaw correction) ---------------------------
        # Displacement and dt derived entirely from successive noisy GNSS
        # positions and NavSatFix message timestamps.
        if self._prev_x is not None and self._prev_stamp_sec is not None:
            dt = stamp_sec - self._prev_stamp_sec
            if dt > 0.0:
                dx = local_x - self._prev_x
                dy = local_y - self._prev_y
                speed_ms = math.hypot(dx, dy) / dt  # GNSS-derived speed, no GT

                if speed_ms >= self._cog_min_speed_ms:
                    # CARLA +Y is south, so increasing latitude maps to decreasing
                    # local_y in the vehicle frame. Negate dy so that heading
                    # follows ROS convention (0=east, pi/2=north, anticlockwise).
                    self._last_heading_rad = math.atan2(-dy, dx)
                    self._last_heading_var = self._cog_heading_variance(speed_ms)

                    if not self._cog_initialised:
                        self._cog_initialised = True
                        self.get_logger().info(
                            f"COG heading initialised: "
                            f"heading={math.degrees(self._last_heading_rad):.2f} deg "
                            f"speed={speed_ms:.2f} m/s"
                        )

                    if self._callback_count % 100 == 0:
                        self.get_logger().info(
                            f"[cog_dbg] dx={dx:.4f} dy={dy:.4f} "
                            f"speed={speed_ms:.2f} m/s "
                            f"heading={math.degrees(self._last_heading_rad):.2f} deg "
                            f"var={self._last_heading_var:.4e} rad^2"
                        )

        self._prev_x = local_x
        self._prev_y = local_y
        self._prev_stamp_sec = stamp_sec

        # Only publish heading once the first valid COG reading has been obtained.
        # Before that the EKF runs on IMU vyaw only.
        if not self._cog_initialised:
            return

        half_h = self._last_heading_rad / 2.0
        heading_msg = PoseWithCovarianceStamped()
        heading_msg.header = out.header
        heading_msg.header.frame_id = "odom"
        heading_msg.pose.pose.orientation.z = math.sin(half_h)
        heading_msg.pose.pose.orientation.w = math.cos(half_h)
        h_cov = list(self._heading_cov_template)
        h_cov[35] = self._last_heading_var
        heading_msg.pose.covariance = h_cov
        self._heading_pub.publish(heading_msg)


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
