"""
@file gnss_noise_relay.py
@brief ROS 2 node that adds dynamic noise to CARLA GNSS and converts to local
       Odometry via flat-earth projection.

This node subscribes to the raw NavSatFix from the CARLA GNSS sensor, adds
Gaussian noise to lat/lon/alt, converts the noisy lat/lon to local XY via a
flat-earth projection, and publishes the result as nav_msgs/Odometry on
/odometry/gps for direct consumption by the robot_localisation EKF. This
replaces navsat_transform_node, eliminating the datum service, startup delay,
and circular EKF dependency that caused persistent NaN output.

IMU covariance stamping is handled separately by ImuNoiseRelayNode
(imu_noise_relay.py). The two nodes are launched together by carla_bridge.launch.py.

The noise multiplier is signalled per-episode via a shared JSON file
(gnss_noise_config.json) written by the training container at each reset().
This follows the same file-based signalling pattern as initial_pose.json.

Mid-episode fix-state transitions are modelled via a discrete-time Markov
chain. At each GNSS callback the current tier can transition to an adjacent
tier with a small probability, simulating real RTK fix losses and
re-acquisitions that occur during a manoeuvre (e.g. driving under a canopy
or past a parked lorry). Transition probabilities are loaded from
gnss_noise_profiles.yaml alongside the tier noise parameters.

@note See documentation/design/gnss_markov_transitions.md for the full
      design rationale and sim-to-real transfer justification.

@author Antonio Galdes
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix

# Ordered tier names -- index position defines row/column in transition matrix.
_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]

# Noise parameters for each tier (fallback if config file is absent).
# Values match configs/deployment/sim/gnss_noise_profiles.yaml.
_TIER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "rtk_fixed":  {"lat_stddev_deg": 0.0000002, "lon_stddev_deg": 0.0000002,
                   "alt_stddev_m": 0.05,  "metric_stddev_m": 0.02},
    "rtk_float":  {"lat_stddev_deg": 0.0000027, "lon_stddev_deg": 0.0000027,
                   "alt_stddev_m": 0.5,   "metric_stddev_m": 0.30},
    "standalone": {"lat_stddev_deg": 0.000018,  "lon_stddev_deg": 0.000018,
                   "alt_stddev_m": 5.0,   "metric_stddev_m": 2.0},
    "degraded":   {"lat_stddev_deg": 0.000045,  "lon_stddev_deg": 0.000045,
                   "alt_stddev_m": 10.0,  "metric_stddev_m": 5.0},
}

# Default per-step transition matrix (rows = from, cols = to).
# Row order matches _TIER_ORDER. Values must sum to 1.0 per row.
# Probabilities are per GNSS callback (20 Hz), so p=0.005 -> ~1 transition
# per 10 seconds on average. Adjacent tiers only; no direct fixed->degraded.
# See documentation/design/gnss_markov_transitions.md for derivation.
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
    @brief Adds per-episode noise to CARLA GNSS and stamps position_covariance.

    The CARLA GNSS sensor already has base noise configured at spawn time
    (matching RTK-fixed ~0.02 m). This node adds EXTRA Gaussian noise on
    top of that base to model degraded RTK states (float, standalone, degraded).

    The noise magnitude is controlled by a JSON config file written by the
    training container at each episode reset. The file sets the INITIAL tier
    for the episode. A discrete-time Markov chain then evolves the active tier
    at each GNSS callback, modelling mid-episode fix-state transitions (e.g.
    RTK loss when driving under a structure).

    The Odometry pose covariance is stamped with the TOTAL metric variance
    (base + extra) so the downstream EKF receives correctly scaled measurement
    noise. The NavSatFix is also republished on /gnss/noisy for diagnostics.

    @see documentation/design/gnss_markov_transitions.md
    """

    # Default shared file path for GNSS noise config. Overridden at runtime
    # by the GNSS_NOISE_CONFIG_FILE environment variable for parallel workers.
    _DEFAULT_CONFIG_PATH: str = "/workspace/outputs/gnss_noise_config.json"

    # Check config file every N callbacks. Every callback (=1) minimises the
    # window during which stale GNSS noise settings take effect after an episode
    # reset. The file read is a single stat+open+json parse -- negligible vs
    # GNSS callback processing at 20 Hz.
    _CONFIG_CHECK_INTERVAL: int = 1

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
        # Enable/disable mid-episode Markov transitions. Disable for ablation
        # runs that require a fixed noise tier throughout each episode.
        self.declare_parameter("enable_markov_transitions", True)

        # Flat-earth projection datum (degrees). Noisy lat/lon are converted to
        # local XY metres relative to this origin: x = (lon - datum_lon) * scale,
        # y = (lat - datum_lat) * 111320. Replaces navsat_transform_node.
        self.declare_parameter("datum_lat", 0.0)
        self.declare_parameter("datum_lon", 0.0)
        # Topic for the local Odometry output (consumed by EKF as odom0).
        self.declare_parameter("odom_output_topic", "/odometry/gps")

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
        self._markov_enabled: bool = bool(
            self.get_parameter("enable_markov_transitions")
            .get_parameter_value()
            .bool_value
        )

        # Flat-earth datum and Odometry output.
        self._datum_lat: float = float(
            self.get_parameter("datum_lat").get_parameter_value().double_value
        )
        self._datum_lon: float = float(
            self.get_parameter("datum_lon").get_parameter_value().double_value
        )
        odom_output_topic = str(
            self.get_parameter("odom_output_topic")
            .get_parameter_value()
            .string_value
        )
        # Precompute flat-earth scale factor (constant for a given datum latitude).
        # At lat=0 the cosine factor is 1.0 so both axes scale identically.
        self._metres_per_deg_lat: float = 111320.0
        self._metres_per_deg_lon: float = (
            111320.0 * math.cos(math.radians(self._datum_lat))
        )

        # Tier noise lookup populated from _TIER_DEFAULTS (updated if
        # gnss_noise_profiles.yaml is loaded via the config file).
        self._tier_params: Dict[str, Dict[str, float]] = dict(_TIER_DEFAULTS)

        # Transition matrix: numpy array shape (4, 4), rows normalised to 1.
        self._transition_matrix: np.ndarray = np.array(
            _DEFAULT_TRANSITION_MATRIX, dtype=np.float64
        )

        # Active tier index into _TIER_ORDER. Initialised to rtk_fixed (0)
        # until the first config file is read.
        self._active_tier_idx: int = 0

        # Current noise state derived from active tier (recalculated on
        # tier change by _apply_tier()).
        self._extra_lat_stddev_deg: float = 0.0
        self._extra_lon_stddev_deg: float = 0.0
        self._extra_alt_stddev_m: float = 0.0
        self._metric_stddev_m: float = self._base_metric_stddev

        self._config_seq: int = -1  # Sequence number from config file
        self._callback_count: int = 0

        # Auto-datum: when datum_lat == datum_lon == 0.0 the datum has not been
        # set explicitly. On the first GNSS callback we latch the raw lat/lon
        # (before noise injection) as the datum so the EKF odom frame is centred
        # near CARLA world (0,0) in metres, not at quasi-UTM scale (~millions).
        # This is equivalent to navsat_transform's "zero datum" mode and makes
        # the odom->world rigid body transform near-identity, which is numerically
        # stable and insensitive to EKF yaw drift.
        self._auto_datum: bool = (
            self._datum_lat == 0.0 and self._datum_lon == 0.0
        )
        self._datum_latched: bool = not self._auto_datum

        # RNG for noise injection and Markov transitions.
        self._rng = np.random.default_rng()

        # QoS: match CARLA bridge default (RELIABLE, keep_last=10).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # IMU covariance stamping is handled by ImuNoiseRelayNode (imu_noise_relay.py).

        self._sub = self.create_subscription(
            NavSatFix, input_topic, self._gnss_callback, qos
        )
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)
        # Odometry publisher: flat-earth converted position for the EKF.
        self._odom_pub = self.create_publisher(Odometry, odom_output_topic, qos)

        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"-> {odom_output_topic} (flat-earth, "
            f"datum={self._datum_lat:.6f}N {self._datum_lon:.6f}E, "
            f"config: {self._config_path}, "
            f"markov_transitions: {self._markov_enabled})"
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
        # Extra noise = tier stddev minus the base RTK-fixed stddev already
        # baked into the CARLA sensor spawn. Clamp to zero for rtk_fixed.
        base_deg = self._base_metric_stddev / 111320.0  # approx metres -> degrees
        extra_lat = max(0.0, params["lat_stddev_deg"] - base_deg)
        extra_lon = max(0.0, params["lon_stddev_deg"] - base_deg)
        extra_alt = max(0.0, params["alt_stddev_m"] - 0.05)

        self._extra_lat_stddev_deg = extra_lat
        self._extra_lon_stddev_deg = extra_lon
        self._extra_alt_stddev_m = extra_alt
        self._metric_stddev_m = params["metric_stddev_m"]

        if tier_name in _TIER_ORDER:
            self._active_tier_idx = _TIER_ORDER.index(tier_name)

    def _check_config_file(self) -> None:
        """
        @brief Read GNSS noise config from shared JSON file if updated.

        The training container writes this file at each episode reset with
        the initial tier for the episode. The Markov chain then evolves
        from this starting tier during the episode.
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

            tier_name: Optional[str] = data.get("tier_name")
            if tier_name and tier_name in _TIER_ORDER:
                self._apply_tier(tier_name)
            else:
                self.get_logger().warn(
                    f"gnss_noise_config.json has unknown or missing tier_name"
                    f" '{tier_name}', keeping current tier."
                )

            # Re-latch the flat-earth datum to the vehicle spawn position.
            # The training container writes datum_lat/datum_lon (the CARLA
            # geolocation of the spawn point) alongside the tier at each
            # episode reset. Re-latching the datum ensures the GNSS local
            # frame is re-zeroed at the spawn, so local (0, 0) == vehicle
            # spawn. This eliminates the drift that occurs when the EKF is
            # reset via /set_pose to (0, 0) but GNSS continues reporting
            # positions relative to a stale datum from a previous episode.
            datum_lat = data.get("datum_lat")
            datum_lon = data.get("datum_lon")
            if datum_lat is not None and datum_lon is not None:
                self._datum_lat = float(datum_lat)
                self._datum_lon = float(datum_lon)
                self._metres_per_deg_lon = (
                    111320.0 * math.cos(math.radians(self._datum_lat))
                )
                self._datum_latched = True
                self.get_logger().info(
                    f"GNSS datum re-latched: lat={self._datum_lat:.7f} "
                    f"lon={self._datum_lon:.7f}"
                )
            else:
                # Fallback: reset auto-latch so the next GNSS callback
                # re-latches to the current vehicle position. This is used
                # when the training container does not provide geolocation
                # (e.g. real-vehicle deployment without CARLA API).
                self._datum_latched = False
                self.get_logger().warn(
                    "No datum_lat/datum_lon in GNSS noise config -- "
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
        @brief Write the current active tier back to gnss_noise_config.json.

        Called after every Markov transition so the training container's
        _read_live_tier() reflects mid-episode changes.  The seq field is
        intentionally preserved from the last episode reset -- writing a new
        seq would cause _check_config_file() to re-read the file and reset
        the Markov chain to the written tier (which is correct behaviour here,
        but the seq guard would then wrongly reject the next genuine reset).
        Instead we write seq=-1 so the guard ignores this file until the next
        proper episode reset overwrites it with a higher seq.

        @note Uses tmp-then-replace to avoid partial reads by the training container.
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
        """
        @brief Advance the Markov chain by one step.

        Samples the next tier from the transition distribution of the current
        tier. If the tier changes, calls _apply_tier() to update noise state,
        writes the new tier back to gnss_noise_config.json (so the inspector's
        live tier display reflects mid-episode transitions), and logs the
        transition for traceability.
        """
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
    # Callbacks
    # ------------------------------------------------------------------

    def _gnss_callback(self, msg: NavSatFix) -> None:
        """
        @brief Process incoming GNSS fix: step Markov chain, add noise, stamp covariance.
        @param msg: Raw NavSatFix from CARLA GNSS sensor.
        """
        self._callback_count += 1

        # Periodically check for a new episode config (resets the starting tier).
        if self._callback_count % self._CONFIG_CHECK_INTERVAL == 0:
            self._check_config_file()

        # Advance Markov chain every callback (20 Hz).
        if self._markov_enabled:
            self._step_markov()

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
        # Diagnostic / logging use only now that navsat_transform is removed.
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

        # -- Auto-datum: latch first raw GNSS reading as projection origin ------
        # When datum was not set explicitly (0,0), use the raw pre-noise lat/lon
        # from the first GNSS callback so the EKF odom frame is centred near
        # CARLA world (0,0) in metres. Noise is added AFTER datum latching so
        # the datum is always the clean simulation reference position.
        if not self._datum_latched:
            self._datum_lat = msg.latitude
            self._datum_lon = msg.longitude
            # Recompute scale factor now that datum latitude is known.
            self._metres_per_deg_lon = (
                111320.0 * math.cos(math.radians(self._datum_lat))
            )
            self._datum_latched = True
            self.get_logger().info(
                f"GnssNoiseRelay: auto-latched datum "
                f"lat={self._datum_lat:.6f} lon={self._datum_lon:.6f}"
            )

        # -- Flat-earth projection: lat/lon -> local XY (metres) ---------------
        # Replaces navsat_transform_node. Simple and accurate for parking-lot
        # scale distances (< 100 m from datum).
        #
        # The EKF odom frame is ROS right-handed convention (Y northward).
        # CovarianceExtractorNode negates y when writing ekf_state.json to
        # convert to CARLA left-handed convention (Y southward) for the
        # training container and calibration code.
        local_x = (out.longitude - self._datum_lon) * self._metres_per_deg_lon
        local_y = (out.latitude - self._datum_lat) * self._metres_per_deg_lat

        odom_msg = Odometry()
        odom_msg.header = out.header
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "ego_vehicle"
        odom_msg.pose.pose.position.x = local_x
        odom_msg.pose.pose.position.y = local_y

        # 6x6 pose covariance (row-major). Set xx (index 0) and yy (index 7)
        # to the GNSS metric variance so the EKF weights GNSS measurements
        # according to the current RTK tier quality.
        pose_cov = [0.0] * 36
        pose_cov[0] = metric_var   # xx
        pose_cov[7] = metric_var   # yy
        pose_cov[35] = 1.0e6       # yaw-yaw: large = "no yaw info from GNSS"
        odom_msg.pose.covariance = pose_cov

        self._odom_pub.publish(odom_msg)


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
