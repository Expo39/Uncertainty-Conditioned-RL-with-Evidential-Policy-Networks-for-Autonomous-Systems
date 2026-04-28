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
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
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
        # When false, no Gaussian noise is added to lat/lon. The relay becomes a
        # pure flat-earth projector. Use for real deployment (real GNSS has its
        # own error) or for dryrun diagnostics with ideal sensors.
        self.declare_parameter("enable_gnss_noise", True)

        # Flat-earth projection datum (degrees). Noisy lat/lon are converted to
        # local XY metres relative to this origin: x = (lon - datum_lon) * scale,
        # y = (lat - datum_lat) * 111320. Replaces navsat_transform_node.
        self.declare_parameter("datum_lat", 0.0)
        self.declare_parameter("datum_lon", 0.0)
        # Topic for the local Odometry output (consumed by EKF as odom0).
        self.declare_parameter("odom_output_topic", "/odometry/gps")
        # Dual-GNSS heading parameters.
        self.declare_parameter("rear_antenna_input_topic", "/carla/ego_vehicle/gnss_rear")
        self.declare_parameter("heading_output_topic", "/gnss/heading")
        # Baseline between front and rear antennas (metres).
        self.declare_parameter("antenna_baseline_m", 1.5)
        # Fraction of per-tier position noise that is common-mode (shared between
        # antennas). ~85% is datasheet-defensible for nearby RTK antennas sharing
        # the same sky view, atmospheric corrections, and satellite clock errors.
        self.declare_parameter("common_mode_fraction", 0.85)

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
        self._gnss_noise_enabled: bool = bool(
            self.get_parameter("enable_gnss_noise").get_parameter_value().bool_value
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
        rear_antenna_topic = str(
            self.get_parameter("rear_antenna_input_topic")
            .get_parameter_value()
            .string_value
        )
        heading_output_topic = str(
            self.get_parameter("heading_output_topic")
            .get_parameter_value()
            .string_value
        )
        self._antenna_baseline_m: float = float(
            self.get_parameter("antenna_baseline_m").get_parameter_value().double_value
        )
        self._common_mode_fraction: float = float(
            self.get_parameter("common_mode_fraction").get_parameter_value().double_value
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

        # Common-mode noise cache: one shared noise draw per 20 Hz cycle so both
        # front and rear antennas see the same atmospheric/satellite-clock error.
        # cycle_key = round(stamp_sec * 20); redrawn when key changes.
        self._cm_noise_x: float = 0.0
        self._cm_noise_y: float = 0.0
        self._cm_noise_cycle_key: int = -1

        # Last front antenna local XY (for per-step trace diagnostics).
        self._front_local_x: Optional[float] = None
        self._front_local_y: Optional[float] = None

        # Last published heading (rad, ROS convention) for per-step trace.
        self._last_heading_raw: Optional[float] = None

        # QoS: match CARLA bridge default (RELIABLE, keep_last=10).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # robot_localization subscribes to pose0 with BEST_EFFORT reliability.
        # A RELIABLE publisher and BEST_EFFORT subscriber are incompatible in
        # ROS 2 -- no messages are delivered. Use BEST_EFFORT for the heading
        # publisher so the QoS policies match and the EKF receives corrections.
        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # IMU covariance stamping is handled by ImuNoiseRelayNode (imu_noise_relay.py).

        # Synchronised front + rear subscriptions so both readings come from the
        # same CARLA tick. ApproximateTimeSynchronizer pairs messages whose
        # timestamps are within `slop` seconds. At 20 Hz one tick = 50 ms;
        # 30 ms slop accepts same-tick pairs while rejecting genuinely stale reads.
        self._front_sub = Subscriber(self, NavSatFix, input_topic, qos_profile=qos)
        self._rear_sub = Subscriber(self, NavSatFix, rear_antenna_topic, qos_profile=qos)
        self._sync = ApproximateTimeSynchronizer(
            [self._front_sub, self._rear_sub], queue_size=10, slop=0.03
        )
        self._sync.registerCallback(self._gnss_sync_callback)
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)
        # Odometry and heading publishers use BEST_EFFORT to match
        # robot_localization's odom0/pose0 subscriber QoS. A RELIABLE publisher
        # paired with a BEST_EFFORT subscriber delivers no messages in ROS 2.
        self._odom_pub = self.create_publisher(Odometry, odom_output_topic, qos_be)
        self._heading_pub = self.create_publisher(
            PoseWithCovarianceStamped, heading_output_topic, qos_be
        )

        # Effective heading stddev at rtk_fixed: sqrt(2*(1-0.85)) * 0.02 / 1.5
        # = sqrt(0.30) * 0.02 / 1.5 = 0.548 * 0.02 / 1.5 = 0.0073 rad = 0.42 deg
        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"-> {odom_output_topic} (flat-earth, "
            f"datum={self._datum_lat:.6f}N {self._datum_lon:.6f}E, "
            f"config: {self._config_path}, "
            f"markov_transitions: {self._markov_enabled})"
        )
        self.get_logger().info(
            f"DualGNSSRelay: heading from front+rear antennas "
            f"({rear_antenna_topic}) -> {heading_output_topic}, "
            f"baseline={self._antenna_baseline_m:.2f}m, "
            f"common_mode={self._common_mode_fraction:.2f}, "
            f"effective heading stddev ~0.42 deg at rtk_fixed"
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
                # Reset front position cache after datum re-latch so stale
                # readings projected into the old datum are not used.
                self._front_local_x = None
                self._front_local_y = None
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

    def _get_common_mode_noise(self, stamp_sec: float) -> tuple:
        """
        @brief Return a shared common-mode noise draw for the current 20 Hz cycle.

        Both front and rear antennas call this per callback. The first caller in
        a new cycle draws fresh N(0, sigma_common) values; the second reuses them.
        This ensures both antennas see the same atmospheric/satellite-clock error
        within the same tick regardless of callback arrival order.

        @param stamp_sec: Message timestamp in seconds.
        @return Tuple (cm_x, cm_y) noise in local XY metres.
        """
        cycle_key = round(stamp_sec * 20)
        if cycle_key != self._cm_noise_cycle_key:
            sigma_common = self._metric_stddev_m * math.sqrt(self._common_mode_fraction)
            self._cm_noise_x = float(self._rng.normal(0.0, sigma_common))
            self._cm_noise_y = float(self._rng.normal(0.0, sigma_common))
            self._cm_noise_cycle_key = cycle_key
        return self._cm_noise_x, self._cm_noise_y

    def _gnss_sync_callback(self, front_msg: NavSatFix, rear_msg: NavSatFix) -> None:
        """
        @brief Process synchronised front + rear GNSS fixes from the same CARLA tick.

        ApproximateTimeSynchronizer guarantees both messages share the same
        simulation timestamp (within 30 ms slop), so the baseline vector
        dx/dy is purely the physical antenna separation with no vehicle-motion
        contamination regardless of speed.

        Publishes:
          - /gnss/noisy         (NavSatFix, front antenna with noise, diagnostics)
          - /odometry/gps       (Odometry, flat-earth XY for EKF odom0)
          - /gnss/heading       (PoseWithCovarianceStamped, baseline heading for EKF pose0)

        @param front_msg: Raw NavSatFix from the front CARLA GNSS sensor.
        @param rear_msg:  Raw NavSatFix from the rear CARLA GNSS sensor.
        """
        self._callback_count += 1

        # Periodically check for a new episode config (resets the starting tier).
        if self._callback_count % self._CONFIG_CHECK_INTERVAL == 0:
            self._check_config_file()

        # Advance Markov chain every callback (20 Hz).
        if self._markov_enabled:
            self._step_markov()

        # Split tier noise into common-mode and independent components.
        sigma_indep = self._metric_stddev_m * math.sqrt(
            max(0.0, 1.0 - self._common_mode_fraction)
        )

        stamp_sec = (
            front_msg.header.stamp.sec + front_msg.header.stamp.nanosec * 1e-9
        )

        # -- Auto-datum: latch first raw GNSS reading as projection origin ------
        if not self._datum_latched:
            self._datum_lat = front_msg.latitude
            self._datum_lon = front_msg.longitude
            self._metres_per_deg_lon = (
                111320.0 * math.cos(math.radians(self._datum_lat))
            )
            self._datum_latched = True
            self.get_logger().info(
                f"GnssNoiseRelay: auto-latched datum "
                f"lat={self._datum_lat:.6f} lon={self._datum_lon:.6f}"
            )

        # -- Apply noise to front antenna ---------------------------------------
        out = NavSatFix()
        out.header = front_msg.header
        out.status = front_msg.status

        if self._gnss_noise_enabled:
            cm_x, cm_y = self._get_common_mode_noise(stamp_sec)
            indep_lon_f = float(self._rng.normal(0.0, sigma_indep)) if sigma_indep > 0.0 else 0.0
            indep_lat_f = float(self._rng.normal(0.0, sigma_indep)) if sigma_indep > 0.0 else 0.0
            out.longitude = front_msg.longitude + (cm_x + indep_lon_f) / self._metres_per_deg_lon
            out.latitude = front_msg.latitude + (cm_y + indep_lat_f) / self._metres_per_deg_lat
            out.altitude = front_msg.altitude + float(
                self._rng.normal(0.0, self._extra_alt_stddev_m)
                if self._extra_alt_stddev_m > 0.0
                else 0.0
            )
        else:
            out.longitude = front_msg.longitude
            out.latitude = front_msg.latitude
            out.altitude = front_msg.altitude

        metric_var = self._metric_stddev_m ** 2
        alt_var = (self._extra_alt_stddev_m + 0.05) ** 2
        out.position_covariance = [
            metric_var, 0.0, 0.0,
            0.0, metric_var, 0.0,
            0.0, 0.0, alt_var,
        ]
        out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self._pub.publish(out)

        # -- Apply noise to rear antenna ----------------------------------------
        if self._gnss_noise_enabled:
            cm_x, cm_y = self._get_common_mode_noise(stamp_sec)
            indep_lon_r = float(self._rng.normal(0.0, sigma_indep)) if sigma_indep > 0.0 else 0.0
            indep_lat_r = float(self._rng.normal(0.0, sigma_indep)) if sigma_indep > 0.0 else 0.0
            rear_lon = rear_msg.longitude + (cm_x + indep_lon_r) / self._metres_per_deg_lon
            rear_lat = rear_msg.latitude + (cm_y + indep_lat_r) / self._metres_per_deg_lat
        else:
            rear_lon = rear_msg.longitude
            rear_lat = rear_msg.latitude

        # -- Flat-earth projection: both antennas -> local XY -------------------
        front_x = (out.longitude - self._datum_lon) * self._metres_per_deg_lon
        front_y = (out.latitude - self._datum_lat) * self._metres_per_deg_lat
        rear_x = (rear_lon - self._datum_lon) * self._metres_per_deg_lon
        rear_y = (rear_lat - self._datum_lat) * self._metres_per_deg_lat

        self._front_local_x = front_x
        self._front_local_y = front_y

        # -- Publish GNSS odometry (EKF odom0) ----------------------------------
        odom_msg = Odometry()
        odom_msg.header = out.header
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "ego_vehicle"
        odom_msg.pose.pose.position.x = front_x
        odom_msg.pose.pose.position.y = front_y

        # Floor at RTK-fixed variance -- never 1e-12 (see heading_variance comment above).
        _rtk_fixed_var = 0.02 ** 2
        xy_var = metric_var if self._gnss_noise_enabled else _rtk_fixed_var
        pose_cov = [0.0] * 36
        pose_cov[0] = xy_var   # xx
        pose_cov[7] = xy_var   # yy
        pose_cov[35] = 1.0e6
        odom_msg.pose.covariance = pose_cov
        self._odom_pub.publish(odom_msg)

        # -- Dual-GNSS baseline heading (EKF pose0) -----------------------------
        # Both readings are from the same CARLA tick so the baseline vector is
        # purely the physical antenna separation: no vehicle-motion contamination.
        dx = front_x - rear_x   # east positive
        dy = front_y - rear_y   # raw flat-earth dy
        # CARLA +Y is south, which maps to INCREASING latitude in CARLA's GNSS.
        # This inverts the flat-earth Y axis: dy is negated relative to true
        # geographic north. Negate dy so heading follows ROS convention:
        # 0=east, +90=north (CARLA yaw=-90), -90=south (CARLA yaw=+90).
        # At yaw=0 CARLA (east): dx=+1.5, dy~0 -> heading=0. OK.
        # At yaw=-90 CARLA (north): dy<0 (inverted), -dy>0 -> heading=+90. OK.
        heading = math.atan2(-dy, dx)
        if self._callback_count % 100 == 0:
            self.get_logger().info(
                f"[heading_dbg] dx={dx:.4f} dy={dy:.4f} "
                f"heading={math.degrees(heading):.2f}deg"
            )
        self._last_heading_raw = heading

        # Minimum heading variance from RTK-fixed position noise (sigma~0.02m, baseline 1.5m).
        # Never use 1e-12: a near-zero measurement variance collapses the EKF yaw covariance
        # to near-zero, making the Kalman gain also near-zero so yaw corrections are ignored.
        _RTK_FIXED_SIGMA_M = 0.02
        _min_heading_var = 2.0 * _RTK_FIXED_SIGMA_M ** 2 / (self._antenna_baseline_m ** 2)
        if not self._gnss_noise_enabled or sigma_indep == 0.0:
            heading_variance = _min_heading_var
        else:
            heading_variance = max(
                _min_heading_var,
                2.0 * sigma_indep ** 2 / (self._antenna_baseline_m ** 2),
            )

        half_h = heading / 2.0
        heading_msg = PoseWithCovarianceStamped()
        heading_msg.header = out.header
        heading_msg.header.frame_id = "odom"
        heading_msg.pose.pose.orientation.x = 0.0
        heading_msg.pose.pose.orientation.y = 0.0
        heading_msg.pose.pose.orientation.z = math.sin(half_h)
        heading_msg.pose.pose.orientation.w = math.cos(half_h)
        h_cov = [0.0] * 36
        h_cov[0] = 1.0e6
        h_cov[7] = 1.0e6
        h_cov[14] = 1.0e6
        h_cov[21] = 1.0e6
        h_cov[28] = 1.0e6
        h_cov[35] = heading_variance
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
