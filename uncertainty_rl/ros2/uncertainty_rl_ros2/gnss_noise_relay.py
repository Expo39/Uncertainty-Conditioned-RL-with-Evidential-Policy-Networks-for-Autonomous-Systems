"""
@file gnss_noise_relay.py
@brief ROS 2 node that adds dynamic noise to CARLA GNSS and stamps covariance.

This node subscribes to the raw NavSatFix from the CARLA GNSS sensor, adds
Gaussian noise to lat/lon/alt, stamps the position_covariance diagonal with
metric variance matching the current episode's RTK fix-state tier, and
republishes on /gnss/noisy.

It also relays the CARLA IMU topic with realistic angular_velocity_covariance
stamped. CARLA's IMU bridge publishes zero covariance on all fields, which
causes robot_localization to treat IMU measurements as infinitely reliable,
suppressing EKF covariance growth entirely. This relay stamps a fixed gyro
noise covariance (mid-grade MEMS IMU, e.g. VectorNav VN-100) so the EKF
correctly inflates uncertainty between GNSS fixes.

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
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix

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

    The position_covariance field is stamped with the TOTAL metric variance
    (base + extra) so navsat_transform_node and the downstream EKF receive
    correctly scaled measurement noise.

    @see documentation/design/gnss_markov_transitions.md
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
        # Enable/disable mid-episode Markov transitions. Disable for ablation
        # runs that require a fixed noise tier throughout each episode.
        self.declare_parameter("enable_markov_transitions", True)

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

        # RNG for noise injection and Markov transitions.
        self._rng = np.random.default_rng()

        # QoS: match CARLA bridge default (RELIABLE, keep_last=10).
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # IMU covariance relay: stamp realistic angular_velocity_covariance
        # so the EKF treats vyaw with finite (not infinite) reliability.
        # CARLA bridge publishes zero covariance on all Imu fields; robot_localization
        # interprets zero as infinite reliability, suppressing covariance growth.
        # Values match a mid-grade MEMS gyro (e.g. VectorNav VN-100):
        #   vyaw stddev = 0.016 rad/s -> variance = 2.5e-4 (rad/s)^2
        # On the real vehicle the IMU driver populates message covariance from
        # hardware; robot_localization uses message values in preference to this
        # relay's stamped values when they are non-zero.
        self.declare_parameter("imu_input_topic", "/carla/ego_vehicle/imu")
        self.declare_parameter("imu_output_topic", "/carla/ego_vehicle/imu/stamped")
        # Gyro noise density for a mid-grade MEMS IMU (rad/s/sqrt(Hz)).
        # Per-sample variance at 20 Hz: (0.0035 * sqrt(20))^2 = 2.45e-4 (rad/s)^2
        self.declare_parameter("imu_gyro_variance", 2.5e-4)

        imu_input_topic = str(
            self.get_parameter("imu_input_topic").get_parameter_value().string_value
        )
        imu_output_topic = str(
            self.get_parameter("imu_output_topic").get_parameter_value().string_value
        )
        self._imu_gyro_variance: float = float(
            self.get_parameter("imu_gyro_variance").get_parameter_value().double_value
        )

        self._sub = self.create_subscription(
            NavSatFix, input_topic, self._gnss_callback, qos
        )
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)

        self._imu_sub = self.create_subscription(
            Imu, imu_input_topic, self._imu_callback, qos
        )
        self._imu_pub = self.create_publisher(Imu, imu_output_topic, qos)

        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"(config: {self._config_path}, "
            f"markov_transitions: {self._markov_enabled})"
        )
        self.get_logger().info(
            f"ImuCovarianceRelay: {imu_input_topic} -> {imu_output_topic} "
            f"(gyro_variance: {self._imu_gyro_variance:.2e} (rad/s)^2)"
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

            # Prefer tier_name field (new format). Fall back to raw stddev
            # fields for backwards compatibility with older training containers.
            tier_name: Optional[str] = data.get("tier_name")
            if tier_name and tier_name in _TIER_ORDER:
                self._apply_tier(tier_name)
            else:
                # Legacy path: read stddev fields directly.
                self._extra_lat_stddev_deg = float(data.get("lat_stddev_deg", 0.0))
                self._extra_lon_stddev_deg = float(data.get("lon_stddev_deg", 0.0))
                self._extra_alt_stddev_m = float(data.get("alt_stddev_m", 0.0))
                self._metric_stddev_m = float(
                    data.get("metric_stddev_m", self._base_metric_stddev)
                )

            self.get_logger().info(
                f"GNSS noise config updated (seq={seq}, tier={tier_name}): "
                f"metric_stddev={self._metric_stddev_m:.3f}m"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to read GNSS noise config: {exc}")

    def _step_markov(self) -> None:
        """
        @brief Advance the Markov chain by one step.

        Samples the next tier from the transition distribution of the current
        tier. If the tier changes, calls _apply_tier() to update noise state
        and logs the transition for traceability.
        """
        row = self._transition_matrix[self._active_tier_idx]
        next_idx = int(self._rng.choice(len(_TIER_ORDER), p=row))
        if next_idx != self._active_tier_idx:
            from_name = _TIER_ORDER[self._active_tier_idx]
            to_name = _TIER_ORDER[next_idx]
            self._apply_tier(to_name)
            self.get_logger().info(
                f"GNSS tier transition: {from_name} -> {to_name} "
                f"(metric_stddev={self._metric_stddev_m:.3f}m)"
            )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _imu_callback(self, msg: Imu) -> None:
        """
        @brief Relay IMU with realistic angular_velocity_covariance stamped.

        CARLA bridge publishes zero covariance on all Imu fields. robot_localization
        interprets zero covariance as infinite sensor reliability, which pins the
        EKF state to the IMU measurement with no uncertainty growth. This relay
        stamps a diagonal angular_velocity_covariance matching a mid-grade MEMS
        gyro so the EKF correctly inflates covariance between GNSS fixes.

        Orientation and linear_acceleration covariance fields are left at zero:
        - Orientation: disabled in imu0_config (CARLA publishes identity quaternion).
        - Linear acceleration: disabled in imu0_config (zero covariance would make
          acceleration infinitely reliable and override GNSS velocity corrections).

        @param msg: Raw sensor_msgs/Imu from CARLA bridge.
        """
        out = Imu()
        out.header = msg.header
        out.orientation = msg.orientation
        out.angular_velocity = msg.angular_velocity
        out.linear_acceleration = msg.linear_acceleration

        # Set orientation_covariance[0] = -1: the ROS convention for
        # "this sensor does not provide orientation". A value of 0.0 (CARLA
        # default) means infinite precision, which causes robot_localization
        # to attempt a Mahalanobis check with a singular covariance and NaN
        # the entire filter -- even when orientation is disabled in imu0_config.
        out.orientation_covariance = [-1.0] + [0.0] * 8

        # Stamp diagonal angular_velocity_covariance. Only the z/z element
        # (index 8, vyaw) is active in imu0_config; set all diagonal elements
        # to the gyro variance for correctness even though only vyaw is used.
        v = self._imu_gyro_variance
        out.angular_velocity_covariance = [
            v,   0.0, 0.0,
            0.0, v,   0.0,
            0.0, 0.0, v,
        ]

        # Set linear_acceleration_covariance[0] = -1: same convention as above.
        # Linear acceleration is disabled in imu0_config but a zero covariance
        # would still trigger the singular-matrix NaN path in robot_localization.
        out.linear_acceleration_covariance = [-1.0] + [0.0] * 8

        self._imu_pub.publish(out)

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
