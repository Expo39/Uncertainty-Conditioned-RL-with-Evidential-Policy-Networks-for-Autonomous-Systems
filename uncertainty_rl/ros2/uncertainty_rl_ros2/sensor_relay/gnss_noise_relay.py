"""
@file gnss_noise_relay.py
@brief ROS 2 node that injects per-episode GNSS noise, projects to local XY,
       and publishes Odometry with Doppler-style velocity + COG heading.

Velocity/heading come from the CLEAN fix, then noised per tier, so the
tiers cannot leave a clean dead-reckoning channel to exploit.
"""

import json
import math
import os
from typing import Dict, List, Optional, cast

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus

# Index position defines the row/column order in the transition matrix.
_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]
_TIER_INDEX: Dict[str, int] = {name: i for i, name in enumerate(_TIER_ORDER)}

# Fallback if gnss_noise_profiles.yaml is absent; mirror any edit there.
# doppler_stddev_ms is 1-sigma velocity noise (m/s) and also sets the COG
# heading noise (course_std = doppler_stddev_ms / speed).
_TIER_DEFAULTS: Dict[str, Dict[str, float]] = {
    "rtk_fixed": {
        "lat_stddev_deg": 0.0000002,
        "lon_stddev_deg": 0.0000002,
        "alt_stddev_m": 0.05,
        # Conservative relative to the cm-level RTK solution, to cover the
        # antenna phase-centre offset.
        "metric_stddev_m": 0.020,
        "doppler_stddev_ms": 0.05,  # m/s - ZED-F9P data-sheet accuracy
    },
    "rtk_float": {
        "lat_stddev_deg": 0.0000032,
        "lon_stddev_deg": 0.0000032,
        "alt_stddev_m": 0.5,
        "metric_stddev_m": 0.360,
        "doppler_stddev_ms": 0.90,  # m/s - 18x base, as position is
    },
    "standalone": {
        "lat_stddev_deg": 0.0000162,
        "lon_stddev_deg": 0.0000162,
        "alt_stddev_m": 5.0,
        "metric_stddev_m": 1.802,
        "doppler_stddev_ms": 4.5,  # m/s - 90x base, as position is
    },
    "degraded": {
        "lat_stddev_deg": 0.0000449,
        "lon_stddev_deg": 0.0000449,
        "alt_stddev_m": 10.0,
        "metric_stddev_m": 5.0,
        "doppler_stddev_ms": 12.5,  # m/s - 250x base, as position is
    },
}

# Used only when the profiles YAML cannot be loaded at startup.
_DEFAULT_TRANSITION_MATRIX: List[List[float]] = [
    # to:  fixed   float   standalone  degraded
    [0.9920, 0.0080, 0.0000, 0.0000],  # from: rtk_fixed
    [0.0400, 0.9550, 0.0050, 0.0000],  # from: rtk_float
    [0.0000, 0.0080, 0.9875, 0.0045],  # from: standalone
    [0.0000, 0.0000, 0.0071, 0.9929],  # from: degraded
]

# The NavSatStatus a real receiver would report in each fix state: RTK
# fix/float use RTCM ground augmentation, hence GBAS; SPP does not. This is
# real-API parity for consumers that branch on fix quality, not data realism.
_TIER_STATUS: Dict[str, int] = {
    "rtk_fixed": NavSatStatus.STATUS_GBAS_FIX,
    "rtk_float": NavSatStatus.STATUS_GBAS_FIX,
    "standalone": NavSatStatus.STATUS_FIX,
    "degraded": NavSatStatus.STATUS_FIX,
}


def mask_recovery_transitions(
    row: np.ndarray,
    active_idx: int,
    degrade_rate_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """
    @brief Mask out recovery (upward) transitions for the monotone-degradation mode.
    @param row: The active tier's row of the per-step transition matrix.
    @param active_idx: Index of the current tier in _TIER_ORDER (0 = best fix).
    @param degrade_rate_scale: Multiplier on the downward transition mass before
            renormalisation. Above 1.0 it compresses the schedule so the walk to
            the worst tier completes inside the episode horizon, which the
            handover-latency measurement needs; the ladder, its ordering and the
            ratchet are unchanged.
    @return Renormalised copy of the row with transitions to a better tier zeroed,
            their mass folded onto the same-or-worse entries. None when no mass
            remains, signalling the caller to hold the current tier.

    Kept free of node state so the one-way ratchet is unit-testable without a
    running ROS 2 node. @see GnssNoiseRelayNode._step_markov.
    """
    masked = np.asarray(row, dtype=np.float64).copy()
    masked[:active_idx] = 0.0
    if degrade_rate_scale != 1.0 and masked.size > active_idx + 1:
        # Scale the strictly-worse-tier mass, capping the total so the self-loop
        # cannot go negative, then let the self-loop absorb whatever remains.
        down = masked[active_idx + 1 :] * float(degrade_rate_scale)
        down_total = down.sum()
        if down_total > 0.0:
            budget = masked[active_idx:].sum()
            if down_total > budget:
                down *= budget / down_total
                down_total = budget
            masked[active_idx + 1 :] = down
            masked[active_idx] = max(0.0, budget - down_total)
    total = masked.sum()
    if total <= 0.0:
        return None
    return cast(np.ndarray, masked / total)


class GnssNoiseRelayNode(Node):
    """
    @class GnssNoiseRelayNode
    @brief Adds per-episode noise to CARLA GNSS, publishes Odometry + COG heading.
    """

    _DEFAULT_CONFIG_PATH: str = "/workspace/outputs/episode_config.json"
    _DEFAULT_NOISE_PROFILES_PATH: str = (
        "/workspace/configs/deployment/sim/gnss_noise_profiles.yaml"
    )

    # The variance of a uniform angle on [-pi, pi]: a COG observation at or
    # above this says no more than "any angle", so it is not worth publishing.
    _MAX_PUBLISH_VARIANCE: float = (math.pi**2) / 3.0

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
        # Minimum per-step displacement (metres) to accept a COG update.
        self.declare_parameter("cog_min_displacement_m", 0.05)
        # Source of the ZUPT (standstill) signal that gates COG.
        self.declare_parameter("imu_topic", "/carla/ego_vehicle/imu/stamped")
        # Master switch: false suppresses all COG heading publication.
        self.declare_parameter("enable_cog_heading", True)
        # Models geometric dilution of precision: real receivers have unequal
        # east/north variances depending on satellite geometry.
        self.declare_parameter("enable_gnss_anisotropy", True)
        self.declare_parameter("aniso_ratio_max", 1.5)
        # Models cycle slips and brief satellite occlusions.
        self.declare_parameter("gnss_dropout_probability", 0.02)
        self.declare_parameter(
            "noise_profiles_path",
            os.environ.get(
                "GNSS_NOISE_PROFILES_PATH", self._DEFAULT_NOISE_PROFILES_PATH
            ),
        )
        # This node runs in the ros2-bridge container, a separate process from
        # training, so the noise RNG cannot inherit the training seed and must
        # be seeded independently to keep a fixed-seed run reproducible.
        self.declare_parameter("seed", 42)

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
            self.get_parameter("heading_output_topic")
            .get_parameter_value()
            .string_value
        )
        self._cog_min_displacement_m: float = float(
            self.get_parameter("cog_min_displacement_m")
            .get_parameter_value()
            .double_value
        )
        imu_topic = str(
            self.get_parameter("imu_topic").get_parameter_value().string_value
        )
        self._enable_cog_heading: bool = bool(
            self.get_parameter("enable_cog_heading").get_parameter_value().bool_value
        )
        self._enable_anisotropy: bool = bool(
            self.get_parameter("enable_gnss_anisotropy")
            .get_parameter_value()
            .bool_value
        )
        self._aniso_ratio_max: float = float(
            self.get_parameter("aniso_ratio_max").get_parameter_value().double_value
        )
        self._dropout_probability: float = float(
            self.get_parameter("gnss_dropout_probability")
            .get_parameter_value()
            .double_value
        )
        # 1.0 == isotropic. Their geometric mean is held at 1.0 so the tier's
        # nominal sigma still describes the average.
        self._sigma_x_factor: float = 1.0
        self._sigma_y_factor: float = 1.0

        self._metres_per_deg_lat: float = 111320.0
        self._metres_per_deg_lon: float = 111320.0 * math.cos(
            math.radians(self._datum_lat)
        )

        self._tier_params: Dict[str, Dict[str, float]] = dict(_TIER_DEFAULTS)
        noise_profiles_path = str(
            self.get_parameter("noise_profiles_path").get_parameter_value().string_value
        )
        # Stage-invariant: the degradation process is loaded once here and
        # never rescaled per episode.
        self._transition_matrix: np.ndarray = self._load_transition_matrix(
            noise_profiles_path
        )
        self._active_tier_idx: int = 0
        # Per-episode override for controlled evaluation conditions: suppresses
        # the Markov chain so the tier stays fixed. Training leaves it False.
        self._hold_tier: bool = False
        # Per-episode override: the chain may only walk down the ladder and
        # never recovers - the monotone-degradation eval condition.
        self._degrade_one_way: bool = False
        # Only read when _degrade_one_way is set. @see mask_recovery_transitions.
        self._degrade_rate_scale: float = 1.0

        self._extra_alt_stddev_m: float = 0.0
        # Stamped on each outgoing NavSatFix when sim noise is enabled.
        self._tier_status: int = NavSatStatus.STATUS_FIX
        self._metric_stddev_m: float = self._base_metric_stddev

        self._config_seq: int = -1
        self._config_mtime_ns: int = 0
        self._callback_count: int = 0

        self._auto_datum: bool = self._datum_lat == 0.0 and self._datum_lon == 0.0
        self._datum_latched: bool = not self._auto_datum

        seed = int(self.get_parameter("seed").get_parameter_value().integer_value)
        self._rng = np.random.default_rng(seed)

        # Pre-allocated 36-element lists copied at each callback.
        self._odom_cov_template: List[float] = [0.0] * 36
        self._heading_cov_template: List[float] = [0.0] * 36
        # Everything but yaw gets infinite variance so the EKF takes only the
        # heading signal from this message; slot 35 is filled per callback.
        self._heading_cov_template[0] = 1.0e6
        self._heading_cov_template[7] = 1.0e6
        self._heading_cov_template[14] = 1.0e6
        self._heading_cov_template[21] = 1.0e6
        self._heading_cov_template[28] = 1.0e6
        # Conversely, GNSS odometry never measures yaw.
        self._odom_cov_template[35] = 1.0e6

        # Inert states get infinite variance so the EKF ignores them; vx
        # (slot 0) and vy (slot 7) are overwritten per callback.
        self._twist_cov_template: List[float] = [0.0] * 36
        self._twist_cov_template[14] = 1.0e6  # vz
        self._twist_cov_template[21] = 1.0e6  # vroll
        self._twist_cov_template[28] = 1.0e6  # vpitch
        self._twist_cov_template[35] = 1.0e6  # vyaw

        # Overwritten on the first _apply_tier call.
        self._doppler_stddev_ms: float = _TIER_DEFAULTS["rtk_fixed"][
            "doppler_stddev_ms"
        ]

        # Previous noisy fix in local XY (metres) and its timestamp (seconds).
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_stamp_sec: Optional[float] = None

        # Doppler source: differencing CLEAN fixes avoids amplifying position
        # noise by 1/dt, which at 20 Hz would be a factor of 20.
        self._prev_clean_x: Optional[float] = None
        self._prev_clean_y: Optional[float] = None

        self._cog_initialised: bool = False
        self._last_heading_rad: float = 0.0
        self._last_heading_var: float = (math.pi**2) / 3.0
        self._last_speed_ms: float = 0.0
        # True only when the last callback met both COG gates, so a rejected
        # callback cannot publish the previous tight variance.
        self._cog_active: bool = False

        # Default True so COG stays gated off until the first IMU sample.
        self._imu_stationary: bool = True

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

        self._sub = self.create_subscription(
            NavSatFix, input_topic, self._gnss_callback, qos
        )
        # ImuNoiseRelayNode publishes the stamped IMU with BEST_EFFORT QoS.
        self._imu_sub = self.create_subscription(
            Imu, imu_topic, self._imu_callback, qos_be
        )
        self._pub = self.create_publisher(NavSatFix, output_topic, qos)
        self._odom_pub = self.create_publisher(Odometry, odom_output_topic, qos_be)
        self._heading_pub = self.create_publisher(
            PoseWithCovarianceStamped, heading_output_topic, qos_be
        )

        self.get_logger().info(
            f"GnssNoiseRelay: {input_topic} -> {output_topic} "
            f"-> {odom_output_topic} (flat-earth, "
            f"COG heading -> {heading_output_topic}, "
            f"cog_min_disp={self._cog_min_displacement_m:.3f} m, "
            f"imu_gate={imu_topic}, "
            f"markov_transitions={self._markov_enabled})"
        )

    def _load_transition_matrix(self, profiles_path: str) -> np.ndarray:
        """
        @brief Load the per-step transition matrix from gnss_noise_profiles.yaml.
        @param profiles_path: Absolute path to gnss_noise_profiles.yaml.
        @return 4x4 ndarray indexed in _TIER_ORDER (rows = from, cols = to).
        """
        fallback = np.array(_DEFAULT_TRANSITION_MATRIX, dtype=np.float64)
        try:
            with open(profiles_path, "r") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as exc:
            self.get_logger().warning(
                f"Could not load transition matrix from {profiles_path} "
                f"({exc}); using hardcoded defaults."
            )
            return fallback

        section = data.get("transition_matrix") if isinstance(data, dict) else None
        if not isinstance(section, dict):
            self.get_logger().warning(
                f"{profiles_path} missing 'transition_matrix' section; "
                "using hardcoded defaults."
            )
            return fallback

        rows: List[List[float]] = []
        for name in _TIER_ORDER:
            row = section.get(name)
            if not isinstance(row, list) or len(row) != len(_TIER_ORDER):
                self.get_logger().warning(
                    f"transition_matrix.{name} malformed in {profiles_path}; "
                    "using hardcoded defaults."
                )
                return fallback
            rows.append([float(x) for x in row])

        P = np.array(rows, dtype=np.float64)
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            self.get_logger().warning(
                f"transition_matrix rows do not sum to 1 ({row_sums.tolist()}); "
                "using hardcoded defaults."
            )
            return fallback

        self.get_logger().info(f"Loaded transition matrix from {profiles_path}.")
        return P

    def _resample_anisotropy(self) -> None:
        """
        @brief Sample per-episode axis-aligned GNSS noise anisotropy.

        The two factors are reciprocal, so their geometric mean stays unity and
        the tier's nominal sigma is preserved on average.
        """
        if not self._enable_anisotropy:
            self._sigma_x_factor = 1.0
            self._sigma_y_factor = 1.0
            return
        ratio = float(self._rng.uniform(1.0, max(1.0, self._aniso_ratio_max)))
        factor = math.sqrt(ratio)
        if self._rng.random() < 0.5:
            self._sigma_x_factor = factor
            self._sigma_y_factor = 1.0 / factor
        else:
            self._sigma_x_factor = 1.0 / factor
            self._sigma_y_factor = factor

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
        self._doppler_stddev_ms = float(
            params.get(
                "doppler_stddev_ms",
                _TIER_DEFAULTS[tier_name]["doppler_stddev_ms"],
            )
        )
        self._tier_status = _TIER_STATUS.get(tier_name, NavSatStatus.STATUS_FIX)

        if tier_name in _TIER_INDEX:
            self._active_tier_idx = _TIER_INDEX[tier_name]

    def _check_config_file(self) -> None:
        """
        @brief Read GNSS noise config from the shared JSON file if updated.

        The training container writes it at each episode reset with that
        episode's start tier.
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

            self._hold_tier = bool(data.get("hold_tier", False))
            self._degrade_one_way = bool(data.get("degrade_one_way", False))
            self._degrade_rate_scale = max(
                1.0, float(data.get("degrade_rate_scale", 1.0))
            )

            tier_name: Optional[str] = data.get("tier_name")
            if tier_name and tier_name in _TIER_ORDER:
                self._apply_tier(tier_name)
                self._write_active_tier(tier_name)
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
                self._metres_per_deg_lon = 111320.0 * math.cos(
                    math.radians(self._datum_lat)
                )
                self._datum_latched = True
                # Invalidate the previous fixes: differencing across a datum
                # change would give a spurious velocity and heading.
                self._prev_x = None
                self._prev_y = None
                self._prev_clean_x = None
                self._prev_clean_y = None
                self._prev_stamp_sec = None
                self._cog_initialised = False
                self._cog_active = False
                # At the episode boundary, so each episode sees a different
                # satellite-geometry pattern.
                self._resample_anisotropy()
                self.get_logger().info(
                    f"GNSS datum re-latched: lat={self._datum_lat:.7f} "
                    f"lon={self._datum_lon:.7f} "
                    f"aniso_factors=({self._sigma_x_factor:.3f},"
                    f"{self._sigma_y_factor:.3f})"
                )

            self.get_logger().info(
                f"GNSS noise config updated (seq={seq}, tier={tier_name}, "
                f"hold={self._hold_tier}, degrade_one_way={self._degrade_one_way}): "
                f"metric_stddev={self._metric_stddev_m:.3f}m"
            )
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self.get_logger().warn(f"Failed to read GNSS noise config: {exc}")

    def _write_active_tier(self, tier_name: str) -> None:
        """
        @brief Write the current active tier back to episode_config.json.

        Read-modify-write, so the datum the training container wrote at episode
        reset survives.

        @param tier_name: Active RTK fix-state tier name.
        """
        tmp_path = self._config_path + ".markov.tmp"
        try:
            try:
                with open(self._config_path, "r") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                data = {}
            data["seq"] = self._config_seq
            data["tier_name"] = tier_name
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, self._config_path)
        except OSError as exc:
            self.get_logger().warn(f"Failed to write active tier to config: {exc}")

    def _step_markov(self) -> None:
        """@brief Advance the Markov chain by one step."""
        row = self._transition_matrix[self._active_tier_idx]
        if self._degrade_one_way:
            row = mask_recovery_transitions(
                row, self._active_tier_idx, self._degrade_rate_scale
            )
            if row is None:
                return  # Already at the worst tier with no self-loop mass; hold.
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

    def _imu_callback(self, msg: Imu) -> None:
        """
        @brief Set _imu_stationary when ZUPT-clamped zeros are seen on all axes.
        @param msg: sensor_msgs/Imu with ZUPT-clamped fields.
        """
        self._imu_stationary = (
            msg.angular_velocity.z == 0.0
            and msg.linear_acceleration.x == 0.0
            and msg.linear_acceleration.y == 0.0
        )

    def _gnss_callback(self, msg: NavSatFix) -> None:
        """
        @brief Process a single GNSS fix: inject noise, project to XY, publish odom + COG.
        @param msg: Raw NavSatFix from the CARLA GNSS sensor.
        """
        self._callback_count += 1
        self._check_config_file()

        if self._markov_enabled and not self._hold_tier:
            self._step_markov()

        # Skipping the fix entirely leaves the EKF open-loop on position for a
        # tick, as a cycle slip or brief satellite occlusion would.
        if (
            self._gnss_noise_enabled
            and self._dropout_probability > 0.0
            and self._rng.random() < self._dropout_probability
        ):
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Fallback path only. Normally the training container writes the datum
        # to episode_config.json before the episode's first GNSS callback and
        # _check_config_file() latches it; reaching here means the config has
        # not arrived (startup race, or an env-side publish failure).
        if not self._datum_latched:
            self._datum_lat = msg.latitude
            self._datum_lon = msg.longitude
            self._metres_per_deg_lon = 111320.0 * math.cos(
                math.radians(self._datum_lat)
            )
            self._datum_latched = True
            self._prev_x = None
            self._prev_y = None
            self._prev_clean_x = None
            self._prev_clean_y = None
            self._prev_stamp_sec = None
            self._cog_initialised = False
            self._cog_active = False
            self.get_logger().info(
                f"GnssNoiseRelay: auto-latched datum "
                f"lat={self._datum_lat:.6f} lon={self._datum_lon:.6f}"
            )

        out = NavSatFix()
        out.header = msg.header
        if self._gnss_noise_enabled:
            # The bridge passes through a fixed status; override it so the
            # reported fix quality tracks the active tier.
            out.status.status = self._tier_status
            out.status.service = msg.status.service
        else:
            out.status = msg.status

        if self._gnss_noise_enabled:
            sigma_x = self._metric_stddev_m * self._sigma_x_factor
            sigma_y = self._metric_stddev_m * self._sigma_y_factor
            n_xy, n_xy2, n_alt = self._rng.standard_normal(3)
            out.longitude = msg.longitude + (
                n_xy * sigma_x / self._metres_per_deg_lon if sigma_x > 0.0 else 0.0
            )
            out.latitude = msg.latitude + (
                n_xy2 * sigma_y / self._metres_per_deg_lat if sigma_y > 0.0 else 0.0
            )
            out.altitude = msg.altitude + (
                n_alt * self._extra_alt_stddev_m
                if self._extra_alt_stddev_m > 0.0
                else 0.0
            )
        else:
            out.longitude = msg.longitude
            out.latitude = msg.latitude
            out.altitude = msg.altitude

        if self._gnss_noise_enabled:
            # Stamp the covariance the injected noise actually has, so the EKF
            # weights the measurement correctly.
            var_x = (self._metric_stddev_m * self._sigma_x_factor) ** 2
            var_y = (self._metric_stddev_m * self._sigma_y_factor) ** 2
            alt_var = (self._extra_alt_stddev_m + 0.05) ** 2
            out.position_covariance = [
                var_x,
                0.0,
                0.0,
                0.0,
                var_y,
                0.0,
                0.0,
                0.0,
                alt_var,
            ]
            out.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        else:
            # Real path: pass through the receiver's reported covariance.
            out.position_covariance = msg.position_covariance
            out.position_covariance_type = msg.position_covariance_type
        self._pub.publish(out)

        # CARLA latitude increases southward, so the raw delta is south-positive.
        # Negate to put +y north (REP-103) for the EKF.
        local_x = (out.longitude - self._datum_lon) * self._metres_per_deg_lon
        local_y = -(out.latitude - self._datum_lat) * self._metres_per_deg_lat

        # The same projection of the ORIGINAL pre-noise fix, as the Doppler
        # source: real carrier frequency-shift velocity derives from the clean
        # signal, and the tier noise added later models tracking-loop
        # sensitivity rather than positional error.
        clean_x = (msg.longitude - self._datum_lon) * self._metres_per_deg_lon
        clean_y = -(msg.latitude - self._datum_lat) * self._metres_per_deg_lat

        dt: float = 0.0
        clean_dx: float = 0.0
        clean_dy: float = 0.0
        clean_disp_m: float = 0.0
        clean_speed: float = 0.0

        if self._prev_clean_x is not None and self._prev_stamp_sec is not None:
            dt = stamp_sec - self._prev_stamp_sec
            if dt > 0.0:
                clean_dx = clean_x - self._prev_clean_x
                clean_dy = clean_y - self._prev_clean_y
                clean_disp_m = math.hypot(clean_dx, clean_dy)
                clean_speed = clean_disp_m / dt
                self._last_speed_ms = clean_speed

        odom_msg = Odometry()
        odom_msg.header = out.header
        odom_msg.header.frame_id = "odom"
        # Naming a child frame is what makes robot_localization read the twist
        # as body-frame, so twist.linear.x is longitudinal speed.
        odom_msg.child_frame_id = "ego_vehicle"
        odom_msg.pose.pose.position.x = local_x
        odom_msg.pose.pose.position.y = local_y

        pose_cov = list(self._odom_cov_template)
        pose_cov[0] = float(out.position_covariance[0])
        pose_cov[7] = float(out.position_covariance[4])
        odom_msg.pose.covariance = pose_cov

        # Doppler speed goes on vx, the only velocity odom0_config fuses. It is
        # valid at rest (~0 +/- sigma), so no moving/stopped gate is needed.
        twist_cov = list(self._twist_cov_template)
        twist_cov[7] = 1.0e6  # vy: never measured
        if self._prev_clean_x is None or dt <= 0.0:
            # No differencing pair yet (first fix or datum re-latch), so the
            # 1e6 variance makes the EKF fall back on its prediction.
            odom_msg.twist.twist.linear.x = 0.0
            twist_cov[0] = 1.0e6
        elif not self._gnss_noise_enabled:
            # Real path: unnoised speed, with the best tier's variance.
            doppler_sigma = _TIER_DEFAULTS["rtk_fixed"]["doppler_stddev_ms"]
            odom_msg.twist.twist.linear.x = clean_speed
            twist_cov[0] = doppler_sigma * doppler_sigma
        else:
            noisy_speed = max(
                0.0,
                clean_speed
                + float(self._rng.standard_normal() * self._doppler_stddev_ms),
            )
            odom_msg.twist.twist.linear.x = noisy_speed
            twist_cov[0] = self._doppler_stddev_ms * self._doppler_stddev_ms
        odom_msg.twist.covariance = twist_cov
        self._odom_pub.publish(odom_msg)

        # COG heading, the EKF's only yaw correction (pose0).
        if clean_disp_m > 0.0 and dt > 0.0:
            # Heading noise grows as speed falls: the same lateral velocity
            # error subtends a larger angle the slower the vehicle moves. The
            # floor keeps the near-standstill variance finite.
            course_std = self._doppler_stddev_ms / max(clean_speed, 0.1)
            candidate_var = course_std * course_std
            # Three gates: enough travel to define a direction, the IMU not
            # reporting standstill, and a heading worth more than a guess.
            if (
                clean_disp_m >= self._cog_min_displacement_m
                and not self._imu_stationary
                and candidate_var <= self._MAX_PUBLISH_VARIANCE
            ):
                self._cog_active = True
                # clean_dy was already made north-positive, so this atan2 is
                # REP-103 heading with no further correction.
                raw_heading = math.atan2(clean_dy, clean_dx)
                if self._gnss_noise_enabled:
                    raw_heading += float(self._rng.standard_normal() * course_std)
                self._last_heading_rad = raw_heading
                self._last_heading_var = candidate_var

                if not self._cog_initialised:
                    self._cog_initialised = True
                    self.get_logger().info(
                        f"COG heading initialised: "
                        f"heading={math.degrees(self._last_heading_rad):.2f} deg "
                        f"speed={clean_speed:.2f} m/s"
                    )

                if self._callback_count % 100 == 0:
                    self.get_logger().info(
                        f"[cog_dbg] clean_dx={clean_dx:.4f} clean_dy={clean_dy:.4f} "
                        f"speed={clean_speed:.2f} m/s "
                        f"heading={math.degrees(self._last_heading_rad):.2f} deg "
                        f"var={self._last_heading_var:.4e} rad^2"
                    )
            else:
                self._cog_active = False

        self._prev_x = local_x
        self._prev_y = local_y
        self._prev_clean_x = clean_x
        self._prev_clean_y = clean_y
        self._prev_stamp_sec = stamp_sec

        if not self._enable_cog_heading:
            return

        if not self._cog_initialised or not self._cog_active:
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
