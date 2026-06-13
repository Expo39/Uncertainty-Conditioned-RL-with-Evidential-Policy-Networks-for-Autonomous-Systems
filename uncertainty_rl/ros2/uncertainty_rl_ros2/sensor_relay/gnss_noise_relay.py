"""
@file gnss_noise_relay.py
@brief ROS 2 node that injects per-episode GNSS noise, projects to local XY,
       and publishes Odometry with Doppler-style velocity + COG heading for
       the robot_localisation EKF.

Subscribes to CARLA NavSatFix, adds tier-appropriate Gaussian noise to
position, converts to metric Odometry via flat-earth projection
(/odometry/gps). Velocity is Doppler-style: clean-source differencing at
20 Hz equals GT velocity; per-tier Gaussian noise (doppler_stddev_ms,
0.05-12.5 m/s) scales by the same factor as the position tier ladder
(0.02-5.0 m), so velocity degrades with the fix state. COG is derived from
the same clean displacement vector with noise proportional to
doppler_std/speed (so it degrades with the same factor) and published on
/gnss/heading.
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus

# Ordered tier names - index position defines row/column in transition matrix.
_TIER_ORDER: List[str] = ["rtk_fixed", "rtk_float", "standalone", "degraded"]
# reverse lookup: tier name -> index in _TIER_ORDER / transition matrix.
_TIER_INDEX: Dict[str, int] = {name: i for i, name in enumerate(_TIER_ORDER)}

# Noise parameters for each tier (fallback if config file is absent).
# doppler_stddev_ms is the 1-sigma velocity noise (m/s); it also sets the COG
# heading noise (course_std = doppler_stddev_ms / speed). It scales by the same
# per-tier factor as position (1 / 18 / 90 / 250x of the rtk_fixed base), so
# velocity and heading degrade with the fix state instead of leaving a clean
# dead-reckoning channel. Mirror these values in gnss_noise_profiles.yaml.
_TIER_DEFAULTS: Dict[str, Dict[str, float]] = {
    # RTK fixed:
    # Conservative 0.020 m used to cover antenna phase-centre offset
    # 0.020 / 111320 = 1.797e-7 deg.
    "rtk_fixed": {
        "lat_stddev_deg": 0.0000002,
        "lon_stddev_deg": 0.0000002,
        "alt_stddev_m": 0.05,
        "metric_stddev_m": 0.020,
        "doppler_stddev_ms": 0.05,  # m/s - 1x base (ZED-F9P data-sheet accuracy)
    },
    # RTK float:
    # 0.360 / 111320 = 3.233e-6 deg.
    "rtk_float": {
        "lat_stddev_deg": 0.0000032,
        "lon_stddev_deg": 0.0000032,
        "alt_stddev_m": 0.5,
        "metric_stddev_m": 0.360,
        "doppler_stddev_ms": 0.90,  # m/s - 18x base (matches position factor)
    },
    # Standalone PVT:
    # 1.802 / 111320 = 1.619e-5 deg.
    "standalone": {
        "lat_stddev_deg": 0.0000162,
        "lon_stddev_deg": 0.0000162,
        "alt_stddev_m": 5.0,
        "metric_stddev_m": 1.802,
        "doppler_stddev_ms": 4.5,  # m/s - 90x base (matches position factor)
    },
    # Degraded:
    # 5.0 / 111320 = 4.492e-5 deg.
    "degraded": {
        "lat_stddev_deg": 0.0000449,
        "lon_stddev_deg": 0.0000449,
        "alt_stddev_m": 10.0,
        "metric_stddev_m": 5.0,
        "doppler_stddev_ms": 12.5,  # m/s - 250x base (matches position factor)
    },
}

# Fallback used only if configs cannot be loaded at startup. Mirror of the
# transition_matrix in gnss_noise_profiles.yaml (standalone ~4 s, degraded ~7 s dwell).
_DEFAULT_TRANSITION_MATRIX: List[List[float]] = [
    # to:  fixed   float   standalone  degraded
    [0.9920, 0.0080, 0.0000, 0.0000],  # from: rtk_fixed
    [0.0400, 0.9550, 0.0050, 0.0000],  # from: rtk_float
    [0.0000, 0.0080, 0.9875, 0.0045],  # from: standalone
    [0.0000, 0.0000, 0.0071, 0.9929],  # from: degraded
]

# Map tier name to the corresponding NavSatStatus value that a real receiver
# would publish in this fix state. RTK FIX/FLOAT solutions use ground-based
# augmentation (RTCM corrections), so both report STATUS_GBAS_FIX. SPP and
# degraded SPP report STATUS_FIX. This is real-API parity, not data realism.
_TIER_STATUS: Dict[str, int] = {
    "rtk_fixed": NavSatStatus.STATUS_GBAS_FIX,
    "rtk_float": NavSatStatus.STATUS_GBAS_FIX,
    "standalone": NavSatStatus.STATUS_FIX,
    "degraded": NavSatStatus.STATUS_FIX,
}


class GnssNoiseRelayNode(Node):
    """
    @class GnssNoiseRelayNode
    @brief Adds per-episode noise to CARLA GNSS, publishes Odometry + COG heading.
    """

    _DEFAULT_CONFIG_PATH: str = "/workspace/outputs/episode_config.json"
    _DEFAULT_NOISE_PROFILES_PATH: str = (
        "/workspace/configs/deployment/sim/gnss_noise_profiles.yaml"
    )

    # Variance ceiling for publishing COG: pi^2/3 is the variance of a uniform
    # distribution on [-pi, pi]. A heading observation with variance >= this is
    # no more informative than "any angle"; do not publish it.
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
        # IMU topic used to detect ZUPT (standstill) and gate COG accordingly.
        self.declare_parameter("imu_topic", "/carla/ego_vehicle/imu/stamped")
        # Master switch: false suppresses all COG heading publication.
        self.declare_parameter("enable_cog_heading", True)
        # Per-episode axis-aligned anisotropy on GNSS position noise. Models
        # geometric dilution of precision: real receivers have unequal east/north
        # variances depending on satellite geometry.
        self.declare_parameter("enable_gnss_anisotropy", True)
        self.declare_parameter("aniso_ratio_max", 1.5)
        # Per-callback probability of skipping a GNSS fix entirely. Models
        # cycle slips and brief satellite occlusions.
        self.declare_parameter("gnss_dropout_probability", 0.02)
        self.declare_parameter(
            "noise_profiles_path",
            os.environ.get(
                "GNSS_NOISE_PROFILES_PATH", self._DEFAULT_NOISE_PROFILES_PATH
            ),
        )
        # Seed for the noise RNG. This node runs in the ros2-bridge container,
        # a separate process from training, so it cannot inherit the training
        # seed; it must be seeded independently. Matches the training seed (42)
        # by default so a fixed-seed training run sees reproducible GNSS noise.
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
        # Per-episode anisotropy factors. 1.0 == isotropic. Geometric mean is
        # always 1.0 so the tier's nominal sigma still describes the average.
        self._sigma_x_factor: float = 1.0
        self._sigma_y_factor: float = 1.0

        # Precompute flat-earth scale factors for the datum latitude.
        self._metres_per_deg_lat: float = 111320.0
        self._metres_per_deg_lon: float = 111320.0 * math.cos(
            math.radians(self._datum_lat)
        )

        self._tier_params: Dict[str, Dict[str, float]] = dict(_TIER_DEFAULTS)
        noise_profiles_path = str(
            self.get_parameter("noise_profiles_path").get_parameter_value().string_value
        )
        # Transition matrix stepped by the Markov chain. The matrix is the GNSS
        # degradation process and is stage-invariant - loaded once from the noise
        # profiles, never rescaled per episode.
        self._transition_matrix: np.ndarray = self._load_transition_matrix(
            noise_profiles_path
        )
        self._active_tier_idx: int = 0
        # Per-episode override: when episode_config.json sets hold_tier, the
        # Markov chain is suppressed for that episode so the level stays fixed
        # (controlled evaluation conditions). Training leaves it False.
        self._hold_tier: bool = False

        self._extra_alt_stddev_m: float = 0.0
        # Current tier's NavSatStatus code, updated by _apply_tier and stamped
        # on each outgoing NavSatFix when sim noise is enabled.
        self._tier_status: int = NavSatStatus.STATUS_FIX
        self._metric_stddev_m: float = self._base_metric_stddev

        self._config_seq: int = -1
        self._config_mtime_ns: int = 0
        self._callback_count: int = 0

        self._auto_datum: bool = self._datum_lat == 0.0 and self._datum_lon == 0.0
        self._datum_latched: bool = not self._auto_datum

        seed = int(self.get_parameter("seed").get_parameter_value().integer_value)
        self._rng = np.random.default_rng(seed)

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

        # Twist covariance template for the Odometry message. Inert states
        # (vz, vroll, vpitch, vyaw) get infinite variance so the EKF ignores
        # them; vx (slot 0) and vy (slot 7) are overwritten per callback.
        self._twist_cov_template: List[float] = [0.0] * 36
        self._twist_cov_template[14] = 1.0e6  # vz
        self._twist_cov_template[21] = 1.0e6  # vroll
        self._twist_cov_template[28] = 1.0e6  # vpitch
        self._twist_cov_template[35] = 1.0e6  # vyaw

        # Doppler-style velocity noise: set by _apply_tier from doppler_stddev_ms.
        # Initialised to rtk_fixed default; overwritten on first _apply_tier call.
        self._doppler_stddev_ms: float = _TIER_DEFAULTS["rtk_fixed"][
            "doppler_stddev_ms"
        ]

        # Previous noisy fix in local XY (metres) and its timestamp (seconds).
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._prev_stamp_sec: Optional[float] = None

        # Previous CLEAN (pre-noise) fix in local XY used as the Doppler
        # velocity source. Clean differencing at 20 Hz = GT velocity;
        # position-noise amplification by 1/dt is avoided entirely.
        self._prev_clean_x: Optional[float] = None
        self._prev_clean_y: Optional[float] = None

        # COG heading state.
        self._cog_initialised: bool = False
        self._last_heading_rad: float = 0.0
        self._last_heading_var: float = (math.pi**2) / 3.0
        self._last_speed_ms: float = 0.0
        # True only when the last callback met both COG gates (speed + displacement).
        # Used to suppress tight-variance publish when gates reject a callback.
        self._cog_active: bool = False

        # True when the stamped IMU shows ZUPT-clamped zeros on all channels.
        # Default True so COG is gated off until the first IMU sample arrives.
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

    # -----------------------------------------------------------------------
    # Config and tier helpers
    # -----------------------------------------------------------------------

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

        Sets sigma_x_factor and sigma_y_factor such that the geometric mean is
        unity (so the tier's nominal sigma is preserved on average), with a
        ratio drawn uniformly from [1.0, aniso_ratio_max]. The major-sigma axis
        is randomly x or y per episode.
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

            # Per-episode hold: when set, the Markov chain is suppressed for the
            # whole episode so the level stays fixed (controlled evaluation
            # conditions). Absent/false in training, where the chain wanders.
            self._hold_tier = bool(data.get("hold_tier", False))

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
                # Invalidate previous fixes so stale positions from the old
                # datum are not used for COG or Doppler velocity after reset.
                self._prev_x = None
                self._prev_y = None
                self._prev_clean_x = None
                self._prev_clean_y = None
                self._prev_stamp_sec = None
                self._cog_initialised = False
                self._cog_active = False
                # Resample anisotropy at the episode boundary so each episode
                # sees a different satellite-geometry pattern.
                self._resample_anisotropy()
                self.get_logger().info(
                    f"GNSS datum re-latched: lat={self._datum_lat:.7f} "
                    f"lon={self._datum_lon:.7f} "
                    f"aniso_factors=({self._sigma_x_factor:.3f},"
                    f"{self._sigma_y_factor:.3f})"
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

        Read-modify-write so that datum_lat and datum_lon written by the
        training container at episode reset are preserved.

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

    # -----------------------------------------------------------------------
    # IMU callback (ZUPT detection)
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Main callback
    # -----------------------------------------------------------------------

    def _gnss_callback(self, msg: NavSatFix) -> None:
        """
        @brief Process a single GNSS fix: inject noise, project to XY, publish odom + COG.

        @param msg: Raw NavSatFix from the CARLA GNSS sensor.
        """
        self._callback_count += 1
        self._check_config_file()

        if self._markov_enabled and not self._hold_tier:
            self._step_markov()

        # Simulated GNSS dropout: occasionally skip a fix entirely so the EKF
        # goes open-loop on position for one tick. Models cycle slips and brief
        # satellite occlusions seen in real RTK operation.
        if (
            self._gnss_noise_enabled
            and self._dropout_probability > 0.0
            and self._rng.random() < self._dropout_probability
        ):
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Datum latch
        # Preferred path: the training container writes datum_lat/datum_lon to
        # episode_config.json before any GNSS callback fires for the new
        # episode, and _apply_episode_config() latches that value. Falling
        # through to auto-latch only happens when the config has not yet
        # arrived (process startup race, or env-side publish failure).
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

        # Inject noise to lat/lon
        out = NavSatFix()
        out.header = msg.header
        if self._gnss_noise_enabled:
            # Sim: override the bridge's pass-through status with one that
            # reflects the active tier (real-API parity for downstream
            # consumers that branch on fix quality).
            out.status.status = self._tier_status
            out.status.service = msg.status.service
        else:
            out.status = msg.status

        if self._gnss_noise_enabled:
            # Per-axis sigmas: tier sigma scaled by per-episode anisotropy.
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
            # Sim path: stamp anisotropic covariance derived from the active
            # tier and the per-episode anisotropy factors.
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

        # Flat-earth projection: noisy lat/lon -> local XY
        # CARLA latitude increases southward, so the raw delta gives a
        # south-positive y. Negate to put +y north (REP-103) for the EKF.
        local_x = (out.longitude - self._datum_lon) * self._metres_per_deg_lon
        local_y = -(out.latitude - self._datum_lat) * self._metres_per_deg_lat

        # Clean projection: project the ORIGINAL (pre-noise) lat/lon.
        # Differencing consecutive clean positions at 20 Hz equals GT velocity
        # without amplifying position noise by 1/dt (at 20 Hz that factor is
        # x20 for position jitter). This is the Doppler source: carrier
        # frequency-shift velocity is physically derived from the clean signal;
        # the tier noise models tracking-loop sensitivity, not positional error.
        clean_x = (msg.longitude - self._datum_lon) * self._metres_per_deg_lon
        clean_y = -(msg.latitude - self._datum_lat) * self._metres_per_deg_lat

        # Doppler-style velocity and COG from clean-source differencing.
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

        # Publish GNSS odometry (EKF odom0: x, y position + vx speed)
        odom_msg = Odometry()
        odom_msg.header = out.header
        odom_msg.header.frame_id = "odom"
        # child_frame_id = ego_vehicle so robot_localization interprets the
        # twist in body frame (twist.linear.x = longitudinal speed).
        odom_msg.child_frame_id = "ego_vehicle"
        odom_msg.pose.pose.position.x = local_x
        odom_msg.pose.pose.position.y = local_y

        # Per-axis position variance from the NavSatFix
        pose_cov = list(self._odom_cov_template)
        pose_cov[0] = float(out.position_covariance[0])
        pose_cov[7] = float(out.position_covariance[4])
        odom_msg.pose.covariance = pose_cov

        # Twist: Doppler-style speed on vx. vy always infinite variance.
        # odom0_config fuses index 6 (vx). Doppler is valid at rest
        # (~0 +/- sigma), so no moving/stopped gate is needed.
        # First fix after a latch: no clean pair -> publish with 1e6 variance
        # so the EKF ignores this measurement and uses its prediction.
        twist_cov = list(self._twist_cov_template)
        twist_cov[7] = 1.0e6  # vy: never measured
        if self._prev_clean_x is None or dt <= 0.0:
            # No differencing pair yet (first fix or datum re-latch).
            odom_msg.twist.twist.linear.x = 0.0
            twist_cov[0] = 1.0e6
        elif not self._gnss_noise_enabled:
            # Noise-disabled path: publish clean speed, rtk_fixed variance.
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

        # COG heading (EKF pose0: yaw correction).
        # Derived from the clean displacement direction + Doppler-proportional
        # heading noise: course_std = doppler_stddev_ms / max(speed, 0.1).
        # Gates: (a) minimum clean displacement, (b) IMU not in ZUPT, (c)
        # heading variance below uniform-distribution ceiling. Same gates as
        # before; the variance model is now analytically derived rather than
        # positional.
        if clean_disp_m > 0.0 and dt > 0.0:
            course_std = self._doppler_stddev_ms / max(clean_speed, 0.1)
            candidate_var = course_std * course_std
            if (
                clean_disp_m >= self._cog_min_displacement_m
                and not self._imu_stationary
                and candidate_var <= self._MAX_PUBLISH_VARIANCE
            ):
                self._cog_active = True
                # clean_dy is north-positive (negated from CARLA's southward
                # latitude), so atan2(clean_dy, clean_dx) gives heading in
                # REP-103 directly.
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
