"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance and lot geometry.

Gymnasium-compatible environment for autonomous parking in CARLA. Localisation
uncertainty comes from the robot_localisation EKF fusing RTK-GNSS and IMU; 2D LiDAR
provides obstacle detection only. CARLA ground truth is used only for reward
computation.

See documentation/detailed_notes/observation_space.md for the full obs breakdown.
"""

import collections
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from gymnasium.core import RenderFrame

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.envs._parking_core import _layout_cache as _shared_layout_cache
from uncertainty_rl.envs._parking_core import (
    build_observation,
    compute_obs_dim,
    extract_obstacle_features,
    load_floor_plan,
    wait_for_ekf,
)
from uncertainty_rl.envs.covariance_subscriber import _CovarianceSubscriber
from uncertainty_rl.envs.sim.helpers import LotSpawner, NPCController, SensorManager
from uncertainty_rl.utils.constants import (
    ACTION_DIM,
    OBSTACLE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    wrap_angle_symmetric,
)
from uncertainty_rl.utils.logging import DebugLogger

logger = logging.getLogger(__name__)

# Trail length for debug overlays and vis state
_TRAJECTORY_MAXLEN = 50

# Shared empty info dict for intermediate action-repeat steps - avoids
# allocating a new dict on each of the (action_repeat - 1) hot-path calls.
_EMPTY_STEP_INFO: Dict[str, Any] = {}

# Re-export geometry helpers so existing imports from this module still work
__all__ = [
    "CARLAParkingEnv",
    "_compute_relative_target_pose",
]


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------


class CARLAParkingEnv(gym.Env):
    """
    @class CARLAParkingEnv
    @brief CARLA parking environment with lot geometry and EKF uncertainty.

    Each episode loads a floor plan from configs/layouts/, places cone perimeters,
    static parked vehicles, NPC patrol vehicles, and pedestrians. The agent must
    manoeuvre the ego vehicle into the target bay. Uncertainty is produced by the
    robot_localisation EKF processing noisy CARLA sensors.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = "FlatPlane",
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        ros2_config: Optional[Dict[str, Any]] = None,
        carla_sensors_config: Optional[Dict[str, Any]] = None,
        parking_scenarios_config: Optional[Dict[str, Any]] = None,
        include_covariance: bool = True,
        include_obstacle_obs: bool = True,
        vis_output_path: Optional[str] = None,
        carla_timestep: float = 0.05,
        eval_mode: bool = False,
        debug: bool = False,
        map_load_sleep: float = 5.0,
        action_repeat: int = 1,
        no_rendering_mode: bool = False,
        max_ego_speed_ms: float = 6.0,
        use_extra_spawns: bool = False,
        gnss_noise_profiles_path: Optional[str] = None,
        gnss_noise_multiplier_override: Optional[float] = None,
        uncertainty_std_max: float = 2.0,
        success_dwell_steps: int = 5,
        success_approach_radius: float = 2.0,
    ) -> None:
        """
        @brief Construct the CARLA parking environment.
        @param carla_host: CARLA server host address.
        @param carla_port: CARLA server port.
        @param town: CARLA town/map to load.
        @param max_steps: Maximum episode length.
        @param render_mode: Rendering mode ('human', 'rgb_array', or None).
        @param ros2_config: ROS 2 settings (covariance_topic, covariance_timeout).
        @param carla_sensors_config: Sensor noise parameters (imu, lidar subsections).
        @param parking_scenarios_config: Parking lot configuration (floor_plans,
               cone spacing, bay occupancy, NPC counts).
        @param include_covariance: If True, obs includes EKF std devs.
               If False, covariance omitted (no ROS 2 subscription).
        @param include_obstacle_obs: If True, obs includes 5 LiDAR clearance dims.
               Set False to ablate obstacle awareness.
        @param vis_output_path: Path for vis_history.jsonl writes. If None,
               defaults to outputs/vis_history.jsonl. Writing only occurs when
               the signal file outputs/.vis_active exists (created by the
               visualiser process).
        @param carla_timestep: Simulation timestep in seconds (default 0.05 = 20 Hz).
               Written into vis frames so the visualiser can pace playback at
               real-time speed.
        @param eval_mode: If True, OOD floor plans are included in sampling.
               If False (training), only non-OOD floor plans are used.
        @param debug: If True, emit per-step diagnostics via DebugLogger and include
               a debug dict in vis frames for the visualiser HUD. Off by default.
        @param map_load_sleep: Seconds to wait after loading the FlatPlane OpenDRIVE
               world before continuing. Increase on slow servers (default 5.0).
        @param max_ego_speed_ms: Maximum ego vehicle speed in m/s. Throttle is cut
               when this speed is exceeded. Default 6.0 m/s (~22 km/h),
               appropriate for parking lot manoeuvres.
        @param use_extra_spawns: If True, extra spawn transforms from the layout
               YAML are included in the spawn pool. If False (default), only the
               primary spawn is used. With RTK-GNSS the odom frame is UTM-aligned
               regardless of spawn location, so extra spawns are safe to enable.
        @param uncertainty_std_max: EKF std (metres) at which progress reward
               reaches zero. Progress is scaled by (1 - clip(std/max, 0, 1))
               so parking attempts under high uncertainty yield no reward.
               Default 2.0 m matches the standalone tier metric_stddev_m.
        @param gnss_noise_profiles_path: Path to GNSS noise profiles YAML. If
               provided, the env samples an RTK fix-state tier each reset() and
               spawns the GNSS sensor with the corresponding noise multiplier.
        @param gnss_noise_multiplier_override: If set, bypasses tier sampling and
               uses this fixed multiplier every episode. Used during evaluation
               to lock GNSS noise to a specific condition.
        @param success_dwell_steps: Number of consecutive steps all success
               criteria (position, orientation, velocity) must be satisfied
               before the episode terminates as a success. Prevents a fast
               drive-through that momentarily satisfies the thresholds from
               being counted as a park. Default 5 steps = 0.25 s at 20 Hz.
        @param success_approach_radius: Radius (metres) around the bay within
               which the bounded approach_term reward peak is active. The term
               pays for being both close and slow, decaying to zero at this
               radius. Installs a reward maximum at the bay so braking becomes
               optimal. Default 2.0 m.
        """
        super().__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._include_covariance = include_covariance
        self._include_obstacle_obs = include_obstacle_obs
        self._eval_mode = eval_mode
        self._map_load_sleep = map_load_sleep
        self._max_ego_speed_ms = max_ego_speed_ms
        self._use_extra_spawns = use_extra_spawns
        self._gnss_noise_multiplier_override = gnss_noise_multiplier_override

        self._uncertainty_std_max: float = max(uncertainty_std_max, 1e-6)
        self._success_approach_radius: float = max(success_approach_radius, 1e-6)
        self._inv_approach_radius: float = 1.0 / self._success_approach_radius

        self._inv_uncertainty_std_max: float = 1.0 / self._uncertainty_std_max
        self._inv_oob_threshold: float = 1.0 / OUT_OF_BOUNDS_THRESHOLD

        # Load GNSS noise profiles for per-episode RTK fix-state sampling.
        self._gnss_noise_tiers: List[Dict[str, Any]] = []
        self._gnss_tier_weights: np.ndarray = np.empty(0, dtype=np.float64)
        self._current_gnss_multiplier: float = 1.0
        self._current_gnss_tier: Optional[Dict[str, Any]] = None
        if gnss_noise_profiles_path:
            self._load_gnss_noise_profiles(gnss_noise_profiles_path)

        ros2_config = ros2_config or {}
        self._ros2_config: Dict[str, Any] = ros2_config
        self._covariance_topic = ros2_config.get(
            "covariance_topic", "/odometry/filtered"
        )
        self._covariance_timeout = ros2_config.get("covariance_timeout", 10.0)
        self._ekf_convergence_timeout: float = ros2_config.get(
            "ekf_convergence_timeout", 15.0
        )

        self._sensors_config = carla_sensors_config or {}

        scenarios = parking_scenarios_config or {}
        self._num_patrol_max: int = scenarios.get("num_patrol_vehicles_max", 1)
        self._patrol_obstacle_distance: float = scenarios.get(
            "patrol_obstacle_stop_distance", 5.0
        )
        self._patrol_pedestrian_distance: float = scenarios.get(
            "patrol_pedestrian_stop_distance", 4.0
        )
        self._patrol_max_speed: float = scenarios.get("patrol_max_speed_ms", 3.0)
        self._patrol_heading_gain: float = scenarios.get("patrol_heading_gain", 0.8)
        # Probability that each pedestrian zone spawns a walker each episode.
        self._pedestrian_spawn_prob: float = scenarios.get(
            "pedestrian_spawn_probability", 1.0
        )
        self._pedestrian_speed: float = scenarios.get("pedestrian_speed_ms", 1.4)
        self._pedestrian_resample_steps: int = scenarios.get(
            "pedestrian_heading_resample_steps", 30
        )
        self._pedestrian_max_lifetime: int = scenarios.get(
            "pedestrian_max_lifetime_steps", 200
        )
        self._floor_plans_config: Dict[str, Any] = scenarios.get("floor_plans", {})

        # When set, force the named floor plan / bay / GNSS tier every episode.
        self._fixed_floor_plan: Optional[str] = scenarios.get("fixed_floor_plan", None)
        self._fixed_target_bay_id: Optional[str] = scenarios.get(
            "fixed_target_bay_id", None
        )
        self._fixed_gnss_tier: Optional[str] = scenarios.get("fixed_gnss_tier", None)

        # CARLA handles
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # The spawn transform chosen for this episode (set in reset() before
        # any world ticks so the GNSS datum config is written first).
        self._chosen_spawn: Dict[str, float] = {}

        # Ego vehicle CoM z after settling under gravity.  Set in _spawn_vehicle()
        # and passed to all static spawners so props and vehicles land on the same
        # ground surface instead of using the hardcoded YAML origin_z.
        self._floor_z: float = 0.3

        # Per-step debug diagnostics.
        # Instantiated here so NPCController / SensorManager can share the reference.
        self._debug_logger: DebugLogger = DebugLogger(debug=debug)

        # Lot spawner - owns static cones and parked vehicles.
        self._lot_spawner = LotSpawner(
            cone_spacing=scenarios.get("perimeter_cone_spacing", 2.0),
            marker_blueprint=scenarios.get(
                "perimeter_marker_blueprint", "static.prop.constructioncone"
            ),
            bay_occupancy_min=scenarios.get("bay_occupancy_min", 0.3),
            bay_occupancy_max=scenarios.get("bay_occupancy_max", 0.8),
        )

        # NPC controller - owns patrol vehicles and pedestrians.
        self._npc_controller = NPCController(
            num_patrol_max=self._num_patrol_max,
            patrol_obstacle_distance=self._patrol_obstacle_distance,
            patrol_pedestrian_distance=self._patrol_pedestrian_distance,
            patrol_max_speed=self._patrol_max_speed,
            patrol_heading_gain=self._patrol_heading_gain,
            pedestrian_spawn_prob=self._pedestrian_spawn_prob,
            pedestrian_speed=self._pedestrian_speed,
            pedestrian_resample_steps=self._pedestrian_resample_steps,
            pedestrian_max_lifetime=self._pedestrian_max_lifetime,
        )

        # Sensor manager - owns IMU, GNSS, 2D LiDAR, collision sensor.
        self._sensor_manager = SensorManager(
            sensors_config=self._sensors_config,
        )

        # Cached actor list for patrol obstacle proximity checks.  Rebuilt
        # once after all vehicles are spawned; avoids per-step world queries.
        self._all_vehicle_actors: List[Any] = []

        # Cached blueprint lists - fetched once on first connect, never re-fetched.
        # Car blueprints are owned by LotSpawner; walker BPs shared with NPCController.
        self._walker_blueprints: List[Any] = []
        self._vehicle_bp: Optional[Any] = None

        # Pre-allocated observation buffer - reused every step to avoid
        # repeated small heap allocations.
        _obs_dim = self._compute_obs_dim()
        self._obs_dim: int = _obs_dim
        self._obs_buffer: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)
        self._obstacle_features_buffer: np.ndarray = np.zeros(
            OBSTACLE_FEATURES_DIM, dtype=np.float32
        )
        # 5-element buffer: [x, y, yaw, vyaw, vx_body]. vx_body is signed
        # body-frame longitudinal velocity (m/s) from the EKF; the CARLA-only
        # fallback path fills it from get_velocity() rotated into body frame.
        self._world_pose_buf: np.ndarray = np.empty(5, dtype=np.float32)

        # Identity odom-to-world transform (tx, ty, cos_r, sin_r, r).
        # The GNSS datum is latched to spawn position each episode reset, so
        # the odom frame coincides with the world frame by construction.
        self._ekf_odom_offset: Tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        # Target bay (world frame, set in reset). dx/dy/dyaw are computed each
        # step by reconstructing vehicle_world via _ekf_odom_offset and then
        # differencing against this.
        self._target_bay: Dict[str, Any] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "width": 2.5,
            "depth": 5.0,
        }
        # Cached float scalars from _target_bay to avoid dict lookup + float()
        # cast on every _compute_reward call.
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_yaw: float = 0.0

        # Current floor plan layout (loaded from YAML in reset)
        self._current_layout: Dict[str, Any] = {}
        self._current_floor_plan_name: str = ""
        # Spawn pool built once per layout load.
        self._spawn_pool: List[Any] = []
        # Bay-type grouping built once per layout load.
        self._bays_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self._bay_type_keys: List[str] = []

        # Trajectory buffer for debug overlays (ring buffer of (x, y) tuples)
        self._trajectory_buffer: Deque[Tuple[float, float]] = collections.deque(
            maxlen=_TRAJECTORY_MAXLEN
        )
        # Last action applied (3-dim: [steer, throttle, brake]). throttle and
        # brake are separate non-negative axes. No reverse gear.
        self._last_action: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float32)

        # Visualisation state writer
        self._vis_history_path: Path = (
            Path(vis_output_path)
            if vis_output_path
            else Path("outputs/vis_history.jsonl")
        )
        self._vis_signal_path: Path = self._vis_history_path.parent / ".vis_active"
        # Persistent append file handle; opened lazily, avoids per-step open().
        self._vis_file: Optional[Any] = None
        # Signal-file stat is cached and refreshed every N steps to avoid per-step
        # filesystem calls when the visualiser is not active (the common case).
        self._vis_active: bool = False
        self._vis_check_counter: int = 0
        # Truncate vis_history.jsonl every N episodes to bound file size.
        self._vis_episodes_since_rotation: int = 0
        self._vis_rotation_interval: int = 10
        self._carla_timestep: float = carla_timestep

        self._action_repeat: int = action_repeat
        self._no_rendering_mode: bool = no_rendering_mode
        self._action_repeat_counter: int = 0

        self._success_dwell_steps: int = max(1, success_dwell_steps)

        # Episode state
        self._episode_id: int = 0
        self.steps = 0
        self._actors_frozen: bool = False
        self._success_counter: int = 0
        # Previous distance to target for potential-based reward shaping
        self._prev_distance: float = 0.0

        # Cached CARLA zero objects - reused across calls to avoid per-call
        # construction overhead in _freeze_all_actors and _teleport_vehicle.
        if carla is not None:
            self._zero_vec3: Any = carla.Vector3D(x=0.0, y=0.0, z=0.0)
            self._zero_walker_ctrl: Any = carla.WalkerControl()
        else:
            self._zero_vec3 = None
            self._zero_walker_ctrl = None

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_obs_dim,),
            dtype=np.float32,
        )

        # Action space: [steer, throttle, brake]
        # steer    : [-1, 1]  left to right
        # throttle : [ 0, 1]  forward throttle (no reverse gear)
        # brake    : [ 0, 1]  friction brake
        # Throttle and brake are separate non-negative axes (was a single
        # bipolar `drive` axis). The bipolar axis put the precise endgame
        # control - brake gently to a stop and hold still - on the
        # throttle/brake discontinuity at zero, where action-sampling noise
        # flips a gentle brake into a throttle. Separate axes make "hold a
        # stop" (throttle ~ 0, brake > 0) a stable region.
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # ROS 2 covariance subscriber (only when covariance included)
        self._cov_subscriber: Optional[_CovarianceSubscriber] = None
        if self._include_covariance:
            self._init_ros2()

        # ------------------------------------------------------------------
        # Stored callables for fixed-flag hot-path branches
        # ------------------------------------------------------------------

        # _read_ekf_state() -> (raw_pose, uncertainty) or (None, None)
        if self._include_covariance and self._cov_subscriber is not None:
            _read_ekf_state: Callable[
                [], Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            ] = self._cov_subscriber.get_latest_state
        else:

            def _read_ekf_state() -> Tuple[None, None]:
                return (None, None)

        self._read_ekf_state = _read_ekf_state

        # _get_lidar_scan() -> scan array or None
        if self._include_obstacle_obs:
            self._get_lidar_scan = self._sensor_manager.get_latest_lidar_scan
        else:

            def _get_lidar_scan() -> None:
                return None

            self._get_lidar_scan = _get_lidar_scan

        # _uncertainty_scale_fn() -> float in [0, 1]
        if self._include_covariance:
            _inv = self._inv_uncertainty_std_max
            _buf = self._obs_buffer
            # std_x and std_y now live at obs[2] and obs[3] (was obs[1], obs[2])
            # after VEHICLE_STATE_DIM grew from 1 (vyaw) to 2 (speed, vyaw).
            _stdx_idx = VEHICLE_STATE_DIM
            _stdy_idx = VEHICLE_STATE_DIM + 1

            def _unc_scale_fn() -> float:
                return min(
                    max(
                        max(float(_buf[_stdx_idx]), float(_buf[_stdy_idx])) * _inv,
                        0.0,
                    ),
                    1.0,
                )

        else:

            def _unc_scale_fn() -> float:  # type: ignore[misc]
                return 0.0

        self._uncertainty_scale_fn = _unc_scale_fn

    # ------------------------------------------------------------------
    # Observation dimension helper
    # ------------------------------------------------------------------

    def _compute_obs_dim(self) -> int:
        """
        @brief Compute the observation dimension based on active feature flags.
        @return Integer observation dimension.
        """
        return compute_obs_dim(self._include_covariance, self._include_obstacle_obs)

    # ------------------------------------------------------------------
    # ROS 2 initialisation
    # ------------------------------------------------------------------

    def _init_ros2(self) -> None:
        """
        @brief Create the covariance reader (file-based, no DDS).

        The covariance reader uses a shared file to avoid cross-distro
        serialisation issues between Humble and Jazzy. The /set_pose
        signal is also file-based: the reader writes initial_pose.json and
        the CovarianceExtractorNode in ros2-bridge publishes it locally.
        No rclpy initialisation is needed.
        """
        node_name = f"covariance_subscriber_{id(self)}"
        self._cov_subscriber = _CovarianceSubscriber(
            covariance_topic=self._covariance_topic,
            node_name=node_name,
            ros2_config=self._ros2_config,
        )
        logger.info("Covariance reader initialised (file-based, no DDS).")

    # ------------------------------------------------------------------
    # GNSS noise profile loading and tier sampling
    # ------------------------------------------------------------------

    def _load_gnss_noise_profiles(self, path: str) -> None:
        """
        @brief Load RTK-GNSS noise tiers from YAML for per-episode sampling.
        @param path: Path to the GNSS noise profiles YAML file.

        Each tier models a different RTK fix state (fixed, float, standalone,
        degraded) with corresponding CARLA GNSS sensor noise and a sampling
        weight. At each reset(), a tier is drawn from this weighted distribution
        and the GNSS sensor is spawned with the corresponding noise multiplier.
        """
        profiles_path = Path(path)
        if not profiles_path.is_absolute():
            profiles_path = Path.cwd() / profiles_path

        with open(profiles_path, "r") as f:
            data = yaml.safe_load(f)

        tiers = data.get("tiers", {})
        if not tiers:
            logger.warning(
                "No GNSS noise tiers found in %s; using multiplier=1.0.",
                profiles_path,
            )
            return

        self._gnss_noise_tiers = []
        raw_weights: List[float] = []
        for name, tier in tiers.items():
            tier["name"] = name
            self._gnss_noise_tiers.append(tier)
            raw_weights.append(float(tier.get("weight", 1.0)))

        # Normalise and store as ndarray so np_random.choice needs no conversion.
        weights_arr = np.array(raw_weights, dtype=np.float64)
        total = float(weights_arr.sum())
        if total > 0:
            weights_arr /= total
        self._gnss_tier_weights = weights_arr

        tier_names = [t["name"] for t in self._gnss_noise_tiers]
        logger.info(
            "Loaded %d GNSS noise tiers: %s",
            len(self._gnss_noise_tiers),
            tier_names,
        )

    def _sample_gnss_noise_tier(self) -> None:
        """
        @brief Sample a GNSS noise tier for the current episode.

        When gnss_noise_multiplier_override is set (evaluation mode),
        bypasses random tier sampling and uses the fixed multiplier.
        """
        if self._gnss_noise_multiplier_override is not None:
            self._current_gnss_multiplier = self._gnss_noise_multiplier_override
            self._current_gnss_tier = None
            logger.info(
                "Episode %d: GNSS multiplier override=%.1f",
                self._episode_id,
                self._current_gnss_multiplier,
            )
            return

        if len(self._gnss_noise_tiers) == 0:
            self._current_gnss_multiplier = 1.0
            self._current_gnss_tier = None
            return

        # Always use that tier instead of sampling from the weight distribution.
        if self._fixed_gnss_tier is not None:
            tier = next(
                (
                    t
                    for t in self._gnss_noise_tiers
                    if t.get("name") == self._fixed_gnss_tier
                ),
                None,
            )
            if tier is None:
                available = [t.get("name", "") for t in self._gnss_noise_tiers]
                raise RuntimeError(
                    f"fixed_gnss_tier='{self._fixed_gnss_tier}' not in "
                    f"loaded tiers: {available}"
                )
            self._current_gnss_tier = tier
        else:
            idx = self.np_random.choice(
                len(self._gnss_noise_tiers),
                p=self._gnss_tier_weights,
            )
            tier = self._gnss_noise_tiers[idx]
            self._current_gnss_tier = tier

        # Multiplier = tier metric stddev / base RTK-fixed stddev.
        # The base GNSS sensor noise in env_config.yaml corresponds to
        # RTK-fixed (~0.02 m).
        base_stddev = 0.02  # RTK-fixed base (metres)
        tier_stddev = float(tier.get("metric_stddev_m", base_stddev))
        self._current_gnss_multiplier = max(1.0, tier_stddev / base_stddev)

        logger.info(
            "Episode %d: GNSS tier '%s' (%.2f m stddev, multiplier=%.1f)",
            self._episode_id,
            tier.get("name", "unknown"),
            tier_stddev,
            self._current_gnss_multiplier,
        )

    def _get_current_gnss_tier(self) -> Optional[Dict[str, Any]]:
        """
        @brief Return the currently sampled GNSS noise tier dict.
        @return Tier dict with lat_stddev_deg, lon_stddev_deg, alt_stddev_m,
                metric_stddev_m, name, weight. None if no tiers loaded.
        """
        return self._current_gnss_tier

    # ------------------------------------------------------------------
    # Floor plan loading and bay sampling
    # ------------------------------------------------------------------

    def _load_floor_plan(self) -> None:
        """
        @brief Select and load a floor plan layout YAML for this episode.
        @see _parking_core.load_floor_plan
        """
        name, layout = load_floor_plan(
            self._floor_plans_config,
            self._eval_mode,
            _shared_layout_cache,
            fixed_name=self._fixed_floor_plan,
        )
        self._current_floor_plan_name = name
        self._current_layout = layout
        logger.info("Loaded floor plan: %s", name)

        primary = layout.get("spawn_transform", {})
        extras: List[Any] = (
            layout.get("extra_spawn_transforms", []) if self._use_extra_spawns else []
        )
        self._spawn_pool = [primary] + list(extras)

        eligible = [
            b for b in layout.get("bays", []) if not b.get("always_empty", False)
        ]
        bays_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for bay in eligible:
            bay_type_key = bay.get("bay_type", "perpendicular")
            bays_by_type.setdefault(bay_type_key, []).append(bay)
        self._bays_by_type = bays_by_type
        self._bay_type_keys = list(bays_by_type.keys())

    def _sample_target_bay(self) -> None:
        """
        @brief Stratified sample of target bay: 1/3 per type, then uniform within type.

        Bay types: perpendicular, angled, parallel.
        Always-empty bays are excluded from target selection.
        """
        if not self._bay_type_keys:
            raise RuntimeError("No eligible bays found in floor plan layout.")

        if self._fixed_target_bay_id is not None:
            target = None
            for bay_type_key, bays in self._bays_by_type.items():
                for bay in bays:
                    if bay.get("id") == self._fixed_target_bay_id:
                        target = bay
                        bay_type = bay_type_key
                        break
                if target is not None:
                    break
            if target is None:
                available_ids = [
                    bay.get("id", "")
                    for bays in self._bays_by_type.values()
                    for bay in bays
                ]
                raise RuntimeError(
                    f"fixed_target_bay_id='{self._fixed_target_bay_id}' "
                    f"not found in floor plan "
                    f"'{self._current_floor_plan_name}'. Available bay ids "
                    f"(first 20): {available_ids[:20]}"
                )
        else:
            bay_type = random.choice(self._bay_type_keys)
            target = random.choice(self._bays_by_type[bay_type])

        tx: float = float(target["x"])
        ty: float = float(target["y"])
        tyaw: float = (
            float(target["yaw"])
            if "yaw" in target
            else math.radians(float(target.get("yaw_deg", 0.0)))
        )

        self._target_bay = {
            "x": tx,
            "y": ty,
            "yaw": tyaw,
            "width": float(target.get("width", 2.5)),
            "depth": float(target.get("depth", 5.0)),
            "bay_type": bay_type,
            "bay_id": target.get("id", target.get("bay_id", "")),
        }

        self._target_x = tx
        self._target_y = ty
        self._target_yaw = tyaw

        logger.info(
            "Target bay: type=%s, id=%s, x=%.1f, y=%.1f",
            bay_type,
            self._target_bay.get("bay_id", ""),
            tx,
            ty,
        )

    # ------------------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------------------

    def _select_spawn(self) -> Dict[str, Any]:
        """
        @brief Choose a spawn transform for this episode from the pre-built pool.
        @return Chosen spawn dict with keys x, y, z, yaw_deg.
        """
        return random.choice(self._spawn_pool)

    def _cache_blueprints(self) -> None:
        """
        @brief Fetch blueprint lists once per reset and distribute to owners.

        LotSpawner receives vehicle blueprints (for static parked cars).
        NPCController receives vehicle + walker blueprints (for patrol + peds).
        """
        if self.world is None:
            return

        # LotSpawner builds its own car blueprint filter internally.
        # Blueprints are static for the lifetime of the CARLA server so both
        # managers skip the fetch on subsequent calls if already populated.
        self._lot_spawner.refresh_blueprints(self.world)

        # Walker blueprints: only fetch once.
        if not self._walker_blueprints:
            bp_lib = self.world.get_blueprint_library()
            self._walker_blueprints = list(bp_lib.filter("walker.pedestrian.*"))

        self._npc_controller.refresh_blueprints(
            self._lot_spawner._car_blueprints, self._walker_blueprints
        )

    def _spawn_lot_statics(self) -> None:
        """
        @brief Spawn all static lot actors (cones and parked vehicles).

        Delegates to LotSpawner.spawn_all().
        """
        self._lot_spawner.spawn_all(
            self.world,
            self._current_layout,
            self._target_bay,
            floor_contact_z=self._floor_z,
            layout_name=self._current_floor_plan_name,
        )

    def _spawn_npc_patrol(self) -> None:
        """
        @brief Spawn scripted patrol vehicles that follow waypoints in the lot.

        Delegates to NPCController.spawn_patrol().
        """
        self._npc_controller.spawn_patrol(
            self.world, self.vehicle, self._current_layout
        )

    def _spawn_pedestrians(self) -> None:
        """
        @brief Spawn random-walk pedestrians inside the lot pedestrian zones.

        Delegates to NPCController.spawn_pedestrians().
        """
        self._npc_controller.spawn_pedestrians(self.world, self._current_layout)

    # ------------------------------------------------------------------
    # Per-step NPC updates
    # ------------------------------------------------------------------

    def _update_patrol_npcs(self) -> None:
        """
        @brief Advance patrol NPC vehicles one step using a proportional heading
               controller.

        Delegates to NPCController.update_patrol().
        """
        self._npc_controller.update_patrol(self.vehicle, self.steps)

    def _update_pedestrians(self) -> None:
        """
        @brief Advance pedestrians one step, re-randomise headings periodically,
               and handle zone boundaries, lifetime expiry, and ego avoidance.

        Delegates to NPCController.update_pedestrians().
        """
        self._npc_controller.update_pedestrians(self.vehicle)

    # ------------------------------------------------------------------
    # Clearance and reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        transform: Optional[Any] = None,
        velocity: Optional[Any] = None,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        """
        @brief Compute reward and termination flags using CARLA ground truth.

        @param transform: Pre-fetched vehicle transform. Fetched internally when None.
        @param velocity: Pre-fetched vehicle velocity. Fetched internally when None.
        @return Tuple of (reward, terminated, success, diagnostics).
        """
        if self.vehicle is None:
            return (
                0.0,
                False,
                False,
                {
                    "pos_error": 0.0,
                    "orientation_error": 0.0,
                    "speed": 0.0,
                    "collision": 0.0,
                    "progress_reward": 0.0,
                },
            )

        if transform is None:
            transform = self.vehicle.get_transform()
        if velocity is None:
            velocity = self.vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        speed = math.hypot(velocity.x, velocity.y)

        position_error = math.hypot(x - self._target_x, y - self._target_y)
        orientation_error = abs(wrap_angle_symmetric(yaw - self._target_yaw))

        diag: Dict[str, float] = {
            "pos_error": position_error,
            "orientation_error": orientation_error,
            "speed": speed,
            "collision": 0.0,
            "progress_reward": 0.0,
        }

        collision_detected, collision_ego_fault = (
            self._sensor_manager.consume_collision()
        )
        if collision_detected:
            self._prev_distance = position_error
            diag["collision"] = 1.0
            # Terminal collision penalty. Lowered from -50/-20 to -25/-10: at
            # -50 the agent became so collision-averse it refused to commit to
            # the final approach near the perimeter and learned to circle
            # instead (run 16052026-1553). -25 still clearly outweighs a
            # successful episode's shaping, keeping the ordering
            # success(+50) > timeout(~0) > collision(-25), but is no longer so
            # punishing that not-approaching beats approaching. Ego-fault
            # (static cone, or a dynamic actor hit while moving) is penalised
            # harder than a non-fault contact.
            reward = -25.0 if collision_ego_fault else -10.0
            return reward, True, False, diag

        in_bay = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
            and speed < SUCCESS_THRESHOLD_VELOCITY
        )
        if in_bay:
            self._success_counter += 1
        else:
            self._success_counter = 0

        if self._success_counter >= self._success_dwell_steps:
            self._prev_distance = position_error
            # Terminal success bonus. Symmetric in magnitude with the ego-fault
            # collision penalty (-50). Must dominate the per-step shaping: with
            # the unbounded final_approach_bonus removed, the per-step potential
            # shaping telescopes to zero over the trajectory, so a one-off +50
            # for success is the largest single reward available - reaching the
            # bay is unambiguously the best outcome.
            return 50.0, True, True, diag

        # Per-step reward: three additive components (Ng et al. 1999 shaping).
        #
        # (1) distance_term: phi(s') - phi(s) with potential phi(r) = -r, the
        #     metres of distance closed this step. The PROGRESS signal:
        #     positive approaching, zero stationary, negative receding. As a
        #     true potential difference it telescopes to zero over any closed
        #     path, so it cannot be farmed by hovering.
        #
        # (2) position_term: small linear penalty proportional to absolute
        #     distance from the bay centre. The PROXIMITY signal: far is bad,
        #     centre is ~0. distance_term alone telescopes to zero and gives no
        #     net reward for *being* close, so on its own it left a flat
        #     per-step landscape and the agent wandered with no gradient (run
        #     16052026-1515: mean_progress ~ 0, pos_error ~ 4 m). position_term
        #     restores a continuous pull toward the bay. It is a bounded linear
        #     PENALTY (negative everywhere except r=0) so, unlike the removed
        #     quadratic final_approach_bonus, it cannot be farmed - it is only
        #     minimised by reaching the centre and finishing.
        #     coefficient 0.005 (see the lowering rationale at the term itself
        #     below): weak enough that timing out near the bay no longer costs
        #     more than crashing.
        #
        # (3) orientation_term: penalty proportional to absolute yaw error
        #     from the bay's target yaw. Drives the agent to arc toward the
        #     bay alignment rather than drive straight at it (the car cannot
        #     reverse - it must approach with the bay's yaw or it cannot
        #     enter). coefficient 0.2 (raised from 0.02): -0.31/step at 90 deg.
        #     At 0.02 it was ~6x weaker than position_term, so the agent
        #     solved (x, y) and ignored heading - it reached the bay area but
        #     never rotated into the bay (runs up to 16052026-1553). At 0.2 it
        #     is comparable to position_term, forcing an arcing approach that
        #     rotates into the bay yaw while closing distance.
        #
        # (4) uncertainty_scale gate: per-step SHAPING (terms 1-3) is multiplied
        #     by (1 - uncertainty_scale). Under high EKF covariance the per-step
        #     gradient shrinks to zero, so the policy is not pushed around by
        #     noisy localisation estimates. Terminal events (success, collision)
        #     and the approach_term (5) bypass this gate.
        #
        # (5) approach_term: a bounded reward PEAK co-located with the bay.
        #     Terms 1-3 are all monotonic in progress - they have no maximum,
        #     so "stop at the bay" is never the optimal action and the agent
        #     learns to drive straight through and crash into the perimeter
        #     (runs up to 17052026-0824: pos_error flatlined at ~3.3 m, 100%
        #     collision, 0% success). approach_term installs the missing
        #     maximum: proximity (0 at the radius, 1 at the bay centre) times
        #     slowness (0 at top speed, 1 stopped) is maximised ONLY when the
        #     car is both at the bay AND stopped - exactly the success state.
        #     Overshooting the bay now sacrifices reward, which makes braking
        #     optimal without a separate speed penalty. Bounded at 0.3/step and
        #     zero beyond success_approach_radius so it cannot be farmed by
        #     circling (the failure mode of the removed quadratic
        #     final_approach_bonus, run 16052026-1332). Un-gated by
        #     uncertainty_scale: "near the bay and slow" is a geometric fact,
        #     not a noisy progress estimate, so it belongs with the terminals.
        progress = self._prev_distance - position_error
        self._prev_distance = position_error

        distance_term = progress
        orientation_term = -0.2 * orientation_error
        # coefficient 0.005 (lowered from 0.05). At 0.05 the per-step proximity
        # penalty made stopping short of the bay and timing out (~337 idle
        # policy steps x -0.165/step ~ -60) cost FAR more than an immediate
        # ego-fault crash (-25), so the optimal policy was to drive straight
        # through the bay and crash quickly to end the episode - exactly the
        # observed failure (runs up to 17052026-0852: pos_error flatlined
        # ~3.3 m, 100% collision, 0% success). The car CAN brake and turn in
        # (confirmed by manual dryrun) - it was rewarded for not doing so.
        # At 0.005 a stop-and-timeout episode costs ~-12, which now beats the
        # -25 crash, so braking near the bay becomes the optimal action. The
        # term still provides a continuous proximity gradient (its original
        # purpose - it was added to stop the agent wandering in a flat
        # landscape, run 16052026-1515), just an order of magnitude weaker so
        # it no longer poisons the brake-vs-crash terminal trade-off.
        position_term = -0.005 * position_error

        # @note An attempt to strengthen this term (run 17052026-1430:
        # coefficient 0.3 -> 0.6, proximity squared, radius 2 m -> 3 m) made
        # training WORSE - success fell from ~0.07 to ~0.01. The stronger
        # slowness-weighted term became a farmable loiter subsidy: crawling
        # slowly anywhere within the radius banked a safe steady positive
        # reward until timeout, so the policy stopped pushing into the success
        # window. Reverted to the run-17052026-1246 form (coefficient 0.3,
        # linear proximity, radius 2 m). Do not strengthen a
        # slowness-weighted shaping term to fix a final-precision gap - it
        # rewards the hovering that IS the problem.
        #
        # @note Slowness reference speed lowered from max_ego_speed_ms (8.0 m/s)
        # to 5 * SUCCESS_THRESHOLD_VELOCITY (1.5 m/s) on 21-05-2026 after run
        # 20052026-1125 demo showed the policy reaching the bay (min pos_error
        # 0.03 m, ep 6) but coasting through at 0.5-0.8 m/s with brake at 0 in
        # all 16 episodes (max brake 0.06, mean 0.00). With the 8.0 m/s
        # reference, slowness at 0.7 m/s was 0.91 vs 1.0 stopped - a 9 percent
        # gap that gave the policy no reason to learn the brake axis. Keyed to
        # 5 * SUCCESS_THRESHOLD_VELOCITY (1.5 m/s) the gap becomes: stopped
        # -> 1.0, success-threshold speed (0.3 m/s) -> 0.8, observed coasting
        # speed (0.75 m/s) -> 0.5, 1.5 m/s -> 0.0. The 5x multiplier keeps the
        # term positive across the speed band the policy actually operates in
        # near the bay (0-1.5 m/s) while making braking distinctly more
        # reward-positive than coasting. The loiter-subsidy failure mode of
        # run 17052026-1430 came from a stronger COEFFICIENT plus squared
        # proximity plus a wider radius - this change touches none of those;
        # it sharpens the slowness curve so the bay itself remains the unique
        # reward peak.
        #
        # @note Alignment gate added 22-05-2026 after run 21052026-2200 demo
        # showed the policy freezing 12-15 m short of the bay in 15 of 16
        # episodes (brake + throttle co-activated; pos_error never closed). The
        # sharpened slowness curve made "slow" a powerful local reward, and the
        # value function bootstrapped that backward in space - low-speed states
        # everywhere became valuable because they correlated with success-window
        # proximity, even though `proximity` is zero outside the 2 m radius.
        # The alignment factor (1 at zero yaw error, 0 at >= SUCCESS_THRESHOLD_
        # ORIENTATION) only collects the slowness bonus when the car is BOTH
        # near the bay AND pointed into it. A random freeze far from the bay
        # has random heading, so alignment kills the bonus and the value-
        # function backward-bootstrap loses its source. Stopping cleanly in the
        # bay (small position_error, small orientation_error, low speed) still
        # collects the full bonus - the success state remains the unique peak.
        # @note Outer-zone pull added 22-05-2026 after run 21052026-2342 demo
        # showed the deterministic mean policy stopping at 7.5-10 m in all 16
        # episodes - just outside the 5 m approach_radius. During training, 23
        # rollout updates achieved success (stochastic action noise teleported
        # the car past the boundary), but PPO's policy gradient could not push
        # the mean past it because the gradient at the boundary is a cliff
        # (zero outside, +0.3/step inside). Replacing the cliff with a gentle
        # ramp in a 5-10 m annulus gives the mean a continuous slope to follow.
        # The outer-zone term has NO slowness or alignment factors - it is a
        # directional gradient (reward grows toward the boundary), not a stop-
        # here bonus. The maximum it can pay if the policy farmed it by sitting
        # at the boundary is 1750 * 0.02 = +35 reward per episode, less than
        # success (+50) and less than the inside-radius bonus the policy gets
        # from actually entering the bay. Real-lot generalisation: the gradient
        # is anchored on the TARGET bay's position_error - in a multi-bay
        # layout the outer-zone pull still points at the target, with no
        # special pull on neighbour bays.
        slowness_reference_speed = 5.0 * SUCCESS_THRESHOLD_VELOCITY
        approach_term = 0.0
        if position_error < self._success_approach_radius:
            proximity = 1.0 - position_error * self._inv_approach_radius
            slowness = max(0.0, 1.0 - speed / slowness_reference_speed)
            alignment = max(
                0.0, 1.0 - orientation_error / SUCCESS_THRESHOLD_ORIENTATION
            )
            approach_term = 0.3 * proximity * slowness * alignment
        elif position_error < 2.0 * self._success_approach_radius:
            outer_proximity = (
                1.0
                - (position_error - self._success_approach_radius)
                * self._inv_approach_radius
            )
            approach_term = 0.02 * outer_proximity

        # @note Co-activation penalty added 22-05-2026 after run 21052026-2200
        # demo showed the policy locking throttle ~0.55 and brake ~0.77
        # simultaneously for hundreds of consecutive steps (15 of 16 episodes,
        # ep 1 never moved at all). Real cars cannot apply both at once
        # (mechanical interlock + ECU veto); CARLA's VehicleControl allows it,
        # so the reward needs to encode the physical constraint. The policy
        # used the combination to "stay still" without committing to either
        # action - a stable region of the action space that should not exist.
        # min(throttle, brake) is zero when one axis is at zero and grows
        # linearly when both are pressed. Coefficient 0.05: 500 steps of full
        # co-activation costs ~12.5 reward (more than a per-step shaping
        # episode is worth, less than collision -25 or success +50). Clamps
        # match step() (throttle and brake clipped to [0, 1]).
        throttle_applied = float(np.clip(self._last_action[1], 0.0, 1.0))
        brake_applied = float(np.clip(self._last_action[2], 0.0, 1.0))
        co_activation_penalty = -0.05 * min(throttle_applied, brake_applied)

        # @note Idle-steer penalty added 22-05-2026 after run 21052026-2342
        # demo showed the policy steering 0.4-0.95 for ~1500 of 1750 steps in
        # every episode AFTER coming to a stop. CARLA ignores wheel angle on a
        # stationary vehicle, but in real-world deployment this is unwanted
        # actuation (rack wear, alarming behaviour to observers). The reward
        # function had no term penalising steering when not moving, so the
        # policy was free to output whatever it wanted on the steer axis. The
        # stationary scale ramps from 1 at zero speed to 0 at the success-
        # threshold velocity (0.3 m/s) and stays at 0 above that, so this
        # penalty is invisible during normal driving and only kicks in below
        # the success-window speed - exactly where the wiggle is happening.
        # Coefficient -0.02: 1500 idle steps at full deflection costs ~30
        # reward, comparable to the collision penalty.
        steer_applied = float(np.clip(self._last_action[0], -1.0, 1.0))
        stationary_scale = max(0.0, 1.0 - speed / SUCCESS_THRESHOLD_VELOCITY)
        idle_steer_penalty = -0.02 * abs(steer_applied) * stationary_scale

        uncertainty_scale = self._uncertainty_scale_fn()
        reward = (distance_term + orientation_term + position_term) * (
            1.0 - uncertainty_scale
        ) + approach_term + co_activation_penalty + idle_steer_penalty

        diag["progress_reward"] = float(progress)
        diag["uncertainty_scale"] = uncertainty_scale
        diag["orientation_penalty"] = float(orientation_term)
        diag["position_penalty"] = float(position_term)
        diag["approach_reward"] = float(approach_term)
        diag["co_activation_penalty"] = float(co_activation_penalty)
        diag["idle_steer_penalty"] = float(idle_steer_penalty)
        return float(reward), False, False, diag

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        @brief Build the observation vector from EKF pose and sim sensors.

        Resolves EKF odom -> world transform (sim-specific), then delegates
        obs construction to build_observation() in _parking_core.

        Falls back to CARLA ground truth when EKF is unavailable (CI/tests).
        """
        if self.vehicle is None or self.world is None:
            return np.zeros(self._obs_dim, dtype=np.float32)

        # Read EKF pose + uncertainty in one file read (no-op when EKF absent).
        raw_ekf_pose, uncertainty = self._read_ekf_state()

        # Resolve EKF odom pose -> world frame.
        world_pose: Optional[np.ndarray] = None
        if raw_ekf_pose is not None:
            ekf_odom_x = float(raw_ekf_pose[0])
            # raw_ekf_pose[1] is already in CARLA convention (extractor negates
            # ROS y to CARLA y on write). No further negation needed.
            ekf_odom_y = float(raw_ekf_pose[1])
            ekf_odom_yaw = float(raw_ekf_pose[2])
            tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
            self._world_pose_buf[0] = cos_r * ekf_odom_x - sin_r * ekf_odom_y + tx
            self._world_pose_buf[1] = sin_r * ekf_odom_x + cos_r * ekf_odom_y + ty
            self._world_pose_buf[2] = ekf_odom_yaw + r
            self._world_pose_buf[3] = raw_ekf_pose[3]
            # vx is body-frame already (EKF publishes twist in base_link);
            # frame-invariant w.r.t. the odom->world rigid transform.
            self._world_pose_buf[4] = raw_ekf_pose[4]
            world_pose = self._world_pose_buf
        else:
            # CI / tests fallback: use CARLA GT (no EKF available)
            self._debug_logger._logger.debug(
                "[state] EKF pose unavailable at step %d - using CARLA GT",
                self.steps,
            )
            t = self.vehicle.get_transform()
            av = self.vehicle.get_angular_velocity()
            gt_vel = self.vehicle.get_velocity()
            yaw_rad = math.radians(t.rotation.yaw)
            # Rotate world-frame velocity into body frame for signed vx.
            vx_body = gt_vel.x * math.cos(yaw_rad) + gt_vel.y * math.sin(yaw_rad)
            self._world_pose_buf[0] = t.location.x
            self._world_pose_buf[1] = t.location.y
            self._world_pose_buf[2] = yaw_rad
            self._world_pose_buf[3] = math.radians(av.z)
            self._world_pose_buf[4] = vx_body
            world_pose = self._world_pose_buf

        obstacle_features = extract_obstacle_features(
            self._get_lidar_scan(),
            self._obstacle_features_buffer,
        )

        return build_observation(
            world_pose,
            uncertainty,
            self._target_bay,
            obstacle_features,
            self._include_covariance,
            self._include_obstacle_obs,
            self._obs_buffer,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _write_vis_state(
        self,
        end_reason: Optional[str] = None,
        transform: Optional[Any] = None,
        velocity: Optional[Any] = None,
    ) -> None:
        """
        @brief Append a visualisation frame to the JSONL history file.

        Writing only occurs when the signal file (outputs/.vis_active) exists,
        which is created by the visualiser process. A persistent
        append file handle is kept open while the visualiser is active.

        @param end_reason: If set, written into the frame as "end_reason" so
                           the visualiser can log why the episode terminated.
                           One of: "collision", "success", "timeout".
                           None for mid-episode frames.
        @param transform: Pre-fetched vehicle transform. Fetched internally when None.
        @param velocity: Pre-fetched vehicle velocity. Fetched internally when None.
        """
        if self.vehicle is None:
            return

        # Refresh the signal-file check every 30 calls.
        self._vis_check_counter += 1
        if self._vis_check_counter >= 30:
            self._vis_check_counter = 0
            new_active = self._vis_signal_path.exists()
            if not new_active and self._vis_file is not None:
                self._vis_file.close()
                self._vis_file = None
            self._vis_active = new_active

        if not self._vis_active:
            return

        if transform is None:
            transform = self.vehicle.get_transform()
        if velocity is None:
            velocity = self.vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw

        # Ground-truth yaw rate in ROS REP-103 convention (left turn =
        # positive, right turn = negative). This matches obs[1]'s convention
        # (the extractor leaves the EKF vyaw un-negated) and the LiDAR
        # bearing sign convention (left obstacles have positive bearings).
        # CARLA's get_angular_velocity().z is in deg/s in CARLA frame where
        # left = negative, so we negate to bring it into the ROS convention.
        gt_av_z_deg = self.vehicle.get_angular_velocity().z
        gt_vyaw = -math.radians(gt_av_z_deg)

        # EKF speed and vyaw are the first two slots of the obs buffer, which
        # has already been populated this step by _get_state() (called by the
        # outer step()/reset() before _write_vis_state). Reading from the buffer
        # avoids a second covariance-subscriber file read.
        ekf_speed = float(self._obs_buffer[0])
        ekf_vyaw = float(self._obs_buffer[1])

        tier_info = self._get_current_gnss_tier()
        gnss_tier_name = str(tier_info.get("name", "")) if tier_info else ""

        patrol_npcs = self._npc_controller.patrol_npcs
        # Build an id-set for O(1) membership test in the actor loop.
        patrol_npc_ids = {id(a) for a in patrol_npcs}
        actor_transforms = []
        for actor in patrol_npcs + self._lot_spawner.spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                at = actor.get_transform()
                actor_transforms.append(
                    {
                        "x": at.location.x,
                        "y": at.location.y,
                        "yaw": at.rotation.yaw,
                        "type": "npc" if id(actor) in patrol_npc_ids else "static",
                    }
                )

        pedestrian_transforms = []
        for walker in self._npc_controller.pedestrian_actors:
            if walker is not None and walker.is_alive:
                wt = walker.get_transform()
                pedestrian_transforms.append({"x": wt.location.x, "y": wt.location.y})

        # Read CARLA world clock for diagnostics
        carla_elapsed = 0.0
        carla_sync = False
        carla_fixed_dt = 0.0
        if self.world is not None:
            snap = self.world.get_snapshot()
            carla_elapsed = snap.timestamp.elapsed_seconds
            ws = self.world.get_settings()
            carla_sync = ws.synchronous_mode
            carla_fixed_dt = ws.fixed_delta_seconds

        state: Dict[str, Any] = {
            "sim_time": self.steps * self._carla_timestep,
            "carla_time": carla_elapsed,
            "carla_sync": carla_sync,
            "carla_fixed_dt": carla_fixed_dt,
            "episode_id": self._episode_id,
            "episode_step": self.steps,
            "carla_timestep": self._carla_timestep,
            "ego": {
                "x": x,
                "y": y,
                "yaw": yaw,
                "vx": velocity.x,
                "vy": velocity.y,
                "speed": math.hypot(velocity.x, velocity.y),
                "gt_vyaw": gt_vyaw,
                "ekf_speed": ekf_speed,
                "ekf_vyaw": ekf_vyaw,
                "gnss_tier": gnss_tier_name,
            },
            # Applied (clamped) action - what the vehicle actually receives,
            # not the policy's raw pre-clip output. steer is clipped to
            # [-1, 1]; throttle and brake to [0, 1] (same clamps as step()).
            # The visualiser HUD shows these, so it reflects vehicle state -
            # showing the raw output would imply the car is braking when a
            # negative raw brake is clamped to 0.
            "action": {
                "steer": float(np.clip(self._last_action[0], -1.0, 1.0)),
                "throttle": float(np.clip(self._last_action[1], 0.0, 1.0)),
                "brake": float(np.clip(self._last_action[2], 0.0, 1.0)),
            },
            "trajectory": list(self._trajectory_buffer),
            "actors": actor_transforms,
            "pedestrians": pedestrian_transforms,
            "target_bay": self._target_bay,
            "floor_plan": self._current_floor_plan_name,
            "bays": self._current_layout.get("bays", []),
            "corners": self._current_layout.get("corners", []),
        }
        if end_reason is not None:
            state["end_reason"] = end_reason

        debug_dict = self._debug_logger.step_debug_dict()
        if debug_dict:
            state["debug"] = debug_dict

        try:
            if self._vis_file is None:
                self._vis_history_path.parent.mkdir(parents=True, exist_ok=True)
                self._vis_file = open(self._vis_history_path, "a")
            self._vis_file.write(json.dumps(state))
            self._vis_file.write("\n")
            self._vis_file.flush()
        except Exception as exc:
            # Non-fatal - visualisation is optional
            logger.debug(f"Could not write vis state: {exc}")

    # ------------------------------------------------------------------
    # CARLA world management
    # ------------------------------------------------------------------

    def _connect_to_carla(self) -> None:
        """
        @brief Connect to CARLA and load the FlatPlane OpenDRIVE world.

        Loads configs/layouts/flat_plane.xodr via generate_opendrive_world()
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(120.0)
            self.world = self.client.get_world()

            current_map_name = self.world.get_map().name.split("/")[-1]
            if current_map_name != "FlatPlane":
                xodr = Path("configs/layouts/flat_plane.xodr").read_text(
                    encoding="utf-8"
                )
                logger.info("Loading configs/layouts/flat_plane.xodr ...")
                self.world = self.client.generate_opendrive_world(
                    xodr,
                    carla.OpendriveGenerationParameters(
                        vertex_distance=2.0,
                        max_road_length=600.0,
                        wall_height=0.0,
                        additional_width=300.0,
                        smooth_junctions=False,
                        enable_mesh_visibility=True,
                        enable_pedestrian_navigation=False,
                    ),
                )
                time.sleep(self._map_load_sleep)
            else:
                logger.info("FlatPlane already loaded.")

            # FlatPlane does not render weather; ClearNoon keeps the scene lit.
            if self.world is not None:
                self.world.set_weather(carla.WeatherParameters.ClearNoon)

        except Exception as exc:
            logger.error(f"Could not connect to CARLA: {exc}")
            self.client = None
            self.world = None

    def _spawn_vehicle(self) -> None:
        """
        @brief Spawn the ego vehicle at a randomly selected spawn transform.

        All spawn transforms (primary + extra_spawn_transforms) are collected
        and one is chosen uniformly at random each episode to ensure the agent
        learns to park from varied entry angles and distances.
        """
        if self.world is None:
            return

        default_z = float(self._current_layout.get("origin", {}).get("z", 0.3))
        # _chosen_spawn is pre-selected in reset() before ticks so the GNSS
        # datum config is written before any GNSS callbacks fire.
        chosen = self._chosen_spawn
        sx = float(chosen.get("x", 0.0))
        sy = float(chosen.get("y", 0.0))
        sz = float(chosen.get("z", default_z))
        syaw = float(chosen.get("yaw_deg", 0.0))

        if self._vehicle_bp is None:
            bp_lib = self.world.get_blueprint_library()
            self._vehicle_bp = bp_lib.filter("vehicle.bmw.grandtourer")[0]
            # The ROS bridge identifies the ego vehicle by role_name and publishes
            # sensor data under /carla/ego_vehicle/* for the EKF.
            self._vehicle_bp.set_attribute("role_name", "ego_vehicle")
        vehicle_bp = self._vehicle_bp

        spawn_transform = carla.Transform(
            carla.Location(x=sx, y=sy, z=sz),
            carla.Rotation(yaw=syaw),
        )

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            logger.error("Ego vehicle could not be spawned at lot spawn point.")

        if self.vehicle is not None and self.world is not None:
            # Tick until the vehicle settles onto the ground plane.
            # In synchronous mode, time.sleep() does not advance physics -
            # world.tick() is required.  Never toggle set_simulate_physics on
            # the ego vehicle: in CARLA 0.9.16 that locks the drivetrain so
            # the wheels steer but the vehicle cannot translate.
            for _ in range(40):
                self.world.tick(10.0)
                if abs(self.vehicle.get_velocity().z) < 0.01:
                    break
            # Record the true floor z so static spawners use the same reference
            # instead of a hardcoded layout_origin_z + guess offset.
            self._floor_z = self.vehicle.get_transform().location.z

    def _teleport_vehicle(self) -> None:
        """
        @brief Teleport the existing ego vehicle to the chosen spawn point.

        Used instead of destroy+respawn to keep sensors alive across episodes,
        which prevents the CARLA ROS bridge from accumulating actor-stream
        registrations and eventually segfaulting mid-training.

        Zeros linear and angular velocity so the vehicle starts stationary,
        then ticks until physics settle (same condition as _spawn_vehicle).
        """
        if self.vehicle is None or self.world is None:
            return

        self.vehicle.disable_constant_velocity()

        default_z = float(self._current_layout.get("origin", {}).get("z", 0.3))
        # _chosen_spawn is pre-selected in reset() before ticks so the GNSS
        # datum config is written before any GNSS callbacks fire.
        chosen = self._chosen_spawn
        sx = float(chosen.get("x", 0.0))
        sy = float(chosen.get("y", 0.0))
        sz = float(chosen.get("z", default_z))
        syaw = float(chosen.get("yaw_deg", 0.0))

        spawn_transform = carla.Transform(
            carla.Location(x=sx, y=sy, z=sz),
            carla.Rotation(yaw=syaw),
        )
        self.vehicle.set_transform(spawn_transform)

        # Zero velocity so the vehicle does not carry momentum from the
        # previous episode into the new one.
        zero = self._zero_vec3
        self.vehicle.set_target_velocity(zero)
        self.vehicle.set_target_angular_velocity(zero)
        self.vehicle.apply_control(carla.VehicleControl())

        # Settle under gravity (same loop as _spawn_vehicle).
        for _ in range(40):
            self.world.tick(10.0)
            if abs(self.vehicle.get_velocity().z) < 0.01:
                break
        self._floor_z = self.vehicle.get_transform().location.z

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn sensors (IMU, GNSS, 2D LiDAR, collision) on the ego vehicle.

        Delegates to SensorManager.spawn(), passing the current episode's GNSS
        noise multiplier (sampled in reset()) and the NPC controller's
        patrol_npc_ids set by reference so the collision callback can identify
        patrol vehicles.
        """
        self._sensor_manager.spawn(
            self.world,
            self.vehicle,
            self._npc_controller.patrol_npc_ids,
        )

    def _wait_for_covariance(self) -> None:
        """
        @brief Block until LiDAR scan and EKF state are both available.
        @see _parking_core.wait_for_ekf
        """
        if self._cov_subscriber is None:
            return

        def _tick_fn() -> None:
            assert self.world is not None
            self.world.tick(10.0)

        tick_fn = _tick_fn if self.world is not None else None
        wait_for_ekf(
            has_lidar=lambda: self._sensor_manager.get_latest_lidar_scan() is not None,
            has_ekf=lambda: self._cov_subscriber.has_data,  # type: ignore[union-attr]
            timeout=self._covariance_timeout,
            tick_fn=tick_fn,
        )

    def _calibrate_ekf_frame_offset(self) -> None:
        """
        @brief Set the EKF odom-to-world transform: spawn translation, no rotation.

        The GNSS datum is latched to the spawn position each episode reset, so
        the EKF odom origin is at the spawn point in world coordinates.
        """
        spawn_x = float(self._chosen_spawn.get("x", 0.0))
        spawn_y = float(self._chosen_spawn.get("y", 0.0))
        self._ekf_odom_offset = (spawn_x, spawn_y, 1.0, 0.0, 0.0)

    def _freeze_all_actors(self) -> None:
        """
        @brief Zero velocity on all moving actors immediately on episode end.

        Called as soon as terminated or truncated is True so actors do not
        continue on their last command while reset() tears down the episode.
        Uses set_target_velocity for instant stops rather than brake control,
        which takes multiple ticks to converge through physics.
        """
        zero = self._zero_vec3

        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.enable_constant_velocity(zero)

        for npc in filter(
            lambda a: a is not None and a.is_alive,
            self._npc_controller.patrol_npcs,
        ):
            # NPCs are destroyed in cleanup so the pin does not carry over.
            npc.enable_constant_velocity(zero)

        for walker in filter(
            lambda w: w is not None and w.is_alive,
            self._npc_controller.pedestrian_actors,
        ):
            walker.apply_control(self._zero_walker_ctrl)

    def _cleanup_actors(self, skip_ego: bool = False) -> None:
        """
        @brief Destroy episode actors.

        @param skip_ego: When True, skip sensor and ego vehicle destruction so
               they can be reused via _teleport_vehicle() in the next episode.
               NPCs, cones, and static vehicles are always destroyed.
        """
        if not skip_ego:
            self._sensor_manager.cleanup()

        self._npc_controller.cleanup()
        self._lot_spawner.cleanup()

        self._all_vehicle_actors.clear()

        if not skip_ego:
            if self.vehicle is not None:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
                self.vehicle = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment for a new episode.
        @param seed: Random seed for reproducibility.
        @param options: Additional options (currently unused).
        @return Tuple of (initial_observation, info_dict).

        Each reset: cleans up previous episode, selects floor plan, samples target
        bay, spawns cones/static vehicles/patrol NPCs/pedestrians.
        """
        super().reset(seed=seed)

        self._episode_id += 1
        self.steps = 0
        self._trajectory_buffer.clear()
        # Force signal-file recheck at episode start so new episodes don't
        # inherit a stale cached value from the previous episode's final step.
        self._vis_check_counter = 30

        # Every _vis_rotation_interval episodes, close the vis file and reopen
        # in write mode to truncate it. The visualiser detects the shrink via
        # the offset-vs-size guard and resets its read pointer to zero.
        self._vis_episodes_since_rotation += 1
        if (
            self._vis_file is not None
            and self._vis_episodes_since_rotation >= self._vis_rotation_interval
        ):
            self._vis_episodes_since_rotation = 0
            self._vis_file.close()
            self._vis_file = open(self._vis_history_path, "w")

        # Sample GNSS noise tier for this episode (RTK fix-state variation).
        # Must happen before _spawn_sensors() so the multiplier is available.
        self._sample_gnss_noise_tier()
        # Draw TiM571 systematic range bias once per training run (NaN sentinel
        # in SensorManager makes subsequent calls no-ops).
        self._sensor_manager.sample_lidar_noise_bias(self.np_random)

        # Connect to CARLA on first reset
        if self.client is None:
            self._connect_to_carla()

        if self.world is None:
            return np.zeros(self._obs_dim, dtype=np.float32), {}

        # Determine whether to reuse the existing vehicle and sensors.
        # On the first episode (vehicle is None) or if the actor has gone stale,
        # do a full spawn.  On all subsequent episodes, teleport instead to avoid
        # the destroy/respawn cycle that causes the CARLA ROS bridge to accumulate
        # actor-stream registrations and eventually segfault (exit code -11).
        reuse_vehicle = self.vehicle is not None and self.vehicle.is_alive

        # Always clean up NPCs, cones, and static vehicles from the previous
        # episode - only skip sensor/ego-vehicle destruction when reusing.
        self._cleanup_actors(skip_ego=reuse_vehicle)

        # Flush pending destroy commands before spawning. Skipped when actors
        # were already frozen at step() termination - no commands in-flight.
        if not self._actors_frozen:
            self.world.tick(10.0)
        self._actors_frozen = False

        # A world reload (generate_opendrive_world / load_world) resets all
        # CARLA settings to async defaults, so sync mode is re-applied here
        # rather than relying on the bridge. Both sync and no_rendering_mode
        # checks share a single get_settings() / apply_settings() pair.
        if self.world is not None:
            settings = self.world.get_settings()
            settings_changed = False

            if not settings.synchronous_mode:
                logger.info(
                    "Applying synchronous mode (fixed_delta=%.3fs) ...",
                    self._carla_timestep,
                )
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self._carla_timestep
                settings_changed = True
            else:
                logger.info(
                    "Synchronous mode already active "
                    f"(fixed_delta={settings.fixed_delta_seconds}s)."
                )

            # Apply no_rendering_mode if enabled. Disables Unreal rendering pipeline
            # but physics and state sensors remain active. For state-based agents
            # (no camera input), this provides 3-4x speedup by skipping GPU rendering.
            if self._no_rendering_mode and not settings.no_rendering_mode:
                logger.info(
                    "Enabling no_rendering_mode (state-based agent, no cameras)..."
                )
                settings.no_rendering_mode = True
                settings_changed = True

            if settings_changed:
                self.world.apply_settings(settings)
                if self._no_rendering_mode:
                    logger.info("No rendering mode enabled. Expected speedup: 3-4x.")

        # Load floor plan and sample target bay
        if self._floor_plans_config:
            self._load_floor_plan()
            self._sample_target_bay()

        # Cache blueprint lists once before any spawning to avoid repeated
        # world queries inside each spawn method.
        self._cache_blueprints()

        # Pre-select spawn and publish GNSS episode config BEFORE any world
        # ticks. The gravity-settle ticks inside _spawn_vehicle/_teleport_vehicle
        # fire GNSS callbacks in the ROS 2 bridge.
        self._chosen_spawn = self._select_spawn()
        sx = float(self._chosen_spawn.get("x", 0.0))
        sy = float(self._chosen_spawn.get("y", 0.0))
        syaw = math.radians(float(self._chosen_spawn.get("yaw_deg", 0.0)))

        if (
            self._include_covariance
            and self._cov_subscriber is not None
            and len(self._gnss_noise_tiers) > 0
        ):
            datum_lat: Optional[float] = None
            datum_lon: Optional[float] = None
            try:
                geo = self.world.get_map().transform_to_geolocation(
                    carla.Location(sx, sy, 0.0)
                )
                datum_lat = float(geo.latitude)
                datum_lon = float(geo.longitude)
            except Exception as exc:
                logger.warning(f"Failed to get spawn geolocation for GNSS datum: {exc}")
            tier = self._get_current_gnss_tier()
            if tier is not None:
                self._cov_subscriber.publish_episode_config(
                    tier_name=str(tier.get("name", "")),
                    datum_lat=datum_lat,
                    datum_lon=datum_lon,
                )

        if reuse_vehicle:
            # Teleport the existing vehicle; sensors stay attached and alive.
            self._teleport_vehicle()
            self._sensor_manager.reset_state()
        else:
            self._spawn_vehicle()
            self._spawn_sensors()

        # Invalidate stale pre-reset EKF data so _wait_for_covariance() blocks
        # until a genuinely post-spawn reading arrives from ekf_state.json.
        if self._include_covariance and self._cov_subscriber is not None:
            self._cov_subscriber.invalidate()

        # Publish spawn pose as local (0, 0, yaw) so /set_pose seeds the EKF
        # at local origin, matching the re-latched GNSS datum frame.
        if (
            self._include_covariance
            and self._cov_subscriber is not None
            and self._ros2_config.get("publish_initial_pose", False)
        ):
            self._cov_subscriber.publish_initial_pose(0.0, 0.0, syaw)

        # Spawn all static lot actors (cones + parked vehicles) then NPCs
        self._spawn_lot_statics()
        self._spawn_npc_patrol()
        self._spawn_pedestrians()

        # Rebuild vehicle actor cache after all vehicles are spawned so
        # update_patrol() can use it without a per-step world query.
        if self.world is not None:
            self._all_vehicle_actors = list(self.world.get_actors().filter("vehicle.*"))
            self._npc_controller.set_vehicle_cache(self._all_vehicle_actors)

        if self._include_covariance:
            # First wait: blocks until the extractor has written any
            # post-invalidation state (ensures the ROS 2 bridge is alive).
            self._wait_for_covariance()
            # Give the extractor's 10 Hz file-watcher time to detect
            # initial_pose.json (up to 100 ms) and the EKF time to process
            # the resulting /set_pose before we sample its state.
            # 10 ticks at 20 Hz = 500 ms - comfortably covers the poll
            # interval plus one EKF prediction cycle.
            if self.world is not None:
                for _ in range(10):
                    self.world.tick(10.0)
            # Second invalidate + wait: the seq barrier is now set after
            # /set_pose has been consumed, so we only accept EKF state that
            # was written after the reset completed.
            if self._cov_subscriber is not None:
                self._cov_subscriber.invalidate()
            self._wait_for_covariance()
            # Always recalibrate: the GNSS datum is re-latched to the spawn
            # position at each episode reset, so the EKF odom origin shifts
            # every episode.
            self._calibrate_ekf_frame_offset()

        # Initialise prev_distance for potential-based reward shaping
        self._success_counter = 0
        if self.vehicle is not None:
            t = self.vehicle.get_transform()
            self._prev_distance = math.hypot(
                t.location.x - self._target_x,
                t.location.y - self._target_y,
            )
        else:
            self._prev_distance = 0.0

        state = self._get_state()

        # Emit debug reset summary (no-op when debug=False)
        self._debug_logger.log_reset(
            self._current_floor_plan_name,
            self._target_bay.get("bay_id", ""),
            sx,
            sy,
        )
        self._debug_logger.log_actors(
            n_static=len(self._lot_spawner.spawned_static_vehicles),
            n_patrol=len(self._npc_controller.patrol_npcs),
            n_peds=len(self._npc_controller.pedestrian_actors),
            n_cones=len(self._lot_spawner.spawned_cones),
        )

        # Write the first vis frame for this episode so the visualiser shows
        # the new layout immediately rather than displaying the previous
        # episode's stale scene during the reset gap.
        self._write_vis_state()

        info: Dict[str, Any] = {
            "floor_plan": self._current_floor_plan_name,
            "target_bay_id": self._target_bay.get("bay_id", ""),
            "target_bay_type": self._target_bay.get("bay_type", ""),
            "episode": {"r": 0.0, "l": 0},
        }

        return state, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Execute one environment step with optional action_repeat.

        When action_repeat > 1, the same action is applied for multiple sim-steps.
        Observations are only constructed on the final step of the repeat sequence,
        reducing EKF covariance reads and state construction by action_repeat factor.

        @param action: 2-dim action vector [steering, drive].
                steering in [-1, 1]: left to right.
                drive    in [-1, 1]: positive = forward throttle, negative = brake.
                                     No reverse gear engaged.
                Mapped to CARLA throttle/brake internally.
        @return Tuple of (observation, reward, terminated, truncated, info).
        """
        # On a new action (or first call), reset the repeat counter
        if self._action_repeat_counter == 0:
            self._last_action[:] = action

        # Execute one sim-step
        self.steps += 1
        self._action_repeat_counter += 1

        # post-tick transform and velocity (fetched once, shared with
        # _compute_reward and _write_vis_state to avoid duplicate CARLA RPCs).
        _post_transform: Optional[Any] = None
        _post_velocity: Optional[Any] = None

        if self.vehicle is not None:
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], 0.0, 1.0))
            brake = float(np.clip(action[2], 0.0, 1.0))

            vel = self.vehicle.get_velocity()
            current_speed = math.hypot(vel.x, vel.y)

            control = carla.VehicleControl()
            control.steer = steer
            control.reverse = False
            # Throttle and brake are independent axes. Cut throttle when the
            # speed limit is exceeded; brake is applied as commanded. The two
            # may be non-zero at once (CARLA resolves throttle-vs-brake) but
            # the policy is free to learn pure-brake / pure-throttle.
            control.throttle = (
                0.0 if current_speed >= self._max_ego_speed_ms else throttle
            )
            control.brake = brake

            self.vehicle.apply_control(control)

            if self.world is not None:
                self._update_patrol_npcs()
                self._update_pedestrians()
                # 10s timeout surfaces a frozen CARLA server as an error rather
                # than hanging the process indefinitely.
                self.world.tick(10.0)

                # Fetch transform + velocity once post-tick; reused by
                # _compute_reward and _write_vis_state below.
                _post_transform = self.vehicle.get_transform()
                _post_velocity = self.vehicle.get_velocity()
                self._trajectory_buffer.append(
                    (_post_transform.location.x, _post_transform.location.y)
                )

        # Only construct observations and compute rewards on the final repeat step.
        # Intermediate steps return the last-built obs buffer directly.
        if self._action_repeat_counter < self._action_repeat:
            return self._obs_buffer, 0.0, False, False, _EMPTY_STEP_INFO

        # Final repeat step: construct full observation and reward
        self._action_repeat_counter = 0  # Reset for next action
        state = self._get_state()
        reward, terminated, success, reward_diag = self._compute_reward(
            transform=_post_transform, velocity=_post_velocity
        )

        # Per-step debug diagnostics (no-op when debug=False).
        # Reuse values already computed by _compute_reward() to avoid duplicate
        # vehicle.get_transform() / get_velocity() calls on the hot path.
        if self._debug_logger.enabled and self.vehicle is not None:
            _unc: Optional[np.ndarray] = None
            if self._cov_subscriber is not None:
                _, _unc = self._cov_subscriber.get_latest_state()
            _ekf_drift = float(max(_unc[0], _unc[1])) if _unc is not None else 0.0
            _obs_dist = (
                float(state[-OBSTACLE_FEATURES_DIM])
                if self._include_obstacle_obs
                else 0.0
            )
            self._debug_logger.log_step(
                step=self.steps,
                reward=reward,
                pos_error=reward_diag["pos_error"],
                yaw_error=reward_diag["orientation_error"],
                speed=reward_diag["speed"],
                action=self._last_action,
                uncertainty=_unc,
                obstacle_dist=_obs_dist,
                ekf_drift=_ekf_drift,
                lidar_points=self._sensor_manager.lidar_point_count(),
            )

        truncated = self.steps >= self.max_steps

        if (terminated or truncated) and self.world is not None:
            self._freeze_all_actors()
            self._actors_frozen = True

        # Distinguish termination cause for vis state writer.
        # Both collision and OOB set terminated=True; success is the third path.
        if success:
            end_reason: Optional[str] = "success"
        elif terminated:
            end_reason = "collision"
        elif truncated:
            end_reason = "timeout"
        else:
            end_reason = None

        self._write_vis_state(
            end_reason=end_reason,
            transform=_post_transform,
            velocity=_post_velocity,
        )

        info: Dict[str, Any] = {
            "steps": self.steps,
            "success": success,
            "collision": bool(reward_diag["collision"]),
            "timeout": truncated,
            "floor_plan": self._current_floor_plan_name,
            "pos_error": reward_diag["pos_error"],
            "orientation_error": reward_diag["orientation_error"],
            "speed": reward_diag["speed"],
            "progress_reward": reward_diag["progress_reward"],
            "uncertainty_scale": reward_diag.get("uncertainty_scale", 0.0),
            "orientation_penalty": reward_diag.get("orientation_penalty", 0.0),
            "position_penalty": reward_diag.get("position_penalty", 0.0),
        }

        return state, reward, terminated, truncated, info

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """
        @brief Render the environment.
        @return RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human" and self.world is not None:
            if self.vehicle is not None:
                transform = self.vehicle.get_transform()
                spectator = self.world.get_spectator()
                spectator.set_transform(
                    carla.Transform(
                        transform.location + carla.Location(z=50),
                        carla.Rotation(pitch=-90),
                    )
                )
        elif self.render_mode == "rgb_array":
            return cast(RenderFrame, np.zeros((600, 800, 3), dtype=np.uint8))
        return None

    def close(self) -> None:
        """
        @brief Clean up all resources including CARLA actors and ROS 2 nodes.
        """
        self._cleanup_actors()
        # Destroy cached cones that _cleanup_actors() (episode reset) preserves.
        self._lot_spawner.cleanup_all()

        # Restore asynchronous mode so CARLA does not freeze waiting for
        # ticks after the training process exits.
        if self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except Exception:
                pass

        if self._cov_subscriber is not None:
            self._cov_subscriber = None

        if self._vis_file is not None:
            try:
                self._vis_file.close()
            except OSError:
                pass
            self._vis_file = None

        self.client = None
        self.world = None

        # Remove the signal file so the visualiser knows training has ended
        # and can exit cleanly rather than waiting indefinitely for new frames.
        try:
            self._vis_signal_path.unlink(missing_ok=True)
        except OSError:
            pass

        super().close()
