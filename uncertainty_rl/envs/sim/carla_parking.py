"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance and lot geometry.

Localisation uncertainty comes from the robot_localisation EKF fusing
RTK-GNSS and IMU; 2D LiDAR is obstacle detection only. CARLA ground truth is
used only for reward computation.
"""

import collections
import json
import logging
import math
import socket
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
    ALONG_TRACK_SCALE,
    APPROACH_INNER_ALIGNMENT_CUTOFF,
    CORRIDOR_HALF_WIDTH,
    CORRIDOR_W_ALONG,
    CORRIDOR_W_CROSS,
    CORRIDOR_W_HEAD,
    ENDGAME_HOLD_COEF,
    ENDGAME_MOVE_COEF,
    OBS_NORM_CLIP,
    OBSTACLE_CLEARANCE_DANGER,
    OBSTACLE_CLEARANCE_SAFE,
    OBSTACLE_FEATURES_DIM,
    OOB_INFLATION_MARGIN,
    OOB_STEP_PENALTY,
    OOB_TERMINATION_PENALTY_LIMIT,
    PHI_NORM_FLOOR,
    PROGRESS_TARGET,
    STALL_GATE_EKF_STD_M,
    STALL_TRUNCATION_DECISIONS,
    SUCCESS_DWELL_STEPS,
    SUCCESS_THRESHOLD_VELOCITY,
    TIMEOUT_PENALTY_FLOOR_NORM,
    TIMEOUT_POS_COEF,
    TIMEOUT_YAW_COEF,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    car_fully_inside_bay,
    inflate_polygon,
    point_in_polygon,
    wrap_angle_symmetric,
)
from uncertainty_rl.utils.logging import DebugLogger

logger = logging.getLogger(__name__)

_TRAJECTORY_MAXLEN = 50

# Chase-camera placement for render_mode="human". Far enough back to keep the
# target bay and its neighbours in frame, angled down so markings stay readable.
_SPECTATOR_BACK_M = 12.0
_SPECTATOR_UP_M = 6.0
_SPECTATOR_PITCH_DEG = -18.0
# Per-tick smoothing: low enough to absorb steering jitter, high enough to keep up.
_SPECTATOR_SMOOTHING = 0.15

# Re-exported so existing imports from this module still work.
__all__ = [
    "CARLAParkingEnv",
    "_compute_relative_target_pose",
]


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
        max_ego_speed_ms: float = 8.0,
        use_extra_spawns: bool = False,
        gnss_noise_profiles_path: Optional[str] = None,
        held_gnss_tier_override: Optional[str] = None,
        degrade_one_way_override: bool = False,
        degrade_rate_scale: float = 1.0,
        success_dwell_steps: int = SUCCESS_DWELL_STEPS,
        bay_margin: float = 0.0,
        actuator_model: Optional[Dict[str, float]] = None,
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
        @param include_covariance: If True, the obs carries the EKF std devs.
        @param include_obstacle_obs: If True, the obs carries the LiDAR clearances.
        @param vis_output_path: Destination for vis_history.jsonl (default
               outputs/vis_history.jsonl); written only while .vis_active exists.
        @param carla_timestep: Simulation timestep (seconds).
        @param eval_mode: If True, OOD floor plans join the sampling pool.
        @param debug: If True, emit per-step diagnostics and a vis HUD debug dict.
        @param map_load_sleep: Seconds to settle after loading the OpenDRIVE world.
        @param max_ego_speed_ms: Speed (m/s) above which throttle is cut.
        @param use_extra_spawns: If True, the layout's extra spawn transforms join
               the spawn pool.
        @param gnss_noise_profiles_path: GNSS noise profiles YAML; enables
               per-episode RTK fix-state tier sampling.
        @param held_gnss_tier_override: Locks GNSS noise to this tier for the whole
               episode (relay drift suppressed), making the level a controlled
               independent variable during evaluation.
        @param degrade_one_way_override: If True, start at rtk_fixed and let the
               chain only degrade. Mutually exclusive with a held tier.
        @param degrade_rate_scale: Multiplier on the one-way chain's downward
               transition mass, so the walk to the worst tier fits inside the
               episode horizon (natively ~27 s against a ~17 s episode).
        @param success_dwell_steps: Consecutive decisions the success criteria must
               hold, so a drive-through cannot count as a park.
        @param bay_margin: Inward margin (metres) for the polygon-fit success check;
               negative values let corners overhang.
        @param actuator_model: Per-axis rate limits and the brake-overrides-throttle
               threshold, applied to the policy command in step().
        """
        super().__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._chase_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # None until the first update, which snaps to the ego rather than
        # easing in from the origin.
        self._chase_yaw: Optional[float] = None
        # Wall-clock budget per tick when a human is watching, so frames arrive at
        # the sim rate. Zero leaves training and evaluation running flat out.
        self._tick_wall_seconds: float = 0.0
        self._include_covariance = include_covariance
        self._include_obstacle_obs = include_obstacle_obs
        self._eval_mode = eval_mode
        self._map_load_sleep = map_load_sleep
        self._max_ego_speed_ms = max_ego_speed_ms
        self._use_extra_spawns = use_extra_spawns

        am = actuator_model or {}
        self._steer_max_delta: float = float(
            am.get("steer_max_delta_per_decision", 0.15)
        )
        self._throttle_max_delta: float = float(
            am.get("throttle_max_delta_per_decision", 0.5)
        )
        self._brake_max_delta: float = float(
            am.get("brake_max_delta_per_decision", 0.5)
        )
        self._brake_override_throttle_threshold: float = float(
            am.get("brake_override_throttle_threshold", 0.1)
        )

        # Post-clamp commands, persisted across calls so the rate limiter measures
        # the delivered command, not the commanded one.
        self._prev_steer_cmd: float = 0.0
        self._prev_throttle_cmd: float = 0.0
        self._prev_brake_cmd: float = 0.0
        self._held_gnss_tier_override = held_gnss_tier_override
        # Ignored when a held tier is set: held tiers have no drift at all.
        self._degrade_one_way_override = bool(degrade_one_way_override) and (
            held_gnss_tier_override is None
        )
        # Floored at 1.0 so a stray config value cannot slow the chain.
        self._degrade_rate_scale: float = max(1.0, float(degrade_rate_scale))

        self._bay_margin: float = float(bay_margin)

        self._gnss_noise_tiers: List[Dict[str, Any]] = []
        self._gnss_tier_weights: np.ndarray = np.empty(0, dtype=np.float64)
        self._current_gnss_multiplier: float = 1.0
        self._current_gnss_tier: Optional[Dict[str, Any]] = None
        # Forwarded to the relay by publish_episode_config: it must HOLD the tier
        # (no Markov drift) so an evaluated level is a clean independent variable.
        self._hold_gnss_tier: bool = False
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

        # Whitelist narrowing the target pool, so the curriculum can introduce bay
        # variety on a subset before opening up the whole lot. None samples every
        # eligible bay; ignored when fixed_target_bay_id already pins the target.
        _allowed = scenarios.get("allowed_bay_ids", None)
        self._allowed_bay_ids: Optional[List[str]] = (
            [str(b) for b in _allowed] if _allowed else None
        )

        # Soft out-of-bounds boundary: the lot polygon inflated by a margin forms a
        # run-off skirt; leaving it costs a per-decision penalty that accumulates
        # until the episode terminates.
        self._oob_step_penalty: float = scenarios.get(
            "oob_step_penalty", OOB_STEP_PENALTY
        )
        self._oob_termination_limit: float = scenarios.get(
            "oob_termination_limit", OOB_TERMINATION_PENALTY_LIMIT
        )
        self._oob_inflation_margin: float = scenarios.get(
            "oob_inflation_margin", OOB_INFLATION_MARGIN
        )
        self._oob_inflated_corners: List[Tuple[float, float]] = []
        self._oob_accumulated_penalty: float = 0.0

        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # Set in reset() before any world tick, so the GNSS datum config is
        # written before the first GNSS callback fires.
        self._chosen_spawn: Dict[str, float] = {}
        # Index into self._spawn_pool (0 = primary, 1+ = extra_spawn_transforms).
        self._chosen_spawn_idx: int = 0

        # Ego CoM z after settling under gravity, passed to the static spawners so
        # props and vehicles land on the same ground surface as the ego rather
        # than on the YAML origin_z.
        self._floor_z: float = 0.3

        # Ego bounding-box half-extents (metres), read from CARLA at spawn time
        # for the polygon-fit success check.
        self._ego_half_length: float = 0.0
        self._ego_half_width: float = 0.0

        # Constructed here so NPCController / SensorManager share the reference.
        self._debug_logger: DebugLogger = DebugLogger(debug=debug)

        self._lot_spawner = LotSpawner(
            cone_spacing=scenarios.get("perimeter_cone_spacing", 2.0),
            marker_blueprint=scenarios.get(
                "perimeter_marker_blueprint", "static.prop.constructioncone"
            ),
            bay_occupancy_min=scenarios.get("bay_occupancy_min", 0.3),
            bay_occupancy_max=scenarios.get("bay_occupancy_max", 0.8),
            spawn_perimeter_cones=scenarios.get("spawn_perimeter_cones", True),
        )

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

        self._sensor_manager = SensorManager(
            sensors_config=self._sensors_config,
        )

        # Rebuilt once after all vehicles spawn, so the patrol proximity check
        # needs no per-step world query.
        self._all_vehicle_actors: List[Any] = []

        # Blueprints are static for the lifetime of the CARLA server, so these
        # are fetched once and never re-fetched.
        self._walker_blueprints: List[Any] = []
        self._vehicle_bp: Optional[Any] = None

        # Pre-allocated and reused every step to avoid repeated heap allocations.
        _obs_dim = self._compute_obs_dim()
        self._obs_dim: int = _obs_dim
        self._obs_buffer: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)
        # Returned in place of a fresh observation when a mid-decision CARLA
        # failure aborts the episode before one can be built.
        self._last_norm_obs: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)
        self._obstacle_features_buffer: np.ndarray = np.zeros(
            OBSTACLE_FEATURES_DIM, dtype=np.float32
        )
        # [x, y, yaw, vyaw, vx_body] in the world frame; vx_body is the signed
        # body-frame longitudinal velocity (m/s).
        self._world_pose_buf: np.ndarray = np.empty(5, dtype=np.float32)

        # EKF world pose snapshotted by the last _get_state(), same layout as
        # _world_pose_buf, surfaced into step() info. _last_ekf_is_real is False
        # on the CI/test ground-truth fallback, so consumers can write NaN
        # instead of logging ground truth in the EKF columns.
        self._last_ekf_world: np.ndarray = np.full(5, np.nan, dtype=np.float32)
        self._last_ekf_is_real: bool = False
        # EKF 1-sigma stds [std_x, std_y, std_yaw] (m, m, rad). The EKF runs for
        # every baseline, so this is populated regardless of include_covariance -
        # that flag controls only whether the policy SEES it.
        self._last_ekf_std: np.ndarray = np.full(3, np.nan, dtype=np.float32)

        # Odom-to-world transform (tx, ty, cos_r, sin_r, r). The GNSS datum is
        # re-latched to the spawn position each reset, so the rotation is always
        # identity and only the translation changes.
        self._ekf_odom_offset: Tuple[float, float, float, float, float] = (
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        )

        # Target bay in the world frame, set in reset(). The obs dx/dy/dyaw come
        # from differencing the odom-to-world EKF pose against this.
        self._target_bay: Dict[str, Any] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "width": 2.5,
            "depth": 5.0,
        }
        # Cached to spare _compute_reward a dict lookup and float() cast per call.
        self._target_x: float = 0.0
        self._target_y: float = 0.0
        self._target_yaw: float = 0.0

        self._current_layout: Dict[str, Any] = {}
        self._current_floor_plan_name: str = ""
        # Spawn pool and bay-type grouping, both built once per layout load.
        self._spawn_pool: List[Any] = []
        self._bays_by_type: Dict[str, List[Dict[str, Any]]] = {}
        self._bay_type_keys: List[str] = []

        self._trajectory_buffer: Deque[Tuple[float, float]] = collections.deque(
            maxlen=_TRAJECTORY_MAXLEN
        )
        # Post-clamp [steer, throttle, brake] actually delivered to CARLA, set on
        # the decision boundary once the actuator model has rate-limited it.
        self._last_action: np.ndarray = np.zeros(ACTION_DIM, dtype=np.float32)

        self._vis_history_path: Path = (
            Path(vis_output_path)
            if vis_output_path
            else Path("outputs/vis_history.jsonl")
        )
        self._vis_signal_path: Path = self._vis_history_path.parent / ".vis_active"
        # Persistent append file handle; opened lazily, avoids per-step open().
        self._vis_file: Optional[Any] = None
        # Cached signal-file stat, refreshed every N steps so the common
        # visualiser-inactive case costs no per-step filesystem call.
        self._vis_active: bool = False
        self._vis_check_counter: int = 0
        # Truncate vis_history.jsonl every N episodes to bound file size.
        self._vis_episodes_since_rotation: int = 0
        self._vis_rotation_interval: int = 10
        self._carla_timestep: float = carla_timestep

        # CARLA 0.9.16 segfaults on long headless runs; `restart: on-failure` on
        # the carla-server service relaunches it, so the crash surfaces here as a
        # world.tick() RuntimeError and the env reconnects rather than killing a
        # multi-hour run. The attempt cap makes a genuinely dead server fail loud.
        recovery = self._ros2_config.get("carla_recovery", {})
        self._max_reconnect_attempts: int = int(
            recovery.get("max_reconnect_attempts", 30)
        )
        self._reconnect_poll_interval: float = float(
            recovery.get("reconnect_poll_interval_s", 5.0)
        )
        # Set when a tick fails mid-episode; consumed by the next reset() to
        # force a reconnect before it touches the (now stale) client handle.
        self._needs_reconnect: bool = False

        self._action_repeat: int = action_repeat
        self._no_rendering_mode: bool = no_rendering_mode

        self._success_dwell_steps: int = max(1, success_dwell_steps)

        self._episode_id: int = 0
        self.steps = 0
        self._actors_frozen: bool = False
        self._success_counter: int = 0
        # Consecutive decisions below the success speed threshold while outside
        # the acceptance box. @see step().
        self._stall_counter: int = 0
        # Previous corridor potential. progress = phi(curr) - phi(prev) telescopes
        # to zero over a loiter, so a stationary car earns nothing from it.
        self._prev_phi: float = 0.0
        # Per-episode normaliser |phi(start)|, set in reset(). Dividing progress
        # and the graded timeout penalty by it puts every bay, near or far, on one
        # scale. @see PHI_NORM_FLOOR for the floor that keeps it from blowing up.
        self._phi_start: float = PHI_NORM_FLOOR

        # Shaping weights, hence code rather than YAML. @see _corridor_potential.
        self._w_along: float = CORRIDOR_W_ALONG
        self._w_cross: float = CORRIDOR_W_CROSS
        self._w_head: float = CORRIDOR_W_HEAD
        # Safety nudge only: sized well below the terminal magnitudes so it cannot
        # outweigh parking. @see _obstacle_clearance_penalty.
        self._clearance_coef: float = 0.02

        # Cached so _freeze_all_actors and _teleport_vehicle need not construct
        # them per call.
        if carla is not None:
            self._zero_vec3: Any = carla.Vector3D(x=0.0, y=0.0, z=0.0)
            self._zero_walker_ctrl: Any = carla.WalkerControl()
        else:
            self._zero_vec3 = None
            self._zero_walker_ctrl = None

        # Bounds are the fixed-normaliser clip: build_observation scales each dim by
        # a physical range and clips to +/-OBS_NORM_CLIP.
        self.observation_space = spaces.Box(
            low=-OBS_NORM_CLIP,
            high=OBS_NORM_CLIP,
            shape=(_obs_dim,),
            dtype=np.float32,
        )

        # [steer, throttle, brake], all uniformly in [-1, 1] because the policy
        # emits a tanh-squashed Gaussian on every axis. step() folds the negative
        # half of throttle and brake to zero. @see step().
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Constructed for EVERY baseline, not just include_covariance ones: the EKF
        # supplies the pose, speed and yaw rate the policy navigates by. Coupling
        # the two would hand the no-covariance baselines perfect ground-truth
        # localisation, making the ablation unfair.
        self._cov_subscriber: Optional[_CovarianceSubscriber] = None
        # File-based (no DDS, no connection), so this cannot fail; without
        # ekf_state.json _get_state() falls back to CARLA ground truth.
        self._init_ros2()

        # Bound once here rather than branched per step on a flag fixed at
        # construction. Returns (raw_pose, uncertainty), or (None, None).
        if self._cov_subscriber is not None:
            _read_ekf_state: Callable[
                [], Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            ] = self._cov_subscriber.get_latest_state
        else:

            def _read_ekf_state() -> Tuple[None, None]:
                return (None, None)

        self._read_ekf_state = _read_ekf_state

        # Returns the latest scan array, or None when obstacle obs are ablated.
        if self._include_obstacle_obs:
            self._get_lidar_scan = self._sensor_manager.get_latest_lidar_scan
        else:

            def _get_lidar_scan() -> None:
                return None

            self._get_lidar_scan = _get_lidar_scan

    def _compute_obs_dim(self) -> int:
        """
        @brief Observation dimension implied by the active feature flags.
        """
        return compute_obs_dim(self._include_covariance, self._include_obstacle_obs)

    def _init_ros2(self) -> None:
        """
        @brief Create the covariance reader (file-based, no DDS).

        A shared file rather than a topic, to avoid cross-distro serialisation
        issues between Humble and Jazzy; /set_pose goes the same way, with
        CovarianceExtractorNode publishing initial_pose.json locally. No rclpy
        initialisation is needed.
        """
        node_name = f"covariance_subscriber_{id(self)}"
        self._cov_subscriber = _CovarianceSubscriber(
            covariance_topic=self._covariance_topic,
            node_name=node_name,
            ros2_config=self._ros2_config,
        )
        logger.info("Covariance reader initialised (file-based, no DDS).")

    def _load_gnss_noise_profiles(self, path: str) -> None:
        """
        @brief Load RTK-GNSS noise tiers from YAML for per-episode sampling.
        @param path: Path to the GNSS noise profiles YAML file.

        Each tier models an RTK fix state with its own measurement noise and
        sampling weight. reset() draws a start tier from the weighted
        distribution and names it to the relay, which injects the noise.
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

        # np_random.choice wants a normalised ndarray of probabilities.
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

    def _resolve_held_tier(self, tier_name: str) -> Dict[str, Any]:
        """
        @brief Map a held-tier name to its loaded fix-state tier dict.
        @param tier_name: Tier name, matching a key in gnss_noise_profiles.yaml.
        @raises RuntimeError if no GNSS tiers are loaded or the name is unknown.
        """
        if len(self._gnss_noise_tiers) == 0:
            raise RuntimeError(
                "GNSS tier requested but no tiers loaded; "
                "pass gnss_noise_profiles_path."
            )
        for tier in self._gnss_noise_tiers:
            if tier.get("name") == tier_name:
                return tier
        available = [t.get("name", "") for t in self._gnss_noise_tiers]
        raise RuntimeError(f"GNSS tier '{tier_name}' not in loaded tiers: {available}")
        return tier

    def _sample_gnss_noise_tier(self) -> None:
        """
        @brief Sample the episode's START GNSS noise tier.

        A held-tier override pins the tier for the whole episode with the relay's
        Markov drift suppressed, so the evaluated level is a clean independent
        variable; otherwise the relay wanders the sampled tier, as in training.
        """
        if self._held_gnss_tier_override is not None:
            tier = self._resolve_held_tier(self._held_gnss_tier_override)
            base_stddev = 0.02  # RTK-fixed base (metres) - matches tier table.
            tier_stddev = float(tier.get("metric_stddev_m", base_stddev))
            self._current_gnss_tier = tier
            self._current_gnss_multiplier = max(1.0, tier_stddev / base_stddev)
            self._hold_gnss_tier = True
            logger.info(
                "Episode %d: held GNSS tier '%s' "
                "(%.2f m stddev, multiplier=%.1f, no drift)",
                self._episode_id,
                tier.get("name", "unknown"),
                tier_stddev,
                self._current_gnss_multiplier,
            )
            return

        # No override: the relay wanders the start tier via the Markov chain.
        self._hold_gnss_tier = False

        if len(self._gnss_noise_tiers) == 0:
            self._current_gnss_multiplier = 1.0
            self._current_gnss_tier = None
            return

        # Monotone degradation must genuinely "start good", so the START tier is
        # pinned to rtk_fixed; the downward-only drift itself is enforced
        # relay-side by the degrade_one_way flag.
        if self._degrade_one_way_override:
            tier = self._resolve_held_tier("rtk_fixed")
        elif self._fixed_gnss_tier is not None:
            tier = self._resolve_held_tier(self._fixed_gnss_tier)
        else:
            idx = self.np_random.choice(
                len(self._gnss_noise_tiers),
                p=self._gnss_tier_weights,
            )
            tier = self._gnss_noise_tiers[idx]
        self._current_gnss_tier = tier

        # Reported relative to the best fix, so the multiplier reads as "how much
        # worse than RTK-fixed". Diagnostic only - the relay injects the noise.
        base_stddev = 0.02  # rtk_fixed metric_stddev_m (metres)
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
        @brief Return the START tier sampled for this episode, or None.
        """
        return self._current_gnss_tier

    def _get_live_gnss_tier_name(self) -> str:
        """
        @brief Return the GNSS fix-state tier name in force at this instant.

        _current_gnss_tier holds only the START tier, while the relay walks the
        Markov chain and writes the live tier back for the subscriber to read.
        Falls back to the start tier when no live tier has been published.

        @return Tier name as in gnss_noise_profiles.yaml, or "" if unknown.
        """
        tier_name = (
            str(self._current_gnss_tier.get("name", ""))
            if self._current_gnss_tier is not None
            else ""
        )
        if self._cov_subscriber is not None:
            live = self._cov_subscriber.get_active_tier()
            if live:
                tier_name = live
        return tier_name

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
            rng=self.np_random,
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

        # Every whitelisted id must resolve, so a typo in a curriculum stage fails
        # loud rather than silently shrinking the target pool.
        if self._allowed_bay_ids is not None and self._fixed_target_bay_id is None:
            allowed = set(self._allowed_bay_ids)
            eligible_ids = {b.get("id") for b in eligible}
            missing = allowed - eligible_ids
            if missing:
                raise RuntimeError(
                    f"allowed_bay_ids contains ids absent from floor plan "
                    f"'{name}' (or always-empty): {sorted(missing)}. "
                    f"Eligible ids (first 20): {sorted(eligible_ids)[:20]}"
                )
            eligible = [b for b in eligible if b.get("id") in allowed]

        bays_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for bay in eligible:
            bay_type_key = bay.get("bay_type", "perpendicular")
            bays_by_type.setdefault(bay_type_key, []).append(bay)
        self._bays_by_type = bays_by_type
        self._bay_type_keys = list(bays_by_type.keys())

    def _sample_target_bay(self) -> None:
        """
        @brief Stratified sample of the target bay: uniform over bay type, then
               uniform within the chosen type.
        @warning The stratification spans every bay type present in the layout,
                 including out-of-scope ones (parallel, motorcycle). Filter
                 self._bay_type_keys before widening the target pool.
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
            # Drawn from the Gymnasium per-env RNG so the (bay, spawn) sequence is
            # reproducible at a fixed seed. Indexed by integer because
            # np_random.choice cannot index a list of dicts.
            type_idx = int(self.np_random.integers(len(self._bay_type_keys)))
            bay_type = self._bay_type_keys[type_idx]
            bays = self._bays_by_type[bay_type]
            bay_idx = int(self.np_random.integers(len(bays)))
            target = bays[bay_idx]

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

    def _select_spawn(self) -> Dict[str, Any]:
        """
        @brief Draw this episode's spawn transform (keys x, y, z, yaw_deg) from
               the per-env RNG, so the spawn sequence is seed-reproducible.
        """
        idx = int(self.np_random.integers(len(self._spawn_pool)))
        # Kept so traces can report where the episode started from.
        self._chosen_spawn_idx = idx
        return cast(Dict[str, Any], self._spawn_pool[idx])

    def _cache_blueprints(self) -> None:
        """
        @brief Fetch blueprint lists once per reset and distribute them to the
               LotSpawner and NPCController.
        """
        if self.world is None:
            return

        # Blueprints are static for the lifetime of the CARLA server, so the
        # managers skip the fetch once populated.
        self._lot_spawner.refresh_blueprints(self.world)

        if not self._walker_blueprints:
            bp_lib = self.world.get_blueprint_library()
            self._walker_blueprints = list(bp_lib.filter("walker.pedestrian.*"))

        self._npc_controller.refresh_blueprints(
            self._lot_spawner._car_blueprints, self._walker_blueprints
        )

    def _spawn_lot_statics(self) -> None:
        """
        @brief Spawn all static lot actors (cones and parked vehicles).
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
        """
        self._npc_controller.spawn_patrol(
            self.world, self.vehicle, self._current_layout
        )

    def _spawn_pedestrians(self) -> None:
        """
        @brief Spawn random-walk pedestrians inside the lot pedestrian zones.
        """
        self._npc_controller.spawn_pedestrians(self.world, self._current_layout)

    def _update_patrol_npcs(self) -> None:
        """
        @brief Advance patrol NPC vehicles one step.
        """
        self._npc_controller.update_patrol(self.vehicle, self.steps)

    def _update_pedestrians(self) -> None:
        """
        @brief Advance pedestrians one step.
        """
        self._npc_controller.update_pedestrians(self.vehicle)

    def _bay_frame_pose(
        self, x: float, y: float, yaw: float
    ) -> Tuple[float, float, float]:
        """
        @brief Express the ego pose (CARLA world frame, metres and radians) in
               the target bay's frame.
        @return (along, cross, heading_err) in metres, metres and radians. Both
                distances are signed but the reward takes abs: the bays are open
                and back-to-back, so either approach side is valid. heading_err
                is already absolute and 180-deg symmetric, in [0, pi/2].
        @note The transpose of _compute_relative_target_pose (target-in-ego):
              here the centreline and depth axes must be the bay's, not the car's.
        """
        dx_world = x - self._target_x
        dy_world = y - self._target_y
        cos_b = math.cos(self._target_yaw)
        sin_b = math.sin(self._target_yaw)
        along = cos_b * dx_world + sin_b * dy_world
        cross = -sin_b * dx_world + cos_b * dy_world
        heading_err = abs(wrap_angle_symmetric(yaw - self._target_yaw))
        return along, cross, heading_err

    def _corridor_potential(
        self, along: float, cross: float, heading_err: float
    ) -> float:
        """
        @brief Bay-frame shaping potential phi (<= 0, zero at the parked pose)
               for the progress term.

        Linear and monotone in each coordinate, so reducing depth, cross-track or
        heading error always raises phi - no traps, and driving in always pays.
        Cross-track and heading outweigh along-track, so a crooked short-cut
        accrues cost en route and scores below an aligned arc ending at the same
        parked pose. @see CORRIDOR_W_ALONG/CROSS/HEAD.
        """
        return -(
            self._w_along * abs(along)
            + self._w_cross * abs(cross)
            + self._w_head * heading_err
        )

    def _obstacle_clearance_penalty(self, on_line: float, aligned: float) -> float:
        """
        @brief Smooth clearance penalty (<= 0) for drifting toward a neighbouring
               car, from the observed LiDAR clearances.
        @param on_line: Corridor on-centreline factor in [0, 1] (1 = on the line).
        @param aligned: Corridor heading-alignment factor in [0, 1] (1 = square).

        Ramps from 0 at OBSTACLE_CLEARANCE_SAFE to its cap at
        OBSTACLE_CLEARANCE_DANGER, then fades out as the car squares up on the
        line so a correct park (~0.98 m from a neighbour) is never penalised.
        """
        if not self._include_obstacle_obs:
            return 0.0

        buf = self._obstacle_features_buffer
        left_dist = float(buf[0])
        right_dist = float(buf[2])
        forward_dist = float(buf[4])

        # 0.0 = no return in that hemisphere -> treat as infinite clearance.
        def _clear(d: float) -> float:
            return d if d > 0.0 else float("inf")

        # Side returns are discounted because a square car alongside a neighbour is
        # safe, so they only register when a corner genuinely swings in close. The
        # forward cone gets no discount: it is the collision-critical direction.
        side_clear = min(_clear(left_dist), _clear(right_dist)) + 0.5
        nearest = min(_clear(forward_dist), side_clear)
        if not math.isfinite(nearest) or nearest >= OBSTACLE_CLEARANCE_SAFE:
            return 0.0

        span = OBSTACLE_CLEARANCE_SAFE - OBSTACLE_CLEARANCE_DANGER
        if nearest <= OBSTACLE_CLEARANCE_DANGER:
            ramp = 1.0
        else:
            ramp = (OBSTACLE_CLEARANCE_SAFE - nearest) / span

        # Gate off once square on the line: an expected neighbour abeam of a
        # correctly parked car incurs no penalty.
        gate = 1.0 - on_line * aligned
        return -self._clearance_coef * ramp * gate

    def _compute_reward(
        self,
        transform: Optional[Any] = None,
        velocity: Optional[Any] = None,
    ) -> Tuple[float, bool, bool, Dict[str, float]]:
        """
        @brief Compute reward and termination flags from CARLA ground truth.
        @param transform: Pre-fetched vehicle transform; fetched here when None.
        @param velocity: Pre-fetched vehicle velocity; fetched here when None.
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

        # Diagnostic and terminal only - the corridor potential, not this radial
        # distance, shapes the per-step reward.
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
            diag["collision"] = 1.0
            # Ego fault costs more; both magnitudes keep the ordering
            # success(+50) > timeout > collision.
            reward = -25.0 if collision_ego_fault else -10.0
            return reward, True, False, diag

        in_bay = (
            car_fully_inside_bay(
                car_x=x,
                car_y=y,
                car_yaw=yaw,
                car_half_length=self._ego_half_length,
                car_half_width=self._ego_half_width,
                bay_x=self._target_x,
                bay_y=self._target_y,
                bay_yaw=self._target_yaw,
                bay_width=float(self._target_bay["width"]),
                bay_depth=float(self._target_bay["depth"]),
                margin=self._bay_margin,
            )
            and speed < SUCCESS_THRESHOLD_VELOCITY
        )
        if in_bay:
            self._success_counter += 1
        else:
            self._success_counter = 0

        if self._success_counter >= self._success_dwell_steps:
            return 50.0, True, True, diag

        along, cross, heading_err = self._bay_frame_pose(x, y, yaw)

        # Telescopes to zero over a loiter, so a stationary car earns nothing.
        # Normalising by the start potential equalises every bay's full-episode
        # progress sum. @see _corridor_potential, PROGRESS_TARGET, PHI_NORM_FLOOR.
        curr_phi = self._corridor_potential(along, cross, heading_err)
        progress = (curr_phi - self._prev_phi) / self._phi_start * PROGRESS_TARGET
        self._prev_phi = curr_phi

        # Corridor shaping factors, each in [0, 1] and 1 at the parked pose.
        # Referenced to the success velocity so "stopped" means stopped enough
        # to park, not merely slow.
        slowness_reference_speed = 5.0 * SUCCESS_THRESHOLD_VELOCITY
        # 1 on the centreline, 0 at the corridor half-width.
        on_line = max(0.0, 1.0 - abs(cross) / CORRIDOR_HALF_WIDTH)
        # 1 at the parked depth, 0 at the along-track scale.
        near_depth = max(0.0, 1.0 - abs(along) / ALONG_TRACK_SCALE)
        # Yaw straightness, saturating at the alignment cutoff.
        aligned = max(0.0, 1.0 - heading_err / APPROACH_INNER_ALIGNMENT_CUTOFF)
        stopped = max(0.0, 1.0 - speed / slowness_reference_speed)

        # A conjunction, so a stop-short-and-crooked pose earns ~0, and the
        # stopped term sharpens the final commit to a held stop.
        endgame = (
            (ENDGAME_MOVE_COEF + ENDGAME_HOLD_COEF * stopped)
            * on_line
            * aligned
            * near_depth
        )

        # Computed from observed LiDAR clearance rather than ground truth, so it
        # transfers to hardware. Raw, like the OOB penalty.
        clearance_term = self._obstacle_clearance_penalty(on_line, aligned)

        # Additive, so no single factor can zero the reward. Uncertainty stays
        # input-only: it enters via the covariance obs and the evidential head,
        # never here.
        reward = progress + endgame + clearance_term

        diag["progress_reward"] = float(progress)
        diag["endgame_reward"] = float(endgame)
        diag["clearance_penalty"] = float(clearance_term)

        # Evaluated last so collision and success take precedence. Raw, because
        # leaving the lot is a safety boundary rather than shaping, and tested on
        # ground truth so it is exact regardless of EKF noise.
        if self._oob_inflated_corners and not point_in_polygon(
            x, y, self._oob_inflated_corners
        ):
            reward += self._oob_step_penalty
            self._oob_accumulated_penalty += -self._oob_step_penalty
            diag["oob"] = 1.0
            if self._oob_accumulated_penalty >= self._oob_termination_limit:
                return float(reward), True, False, diag
        else:
            diag["oob"] = 0.0

        return float(reward), False, False, diag

    def _get_state(self) -> np.ndarray:
        """
        @brief Build the observation vector from EKF pose and sim sensors.

        Resolves the sim-specific EKF odom -> world transform, then delegates to
        build_observation(). Falls back to CARLA ground truth when the EKF is
        unavailable (CI/tests).
        """
        if self.vehicle is None or self.world is None:
            return np.zeros(self._obs_dim, dtype=np.float32)

        # Pose and uncertainty come from a single file read.
        raw_ekf_pose, uncertainty = self._read_ekf_state()

        world_pose: Optional[np.ndarray] = None
        if raw_ekf_pose is not None:
            ekf_odom_x = float(raw_ekf_pose[0])
            ekf_odom_y = float(raw_ekf_pose[1])
            ekf_odom_yaw = float(raw_ekf_pose[2])
            tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
            self._world_pose_buf[0] = cos_r * ekf_odom_x - sin_r * ekf_odom_y + tx
            self._world_pose_buf[1] = sin_r * ekf_odom_x + cos_r * ekf_odom_y + ty
            self._world_pose_buf[2] = ekf_odom_yaw + r
            self._world_pose_buf[3] = raw_ekf_pose[3]
            # vx is body-frame; invariant under the odom-to-world rigid transform.
            self._world_pose_buf[4] = raw_ekf_pose[4]
            world_pose = self._world_pose_buf
            # Snapshotted for step() info and the trace CSV.
            self._last_ekf_world[:] = self._world_pose_buf
            self._last_ekf_is_real = True
            if uncertainty is not None:
                self._last_ekf_std[:] = uncertainty
            else:
                self._last_ekf_std[:] = np.nan
        else:
            # CI / tests fallback: no EKF available, so navigate on CARLA GT.
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
            # Marked not-real so trace consumers write NaN in the EKF columns
            # rather than a copy of ground truth.
            self._last_ekf_world[:] = np.nan
            self._last_ekf_is_real = False
            self._last_ekf_std[:] = np.nan

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

    def _write_vis_state(
        self,
        end_reason: Optional[str] = None,
        transform: Optional[Any] = None,
        velocity: Optional[Any] = None,
    ) -> None:
        """
        @brief Append a visualisation frame to the JSONL history file.

        Writes only while the visualiser's signal file exists, keeping one append
        handle open for as long as it does.

        @param end_reason: One of "collision", "out_of_bounds", "success" or
                           "timeout"; None for mid-episode frames.
        @param transform: Pre-fetched vehicle transform; fetched here when None.
        @param velocity: Pre-fetched vehicle velocity; fetched here when None.
        """
        if self.vehicle is None:
            return

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

        # Negated into REP-103 (left turn positive) to match the EKF vyaw at
        # obs[1] and the LiDAR bearing convention.
        gt_av_z_deg = self.vehicle.get_angular_velocity().z
        gt_vyaw = -math.radians(gt_av_z_deg)

        # Reuses the buffer _get_state() filled this step rather than re-reading
        # the EKF file.
        ekf_speed = float(self._obs_buffer[0])
        ekf_vyaw = float(self._obs_buffer[1])

        # The live tier, not the reset-time one: this is the visualiser's headline
        # field, so it must track the mid-episode Markov drift.
        gnss_tier_name = self._get_live_gnss_tier_name()

        patrol_npcs = self._npc_controller.patrol_npcs
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
            # The clamped command, not the policy's raw output, so the HUD
            # reflects what actually drove the car.
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
            # Non-fatal: visualisation must never take down a training run.
            logger.debug(f"Could not write vis state: {exc}")

    def _connect_to_carla(self) -> None:
        """
        @brief Connect to CARLA and load configs/layouts/flat_plane.xodr as the
               FlatPlane OpenDRIVE world.
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(120.0)
            self.world = self.client.get_world()

            current_map_name = self.world.get_map().name.split("/")[-1]
            # Regenerating an already-loaded world zeroes CARLA's elapsed-seconds
            # sensor clock, after which the long-lived ros2-bridge EKF rejects
            # every fresh transform as TF_OLD_DATA and stops writing ekf_state.json.
            # Either name below means FlatPlane is already present.
            if current_map_name not in ("FlatPlane", "OpenDriveMap"):
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

    def _carla_port_open(self) -> bool:
        """
        @brief Probe whether the CARLA RPC port accepts connections, so a
               reconnect does not race a restarted engine's RPC startup.
        """
        try:
            with socket.create_connection(
                (self.carla_host, self.carla_port), timeout=2.0
            ):
                return True
        except OSError:
            return False

    def _reconnect_to_carla(self) -> bool:
        """
        @brief Tear down the stale CARLA client and reconnect to a fresh server.
        @return True if a fresh world handle was obtained, False otherwise.

        @note Episode actors are NOT respawned here; the fresh world starts empty
              and reset() repopulates it on the next episode.
        """
        # The actors live in the crashed engine and cannot be destroyed over RPC,
        # so forget the handles rather than calling _cleanup_actors, whose
        # destroy() RPCs would themselves time out.
        self.client = None
        self.world = None
        self.vehicle = None
        self._vehicle_bp = None
        self._all_vehicle_actors = []
        self._sensor_manager.forget_actors()
        self._lot_spawner.forget_actors()
        self._npc_controller.forget_actors()

        for attempt in range(1, self._max_reconnect_attempts + 1):
            if self._carla_port_open():
                logger.info(
                    "CARLA RPC port reopened on attempt %d; reconnecting ...",
                    attempt,
                )
                self._connect_to_carla()
                if self.world is not None:
                    logger.info("Reconnected to CARLA after server restart.")
                    return True
            logger.warning(
                "Waiting for CARLA server to come back (attempt %d/%d) ...",
                attempt,
                self._max_reconnect_attempts,
            )
            time.sleep(self._reconnect_poll_interval)

        logger.error(
            "CARLA did not come back after %d attempts (%.0fs). Is the "
            "carla-server container restarting?",
            self._max_reconnect_attempts,
            self._max_reconnect_attempts * self._reconnect_poll_interval,
        )
        return False

    def _spawn_vehicle(self) -> None:
        """
        @brief Spawn the ego vehicle at this episode's spawn transform.
        """
        if self.world is None:
            return

        default_z = float(self._current_layout.get("origin", {}).get("z", 0.3))
        # Pre-selected in reset() before any tick, so the GNSS datum config is
        # written before the first GNSS callback fires.
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

        if self.vehicle is not None:
            # CARLA's extent is already half-lengths on the body axes: x forward,
            # y lateral.
            bb_extent = self.vehicle.bounding_box.extent
            self._ego_half_length = float(bb_extent.x)
            self._ego_half_width = float(bb_extent.y)

        if self.vehicle is not None and self.world is not None:
            # Settle onto the ground plane by ticking: in synchronous mode
            # time.sleep() does not advance physics. Never toggle
            # set_simulate_physics on the ego - in CARLA 0.9.16 that locks the
            # drivetrain, so the wheels steer but the vehicle cannot translate.
            for _ in range(40):
                self.world.tick(10.0)
                if abs(self.vehicle.get_velocity().z) < 0.01:
                    break
            # The settled z, so the static spawners share the ego's ground
            # reference rather than guessing from the layout origin.
            self._floor_z = self.vehicle.get_transform().location.z

    def _teleport_vehicle(self) -> None:
        """
        @brief Teleport the existing ego vehicle to the chosen spawn point.

        Preferred over destroy+respawn because keeping the sensors alive stops
        the CARLA ROS bridge accumulating actor-stream registrations and
        eventually segfaulting mid-training.
        """
        if self.vehicle is None or self.world is None:
            return

        self.vehicle.disable_constant_velocity()

        default_z = float(self._current_layout.get("origin", {}).get("z", 0.3))
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

        # Zeroed so the vehicle does not carry the previous episode's momentum
        # into the new one.
        zero = self._zero_vec3
        self.vehicle.set_target_velocity(zero)
        self.vehicle.set_target_angular_velocity(zero)
        self.vehicle.apply_control(carla.VehicleControl())

        # Settle under gravity, as in _spawn_vehicle.
        for _ in range(40):
            self.world.tick(10.0)
            if abs(self.vehicle.get_velocity().z) < 0.01:
                break
        self._floor_z = self.vehicle.get_transform().location.z

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn sensors (IMU, GNSS, 2D LiDAR, collision) on the ego vehicle.

        The NPC controller's patrol_npc_ids set is passed by reference so the
        collision callback can tell a patrol vehicle from a static one.
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
        @brief Zero velocity on all moving actors immediately on episode end, so
               they do not run on their last command while reset() tears down.

        Pins the velocity rather than braking: brake control needs several ticks
        to converge through physics.
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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment for a new episode: clean up the previous
               one, select a floor plan, sample the target bay, then spawn the
               cones, static vehicles, patrol NPCs and pedestrians.
        @param seed: Random seed for reproducibility.
        @param options: Additional options (currently unused).
        @return Tuple of (initial_observation, info_dict).
        """
        super().reset(seed=seed)

        # Sharing the seeded per-env RNG puts every per-episode placement draw on
        # one reproducible stream.
        self._lot_spawner.set_rng(self.np_random)
        self._npc_controller.set_rng(self.np_random)

        self._episode_id += 1
        self.steps = 0
        # Cleared ahead of every early-return path below, so a stall can never
        # leak into the next episode.
        self._stall_counter = 0
        self._trajectory_buffer.clear()
        # Snaps the chase camera to the new spawn instead of gliding across the
        # lot from wherever the last episode ended.
        self._chase_yaw = None
        # Forces a signal-file recheck, so the episode cannot inherit a stale
        # cached value from the previous episode's final step.
        self._vis_check_counter = 30

        # Periodic truncation bounds the file size; the visualiser detects the
        # shrink and resets its read offset.
        self._vis_episodes_since_rotation += 1
        if (
            self._vis_file is not None
            and self._vis_episodes_since_rotation >= self._vis_rotation_interval
        ):
            self._vis_episodes_since_rotation = 0
            self._vis_file.close()
            self._vis_file = open(self._vis_history_path, "w")

        # Must precede publish_episode_config below, which names the start tier
        # to the relay.
        self._sample_gnss_noise_tier()
        # Redraws the LiDAR systematic range bias for this episode.
        self._sensor_manager.sample_lidar_noise_bias(self.np_random)

        # Recovery from a crash the previous step() flagged. Must precede
        # _cleanup_actors(), whose destroy() RPCs would time out against the dead
        # engine; the reset then proceeds as a fresh full spawn into a new world.
        if self._needs_reconnect:
            reconnected = self._reconnect_to_carla()
            self._needs_reconnect = False
            # Nothing was destroyed over RPC against the dead server, so no
            # destroy commands are in flight - skip the flush tick below.
            self._actors_frozen = True
            if not reconnected:
                return np.zeros(self._obs_dim, dtype=np.float32), {}

        if self.client is None:
            self._connect_to_carla()

        if self.world is None:
            return np.zeros(self._obs_dim, dtype=np.float32), {}

        # Teleporting rather than respawning avoids the destroy/respawn cycle
        # that makes the CARLA ROS bridge accumulate actor-stream registrations
        # and eventually segfault. Only a first or stale actor needs a spawn.
        reuse_vehicle = self.vehicle is not None and self.vehicle.is_alive

        # NPCs, cones and static vehicles always go; only the sensors and ego
        # survive, and only when reused.
        self._cleanup_actors(skip_ego=reuse_vehicle)

        # Flushes the pending destroy commands before spawning. Unnecessary when
        # step() already froze the actors, as nothing is then in flight.
        if not self._actors_frozen:
            self.world.tick(10.0)
        self._actors_frozen = False

        # A world reload resets every CARLA setting to the async defaults, so
        # sync mode is re-applied here rather than trusted to the bridge. Both
        # checks share one get_settings() / apply_settings() pair.
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

            # Skipping the Unreal rendering pipeline is a 3-4x speedup and costs
            # a state-based agent nothing, but it blacks out the viewport the
            # moment it is applied, so a human watcher overrides it.
            want_no_rendering = self._no_rendering_mode and self.render_mode != "human"
            if want_no_rendering and not settings.no_rendering_mode:
                logger.info(
                    "Enabling no_rendering_mode (state-based agent, no cameras)..."
                )
                settings.no_rendering_mode = True
                settings_changed = True
            elif not want_no_rendering and settings.no_rendering_mode:
                # A previous run may have left the server in no-rendering mode;
                # the setting is server-side and persists across clients.
                logger.info("Disabling no_rendering_mode for the windowed view...")
                settings.no_rendering_mode = False
                settings_changed = True

            if settings_changed:
                self.world.apply_settings(settings)
                if want_no_rendering:
                    logger.info("No rendering mode enabled. Expected speedup: 3-4x.")

        if self._floor_plans_config:
            self._load_floor_plan()
            self._sample_target_bay()

        # Cached before any spawning, so no spawn method repeats the world query.
        self._cache_blueprints()

        # The spawn and the GNSS episode config must both be settled BEFORE any
        # world tick: the gravity-settle ticks in _spawn_vehicle /
        # _teleport_vehicle already fire GNSS callbacks in the ROS 2 bridge.
        self._chosen_spawn = self._select_spawn()
        sx = float(self._chosen_spawn.get("x", 0.0))
        sy = float(self._chosen_spawn.get("y", 0.0))
        syaw = math.radians(float(self._chosen_spawn.get("yaw_deg", 0.0)))

        if self._cov_subscriber is not None and len(self._gnss_noise_tiers) > 0:
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
                    hold_tier=self._hold_gnss_tier,
                    degrade_one_way=self._degrade_one_way_override,
                    degrade_rate_scale=self._degrade_rate_scale,
                )

        if reuse_vehicle:
            # Sensors stay attached and alive across the teleport.
            self._teleport_vehicle()
            self._sensor_manager.reset_state()
        else:
            self._spawn_vehicle()
            self._spawn_sensors()

        # Drops the pre-reset reading so _wait_for_covariance() blocks until a
        # genuinely post-spawn one arrives.
        if self._cov_subscriber is not None:
            self._cov_subscriber.invalidate()

        # Local (0, 0, yaw), so /set_pose seeds the EKF at the origin of the
        # re-latched GNSS datum frame.
        if self._cov_subscriber is not None and self._ros2_config.get(
            "publish_initial_pose", False
        ):
            self._cov_subscriber.publish_initial_pose(0.0, 0.0, syaw)

        self._spawn_lot_statics()
        self._spawn_npc_patrol()
        self._spawn_pedestrians()

        # Rebuilt only now that every vehicle exists, so update_patrol() needs
        # no per-step world query.
        if self.world is not None:
            self._all_vehicle_actors = list(self.world.get_actors().filter("vehicle.*"))
            self._npc_controller.set_vehicle_cache(self._all_vehicle_actors)

        if self._cov_subscriber is not None:
            # Waits twice on purpose: the first confirms the extractor is
            # writing and gives the EKF ticks to consume /set_pose, the second
            # (after invalidate) accepts only post-reset EKF state.
            self._wait_for_covariance()
            if self.world is not None:
                for _ in range(10):
                    self.world.tick(10.0)
            self._cov_subscriber.invalidate()
            self._wait_for_covariance()
            # Unconditional, because the GNSS datum is re-latched to the spawn
            # each reset and the odom origin moves with it.
            self._calibrate_ekf_frame_offset()

        self._success_counter = 0
        # Recomputed per episode, the lot corners being fixed for a floor plan.
        self._oob_accumulated_penalty = 0.0
        lot_corners = [
            (float(c["x"]), float(c["y"]))
            for c in self._current_layout.get("corners", [])
        ]
        self._oob_inflated_corners = inflate_polygon(
            lot_corners, self._oob_inflation_margin
        )
        self._last_action[:] = 0.0
        # Back to the rest state, so the first decision starts from zero pedal
        # and centred steering, matching the policy's first observation.
        self._prev_steer_cmd = 0.0
        self._prev_throttle_cmd = 0.0
        self._prev_brake_cmd = 0.0
        if self.vehicle is not None:
            t = self.vehicle.get_transform()
            along0, cross0, head0 = self._bay_frame_pose(
                t.location.x,
                t.location.y,
                math.radians(t.rotation.yaw),
            )
            self._prev_phi = self._corridor_potential(along0, cross0, head0)
            # |phi(start)| is the whole potential the car must close to park, so
            # dividing progress by it equalises the full-episode sum across bays.
            # The floor stops a spawn that already sits near the bay blowing it up.
            self._phi_start = max(abs(self._prev_phi), PHI_NORM_FLOOR)
        else:
            self._prev_phi = 0.0
            self._phi_start = PHI_NORM_FLOOR

        state = self._get_state()
        self._last_norm_obs = state

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

        # A first frame now, so the visualiser shows the new layout during the
        # reset gap instead of the previous episode's stale scene.
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
        @brief Execute one policy decision (action_repeat sim ticks).

        One call is one agent transition: the rate-limited command is held while
        CARLA advances action_repeat fixed timesteps, and the observation and
        reward are built once from the post-tick state. The agent never sees the
        intermediate ticks, so SB3 timesteps count decisions, not sim ticks.

        @param action: [steer, throttle, brake], each in [-1, 1]. The actuator
                model rate-limits the command before CARLA sees it, so what the
                policy asks for and what the vehicle receives may differ.
        @return Tuple of (observation, reward, terminated, truncated, info).
        """
        # Every policy axis is uniformly [-1, 1] (tanh-squashed), so steer passes
        # through while throttle and brake fold their negative half to zero: the
        # symmetric prior gamma = 0 must map to pedal-off. A linear (a + 1) / 2
        # would instead sit both pedals at 0.5 and trip the brake override.
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], 0.0, 1.0))
        brake_cmd = float(np.clip(action[2], 0.0, 1.0))

        # Rate-limited to a physical actuator, then persisted in _prev_*_cmd and
        # held for this decision's action_repeat sim ticks.
        steer_cmd = float(
            np.clip(
                steer_cmd,
                self._prev_steer_cmd - self._steer_max_delta,
                self._prev_steer_cmd + self._steer_max_delta,
            )
        )
        brake_cmd = float(
            np.clip(
                brake_cmd,
                self._prev_brake_cmd - self._brake_max_delta,
                self._prev_brake_cmd + self._brake_max_delta,
            )
        )

        # Brake-overrides-throttle, as a real driver-assistance system does, so
        # the policy cannot learn to fight itself. Applied BEFORE the throttle
        # rate limit: a real throttle plate cannot slam closed instantly, so the
        # cut must respect the actuator slew rate too.
        if brake_cmd > self._brake_override_throttle_threshold:
            throttle_cmd = 0.0
        throttle_cmd = float(
            np.clip(
                throttle_cmd,
                self._prev_throttle_cmd - self._throttle_max_delta,
                self._prev_throttle_cmd + self._throttle_max_delta,
            )
        )

        self._prev_steer_cmd = steer_cmd
        self._prev_throttle_cmd = throttle_cmd
        self._prev_brake_cmd = brake_cmd

        self._last_action[0] = steer_cmd
        self._last_action[1] = throttle_cmd
        self._last_action[2] = brake_cmd

        # Advance the simulation under the held setpoint. The post-tick
        # transform and velocity of the last tick are shared with
        # _compute_reward and _write_vis_state to avoid duplicate CARLA RPCs.
        _post_transform: Optional[Any] = None
        _post_velocity: Optional[Any] = None

        for _ in range(self._action_repeat):
            tick_start = time.monotonic()
            self.steps += 1

            if self.vehicle is None:
                continue

            vel = self.vehicle.get_velocity()
            current_speed = math.hypot(vel.x, vel.y)

            control = carla.VehicleControl()
            control.steer = float(self._last_action[0])
            control.reverse = False
            # Cut throttle above the speed limit; brake is forwarded as-is.
            control.throttle = (
                0.0
                if current_speed >= self._max_ego_speed_ms
                else float(self._last_action[1])
            )
            control.brake = float(self._last_action[2])

            self.vehicle.apply_control(control)

            if self.world is None:
                continue

            self._update_patrol_npcs()
            self._update_pedestrians()
            # CARLA 0.9.16 segfaults on long headless runs. The timeout surfaces
            # a frozen server as an error, and the episode truncates so SB3 ends
            # it cleanly instead of a multi-hour run dying on the exception.
            # @see _reconnect_to_carla.
            try:
                self.world.tick(10.0)
            except RuntimeError as exc:
                logger.error(
                    "CARLA tick failed (server crash?): %s. Aborting episode "
                    "and scheduling reconnect on next reset.",
                    exc,
                )
                self._needs_reconnect = True
                self._actors_frozen = True
                return (
                    self._last_norm_obs,
                    0.0,
                    False,
                    True,
                    {"carla_reconnect": True},
                )

            # Fetch transform + velocity once post-tick; reused by
            # _compute_reward and _write_vis_state below.
            _post_transform = self.vehicle.get_transform()
            _post_velocity = self.vehicle.get_velocity()
            self._trajectory_buffer.append(
                (_post_transform.location.x, _post_transform.location.y)
            )

            # Follow on every TICK, not once per policy decision: in synchronous
            # mode the server renders a frame per tick, so a per-decision update
            # would emit action_repeat frames back to back and the caller's sleep
            # would show them as one jump per decision - a slideshow.
            if self.render_mode == "human":
                self._update_chase_camera()
                if self._tick_wall_seconds > 0.0:
                    lag = time.monotonic() - tick_start
                    if lag < self._tick_wall_seconds:
                        time.sleep(self._tick_wall_seconds - lag)

        # Construct the observation and reward once per decision.
        state = self._get_state()
        self._last_norm_obs = state
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
            # Read the RAW (unscaled) obs buffer, not the normalised return
            # value, so the debug distance stays in metres.
            _obs_dist = (
                float(self._obs_buffer[-OBSTACLE_FEATURES_DIM])
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

        # A near-stop OUTSIDE the acceptance box is parked in the wrong place and,
        # forward-only, almost never recovers. Truncates through the same path as
        # the clock running out so the graded penalty applies identically.
        # @see STALL_TRUNCATION_DECISIONS.
        if not terminated:
            # Live EKF position std (m): mean of the x/y diagonal stds, matching
            # how the observation and eval combine them. NaN when the EKF is
            # unavailable (CI/tests), in which case the gate stays closed (treat
            # as good localisation) so the stall rule is unchanged off-stack.
            ekf_std_pos = float(
                np.nanmean(self._last_ekf_std[:2])
                if np.any(np.isfinite(self._last_ekf_std[:2]))
                else 0.0
            )
            # Waiting out a degraded fix is legitimate, not a stall - punishing it
            # would suppress the behaviour the covariance is meant to induce.
            waiting_out_bad_fix = ekf_std_pos > STALL_GATE_EKF_STD_M
            # Counter > 0 means slow INSIDE the box, i.e. the success dwell.
            near_stop = (
                reward_diag["speed"] < SUCCESS_THRESHOLD_VELOCITY
                and self._success_counter == 0
            )
            # Held, not reset, during the wait: a genuine good-fix stall briefly
            # interrupted by a degraded blip resumes rather than restarting.
            if not waiting_out_bad_fix:
                if near_stop:
                    self._stall_counter += 1
                else:
                    self._stall_counter = 0
            if self._stall_counter >= STALL_TRUNCATION_DECISIONS:
                truncated = True

        # Graded on final distance and misalignment, so ending closer and
        # straighter always costs less than stopping short. Normalised by
        # |phi(start)| like the corridor progress, so a far-bay timeout is not
        # penalised merely for being far. @see TIMEOUT_PENALTY_FLOOR_NORM.
        if truncated and not terminated:
            reward += max(
                -(
                    TIMEOUT_POS_COEF * reward_diag["pos_error"]
                    + TIMEOUT_YAW_COEF * reward_diag["orientation_error"]
                )
                / self._phi_start
                * PROGRESS_TARGET,
                TIMEOUT_PENALTY_FLOOR_NORM,
            )

        if (terminated or truncated) and self.world is not None:
            self._freeze_all_actors()
            self._actors_frozen = True

        # Distinguish termination cause for vis state writer. A terminated
        # episode is either an out-of-bounds run-off (oob flag set) or a
        # collision; success and timeout are the other two paths.
        if success:
            end_reason: Optional[str] = "success"
        elif terminated and reward_diag.get("oob", 0.0) > 0.0:
            end_reason = "out_of_bounds"
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

        # Ground-truth world pose for trace logging and EKF accuracy checks.
        # vx is the signed body-frame longitudinal velocity (matches the EKF
        # vx convention); vyaw is in REP-103 (left turn positive), matching the
        # EKF vyaw at obs[1] and the GT vyaw written by _write_vis_state().
        if (
            _post_transform is not None
            and _post_velocity is not None
            and self.vehicle is not None
        ):
            gt_yaw = math.radians(_post_transform.rotation.yaw)
            gt_vx = _post_velocity.x * math.cos(gt_yaw) + _post_velocity.y * math.sin(
                gt_yaw
            )
            gt_vyaw = -math.radians(self.vehicle.get_angular_velocity().z)
            gt_x = _post_transform.location.x
            gt_y = _post_transform.location.y
        else:
            gt_x = gt_y = gt_yaw = gt_vx = gt_vyaw = float("nan")

        # EKF world pose snapshotted by the most recent _get_state(). NaN on the
        # CI/test GT-fallback path so the CSV never reports GT as if it were EKF.
        ekf = self._last_ekf_world

        # Live GNSS fix-state tier for this step. @see _get_live_gnss_tier_name.
        _live_tier = self._get_live_gnss_tier_name()
        # Noise multiplier consistent with the live tier, computed the same way
        # as _sample_gnss_noise_tier (tier metric stddev / RTK-fixed base 0.02 m).
        # Falls back to the per-episode multiplier when the live tier is
        # unavailable/unknown.
        _live_multiplier = float(self._current_gnss_multiplier)
        for _t in self._gnss_noise_tiers:
            if _t.get("name") == _live_tier:
                _tier_stddev = float(_t.get("metric_stddev_m", 0.02))
                _live_multiplier = max(1.0, _tier_stddev / 0.02)
                break

        info: Dict[str, Any] = {
            "steps": self.steps,
            "success": success,
            "collision": bool(reward_diag["collision"]),
            "timeout": truncated,
            "oob": bool(reward_diag.get("oob", 0.0)),
            "floor_plan": self._current_floor_plan_name,
            # Episode routing: which bay the car is parking into and which spawn
            # it started from (0 = primary, 1+ = extra). Constant per episode;
            # logged per row so a trace shows "from where to where".
            "target_bay_id": self._target_bay.get("bay_id", ""),
            "spawn_id": self._chosen_spawn_idx,
            "pos_error": reward_diag["pos_error"],
            "orientation_error": reward_diag["orientation_error"],
            "speed": reward_diag["speed"],
            "progress_reward": reward_diag["progress_reward"],
            "endgame_reward": reward_diag.get("endgame_reward", 0.0),
            "clearance_penalty": reward_diag.get("clearance_penalty", 0.0),
            # Post-clamp commands actually delivered to CARLA. Distinct from
            # the policy's raw output so diagnostics can verify the actuator
            # model is doing its job.
            "steer_cmd": float(self._last_action[0]),
            "throttle_cmd": float(self._last_action[1]),
            "brake_cmd": float(self._last_action[2]),
            # Ground-truth world pose (CARLA). Reward uses GT; these expose it.
            "gt_x": float(gt_x),
            "gt_y": float(gt_y),
            "gt_yaw": float(gt_yaw),
            "gt_vx": float(gt_vx),
            "gt_vyaw": float(gt_vyaw),
            # EKF world pose (what the policy actually sees via _get_state).
            # NaN when no genuine EKF estimate was available this step.
            "ekf_x": float(ekf[0]),
            "ekf_y": float(ekf[1]),
            "ekf_yaw": float(ekf[2]),
            "ekf_vx": float(ekf[4]),
            "ekf_vyaw": float(ekf[3]),
            # EKF 1-sigma localisation stds (m, m, rad). Populated for every
            # baseline (the EKF always runs); include_covariance only controls
            # whether the policy observes them. NaN when no EKF estimate.
            "ekf_std_x": float(self._last_ekf_std[0]),
            "ekf_std_y": float(self._last_ekf_std[1]),
            "ekf_std_yaw": float(self._last_ekf_std[2]),
            # Live GNSS fix-state tier (name + noise multiplier vs RTK-fixed),
            # read back from the relay's Markov drift. Lets timing analysis
            # recover the degraded-tier onset step (the handover-latency
            # reference), which varies per episode under degrade_one_way.
            "gnss_tier": _live_tier,
            "gnss_multiplier": _live_multiplier,
            # Target bay this episode (id, world pose, dimensions). Constant
            # within an episode; surfaced so trace tooling can record which bay
            # the run targeted without reaching into the env internals.
            "target_bay": dict(self._target_bay),
        }

        return state, reward, terminated, truncated, info

    def _update_chase_camera(self) -> None:
        """
        @brief Move the CARLA spectator to a smoothed chase view of the ego.

        Called every simulation tick (not once per policy decision) so the view
        does not stutter at 1 / action_repeat of the sim rate. Both the target
        point and the yaw are low-pass filtered: the raw pose jitters with every
        steering correction, which reads as camera shake on a recording. Yaw is
        interpolated on the shortest arc so the +/-180 deg wrap does not spin the
        camera the long way round.
        """
        if self.vehicle is None or self.world is None:
            return

        transform = self.vehicle.get_transform()
        location = transform.location
        yaw = transform.rotation.yaw

        if self._chase_yaw is None:
            smooth_x, smooth_y, smooth_z = location.x, location.y, location.z
            smooth_yaw = yaw
        else:
            alpha = _SPECTATOR_SMOOTHING
            prev_x, prev_y, prev_z = self._chase_xyz
            smooth_x = prev_x + alpha * (location.x - prev_x)
            smooth_y = prev_y + alpha * (location.y - prev_y)
            smooth_z = prev_z + alpha * (location.z - prev_z)
            # Shortest-arc yaw blend: wrap the delta into [-180, 180) first.
            delta = (yaw - self._chase_yaw + 180.0) % 360.0 - 180.0
            smooth_yaw = self._chase_yaw + alpha * delta

        self._chase_xyz = (smooth_x, smooth_y, smooth_z)
        self._chase_yaw = smooth_yaw

        # Chase camera: behind and above the ego, looking down at it. A top-down
        # view would only duplicate the 2D bird's-eye viewer, whereas the 3D
        # window earns its place by showing the vehicle against the lot.
        yaw_rad = math.radians(smooth_yaw)
        self.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(
                    x=smooth_x - _SPECTATOR_BACK_M * math.cos(yaw_rad),
                    y=smooth_y - _SPECTATOR_BACK_M * math.sin(yaw_rad),
                    z=smooth_z + _SPECTATOR_UP_M,
                ),
                carla.Rotation(pitch=_SPECTATOR_PITCH_DEG, yaw=smooth_yaw),
            )
        )

    def render(self) -> "RenderFrame | list[RenderFrame] | None":
        """
        @brief Render the environment.
        @return RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human" and self.world is not None:
            if self.vehicle is not None:
                self._update_chase_camera()
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
