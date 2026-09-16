"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance and lot geometry.

Gymnasium-compatible environment for autonomous parking in CARLA. Localisation
uncertainty comes from the robot_localisation EKF fusing RTK-GNSS and IMU; 2D LiDAR
provides obstacle detection only. CARLA ground truth is used only for reward
computation.
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

# Trail length for debug overlays and vis state
_TRAJECTORY_MAXLEN = 50

# Chase-camera placement for render_mode="human" (windowed demo only). Far
# enough back to keep the target bay and its neighbours in frame while the car
# manoeuvres, angled down so the bay markings stay readable.
_SPECTATOR_BACK_M = 12.0
_SPECTATOR_UP_M = 6.0
_SPECTATOR_PITCH_DEG = -18.0

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

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

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
               when this speed is exceeded. Default 8.0 m/s (~29 km/h),
               appropriate for parking lot manoeuvres. The live value comes from
               max_ego_speed_ms in agent_config.yaml; this default is the guard
               used only when that key is absent.
        @param use_extra_spawns: If True, extra spawn transforms from the layout
               YAML are included in the spawn pool. If False (default), only the
               primary spawn is used. With RTK-GNSS the odom frame is UTM-aligned
               regardless of spawn location, so extra spawns are safe to enable.
        @param gnss_noise_profiles_path: Path to GNSS noise profiles YAML. If
               provided, the env samples an RTK fix-state tier each reset() and
               spawns the GNSS sensor with the corresponding noise multiplier.
        @param held_gnss_tier_override: If set, locks the GNSS noise to the named
               fix-state tier for the WHOLE episode (Markov drift held off in the
               relay) every reset. The name must match a tier in the loaded GNSS
               noise profiles; that tier drives both the relay noise and the
               held-level signalling, so the level is a controlled independent
               variable. Used during evaluation to lock GNSS noise to a specific
               condition.
        @param degrade_one_way_override: If True, the episode STARTS at rtk_fixed
               and the relay lets the Markov chain only degrade (never recover) -
               the monotone-degradation eval condition (starts good, drifts to
               degraded, stays there). Mutually exclusive with a held tier (a held
               tier suppresses drift entirely). Training leaves it False.
        @param degrade_rate_scale: Multiplier on the one-way chain's downward
               transition mass, compressing the drift so the walk to the worst
               tier completes inside the episode horizon (at the native rate it
               takes ~27 s in expectation against a ~17 s episode). 1.0 is the
               datasheet-anchored schedule and the default; only the drift
               evaluation condition raises it. Ignored unless
               degrade_one_way_override is active.
        @param success_dwell_steps: Number of consecutive steps all success
               criteria (position, orientation, velocity) must be satisfied
               before the episode terminates as a success. Prevents a fast
               drive-through that momentarily satisfies the thresholds from
               being counted as a park. Default 5 steps = 0.25 s at 20 Hz.
        @param actuator_model: Per-axis rate limits and the brake-overrides-
               throttle threshold. Keys: steer_max_delta_per_decision,
               throttle_max_delta_per_decision, brake_max_delta_per_decision,
               brake_override_throttle_threshold. Clamps the policy command
               in step() before forwarding to CARLA. If None or missing keys,
               defaults match production drive-by-wire literature.
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

        # Actuator rate limits and brake-overrides-throttle threshold.
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

        # Previous actuator command (post-clamp), persisted across calls so the
        # rate limiter measures the delivered command, not the commanded one.
        # Reset to zero on episode start.
        self._prev_steer_cmd: float = 0.0
        self._prev_throttle_cmd: float = 0.0
        self._prev_brake_cmd: float = 0.0
        self._held_gnss_tier_override = held_gnss_tier_override
        # Monotone-degradation eval condition: start at rtk_fixed and let the
        # relay drift only downward. Ignored when a held tier is set (held tiers
        # have no drift at all).
        self._degrade_one_way_override = bool(degrade_one_way_override) and (
            held_gnss_tier_override is None
        )
        # Drift-schedule compression, forwarded to the relay with the one-way
        # flag. Floored at 1.0 so a stray config value cannot slow the chain.
        self._degrade_rate_scale: float = max(1.0, float(degrade_rate_scale))

        self._bay_margin: float = float(bay_margin)

        # Load GNSS noise profiles for per-episode RTK fix-state sampling.
        self._gnss_noise_tiers: List[Dict[str, Any]] = []
        self._gnss_tier_weights: np.ndarray = np.empty(0, dtype=np.float64)
        self._current_gnss_multiplier: float = 1.0
        self._current_gnss_tier: Optional[Dict[str, Any]] = None
        # When a held-tier override locks a single tier for evaluation, the
        # relay must HOLD that tier (no Markov drift) so the level is a clean
        # independent variable. publish_episode_config carries this flag.
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

        # Optional whitelist restricting which bays the per-episode sampler may
        # target. None (default) samples from every eligible bay in the layout.
        # A non-empty list restricts the target pool to those bay ids, used by
        # the curriculum to introduce bay variety on a SUBSET (e.g. one row at a
        # single approach orientation) before opening up to the whole lot. It is
        # ignored when fixed_target_bay_id is set (a single fixed bay already
        # pins the target). Bay ids absent from the layout raise at pool build.
        _allowed = scenarios.get("allowed_bay_ids", None)
        self._allowed_bay_ids: Optional[List[str]] = (
            [str(b) for b in _allowed] if _allowed else None
        )

        # Soft out-of-bounds boundary: the lot polygon inflated by a margin forms
        # a run-off skirt; leaving it costs a small per-decision penalty that
        # accumulates until the episode terminates. Config overrides the
        # structural defaults in constants.py.
        self._oob_step_penalty: float = scenarios.get(
            "oob_step_penalty", OOB_STEP_PENALTY
        )
        self._oob_termination_limit: float = scenarios.get(
            "oob_termination_limit", OOB_TERMINATION_PENALTY_LIMIT
        )
        self._oob_inflation_margin: float = scenarios.get(
            "oob_inflation_margin", OOB_INFLATION_MARGIN
        )
        # Inflated boundary polygon and accumulated cost - both reset per episode.
        self._oob_inflated_corners: List[Tuple[float, float]] = []
        self._oob_accumulated_penalty: float = 0.0

        # CARLA handles
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # The spawn transform chosen for this episode (set in reset() before
        # any world ticks so the GNSS datum config is written first).
        self._chosen_spawn: Dict[str, float] = {}
        # Index of the chosen spawn in self._spawn_pool (0 = primary spawn,
        # 1+ = extra_spawn_transforms). Surfaced in step info as "spawn_id".
        self._chosen_spawn_idx: int = 0

        # Ego vehicle CoM z after settling under gravity.  Set in _spawn_vehicle()
        # and passed to all static spawners so props and vehicles land on the same
        # ground surface instead of using the hardcoded YAML origin_z.
        self._floor_z: float = 0.3

        # Ego bounding-box half-extents (metres) read from CARLA at spawn time.
        # Used by the polygon-fit success check in _compute_reward to test
        # whether every corner of the car lies inside the target bay polygon.
        self._ego_half_length: float = 0.0
        self._ego_half_width: float = 0.0

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
            spawn_perimeter_cones=scenarios.get("spawn_perimeter_cones", True),
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
        # Most recent normalised observation returned to the agent; reused as
        # the safe return value when a mid-decision CARLA failure aborts the
        # episode before a fresh observation can be built.
        self._last_norm_obs: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)
        self._obstacle_features_buffer: np.ndarray = np.zeros(
            OBSTACLE_FEATURES_DIM, dtype=np.float32
        )
        # World-frame pose buffer: [x, y, yaw, vyaw, vx_body]. vx_body is the
        # signed body-frame longitudinal velocity (m/s).
        self._world_pose_buf: np.ndarray = np.empty(5, dtype=np.float32)

        # Snapshot of the EKF-derived world pose from the most recent
        # _get_state() call, surfaced into step() info for trace logging and
        # EKF-vs-ground-truth accuracy checks. Layout matches _world_pose_buf:
        # [x, y, yaw, vyaw, vx_body]. _last_ekf_is_real is False on the CI/test
        # fallback path (no EKF available), so consumers can mark EKF columns
        # as not-a-number rather than logging ground truth as if it were EKF.
        self._last_ekf_world: np.ndarray = np.full(5, np.nan, dtype=np.float32)
        self._last_ekf_is_real: bool = False
        # EKF 1-sigma localisation stds [std_x, std_y, std_yaw] (m, m, rad)
        # from the same _get_state() read. The EKF runs for every baseline, so
        # this is populated regardless of include_covariance - the obs flag
        # only controls whether the policy SEES it, not whether it exists.
        # Surfaced into step() info for uncertainty-gating analysis.
        self._last_ekf_std: np.ndarray = np.full(3, np.nan, dtype=np.float32)

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
        # Last action applied (post-clamp): [steer, throttle, brake]. Updated
        # in step() on the policy decision boundary after the actuator model
        # has rate-limited the policy command.
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

        # CARLA 0.9.16 segfaults (SIGSEGV, container exit 139) on long headless
        # runs. With `restart: on-failure` on the carla-server service Docker
        # relaunches the engine within seconds, so the failure surfaces here as
        # a world.tick() RuntimeError. Rather than killing a multi-hour run, the
        # env reconnects to the fresh server and drops the in-flight episode.
        # max_reconnect_attempts caps the wait so a genuinely dead server (no
        # restart policy, crashed host) still fails loudly instead of hanging.
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

        # Episode state
        self._episode_id: int = 0
        self.steps = 0
        self._actors_frozen: bool = False
        self._success_counter: int = 0
        # Consecutive decisions below the success speed threshold while outside
        # the acceptance box; truncates the episode as a stall at
        # STALL_TRUNCATION_DECISIONS. @see step().
        self._stall_counter: int = 0
        # Previous corridor potential phi for the potential-based progress term
        # in _compute_reward. progress = phi(curr) - phi(prev) telescopes to zero
        # over any loiter, so a stationary car earns nothing from it.
        self._prev_phi: float = 0.0
        # Per-episode reward normaliser: |phi(start)| (floored). progress and the
        # graded timeout penalty are divided by this (then scaled by PROGRESS_TARGET)
        # so every bay - near or far - yields a full-episode progress sum of
        # PROGRESS_TARGET, and the timeout penalty stays on the same scale regardless
        # of how far the sampled bay is from the spawn. Set on reset; the floor avoids
        # a blow-up when the spawn is already near the bay. @see PHI_NORM_FLOOR.
        self._phi_start: float = PHI_NORM_FLOOR

        # Corridor potential weights (shaping, hence code not YAML; values in
        # constants.py). cross-track and heading are weighted above along-track so
        # the dominant progress gradient pulls the car onto the centreline and
        # square before advancing in depth. @see _corridor_potential.
        self._w_along: float = CORRIDOR_W_ALONG
        self._w_cross: float = CORRIDOR_W_CROSS
        self._w_head: float = CORRIDOR_W_HEAD
        # Clearance penalty coefficient (Gap B, safety nudge). NOTE: not yet
        # tuned against the offline scenario suite - the late-turn-crash vs
        # clean-park margin (B2 vs B1) is the acceptance bar; raise this if
        # neighbour-clipping persists while alignment improves.
        self._clearance_coef: float = 0.02

        # Cached CARLA zero objects - reused across calls to avoid per-call
        # construction overhead in _freeze_all_actors and _teleport_vehicle.
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

        # Action space: [steer, throttle, brake], uniformly in [-1, 1].
        # The policy outputs a tanh-squashed Gaussian on every axis. step()
        # passes steer through and remaps throttle / brake from [-1, 1] to
        # [0, 1] via (a + 1) / 2 so a held stop (throttle = 0, brake > 0)
        # remains a stable region of the policy's action space.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # ROS 2 covariance subscriber (only when covariance included)
        # EKF localisation runs for EVERY baseline - it is the source of the
        # pose, speed, and yaw-rate the policy navigates by, and the per-episode
        # GNSS noise that the EKF estimates. include_covariance controls ONLY
        # whether the 3 covariance feature dims (std_x/y/yaw) are written into
        # the observation vector (an obs-layout ablation), NOT whether the EKF
        # runs. Coupling the two would give the no-covariance baselines perfect
        # ground-truth localisation while the covariance baselines run on noisy
        # EKF - an unfair ablation. The subscriber is file-based (no DDS, no
        # connection), so constructing it is always safe; when ekf_state.json is
        # absent (CI/tests) get_latest_state() returns (None, None) and the env
        # falls back to CARLA ground truth in _get_state().
        self._cov_subscriber: Optional[_CovarianceSubscriber] = None
        self._init_ros2()

        # -------------------------------------------------------------------
        # Stored callables for fixed-flag hot-path branches
        # -------------------------------------------------------------------

        # _read_ekf_state() -> (raw_pose, uncertainty) or (None, None)
        if self._cov_subscriber is not None:
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

    # -----------------------------------------------------------------------
    # Observation dimension helper
    # -----------------------------------------------------------------------

    def _compute_obs_dim(self) -> int:
        """
        @brief Compute the observation dimension based on active feature flags.
        @return Integer observation dimension.
        """
        return compute_obs_dim(self._include_covariance, self._include_obstacle_obs)

    # -----------------------------------------------------------------------
    # ROS 2 initialisation
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # GNSS noise profile loading and tier sampling
    # -----------------------------------------------------------------------

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

    def _resolve_held_tier(self, tier_name: str) -> Dict[str, Any]:
        """
        @brief Map a held-tier name to its loaded fix-state tier dict.

        The eval conditions name the RTK fix-state tier to hold directly (e.g.
        rtk_float), so resolution is an exact name lookup against the loaded
        profiles. Names must match a tier in
        configs/deployment/sim/gnss_noise_profiles.yaml.

        @param tier_name: Fix-state tier name (from a held-tier override or the
               curriculum fixed_gnss_tier).
        @return The matching loaded tier dict.
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
        @brief Sample a GNSS noise tier for the current episode.

        When held_gnss_tier_override is set (evaluation mode), the named
        fix-state tier is resolved and HELD for the whole episode (the relay's
        Markov drift is suppressed via the hold flag), so the condition is a
        clean independent variable. Otherwise a tier is sampled and the relay
        wanders it via the Markov chain, exactly as in training.
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

        # Monotone-degradation condition: pin the START tier to rtk_fixed (so the
        # episode genuinely "starts good") and let the relay's one-way chain drift
        # it down to degraded without recovery. The downward-only behaviour is
        # enforced relay-side via the degrade_one_way flag in publish_episode_config.
        if self._degrade_one_way_override:
            tier = self._resolve_held_tier("rtk_fixed")
        # When a fixed tier is configured, bypass the weighted sampler.
        elif self._fixed_gnss_tier is not None:
            tier = self._resolve_held_tier(self._fixed_gnss_tier)
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

    def _get_live_gnss_tier_name(self) -> str:
        """
        @brief Return the GNSS fix-state tier name in force at this instant.

        self._current_gnss_tier holds only the START tier sampled at reset, while
        the relay walks the Markov chain at 20 Hz and writes the live tier back
        for the subscriber to read. Falls back to the start tier when the
        subscriber is absent (baselines without covariance, CI/tests) or no live
        tier has been published yet.

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

    # -----------------------------------------------------------------------
    # Floor plan loading and bay sampling
    # -----------------------------------------------------------------------

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

        # Restrict the target pool to the whitelist when one is set (and no
        # single bay is pinned). Validate every requested id resolves to an
        # eligible bay so a typo in a curriculum stage fails loud rather than
        # silently shrinking the pool.
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
        @brief Stratified sample of target bay: uniform over type, then uniform
               within type.

        In-scope target bay types are perpendicular and angled (forward-only
        parking). When `fixed_target_bay_id` is set the sampler returns that
        single bay and the type stratification is bypassed. Always-empty bays
        are excluded from target selection.

        @warning When the bay is NOT fixed, this samples uniformly over every
                 bay type present in the layout, which may include out-of-scope
                 types (e.g. motorcycle, parallel) if the layout contains them.
                 Filter `self._bay_type_keys` to the in-scope set before
                 unfixing the target bay for multi-bay training.
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
            # Drive task selection from the Gymnasium per-env RNG (seeded via
            # super().reset(seed=)) so the (target bay, spawn) sequence is
            # reproducible at a fixed training seed. np_random.choice cannot
            # index a list of dicts directly, so choose by integer index.
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

    # -----------------------------------------------------------------------
    # Actor spawning
    # -----------------------------------------------------------------------

    def _select_spawn(self) -> Dict[str, Any]:
        """
        @brief Choose a spawn transform for this episode from the pre-built pool.
        @return Chosen spawn dict with keys x, y, z, yaw_deg.

        Drawn from the Gymnasium per-env RNG so the spawn sequence is
        reproducible at a fixed training seed.
        """
        idx = int(self.np_random.integers(len(self._spawn_pool)))
        # Record the pool index as the episode's spawn id (0 = primary spawn,
        # 1+ = extra_spawn_transforms in layout order) so traces can report
        # where the episode started from.
        self._chosen_spawn_idx = idx
        return cast(Dict[str, Any], self._spawn_pool[idx])

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

    # -----------------------------------------------------------------------
    # Per-step NPC updates
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Clearance and reward
    # -----------------------------------------------------------------------

    def _bay_frame_pose(
        self, x: float, y: float, yaw: float
    ) -> Tuple[float, float, float]:
        """
        @brief Express the ego pose in the target bay's frame.
        @param x: Ego x position (metres, CARLA world frame).
        @param y: Ego y position (metres, CARLA world frame).
        @param yaw: Ego heading (radians, CARLA world frame).
        @return Tuple (along, cross, heading_err):
                - along: signed distance along the bay depth axis from the parked
                  position (the drive-in axis). The reward uses abs(along) because
                  the bays are open / back-to-back and either side is a valid
                  approach.
                - cross: perpendicular (cross-track) distance from the bay
                  centreline. Sign is irrelevant to the reward (abs is used).
                - heading_err: absolute heading error from the bay axis, wrapped
                  with 180-deg parking symmetry (in [0, pi/2]).
        @note This is the transpose of `_compute_relative_target_pose` (which
              gives target-in-ego); here we want ego-in-bay so the centreline and
              depth axes are the bay's, not the car's.
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
        @brief Bay-frame shaping potential phi for the progress term.
        @param along: Along-track distance from the parked depth (metres).
        @param cross: Cross-track distance from the centreline (metres).
        @param heading_err: Absolute heading error from the bay axis (radians).
        @return Potential phi (always <= 0); 0 at the perfectly parked pose.

        phi = -(W_ALONG*|along| + W_CROSS*|cross| + W_HEAD*heading_err). Linear and
        monotone in each coordinate, so reducing any of depth, cross-track, or
        heading error always raises phi (no traps, and driving in always pays). The
        cross-track and heading weights exceed the along-track weight so the
        gradient pulls the car onto the centreline and square before advancing in
        depth: a crooked short-cut accrues cross/heading cost en route and scores
        below an aligned arc that ends at the same parked pose.
        @see CORRIDOR_W_ALONG/CROSS/HEAD.
        """
        return -(
            self._w_along * abs(along)
            + self._w_cross * abs(cross)
            + self._w_head * heading_err
        )

    def _obstacle_clearance_penalty(self, on_line: float, aligned: float) -> float:
        """
        @brief Smooth clearance penalty for drifting toward a neighbouring car.
        @param on_line: Corridor on-centreline factor in [0, 1] (1 = on the line).
        @param aligned: Corridor heading-alignment factor in [0, 1] (1 = square).
        @return Penalty <= 0; 0 when no obstacle is inside the danger band or when
                the car is square on the centreline.

        Reads the hemispheric LiDAR clearance features in
        `self._obstacle_features_buffer` (a hemisphere with no return is 0.0, treated
        as infinite clearance). The forward cone is weighted above the side clearances
        since a return ahead while moving is the real collision risk. The penalty
        ramps from 0 at OBSTACLE_CLEARANCE_SAFE to its cap at OBSTACLE_CLEARANCE_DANGER
        and is multiplied by (1 - on_line*aligned), fading to zero once the car is
        square on the line, so a correct park (~0.98 m from a neighbour) is never
        penalised.
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

        # Forward cone is the collision-critical direction; side returns are
        # discounted (a square car alongside a neighbour is safe). The discount
        # widens the effective clearance of the side returns so they only register
        # when a corner genuinely swings in close.
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

        # position_error (Euclidean to bay centre) is diagnostic / terminal only: it
        # feeds the diag, the graded timeout penalty, and the metric, not the per-step
        # reward (the corridor potential shapes that).
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
            # Ego-fault penalised harder than non-fault. Magnitudes preserve
            # success(+50) > timeout(~0) > collision.
            reward = -25.0 if collision_ego_fault else -10.0
            return reward, True, False, diag

        # Success: every corner of the ego bounding box inside the bay
        # polygon and the vehicle essentially stopped.
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

        # Ego pose in the bay frame: how far in (along), how far off the
        # centreline (cross), and how square to the bay axis (heading_err).
        along, cross, heading_err = self._bay_frame_pose(x, y, yaw)

        # Corridor potential progress. progress = phi(curr) - phi(prev) telescopes to
        # zero over a loiter; the cross/heading weighting pulls the car onto the
        # centreline and square before advancing. Divided by the start potential and
        # scaled by PROGRESS_TARGET so every bay's full-episode progress sum is equal.
        # @see _corridor_potential, PROGRESS_TARGET, PHI_NORM_FLOOR.
        curr_phi = self._corridor_potential(along, cross, heading_err)
        progress = (curr_phi - self._prev_phi) / self._phi_start * PROGRESS_TARGET
        self._prev_phi = curr_phi

        # Corridor shaping factors, each in [0, 1]:
        #   on_line    - 1 on the centreline, 0 at the corridor half-width.
        #   near_depth - 1 at the parked depth, 0 at the along-track scale.
        #   aligned    - yaw straightness, saturates at the alignment cutoff.
        #   stopped    - sharp slowness against the success velocity (held stop).
        slowness_reference_speed = 5.0 * SUCCESS_THRESHOLD_VELOCITY
        on_line = max(0.0, 1.0 - abs(cross) / CORRIDOR_HALF_WIDTH)
        near_depth = max(0.0, 1.0 - abs(along) / ALONG_TRACK_SCALE)
        aligned = max(0.0, 1.0 - heading_err / APPROACH_INNER_ALIGNMENT_CUTOFF)
        stopped = max(0.0, 1.0 - speed / slowness_reference_speed)

        # Endgame finisher gated on the corridor factors: their conjunction
        # (on_line AND aligned AND near_depth) means a stop-short-and-crooked pose
        # earns ~0, and +HOLD*stopped sharpens the final commit to a held stop.
        endgame = (
            (ENDGAME_MOVE_COEF + ENDGAME_HOLD_COEF * stopped)
            * on_line
            * aligned
            * near_depth
        )

        # Obstacle clearance: smooth penalty for drifting toward a neighbouring parked
        # car during a crooked approach. Uses the observed LiDAR clearance (so it
        # transfers to hardware), anchored below the ~0.98 m parked-square side gap and
        # gated off once the car is square on the line. Raw, like the OOB penalty.
        clearance_term = self._obstacle_clearance_penalty(on_line, aligned)

        # Pure additive task reward; no single factor can zero it. Uncertainty is
        # input-only: it enters via the EKF covariance observation and the evidential
        # head, never the reward.
        reward = progress + endgame + clearance_term

        diag["progress_reward"] = float(progress)
        diag["endgame_reward"] = float(endgame)
        diag["clearance_penalty"] = float(clearance_term)

        # Soft out-of-bounds boundary, evaluated last so collision and success take
        # precedence. Applied raw (leaving the lot is a safety boundary, not shaping);
        # accumulates until the limit, then terminates with no extra crash cost. Tested
        # on CARLA ground truth, so it is exact regardless of EKF noise.
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

    # -----------------------------------------------------------------------
    # State
    # -----------------------------------------------------------------------

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
            # Snapshot the genuine EKF world pose for step() info / trace CSV.
            self._last_ekf_world[:] = self._world_pose_buf
            self._last_ekf_is_real = True
            if uncertainty is not None:
                self._last_ekf_std[:] = uncertainty
            else:
                self._last_ekf_std[:] = np.nan
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
            # GT fallback path: no genuine EKF estimate this step. Mark the
            # snapshot as not-real so trace consumers log EKF columns as NaN
            # rather than as a copy of ground truth.
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

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

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
                           One of: "collision", "out_of_bounds", "success",
                           "timeout". None for mid-episode frames.
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

        # Ground-truth yaw rate in REP-103 convention (left turn positive),
        # matching the EKF vyaw at obs[1] and the LiDAR bearing convention.
        gt_av_z_deg = self.vehicle.get_angular_velocity().z
        gt_vyaw = -math.radians(gt_av_z_deg)

        # Reuse the obs buffer populated earlier this step by _get_state()
        # rather than re-reading the EKF file.
        ekf_speed = float(self._obs_buffer[0])
        ekf_vyaw = float(self._obs_buffer[1])

        # Live tier, not the reset-time one: the visualiser shows this as the
        # headline field, so it must track the mid-episode Markov drift.
        gnss_tier_name = self._get_live_gnss_tier_name()

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
            # Clamped action sent to the vehicle (not the policy's pre-clip
            # output), so the HUD reflects what actually drove the car.
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

    # -----------------------------------------------------------------------
    # CARLA world management
    # -----------------------------------------------------------------------

    def _connect_to_carla(self) -> None:
        """
        @brief Connect to CARLA and load the FlatPlane OpenDRIVE world.

        Loads configs/layouts/flat_plane.xodr via generate_opendrive_world()
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(120.0)
            self.world = self.client.get_world()

            # generate_opendrive_world() leaves the world reporting
            # "Carla/Maps/OpenDriveMap", not "FlatPlane", so both names mean
            # the FlatPlane OpenDRIVE world is already present. Regenerating an
            # already-loaded world resets CARLA's elapsed-seconds sensor clock
            # to zero; the long-lived ros2-bridge EKF then rejects every fresh
            # transform as TF_OLD_DATA and stops writing ekf_state.json (the
            # evaluation sweep builds a fresh env per condition, so this fired
            # at every condition boundary). Skip regen when either name is seen.
            current_map_name = self.world.get_map().name.split("/")[-1]
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
        @brief Probe whether the CARLA RPC port is accepting connections.
        @return True if a TCP connection to (carla_host, carla_port) succeeds.

        Used to wait for a restarted CARLA server to finish booting before a
        reconnect attempt, so we do not race the engine's RPC startup.
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

        @note CARLA 0.9.16 segfaults on long headless runs; `restart: on-failure`
              on the carla-server service relaunches the engine. This polls for
              the RPC port to reopen, then rebuilds the client and reloads the
              FlatPlane world. Episode actors are NOT respawned here - the caller
              (reset) does that on the next episode. @see _connect_to_carla.
        """
        # Drop every handle into the dead server. The vehicle / sensor / NPC
        # actors live in the crashed engine and cannot be destroyed over RPC, so
        # forget them rather than calling _cleanup_actors (which would itself
        # time out). The fresh world starts empty; reset() repopulates it.
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

        if self.vehicle is not None:
            # CARLA's bounding_box.extent is half-lengths along the body axes:
            # x = forward (length/2), y = lateral (width/2).
            bb_extent = self.vehicle.bounding_box.extent
            self._ego_half_length = float(bb_extent.x)
            self._ego_half_width = float(bb_extent.y)

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

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

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

        # Push the Gymnasium per-env RNG (seeded above) into the spawner and NPC
        # controller so all per-episode placement draws (static cars, patrol
        # vehicles, pedestrians) share one reproducible stream at a fixed seed.
        self._lot_spawner.set_rng(self.np_random)
        self._npc_controller.set_rng(self.np_random)

        self._episode_id += 1
        self.steps = 0
        # Reset here, before any early-return path (CARLA unavailable /
        # reconnect failure), so a stall can never leak into the next episode.
        self._stall_counter = 0
        self._trajectory_buffer.clear()
        # Force signal-file recheck at episode start so new episodes don't
        # inherit a stale cached value from the previous episode's final step.
        self._vis_check_counter = 30

        # Reopen the vis history file in write mode periodically to truncate
        # it. The visualiser detects the shrink and resets its read offset.
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

        # Recover from a CARLA server crash flagged by the previous step().
        # Must run before _cleanup_actors() below, whose destroy() RPCs would
        # time out against the dead engine. _reconnect_to_carla forgets all
        # stale actor handles and waits for the restarted server, so the reset
        # then proceeds as a fresh full spawn into the new world.
        if self._needs_reconnect:
            reconnected = self._reconnect_to_carla()
            self._needs_reconnect = False
            # Actors were never destroyed over RPC (server was dead), so no
            # destroy commands are pending - skip the flush tick below.
            self._actors_frozen = True
            if not reconnected:
                return np.zeros(self._obs_dim, dtype=np.float32), {}

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
            # Teleport the existing vehicle; sensors stay attached and alive.
            self._teleport_vehicle()
            self._sensor_manager.reset_state()
        else:
            self._spawn_vehicle()
            self._spawn_sensors()

        # Invalidate stale pre-reset EKF data so _wait_for_covariance() blocks
        # until a genuinely post-spawn reading arrives from ekf_state.json.
        if self._cov_subscriber is not None:
            self._cov_subscriber.invalidate()

        # Publish spawn pose as local (0, 0, yaw) so /set_pose seeds the EKF
        # at local origin, matching the re-latched GNSS datum frame.
        if self._cov_subscriber is not None and self._ros2_config.get(
            "publish_initial_pose", False
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

        if self._cov_subscriber is not None:
            # Wait once to confirm the extractor is writing, tick the world
            # long enough for the EKF to consume the /set_pose published from
            # initial_pose.json, then invalidate and wait again so only
            # post-reset EKF state is accepted.
            self._wait_for_covariance()
            if self.world is not None:
                for _ in range(10):
                    self.world.tick(10.0)
            self._cov_subscriber.invalidate()
            self._wait_for_covariance()
            # Always recalibrate: the GNSS datum is re-latched to the spawn
            # position at each episode reset, so the EKF odom origin shifts
            # every episode.
            self._calibrate_ekf_frame_offset()

        # Initialise prev_distance for potential-based reward shaping
        self._success_counter = 0
        # Reset the soft out-of-bounds accumulator and recompute the inflated
        # boundary once per episode (lot corners are fixed for a floor plan).
        self._oob_accumulated_penalty = 0.0
        lot_corners = [
            (float(c["x"]), float(c["y"]))
            for c in self._current_layout.get("corners", [])
        ]
        self._oob_inflated_corners = inflate_polygon(
            lot_corners, self._oob_inflation_margin
        )
        self._last_action[:] = 0.0
        # Actuator-side previous commands reset to the zero rest state each
        # episode so the first decision starts from zero throttle / brake /
        # centred steering, matching the policy's first observation.
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
            # Fix the per-episode normaliser from the start potential. phi is <= 0
            # and 0 at the parked pose, so |phi(start)| is the total potential the
            # car must close to park; dividing progress by it makes the full-episode
            # progress sum equal to PROGRESS_TARGET for every bay. Floored to avoid a
            # blow-up when the spawn already sits near the bay.
            self._phi_start = max(abs(self._prev_phi), PHI_NORM_FLOOR)
        else:
            self._prev_phi = 0.0
            self._phi_start = PHI_NORM_FLOOR

        state = self._get_state()
        self._last_norm_obs = state

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
        @brief Execute one policy decision (action_repeat sim ticks).

        One step() call is one agent transition: the rate-limited command is
        held constant while CARLA advances action_repeat fixed timesteps, then
        the observation and reward are built once from the post-tick state.
        The agent never sees intermediate sim ticks, so every transition pairs
        the decision with the normalised observation that resulted from it,
        and SB3 timesteps count policy decisions, not sim ticks.

        @param action: 3-dim action vector [steer, throttle, brake] in the
                policy's normalised command space. Steer in [-1, 1], throttle
                and brake in [0, 1]. The env rate-limits the command via the
                actuator model before forwarding to CARLA, so what the policy
                commands and what the vehicle physically receives may differ.
        @return Tuple of (observation, reward, terminated, truncated, info).
        """
        # Rate-limit the raw policy command so the action delivered to CARLA
        # matches a physical actuator. The post-clamp values are persisted via
        # _prev_*_cmd and held constant for the action_repeat sim ticks of this
        # decision (~0.2 s).
        # Policy axes are uniformly [-1, 1] (tanh-squashed). Steer passes
        # through. Throttle / brake fold the negative half to zero so the
        # symmetric prior gamma = 0 maps to pedal-off; only the positive
        # half engages the pedal. A linear (a + 1) / 2 remap put both
        # pedals at 0.5 at init, which triggered the brake-overrides-
        # throttle override and prevented any movement at training start.
        steer_cmd = float(np.clip(action[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(action[1], 0.0, 1.0))
        brake_cmd = float(np.clip(action[2], 0.0, 1.0))

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

        # Brake-overrides-throttle. A real driver-assistance system cuts
        # throttle whenever the brake is meaningfully pressed; modelling
        # the same here keeps the policy from learning to fight itself.
        # Apply the override BEFORE the throttle rate limit so the cut
        # itself still respects the actuator rate (a real throttle plate
        # cannot slam closed faster than its slew rate).
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
            # 10s timeout surfaces a frozen CARLA server as an error rather
            # than hanging the process indefinitely. CARLA 0.9.16 segfaults
            # on long headless runs; a crash here is recoverable - flag the
            # env for reconnect on the next reset() and abort the episode as
            # a truncation so SB3 ends it cleanly rather than the whole
            # multi-hour run dying on the exception. @see _reconnect_to_carla.
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

        # Stall truncation: a car holding a near-stop OUTSIDE the acceptance box
        # for STALL_TRUNCATION_DECISIONS consecutive decisions is parked in the
        # wrong place and (forward-only, no reverse) almost never recovers -
        # truncate through the same path as the clock running out, so the graded
        # timeout penalty below applies identically. _success_counter > 0 means
        # the car is slow INSIDE the box (the success dwell), which must not
        # count as a stall. The window is sized above the GNSS Markov chain's
        # full recovery time, so stopping to wait out a degraded fix is never
        # cut short. @see STALL_TRUNCATION_DECISIONS.
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
            # Suspend the stall counter while localisation is genuinely degraded:
            # a near-stop there is a legitimate wait-for-recovery, not a stall,
            # and must not be punished (the behaviour the covariance induces). The
            # counter is HELD (neither incremented nor reset) during the wait, so
            # a real good-fix stall briefly interrupted by a degraded blip resumes
            # rather than restarting from zero.
            waiting_out_bad_fix = ekf_std_pos > STALL_GATE_EKF_STD_M
            near_stop = (
                reward_diag["speed"] < SUCCESS_THRESHOLD_VELOCITY
                and self._success_counter == 0
            )
            if not waiting_out_bad_fix:
                if near_stop:
                    self._stall_counter += 1
                else:
                    self._stall_counter = 0
            if self._stall_counter >= STALL_TRUNCATION_DECISIONS:
                truncated = True

        # Graded timeout penalty: judge the final state when the clock runs out
        # without a park. Scaled by how far and how misaligned the car ended, so
        # ending closer and straighter is always less costly than stopping short.
        # Transformed by /|phi(start)| * PROGRESS_TARGET (the same normaliser the
        # corridor progress uses) so the penalty lives on the dense progress scale
        # and a far-bay timeout is not penalised more heavily than a near-bay one
        # merely for being far. Clamped to TIMEOUT_PENALTY_FLOOR_NORM so timing out
        # costs about as much as forfeiting the whole progress reward and the fixed
        # terminals stay far larger, keeping success(+50) > timeout > ego
        # collision(-25).
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
            # read back from the relay's Markov drift (see above). Lets timing
            # analysis recover the step the drift first reaches a degraded tier
            # (the onset reference for handover-latency), which varies per
            # episode under the degrade_one_way drift.
            "gnss_tier": _live_tier,
            "gnss_multiplier": _live_multiplier,
            # Target bay this episode (id, world pose, dimensions). Constant
            # within an episode; surfaced so trace tooling can record which bay
            # the run targeted without reaching into the env internals.
            "target_bay": dict(self._target_bay),
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
                # Chase camera: behind and above the ego, looking down at it. A
                # top-down view would only duplicate the 2D bird's-eye viewer,
                # whereas the 3D window earns its place by showing the vehicle
                # against the lot. Offsets are in metres.
                yaw_rad = math.radians(transform.rotation.yaw)
                offset = carla.Location(
                    x=-_SPECTATOR_BACK_M * math.cos(yaw_rad),
                    y=-_SPECTATOR_BACK_M * math.sin(yaw_rad),
                    z=_SPECTATOR_UP_M,
                )
                spectator.set_transform(
                    carla.Transform(
                        transform.location + offset,
                        carla.Rotation(
                            pitch=_SPECTATOR_PITCH_DEG,
                            yaw=transform.rotation.yaw,
                        ),
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
