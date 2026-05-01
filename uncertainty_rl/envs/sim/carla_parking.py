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
from typing import Any, Deque, Dict, List, Optional, Tuple, cast

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from gymnasium.core import RenderFrame

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.envs.sim.helpers import LotSpawner, NPCController, SensorManager
from uncertainty_rl.envs.covariance_subscriber import _CovarianceSubscriber
from uncertainty_rl.envs._parking_core import (
    _layout_cache as _shared_layout_cache,
    build_observation,
    calibrate_ekf_frame_offset,
    compute_obs_dim,
    extract_obstacle_features,
    load_floor_plan,
    wait_for_ekf,
)
from uncertainty_rl.utils.constants import (
    OBSTACLE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    wrap_angle_symmetric,
)
from uncertainty_rl.utils.logging import DebugLogger

logger = logging.getLogger(__name__)

# Trail length for debug overlays and vis state
_TRAJECTORY_MAXLEN = 50

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

        # Per-step uncertainty estimates set externally (by policy or wrapper).
        # Used for uncertainty-aware reward shaping when enabled.
        self._step_epistemic: float = 0.0
        self._step_aleatoric: float = 0.0

        # Load GNSS noise profiles for per-episode RTK fix-state sampling.
        self._gnss_noise_tiers: List[Dict[str, Any]] = []
        self._gnss_tier_weights: List[float] = []
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

        # CARLA handles
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.vehicle: Optional[Any] = None

        # The spawn transform chosen for this episode (set in _spawn_vehicle()).
        # Used for /set_pose publishing to seed the EKF on reset.
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
        self._obs_buffer: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)
        self._obstacle_features_buffer: np.ndarray = np.zeros(
            OBSTACLE_FEATURES_DIM, dtype=np.float32
        )

        # Full 2D rigid body transform from EKF odom frame to CARLA world frame,
        # computed once per episode in _calibrate_ekf_frame_offset() using the
        # known YAML spawn position paired with the EKF odom reading at spawn.
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

        # Current floor plan layout (loaded from YAML in reset)
        self._current_layout: Dict[str, Any] = {}
        self._current_floor_plan_name: str = ""

        # Trajectory buffer for debug overlays (ring buffer of (x, y) tuples)
        self._trajectory_buffer: Deque[Tuple[float, float]] = collections.deque(
            maxlen=_TRAJECTORY_MAXLEN
        )
        # Last action applied 
        self._last_action = np.zeros(3, dtype=np.float32)

        # Visualisation state writer
        self._vis_history_path: Path = (
            Path(vis_output_path)
            if vis_output_path
            else Path("outputs/vis_history.jsonl")
        )
        self._vis_signal_path: Path = self._vis_history_path.parent / ".vis_active"
        self._carla_timestep: float = carla_timestep

        self._action_repeat: int = action_repeat
        self._no_rendering_mode: bool = no_rendering_mode
        self._action_repeat_counter: int = 0

        # Episode state
        self._episode_id: int = 0
        self.steps = 0
        self._actors_frozen: bool = False
        # Previous distance to target for potential-based reward shaping
        self._prev_distance: float = 0.0

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(_obs_dim,),
            dtype=np.float32,
        )

        # Action space: [throttle, steer, brake], all in [-1, 1]. Throttle and brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

        # ROS 2 covariance subscriber (only when covariance included)
        self._cov_subscriber: Optional[_CovarianceSubscriber] = None
        if self._include_covariance:
            self._init_ros2()

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
        self._gnss_tier_weights = []
        for name, tier in tiers.items():
            tier["name"] = name
            self._gnss_noise_tiers.append(tier)
            self._gnss_tier_weights.append(float(tier.get("weight", 1.0)))

        # Normalise weights to sum to 1.0
        total = sum(self._gnss_tier_weights)
        if total > 0:
            self._gnss_tier_weights = [w / total for w in self._gnss_tier_weights]

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

        if not self._gnss_noise_tiers:
            self._current_gnss_multiplier = 1.0
            self._current_gnss_tier = None
            return

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
        )
        self._current_floor_plan_name = name
        self._current_layout = layout
        logger.info("Loaded floor plan: %s", name)

    def _sample_target_bay(self) -> None:
        """
        @brief Stratified sample of target bay: 1/3 per type, then uniform within type.

        Bay types: perpendicular, angled, parallel.
        Always-empty bays are excluded from target selection.
        """
        bays = [
            b for b in self._current_layout.get("bays", [])
            if not b.get("always_empty", False)
        ]

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for bay in bays:
            bay_type = bay.get("bay_type", "perpendicular")
            by_type.setdefault(bay_type, []).append(bay)

        if not by_type:
            raise RuntimeError("No eligible bays found in floor plan layout.")

        bay_type = random.choice(list(by_type.keys()))
        target = random.choice(by_type[bay_type])

        self._target_bay = {
            "x": float(target["x"]),
            "y": float(target["y"]),
            "yaw": (
                float(target["yaw"])
                if "yaw" in target
                else math.radians(float(target.get("yaw_deg", 0.0)))
            ),
            "width": float(target.get("width", 2.5)),
            "depth": float(target.get("depth", 5.0)),
            "bay_type": bay_type,
            "bay_id": target.get("id", target.get("bay_id", "")),
        }
        logger.info(
            "Target bay: type=%s, id=%s, x=%.1f, y=%.1f",
            self._target_bay.get("bay_type", ""),
            self._target_bay.get("bay_id", ""),
            self._target_bay["x"],
            self._target_bay["y"],
        )

    # ------------------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------------------

    def _select_spawn(self) -> Dict[str, Any]:
        """
        @brief Choose a spawn transform for this episode from the layout YAML.
        @return Chosen spawn dict with keys x, y, z, yaw_deg.
        """
        primary = self._current_layout.get("spawn_transform", {})
        extras: List[Any] = (
            self._current_layout.get("extra_spawn_transforms", [])
            if self._use_extra_spawns
            else []
        )
        return random.choice([primary] + list(extras))

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

    def set_step_uncertainty(
        self,
        epistemic: float,
        aleatoric: float,
    ) -> None:
        """
        @brief Set the current step's uncertainty estimates from the policy.
        @param epistemic: Mean epistemic uncertainty from evidential actor.
        @param aleatoric: Mean aleatoric uncertainty from evidential actor.
        """
        self._step_epistemic = epistemic
        self._step_aleatoric = aleatoric

    def _compute_reward(self) -> Tuple[float, bool, bool, Dict[str, float]]:
        """
        @brief Compute reward and termination flags using CARLA ground truth.

        @return Tuple of (reward, terminated, success, diagnostics).
        """
        if self.vehicle is None:
            return 0.0, False, False, {
                "pos_error": 0.0,
                "orientation_error": 0.0,
                "speed": 0.0,
                "collision": 0.0,
                "progress_reward": 0.0,
            }

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2)

        target_x = float(self._target_bay["x"])
        target_y = float(self._target_bay["y"])
        target_yaw = float(self._target_bay["yaw"])

        position_error = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        orientation_error = abs(wrap_angle_symmetric(yaw - target_yaw))

        diag: Dict[str, float] = {
            "pos_error": position_error,
            "orientation_error": orientation_error,
            "speed": speed,
            "collision": 0.0,
            "progress_reward": 0.0,
        }

        collision_detected, collision_ego_fault = self._sensor_manager.consume_collision()
        if collision_detected:
            self._prev_distance = position_error
            diag["collision"] = 1.0
            reward = -10.0 if collision_ego_fault else 0.0
            return reward, True, False, diag

        success = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
            and speed < SUCCESS_THRESHOLD_VELOCITY
        )
        if success:
            self._prev_distance = position_error
            return 10.0, True, True, diag

        progress = (self._prev_distance - position_error) / OUT_OF_BOUNDS_THRESHOLD
        self._prev_distance = position_error

        # Scale progress reward by localisation quality. std_x/std_y are at
        # obs indices 1 and 2; read directly from the pre-built obs buffer so
        # the reward sees the same values the policy sees.
        std_x = float(self._obs_buffer[1]) if self._include_covariance else 0.0
        std_y = float(self._obs_buffer[2]) if self._include_covariance else 0.0
        uncertainty_scale = float(
            np.clip(max(std_x, std_y) / self._uncertainty_std_max, 0.0, 1.0)
        )
        reward = progress * (1.0 - uncertainty_scale) - 0.01

        diag["progress_reward"] = float(progress)
        diag["uncertainty_scale"] = uncertainty_scale
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
            return np.zeros(len(self._obs_buffer), dtype=np.float32)

        # Read EKF pose + uncertainty in one file read.
        raw_ekf_pose: Optional[np.ndarray] = None
        uncertainty: Optional[np.ndarray] = None
        if self._cov_subscriber is not None:
            raw_ekf_pose, uncertainty = self._cov_subscriber.get_latest_state()

        # Resolve EKF odom pose -> world frame.
        world_pose: Optional[np.ndarray] = None
        if raw_ekf_pose is not None:
            ekf_odom_x = float(raw_ekf_pose[0])
            ekf_odom_y = -float(raw_ekf_pose[1])
            ekf_odom_yaw = float(raw_ekf_pose[2])
            tx, ty, cos_r, sin_r, r = self._ekf_odom_offset
            wx = cos_r * ekf_odom_x - sin_r * ekf_odom_y + tx
            wy = sin_r * ekf_odom_x + cos_r * ekf_odom_y + ty
            wyaw = ekf_odom_yaw + r
            world_pose = np.array(
                [wx, wy, wyaw, float(raw_ekf_pose[3])],
                dtype=np.float32,
            )
        else:
            # CI / tests fallback: use CARLA GT (no EKF available)
            self._debug_logger._logger.debug(
                "[state] EKF pose unavailable at step %d - using CARLA GT",
                self.steps,
            )
            t = self.vehicle.get_transform()
            av = self.vehicle.get_angular_velocity()
            world_pose = np.array(
                [t.location.x, t.location.y, math.radians(t.rotation.yaw),
                 math.radians(av.z)],
                dtype=np.float32,
            )

        obstacle_features = extract_obstacle_features(
            self._sensor_manager.get_latest_lidar_scan()
            if self._include_obstacle_obs else None,
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

    def _write_vis_state(self, end_reason: Optional[str] = None) -> None:
        """
        @brief Append a visualisation frame to the JSONL history file.

        Writing only occurs when the signal file (outputs/.vis_active) exists,
        which is created by the visualiser process. This avoids unnecessary I/O
        when nobody is watching. Each line is a complete JSON object with
        sim_time so the visualiser can pace playback at real-time speed.

        @param end_reason: If set, written into the frame as "end_reason" so
                           the visualiser can log why the episode terminated.
                           One of: "collision", "success", "timeout".
                           None for mid-episode frames.
        """
        if self.vehicle is None:
            return

        # Only write when the visualiser is actively watching
        if not self._vis_signal_path.exists():
            return

        transform = self.vehicle.get_transform()
        vel = self.vehicle.get_velocity()
        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw

        patrol_npcs = self._npc_controller.patrol_npcs
        actor_transforms = []
        for actor in patrol_npcs + self._lot_spawner.spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                at = actor.get_transform()
                actor_transforms.append(
                    {
                        "x": at.location.x,
                        "y": at.location.y,
                        "yaw": at.rotation.yaw,
                        "type": "npc" if actor in patrol_npcs else "static",
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
                "vx": vel.x,
                "vy": vel.y,
                "speed": math.sqrt(vel.x**2 + vel.y**2),
            },
            "action": {
                "steer": float(self._last_action[0]),
                "longitudinal": float(self._last_action[1]),
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
            json_str = json.dumps(state)
            self._vis_history_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._vis_history_path, "a") as f:
                f.write(json_str + "\n")
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
        chosen = self._select_spawn()
        # Store spawn transform so LotSpawner can open this entry gap in cones.
        self._chosen_spawn = chosen
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
        chosen = self._select_spawn()
        self._chosen_spawn = chosen

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
        zero = carla.Vector3D(x=0.0, y=0.0, z=0.0)
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

        tick_fn = (lambda: self.world.tick(10.0)) if self.world is not None else None
        wait_for_ekf(
            has_lidar=lambda: self._sensor_manager.get_latest_lidar_scan() is not None,
            has_ekf=lambda: self._cov_subscriber.has_data,  # type: ignore[union-attr]
            timeout=self._covariance_timeout,
            tick_fn=tick_fn,
        )

    def _calibrate_ekf_frame_offset(self) -> None:
        """
        @brief Resolve the spawn reference position and delegate EKF
               convergence to calibrate_ekf_frame_offset() in _parking_core.
        """
        if self._cov_subscriber is None:
            return

        world_x = float(self._chosen_spawn.get("x", 0.0))
        world_y = float(self._chosen_spawn.get("y", 0.0))
        world_yaw = math.radians(float(self._chosen_spawn.get("yaw_deg", 0.0)))

        tick_fn = (lambda: self.world.tick(10.0)) if self.world is not None else None
        self._ekf_odom_offset = calibrate_ekf_frame_offset(
            world_x=world_x,
            world_y=world_y,
            world_yaw=world_yaw,
            get_pose=self._cov_subscriber.get_latest_pose,
            timeout=self._ekf_convergence_timeout,
            tick_fn=tick_fn,
        )

    def _freeze_all_actors(self) -> None:
        """
        @brief Zero velocity on all moving actors immediately on episode end.

        Called as soon as terminated or truncated is True so actors do not
        continue on their last command while reset() tears down the episode.
        Uses set_target_velocity for instant stops rather than brake control,
        which takes multiple ticks to converge through physics.
        """
        zero = carla.Vector3D(x=0.0, y=0.0, z=0.0)

        if self.vehicle is not None and self.vehicle.is_alive:
            self.vehicle.enable_constant_velocity(zero)

        for npc in self._npc_controller.patrol_npcs:
            if npc is not None and npc.is_alive:
                # NPCs are destroyed in cleanup so the pin does not carry over.
                npc.enable_constant_velocity(zero)

        for walker in self._npc_controller.pedestrian_actors:
            if walker is not None and walker.is_alive:
                ctrl = carla.WalkerControl()
                ctrl.speed = 0.0
                walker.apply_control(ctrl)

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

        # Sample GNSS noise tier for this episode (RTK fix-state variation).
        # Must happen before _spawn_sensors() so the multiplier is available.
        self._sample_gnss_noise_tier()

        # Connect to CARLA on first reset
        if self.client is None:
            self._connect_to_carla()

        if self.world is None:
            return np.zeros(len(self._obs_buffer), dtype=np.float32), {}

        # Determine whether to reuse the existing vehicle and sensors.
        # On the first episode (vehicle is None) or if the actor has gone stale,
        # do a full spawn.  On all subsequent episodes, teleport instead to avoid
        # the destroy/respawn cycle that causes the CARLA ROS bridge to accumulate
        # actor-stream registrations and eventually segfault (exit code -11).
        vehicle_alive = self.vehicle is not None and self.vehicle.is_alive
        reuse_vehicle = vehicle_alive

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
        # rather than relying on the bridge.
        if self.world is not None:
            settings = self.world.get_settings()
            if not settings.synchronous_mode:
                logger.info(
                    "Applying synchronous mode (fixed_delta=%.3fs) ...",
                    self._carla_timestep,
                )
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self._carla_timestep
                self.world.apply_settings(settings)
                logger.info("Synchronous mode enabled.")
            else:
                logger.info(
                    "Synchronous mode already active "
                    f"(fixed_delta={settings.fixed_delta_seconds}s)."
                )

            # Apply no_rendering_mode if enabled. Disables Unreal rendering pipeline
            # but physics and state sensors remain active. For state-based agents
            # (no camera input), this provides 3-4x speedup by skipping GPU rendering.
            if self._no_rendering_mode:
                settings = self.world.get_settings()
                if not settings.no_rendering_mode:
                    logger.info("Enabling no_rendering_mode (state-based agent, no cameras)...")
                    settings.no_rendering_mode = True
                    self.world.apply_settings(settings)
                    logger.info("No rendering mode enabled. Expected speedup: 3-4x.")

        # Load floor plan and sample target bay
        if self._floor_plans_config:
            self._load_floor_plan()
            self._sample_target_bay()

        # Cache blueprint lists once before any spawning to avoid repeated
        # world queries inside each spawn method.
        self._cache_blueprints()

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

        # Spawn coordinates used for GNSS datum and /set_pose.
        sx = float(self._chosen_spawn.get("x", 0.0))
        sy = float(self._chosen_spawn.get("y", 0.0))
        syaw = math.radians(float(self._chosen_spawn.get("yaw_deg", 0.0)))

        # Step 1: Signal GNSS tier AND spawn datum to the ros2-bridge.
        # The datum re-latch MUST happen before /set_pose so the GNSS local
        # frame is re-zeroed at the spawn position before the EKF receives
        # its initial state. Without this, /set_pose and GNSS Odometry use
        # different frame origins and the EKF drifts toward the stale datum
        # over the first 5-10 seconds of every episode.
        if (
            self._include_covariance
            and self._cov_subscriber is not None
            and self._gnss_noise_tiers
        ):
            datum_lat: Optional[float] = None
            datum_lon: Optional[float] = None
            if self.world is not None:
                try:
                    geo = self.world.get_map().transform_to_geolocation(
                        carla.Location(sx, sy, 0.0)
                    )
                    datum_lat = float(geo.latitude)
                    datum_lon = float(geo.longitude)
                except Exception as exc:
                    logger.warning(
                        f"Failed to get spawn geolocation for GNSS datum: {exc}"
                    )
            tier = self._get_current_gnss_tier()
            if tier is not None:
                self._cov_subscriber.publish_gnss_noise_config(
                    tier_name=str(tier.get("name", "")),
                    datum_lat=datum_lat,
                    datum_lon=datum_lon,
                    spawn_yaw=-syaw,
                )

        # Step 2: Publish spawn pose as local (0, 0, yaw) so /set_pose agrees
        # with the re-latched GNSS frame. The datum is at the spawn position,
        # so the vehicle starts at local origin (0, 0) in both the GNSS
        # Odometry and the EKF state - no frame mismatch, no drift.
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
            self._wait_for_covariance()
            # Always recalibrate: the GNSS datum is re-latched to the spawn
            # position at each episode reset, so the EKF odom origin shifts
            # every episode.
            self._calibrate_ekf_frame_offset()

        # Initialise prev_distance for potential-based reward shaping
        if self.vehicle is not None:
            t = self.vehicle.get_transform()
            self._prev_distance = math.sqrt(
                (t.location.x - self._target_bay["x"]) ** 2
                + (t.location.y - self._target_bay["y"]) ** 2
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

        @param action: 2-dim action vector [steering, longitudinal].
                steering     in [-1, 1]: left to right.
                longitudinal in [-1, 1]: negative = brake, positive = throttle.
                Mapped to CARLA throttle/brake internally.
        @return Tuple of (observation, reward, terminated, truncated, info).
        """
        # On a new action (or first call), reset the repeat counter
        if self._action_repeat_counter == 0:
            self._last_action = action.copy()

        # Execute one sim-step
        self.steps += 1
        self._action_repeat_counter += 1

        if self.vehicle is not None:
            steer = float(np.clip(action[0], -1.0, 1.0))
            longitudinal = float(np.clip(action[1], -1.0, 1.0))

            # Split longitudinal into CARLA throttle/brake.
            # Positive longitudinal -> throttle, negative -> brake.
            control = carla.VehicleControl()
            control.steer = steer
            control.brake = float(max(-longitudinal, 0.0))

            # Cut throttle when speed limit is exceeded. Appropriate for
            # parking lot manoeuvres (< 3 m/s).
            vel = self.vehicle.get_velocity()
            current_speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
            if current_speed >= self._max_ego_speed_ms:
                control.throttle = 0.0
            else:
                control.throttle = float(max(longitudinal, 0.0))

            self.vehicle.apply_control(control)

            if self.world is not None:
                self._update_patrol_npcs()
                self._update_pedestrians()
                # 10s timeout surfaces a frozen CARLA server as an error rather
                # than hanging the process indefinitely.
                self.world.tick(10.0)

                # Update trajectory buffer for vis state writer
                t = self.vehicle.get_transform()
                self._trajectory_buffer.append((t.location.x, t.location.y))

        # Only construct observations and compute rewards on the final repeat step.
        # Intermediate steps return minimal state to avoid EKF covariance reads.
        if self._action_repeat_counter < self._action_repeat:
            # Intermediate repeat step: return cached state, continue action
            cached_state = self._get_state()
            return cached_state, 0.0, False, False, {}

        # Final repeat step: construct full observation and reward
        self._action_repeat_counter = 0  # Reset for next action
        state = self._get_state()
        reward, terminated, success, reward_diag = self._compute_reward()

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

        self._write_vis_state(end_reason=end_reason)

        info: Dict[str, Any] = {
            "steps": self.steps,
            "success": success,
            "collision": bool(reward_diag["collision"]),
            "timeout": truncated,
            "floor_plan": self._current_floor_plan_name,
            # Per-step reward diagnostics for TensorBoard callback
            "pos_error": reward_diag["pos_error"],
            "orientation_error": reward_diag["orientation_error"],
            "speed": reward_diag["speed"],
            "progress_reward": reward_diag["progress_reward"],
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

        self.client = None
        self.world = None

        # Remove the signal file so the visualiser knows training has ended
        # and can exit cleanly rather than waiting indefinitely for new frames.
        try:
            self._vis_signal_path.unlink(missing_ok=True)
        except OSError:
            pass

        super().close()
