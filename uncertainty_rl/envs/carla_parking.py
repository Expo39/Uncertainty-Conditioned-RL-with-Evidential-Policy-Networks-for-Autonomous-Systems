"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance and lot geometry.

This module implements a Gymnasium-compatible environment for autonomous parking
in CARLA simulator. The agent parks in one of three floor-plan geometries loaded
from pre-computed layout YAMLs (configs/layouts/). Localisation uncertainty comes
from the robot_localisation EKF node (via ROS 2 DDS), driven by noisy CARLA
sensors, weather conditions, and dynamic traffic - not from a simulated noise model.

The observation comprises up to 20 dimensions (default, include_obstacle_obs=true):
  - indices  0-5:  EKF filtered pose (x, y, yaw, vx, vy, vyaw)
  - indices  6-14: EKF covariance features (std_x, std_y, std_yaw,
                   cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw)
  - indices 15-17: target bay in ego body frame (dx, dy, dyaw)
  - indices 18-19: nearest obstacle (distance_m, bearing_rad)
                   only present when include_obstacle_obs=true (default)

Actual obs dim depends on include_covariance and include_obstacle_obs flags;
use _compute_obs_dim() rather than hardcoding dimensions directly inside the env.

CARLA ground truth is used only for reward computation (position error, collision
detection) not in the observation. This ensures sim-to-real transfer without retraining.
"""

import collections
import json
import logging
import math
import os
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

try:
    import rclpy

    _ROS2_AVAILABLE = True
except ImportError:
    _ROS2_AVAILABLE = False

from uncertainty_rl.envs._lot_spawner import LotSpawner
from uncertainty_rl.envs._npc_controller import NPCController
from uncertainty_rl.envs._sensor_manager import SensorManager
from uncertainty_rl.envs.covariance_subscriber import _CovarianceSubscriber
from uncertainty_rl.utils.logging import DebugLogger
from uncertainty_rl.utils.constants import (
    COVARIANCE_FEATURES_DIM,
    MAX_PARKING_SPEED,
    OBSTACLE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
    TARGET_POSE_DIM,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.geometry import _compute_relative_target_pose

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
    manoeuvre the ego vehicle into the target bay. Uncertainty is produced
    naturally by the robot_localisation EKF processing noisy CARLA sensors.

    Observation space when include_covariance=True, include_obstacle_obs=True (20-dim):
      [0-5]   EKF pose: x, y, yaw, vx, vy, vyaw
      [6-14]  EKF covariance features
      [15-17] target bay in ego body frame
      [18-19] obstacle awareness: nearest_dist_m, nearest_bearing_rad

    When include_covariance=False (9-dim or 11-dim depending on include_obstacle_obs):
      [0-5]   EKF pose
      [6-8]   target bay in ego body frame
      [9-10]  obstacle awareness (only when include_obstacle_obs=True)

    Setting include_obstacle_obs=False removes obs indices 18-19 (reverts 20->18 dim,
    or 11->9 dim). Set in train_config.yaml: include_obstacle_obs: false.

    @note Docker + ROS 2 required for training. No standalone fallback.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Class-level layout cache shared across all instances to avoid re-reading
    # the same YAML from disk on every episode reset.
    _layout_cache: Dict[str, Any] = {}

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
        sensor_suite: str = "suite_a",
        vis_output_path: Optional[str] = None,
        carla_timestep: float = 0.05,
        eval_mode: bool = False,
        debug: bool = False,
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
        @param include_covariance: If True, obs includes EKF covariance features
               (indices 6-14). If False, covariance omitted and no ROS 2 subscription.
        @param include_obstacle_obs: If True, obs includes 2 obstacle awareness dims
               (nearest_dist_m, nearest_bearing_rad). Set False to revert to 18-dim
               obs without changing any other code.
        @param sensor_suite: Sensor suite to spawn ('suite_a', 'suite_b', 'suite_c').
               suite_a = 2D LiDAR + IMU. suite_b = 3D LiDAR + IMU.
               suite_c = 3D LiDAR + camera + IMU.
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
        # 1.0 = always spawn; 0.0 = never spawn. Evaluated independently per zone.
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
        # Used for /initialpose publishing to seed the EKF on reset.
        self._chosen_spawn: Dict[str, float] = {}

        # Ego vehicle CoM z after settling under gravity.  Set in _spawn_vehicle()
        # and passed to all static spawners so props and vehicles land on the same
        # ground surface instead of using the hardcoded YAML origin_z.
        self._floor_z: float = 0.3

        # Per-step debug diagnostics (emits structured logs + populates vis HUD).
        # Instantiated here so NPCController / SensorManager can share the reference.
        self._debug_logger: DebugLogger = DebugLogger(debug=debug)

        # Lot spawner -- owns static cones and parked vehicles.
        self._lot_spawner = LotSpawner(
            cone_spacing=scenarios.get("perimeter_cone_spacing", 2.0),
            marker_blueprint=scenarios.get(
                "perimeter_marker_blueprint", "static.prop.constructioncone"
            ),
            bay_occupancy_min=scenarios.get("bay_occupancy_min", 0.3),
            bay_occupancy_max=scenarios.get("bay_occupancy_max", 0.8),
            spawn_perimeter_cones=scenarios.get("spawn_perimeter_cones", False),
        )

        # NPC controller -- owns patrol vehicles and pedestrians.
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
            debug_logger=self._debug_logger,
        )

        # Sensor manager -- owns IMU, LiDAR, collision sensor, camera.
        self._sensor_manager = SensorManager(
            sensors_config=self._sensors_config,
            sensor_suite=sensor_suite,
            debug_logger=self._debug_logger,
        )

        # Cached actor list for patrol obstacle proximity checks.  Rebuilt
        # once after all vehicles are spawned; avoids per-step world queries.
        self._all_vehicle_actors: List[Any] = []

        # Cached blueprint lists -- fetched once on first connect, never re-fetched.
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

        # Full 2D rigid body transform from Cartographer odom frame to CARLA
        # world frame, computed once per episode in _calibrate_ekf_frame_offset().
        # Stored as (tx, ty, cos_r, sin_r, r) where:
        #   tx, ty   -- world translation after applying the rotation
        #   cos_r, sin_r -- 2D rotation matrix components (odom->world)
        #   r        -- rotation angle (world_yaw - ekf_yaw at convergence)
        # Used only by the debug drift calculation in step(); the observation
        # path uses odom-frame coordinates directly via _target_bay_odom.
        self._ekf_odom_offset: Tuple[float, float, float, float, float] = (
            0.0, 0.0, 1.0, 0.0, 0.0
        )

        # Target bay (world frame, set in reset)
        self._target_bay: Dict[str, Any] = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "width": 2.5,
            "depth": 5.0,
        }

        # Target bay expressed in Cartographer odom frame (set each episode after
        # _calibrate_ekf_frame_offset() resolves the odom->world transform).
        # Using odom-frame coordinates for the relative target computation means
        # both the vehicle pose (EKF output) and the target are in the same
        # drifting frame, so mid-episode odom drift cancels in the subtraction.
        self._target_bay_odom: Dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0}

        # Current floor plan layout (loaded from YAML in reset)
        self._current_layout: Dict[str, Any] = {}
        self._current_floor_plan_name: str = ""

        # Trajectory buffer for debug overlays (ring buffer of (x, y) tuples)
        self._trajectory_buffer: Deque[Tuple[float, float]] = collections.deque(
            maxlen=_TRAJECTORY_MAXLEN
        )
        # Last action applied (for vis diagnostics)
        self._last_action = np.zeros(3, dtype=np.float32)

        # Visualisation state writer (demand-driven via signal file)
        self._vis_history_path: Path = (
            Path(vis_output_path) if vis_output_path
            else Path("outputs/vis_history.jsonl")
        )
        self._vis_signal_path: Path = (
            self._vis_history_path.parent / ".vis_active"
        )
        self._carla_timestep: float = carla_timestep

        # Episode state
        self._episode_id: int = 0
        self.steps = 0
        # Previous distance to target for potential-based reward shaping
        self._prev_distance: float = 0.0

        # Observation and action spaces
        obs_dim = self._compute_obs_dim()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

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
    # Observation dimension helper
    # ------------------------------------------------------------------

    def _compute_obs_dim(self) -> int:
        """
        @brief Compute the observation dimension based on active feature flags.
        @return Integer observation dimension.

        Base: VEHICLE_STATE_DIM (6) + TARGET_POSE_DIM (3) = 9
        With include_covariance: +COVARIANCE_FEATURES_DIM (9) = 18
        With include_obstacle_obs: +OBSTACLE_FEATURES_DIM (2) = 20 (or 11 without cov)
        """
        dim = VEHICLE_STATE_DIM + TARGET_POSE_DIM
        if self._include_covariance:
            dim += COVARIANCE_FEATURES_DIM
        if self._include_obstacle_obs:
            dim += OBSTACLE_FEATURES_DIM
        return dim

    # ------------------------------------------------------------------
    # ROS 2 initialisation
    # ------------------------------------------------------------------

    def _init_ros2(self) -> None:
        """
        @brief Initialise rclpy and create the covariance reader.

        The covariance reader uses a shared file (no DDS subscription) to avoid
        cross-distro serialisation issues between Humble and Jazzy. rclpy is
        still needed for /initialpose publishing at episode reset.

        When rclpy is unavailable (CI/tests), the reader still works for file
        reading but /initialpose publishing is disabled.
        """
        if _ROS2_AVAILABLE:
            if not rclpy.ok():
                rclpy.init()

        node_name = f"covariance_subscriber_{id(self)}"
        self._cov_subscriber = _CovarianceSubscriber(
            covariance_topic=self._covariance_topic,
            node_name=node_name,
            ros2_config=self._ros2_config,
        )
        logger.info("Covariance reader initialised (file-based, no DDS).")

    # ------------------------------------------------------------------
    # Floor plan loading and bay sampling
    # ------------------------------------------------------------------

    def _load_floor_plan(self) -> None:
        """
        @brief Select and load a floor plan layout YAML for this episode.

        During training (eval_mode=False), only floor plans with ood=false are
        eligible. During evaluation (eval_mode=True), all plans are eligible.

        @warning Raises RuntimeError if no eligible floor plans are configured.
        """
        eligible = {}
        for name, cfg in self._floor_plans_config.items():
            is_ood = cfg.get("ood", False)
            if self._eval_mode or not is_ood:
                eligible[name] = cfg

        if not eligible:
            raise RuntimeError(
                "No eligible floor plans found. "
                "Check parking_scenarios.floor_plans in train_config.yaml."
            )

        name = random.choice(list(eligible.keys()))
        layout_file = eligible[name].get("layout_file", "")

        layout_path = Path(layout_file)
        if not layout_path.exists():
            raise FileNotFoundError(
                f"Floor plan layout file not found: {layout_path}. "
                f"Run 'make generate-layouts' to create it."
            )

        cache_key = str(layout_path.resolve())
        if cache_key in CARLAParkingEnv._layout_cache:
            self._current_layout = CARLAParkingEnv._layout_cache[cache_key]
        else:
            with open(layout_path, "r") as fh:
                self._current_layout = yaml.safe_load(fh)
            CARLAParkingEnv._layout_cache[cache_key] = self._current_layout
            logger.debug(f"Cached floor plan layout: {layout_path}")

        self._current_floor_plan_name = name
        logger.info(f"Loaded floor plan: {name} from {layout_path}")

    def _sample_target_bay(self) -> None:
        """
        @brief Stratified sample of target bay: 1/3 per type, then uniform within type.

        Bay types: perpendicular, angled, parallel.

        @note Sets self._target_bay with world-frame coordinates.
        """
        bays = self._current_layout.get("bays", [])

        # Exclude always-empty bays before sampling the target
        bays = [b for b in bays if not b.get("always_empty", False)]

        # Group by bay type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for bay in bays:
            bay_type = bay.get("bay_type", "perpendicular")
            by_type.setdefault(bay_type, []).append(bay)

        if not by_type:
            raise RuntimeError("No eligible bays found in floor plan layout.")

        # Stratified: sample type uniformly, then bay uniformly within type
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
            f"Target bay: type={bay_type}, id={self._target_bay['bay_id']}, "
            f"x={self._target_bay['x']:.1f}, y={self._target_bay['y']:.1f}"
        )

    # ------------------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------------------

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
            self.world, self._current_layout, self._target_bay,
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
        self._npc_controller.update_patrol(
            self.vehicle, self.steps, self._current_layout
        )

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

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """
        @brief Compute reward and termination flags for the current step.
        @return Tuple of (reward, terminated, success).

        Uses CARLA ground truth transform (not EKF pose) for position and
        orientation errors. Potential-based reward shaping (Ng et al. 1999)
        ensures the success bonus is never dominated by the distance penalty.

        Reward structure:
          - Progress: (prev_distance - curr_distance) / OUT_OF_BOUNDS_THRESHOLD
            Positive when closing on target, negative when moving away.
            OUT_OF_BOUNDS_THRESHOLD used as a fixed normalisation scale only.
          - Time penalty: -0.01 per step to discourage stalling.
          - Collision: -10.0 + termination.
          - Success: +10.0 + termination.

        Termination conditions (priority order):
          1. Collision: physical contact detected by collision sensor
             (includes hitting perimeter cones -- the lot boundary).
          2. Success: position < SUCCESS_THRESHOLD_POSITION,
             yaw < SUCCESS_THRESHOLD_ORIENTATION,
             velocity < SUCCESS_THRESHOLD_VELOCITY
          3. Time limit handled externally via truncated flag in step()
        """
        if self.vehicle is None:
            return 0.0, False, False

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)

        vx = velocity.x
        vy = velocity.y
        speed = math.sqrt(vx * vx + vy * vy)

        target_x = self._target_bay["x"]
        target_y = self._target_bay["y"]
        target_yaw = self._target_bay["yaw"]

        position_error = math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
        # Both nose-in and nose-out are valid -- use the smaller of the two errors.
        yaw_error_raw = yaw - target_yaw
        orientation_error = min(
            abs(math.atan2(math.sin(yaw_error_raw), math.cos(yaw_error_raw))),
            abs(math.atan2(
                math.sin(yaw_error_raw + math.pi),
                math.cos(yaw_error_raw + math.pi),
            )),
        )

        # Check collision (penalty + termination).  Flag set by SensorManager
        # collision callback; consume_collision() reads and resets atomically.
        if self._sensor_manager.consume_collision():
            self._prev_distance = position_error
            return -10.0, True, False

        # Out-of-bounds: terminate when any part of the car body exits the lot.
        # Uses the layout corners AABB shrunk by the vehicle half-width (1.0 m)
        # so the check triggers when the car body -- not just its centre -- crosses
        # the perimeter cone line. Perimeter cones are static props (no physics)
        # so the collision sensor cannot detect them; this is the boundary enforcer.
        # On real-world deployment: replace the YAML corners with the surveyed lot
        # boundary -- no code changes needed.
        corners = self._current_layout.get("corners", [])
        if corners:
            xs = [c["x"] for c in corners]
            ys = [c["y"] for c in corners]
            _half_width = 1.0  # BMW Grand Tourer half-width (metres)
            _oob = (
                x < min(xs) + _half_width
                or x > max(xs) - _half_width
                or y < min(ys) + _half_width
                or y > max(ys) - _half_width
            )
            if _oob:
                self._prev_distance = position_error
                return -5.0, True, False

        # Success condition
        success = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
            and speed < SUCCESS_THRESHOLD_VELOCITY
        )
        if success:
            self._prev_distance = position_error
            return 10.0, True, True

        # Potential-based progress reward (Ng et al. 1999)
        # Positive when closing on the target, negative when drifting away.
        # Normalised by a fixed scale factor to keep rewards in a consistent range.
        progress = (self._prev_distance - position_error) / OUT_OF_BOUNDS_THRESHOLD
        reward = progress - 0.01  # 0.01/step time penalty
        self._prev_distance = position_error

        return float(reward), False, False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        @brief Build the observation vector.
        @return Float32 array of shape (_compute_obs_dim(),).

        Indices 0-5: EKF filtered pose (x, y, yaw, vx, vy, vyaw).
                     Read from _CovarianceSubscriber.get_latest_pose() when the
                     EKF subscriber is available. Falls back to CARLA ground truth
                     when rclpy is unavailable (CI / unit tests).
        Indices 6-14: EKF covariance features log1p-transformed
                     (only when include_covariance=True).
        Indices 15-17 (or 6-8 without covariance): relative target bay pose.
        Indices 18-19 (or 9-10 without covariance): obstacle awareness dims
                     (only when include_obstacle_obs=True).
        """
        if self.vehicle is None or self.world is None:
            return np.zeros(self._compute_obs_dim(), dtype=np.float32)

        # -- Pose (indices 0-5) -------------------------------------------
        # Prefer EKF filtered estimate for sim-to-real transfer; fall back to
        # CARLA ground truth when rclpy is unavailable (CI / unit tests).
        ekf_pose: Optional[np.ndarray] = None
        if self._cov_subscriber is not None:
            ekf_pose = self._cov_subscriber.get_latest_pose()

        if ekf_pose is not None:
            # Use EKF pose directly in Cartographer odom frame.
            # Both the vehicle pose and the target bay (_target_bay_odom) are
            # expressed in this frame, so mid-episode odom drift cancels in the
            # relative-target subtraction.  No world-frame transform is needed.
            x = float(ekf_pose[0])
            y = float(ekf_pose[1])
            yaw = float(ekf_pose[2])
            # Clamp EKF velocities to physically plausible range.
            # The IMU prediction step can integrate large noise spikes
            # before the first scan-match correction at episode start,
            # producing transient velocity readings in the hundreds of m/s.
            vx = float(np.clip(ekf_pose[3], -MAX_PARKING_SPEED, MAX_PARKING_SPEED))
            vy = float(np.clip(ekf_pose[4], -MAX_PARKING_SPEED, MAX_PARKING_SPEED))
            vyaw = float(ekf_pose[5])
        else:
            # Fallback: EKF not yet initialised or ROS 2 unavailable (CI / unit tests)
            self._debug_logger._logger.debug(
                "[state] EKF pose unavailable at step %d -- using CARLA ground truth",
                self.steps,
            )
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_vel = self.vehicle.get_angular_velocity()
            x = transform.location.x
            y = transform.location.y
            yaw = math.radians(transform.rotation.yaw)
            vx = velocity.x
            vy = velocity.y
            vyaw = math.radians(angular_vel.z)

        # Relative target pose in ego body frame.
        # Both vehicle pose (x, y, yaw) and target are in Cartographer odom frame,
        # so odom drift cancels in the subtraction.
        dx, dy, dyaw = _compute_relative_target_pose(
            x,
            y,
            yaw,
            self._target_bay_odom["x"],
            self._target_bay_odom["y"],
            self._target_bay_odom["yaw"],
        )

        # -- Obstacle features (computed once, used in both branches) ------
        if self._include_obstacle_obs:
            obstacle_features = self._get_obstacle_features()
        else:
            obstacle_features = np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)

        if not self._include_covariance:
            # Without covariance: [pose(6), target(3)] = 9-dim
            # With obstacle obs:  [pose(6), target(3), obstacle(3)] = 12-dim
            self._obs_buffer[0] = x
            self._obs_buffer[1] = y
            self._obs_buffer[2] = yaw
            self._obs_buffer[3] = vx
            self._obs_buffer[4] = vy
            self._obs_buffer[5] = vyaw
            self._obs_buffer[6] = dx
            self._obs_buffer[7] = dy
            self._obs_buffer[8] = dyaw
            if self._include_obstacle_obs:
                self._obs_buffer[9:11] = obstacle_features
            return self._obs_buffer.copy()

        # -- EKF covariance features (indices 6-14) -----------------------
        if self._cov_subscriber is not None:
            uncertainty = self._cov_subscriber.get_latest_uncertainty()
        else:
            uncertainty = None

        if uncertainty is None:
            self._debug_logger._logger.debug(
                "[state] EKF covariance unavailable at step %d -- zeroing features",
                self.steps,
            )
            uncertainty = np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32)
        else:
            uncertainty = uncertainty.astype(np.float32)
            if not np.any(uncertainty):
                self._debug_logger._logger.debug(
                    "[state] EKF covariance all-zeros at step %d "
                    "-- uncertainty signal absent",
                    self.steps,
                )

        # No obstacle obs:  [pose(6), cov(9), target(3)] = 18-dim
        # With obstacle obs: [pose(6), cov(9), target(3), obs(3)] = 21-dim
        self._obs_buffer[0] = x
        self._obs_buffer[1] = y
        self._obs_buffer[2] = yaw
        self._obs_buffer[3] = vx
        self._obs_buffer[4] = vy
        self._obs_buffer[5] = vyaw
        # log1p compresses heavy tails from high-uncertainty conditions (rain, sensor
        # noise) that would otherwise distort VecNormalize running statistics.
        # np.sign preserves the sign of off-diagonal covariance terms (cov_xy,
        # cov_xyaw, cov_yyaw) which can be negative.
        self._obs_buffer[6:15] = np.sign(uncertainty) * np.log1p(np.abs(uncertainty))
        self._obs_buffer[15] = dx
        self._obs_buffer[16] = dy
        self._obs_buffer[17] = dyaw
        if self._include_obstacle_obs:
            self._obs_buffer[18:20] = obstacle_features
        return self._obs_buffer.copy()

    def _get_obstacle_features(self) -> np.ndarray:
        """
        @brief Extract nearest obstacle features from the cached LiDAR scan.
        @return Float32 array [nearest_dist_m, bearing_rad].

        nearest_dist_m: distance to nearest LiDAR return (metres, clamped to
                        sensor range). Zero if no scan available.
        bearing_rad:    bearing to nearest return in ego body frame (radians,
                        0 = forward, positive = left per ROS convention).

        Returns zeros when no LiDAR scan is available (sensor not yet ticked).
        """
        scan = self._sensor_manager.get_latest_lidar_scan()

        if scan is None or len(scan) == 0:
            self._obstacle_features_buffer[:] = 0.0
            return self._obstacle_features_buffer

        # Keep only the forward hemisphere (x > 0 in vehicle frame).
        # The LiDAR is front-bumper mounted -- rearward rays are physically
        # blocked by the car body on the real robot (270 deg FOV) and produce
        # self-returns in CARLA's 360 deg simulation.  Discarding x <= 0
        # replicates the real sensor FOV and eliminates all self-returns.
        dists = np.sqrt(scan[:, 0] ** 2 + scan[:, 1] ** 2)
        valid = scan[:, 0] > 0.0
        if not np.any(valid):
            self._obstacle_features_buffer[:] = 0.0
            return self._obstacle_features_buffer
        dists = dists[valid]
        scan = scan[valid]
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])

        # Bearing: atan2(y, x) in vehicle frame (y-left, x-forward convention)
        nearest_bearing = float(np.arctan2(scan[nearest_idx, 1], scan[nearest_idx, 0]))

        self._obstacle_features_buffer[0] = nearest_dist
        self._obstacle_features_buffer[1] = nearest_bearing
        return self._obstacle_features_buffer

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

        # Read CARLA world clock for diagnostics (not the computed sim_time)
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
                "x": x, "y": y, "yaw": yaw,
                "vx": self.vehicle.get_velocity().x,
                "vy": self.vehicle.get_velocity().y,
                "speed": math.sqrt(
                    self.vehicle.get_velocity().x ** 2
                    + self.vehicle.get_velocity().y ** 2
                ),
            },
            "action": {
                "steer": float(self._last_action[0])
                if hasattr(self, "_last_action")
                else 0.0,
                "throttle": float(self._last_action[1])
                if hasattr(self, "_last_action")
                else 0.0,
                "brake": float(self._last_action[2])
                if hasattr(self, "_last_action")
                else 0.0,
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
            # Non-fatal -- visualisation is optional
            logger.debug(f"Could not write vis state: {exc}")

    # ------------------------------------------------------------------
    # CARLA world management
    # ------------------------------------------------------------------

    def _connect_to_carla(self) -> None:
        """
        @brief Connect to CARLA and load the world.

        @note When town is "FlatPlane", loads configs/layouts/flat_plane.xodr via
              generate_opendrive_world() - a clean flat plane with no roads or
              buildings. Otherwise uses load_world() for named CARLA towns.
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(120.0)
            self.world = self.client.get_world()

            current_map_name = self.world.get_map().name.split("/")[-1]
            if self.town == "FlatPlane":
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
                    time.sleep(5.0)
                else:
                    logger.info("FlatPlane already loaded.")
            elif current_map_name != self.town:
                logger.info(f"Loading map: {self.town}")
                self.world = self.client.load_world(self.town)
                time.sleep(8.0)

            # Default to ClearNoon so the scene is always daytime.
            # Training randomises weather per episode via _configure_weather().
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
        primary = self._current_layout.get("spawn_transform", {})
        extras: List[Any] = self._current_layout.get("extra_spawn_transforms", [])
        all_spawns = [primary] + list(extras)

        chosen = random.choice(all_spawns)
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
            # sensor data under /carla/ego_vehicle/* for Cartographer and the EKF.
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
            # In synchronous mode, time.sleep() does not advance physics --
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

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn sensors for the configured suite attached to the ego vehicle.

        Delegates to SensorManager.spawn(), passing the NPC controller's patrol_npc_ids
        set by reference so the collision callback can identify patrol vehicles.
        """
        self._sensor_manager.spawn(
            self.world, self.vehicle, self._npc_controller.patrol_npc_ids
        )

    def _wait_for_covariance(self) -> None:
        """
        @brief Block until the first EKF covariance message arrives.

        Ticks the CARLA simulation while waiting. Raises RuntimeError on timeout.
        """
        if self._cov_subscriber is None:
            return

        start = time.monotonic()
        tick_interval = 0.05  # 20 Hz -- matches simulation timestep

        while not self._cov_subscriber.has_data:
            elapsed = time.monotonic() - start
            if elapsed > self._covariance_timeout:
                raise RuntimeError(
                    f"No covariance message received within "
                    f"{self._covariance_timeout}s. "
                    f"Ensure ros2-bridge container is healthy."
                )
            if self.world is not None:
                self.world.tick(10.0)
            time.sleep(tick_interval)

        logger.debug(
            f"First covariance received after {time.monotonic() - start:.2f}s."
        )

    def _compute_target_bay_odom(
        self, tx: float, ty: float, cos_r: float, sin_r: float, r: float
    ) -> None:
        """
        @brief Project the world-frame target bay into Cartographer odom frame.

        The forward transform is p_world = R * p_odom + t.
        The inverse (world -> odom) is p_odom = R^T * (p_world - t),
        where R^T = [[cos_r, sin_r], [-sin_r, cos_r]].

        Storing the target in odom frame means both the EKF pose estimate and
        the target are expressed in the same (drifting) frame.  Mid-episode
        odom drift therefore cancels in the relative-target subtraction.

        @param tx: World translation X after rotation (from _ekf_odom_offset).
        @param ty: World translation Y after rotation (from _ekf_odom_offset).
        @param cos_r: cos of the odom-to-world rotation angle.
        @param sin_r: sin of the odom-to-world rotation angle.
        @param r: Odom-to-world rotation angle (radians).
        """
        bx = self._target_bay["x"] - tx
        by = self._target_bay["y"] - ty
        # Apply inverse rotation: R^T * (p_world - t)
        self._target_bay_odom = {
            "x": cos_r * bx + sin_r * by,
            "y": -sin_r * bx + cos_r * by,
            "yaw": math.atan2(
                math.sin(self._target_bay["yaw"] - r),
                math.cos(self._target_bay["yaw"] - r),
            ),
        }

    def _calibrate_ekf_frame_offset(self) -> None:
        """
        @brief Compute the full 2D rigid body transform from Cartographer odom
               frame to CARLA world frame and store it in self._ekf_odom_offset.

        Cartographer's odom x-axis aligns with the vehicle's heading at
        Cartographer startup. For a spawn at yaw=0 this matches CARLA world x,
        but for yaw=90 the odom x-axis points along CARLA world y -- so a
        pure translation offset is insufficient. A rotation must be applied
        first to bring odom coordinates into CARLA world axis alignment.

        The transform is: p_world = R * p_odom + t
        where R is a 2D rotation matrix by angle `r = world_yaw - ekf_yaw`
        and t is the translation after rotation.

        Computed once per episode while the vehicle is stationary at reset,
        after the EKF velocity has settled below 0.5 m/s (convergence signal).
        """
        if self._cov_subscriber is None or self.vehicle is None:
            return

        carla_t = self.vehicle.get_transform()
        world_x = carla_t.location.x
        world_y = carla_t.location.y
        world_yaw = math.radians(carla_t.rotation.yaw)

        start = time.monotonic()
        tick_interval = 0.05  # 20 Hz

        # Convergence detection: the vehicle is stationary at episode reset,
        # so the EKF velocity should be near zero once Cartographer has matched
        # the map and the IMU initialisation transient has settled.
        _VEL_CONVERGED = 0.5  # m/s

        while True:
            ekf_pose = self._cov_subscriber.get_latest_pose()
            elapsed = time.monotonic() - start

            if ekf_pose is None:
                if elapsed > self._ekf_convergence_timeout:
                    logger.warning(
                        "EKF pose unavailable after "
                        f"{self._ekf_convergence_timeout:.0f}s -- "
                        "odom transform will be identity."
                    )
                    self._ekf_odom_offset = (0.0, 0.0, 1.0, 0.0, 0.0)
                    # Identity transform: odom frame = world frame.
                    self._compute_target_bay_odom(0.0, 0.0, 1.0, 0.0, 0.0)
                    return
                if self.world is not None:
                    self.world.tick(10.0)
                time.sleep(tick_interval)
                continue

            ekf_x = float(ekf_pose[0])
            ekf_y = float(ekf_pose[1])
            ekf_yaw = float(ekf_pose[2])
            ekf_vx = float(ekf_pose[3])
            ekf_vy = float(ekf_pose[4])

            # Rotation angle: odom frame -> CARLA world frame.
            # Cartographer odom x-axis points along the vehicle heading at
            # startup, so the rotation is world_yaw - ekf_yaw.
            r = math.atan2(
                math.sin(world_yaw - ekf_yaw),
                math.cos(world_yaw - ekf_yaw),
            )
            cos_r = math.cos(r)
            sin_r = math.sin(r)

            # Rotate the odom-frame origin into world-aligned coordinates,
            # then compute the translation to reach the CARLA world position.
            rotated_x = cos_r * ekf_x - sin_r * ekf_y
            rotated_y = sin_r * ekf_x + cos_r * ekf_y
            tx = world_x - rotated_x
            ty = world_y - rotated_y

            ekf_speed = math.sqrt(ekf_vx ** 2 + ekf_vy ** 2)
            if ekf_speed <= _VEL_CONVERGED:
                self._ekf_odom_offset = (tx, ty, cos_r, sin_r, r)
                self._compute_target_bay_odom(tx, ty, cos_r, sin_r, r)
                logger.info(
                    f"EKF converged after {elapsed:.2f}s: "
                    f"rotation={math.degrees(r):.1f}deg "
                    f"tx={tx:.3f}m ty={ty:.3f}m "
                    f"ekf_speed={ekf_speed:.3f}m/s "
                    f"(CARLA spawn=({world_x:.2f},{world_y:.2f}) "
                    f"EKF odom=({ekf_x:.2f},{ekf_y:.2f})) "
                    f"target_odom=({self._target_bay_odom['x']:.2f},"
                    f"{self._target_bay_odom['y']:.2f})"
                )
                return

            if elapsed > self._ekf_convergence_timeout:
                self._ekf_odom_offset = (tx, ty, cos_r, sin_r, r)
                self._compute_target_bay_odom(tx, ty, cos_r, sin_r, r)
                logger.warning(
                    f"EKF convergence timeout ({self._ekf_convergence_timeout:.0f}s) "
                    f"-- ekf_speed={ekf_speed:.2f}m/s still above threshold. "
                    "Using best available transform; position obs may be inaccurate."
                )
                return

            logger.debug(
                f"Waiting for EKF convergence: "
                f"ekf_speed={ekf_speed:.2f}m/s elapsed={elapsed:.1f}s"
            )
            if self.world is not None:
                self.world.tick(10.0)
            time.sleep(tick_interval)

    def _cleanup_actors(self) -> None:
        """
        @brief Destroy all episode actors (sensors, cones, static vehicles, NPCs,
               pedestrians, ego vehicle).
        """
        # Delegates lifecycle to the three actor managers
        self._sensor_manager.cleanup()
        self._npc_controller.cleanup()
        self._lot_spawner.cleanup()

        self._all_vehicle_actors.clear()

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

        self._cleanup_actors()

        # Connect to CARLA on first reset
        if self.client is None:
            self._connect_to_carla()

        if self.world is None:
            return np.zeros(self._compute_obs_dim(), dtype=np.float32), {}

        # Flush pending destroy commands in synchronous mode.  Without this
        # tick, CARLA queues the destroys from _cleanup_actors() and only
        # processes them on the next tick, which happens inside LotSpawner.
        # By then new actors are already spawned, causing ghost collisions and
        # steadily increasing actor IDs (memory leak).  One explicit tick here
        # ensures a clean slate before spawning.
        self.world.tick(10.0)

        # When covariance is included in the observation, synchronous mode must
        # be active so the EKF runs in lock-step with the simulation.  A world
        # reload (generate_opendrive_world / load_world) resets all CARLA
        # settings to async defaults, so we apply sync mode ourselves rather
        # than relying on the bridge to re-apply it after a reload.
        # The bridge is configured with synchronous_mode: true in its launch
        # YAML, which tells it to *expect* sync mode -- it does not re-set it
        # after a world reload.
        # When include_covariance=False (inspector, ablation baselines) the
        # ROS bridge is not required and this block is skipped entirely.
        if self._include_covariance:
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

        # Load floor plan and sample target bay
        if self._floor_plans_config:
            self._load_floor_plan()
            self._sample_target_bay()

        # Cache blueprint lists once before any spawning to avoid repeated
        # world queries inside each spawn method.
        self._cache_blueprints()

        self._spawn_vehicle()
        self._spawn_sensors()

        # Invalidate stale pre-reset EKF data so _wait_for_covariance() blocks
        # until a genuinely post-spawn reading arrives from ekf_state.json.
        if self._include_covariance and self._cov_subscriber is not None:
            self._cov_subscriber.invalidate()

        # Publish spawn pose to /initialpose so Cartographer pure localisation
        # can converge quickly at the start of each episode. Safe no-op in SLAM mode.
        if (
            self._include_covariance
            and self._cov_subscriber is not None
            and self._ros2_config.get("publish_initial_pose", False)
        ):
            sx = float(self._chosen_spawn.get("x", 0.0))
            sy = float(self._chosen_spawn.get("y", 0.0))
            syaw = math.radians(float(self._chosen_spawn.get("yaw_deg", 0.0)))
            self._cov_subscriber.publish_initial_pose(sx, sy, syaw)

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
        sx = float(self._chosen_spawn.get("x", 0.0))
        sy = float(self._chosen_spawn.get("y", 0.0))
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
        @brief Execute one environment step.
        @param action: 3-dim action vector [steering, throttle, brake].
        @return Tuple of (observation, reward, terminated, truncated, info).
        """
        self.steps += 1

        # Cache the applied action for vis state diagnostics
        self._last_action = action.copy()

        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(np.clip(action[0], -1.0, 1.0))
            control.throttle = float(np.clip(action[1], 0.0, 1.0))
            control.brake = float(np.clip(action[2], 0.0, 1.0))

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

        state = self._get_state()
        reward, terminated, success = self._compute_reward()

        # Per-step debug diagnostics (no-op when debug=False)
        if self._debug_logger.enabled and self.vehicle is not None:
            _t = self.vehicle.get_transform()
            _v = self.vehicle.get_velocity()
            _pos_err = math.sqrt(
                (_t.location.x - self._target_bay["x"]) ** 2
                + (_t.location.y - self._target_bay["y"]) ** 2
            )
            _yaw_raw = math.radians(_t.rotation.yaw) - self._target_bay["yaw"]
            _yaw_err = min(
                abs(math.atan2(math.sin(_yaw_raw), math.cos(_yaw_raw))),
                abs(math.atan2(
                    math.sin(_yaw_raw + math.pi),
                    math.cos(_yaw_raw + math.pi),
                )),
            )
            _speed = math.sqrt(_v.x ** 2 + _v.y ** 2)
            _unc: Optional[np.ndarray] = None
            _ekf_pose: Optional[np.ndarray] = None
            if self._cov_subscriber is not None:
                _unc = self._cov_subscriber.get_latest_uncertainty()
                _ekf_pose = self._cov_subscriber.get_latest_pose()
            # EKF vs ground truth drift (metres) -- key sim-to-real signal.
            # Apply the full odom->world rigid transform so both positions are
            # in CARLA world frame before computing the error.
            _ekf_drift = 0.0
            if _ekf_pose is not None:
                _tx, _ty, _cos_r, _sin_r, _ = self._ekf_odom_offset
                _world_ex = _cos_r * _ekf_pose[0] - _sin_r * _ekf_pose[1] + _tx
                _world_ey = _sin_r * _ekf_pose[0] + _cos_r * _ekf_pose[1] + _ty
                _ekf_drift = math.sqrt(
                    (_world_ex - _t.location.x) ** 2
                    + (_world_ey - _t.location.y) ** 2
                )
            _obs_dist = float(state[18]) if len(state) > 18 else 0.0
            _lidar_pts = self._sensor_manager.lidar_point_count()
            self._debug_logger.log_step(
                step=self.steps,
                reward=reward,
                pos_error=_pos_err,
                yaw_error=_yaw_err,
                speed=_speed,
                action=self._last_action,
                uncertainty=_unc,
                obstacle_dist=_obs_dist,
                ekf_drift=_ekf_drift,
                lidar_points=_lidar_pts,
            )

        truncated = self.steps >= self.max_steps

        # Distinguish termination cause. Only collision (-10.0) terminates the
        # episode; out-of-bounds is no longer a termination condition (the
        # perimeter cones handle physical lot boundary via collision detection).
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
            "floor_plan": self._current_floor_plan_name,
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
