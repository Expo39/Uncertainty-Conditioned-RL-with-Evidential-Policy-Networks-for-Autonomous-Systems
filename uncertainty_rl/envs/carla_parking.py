"""
@file carla_parking.py
@brief CARLA parking environment with EKF covariance and lot geometry.

This module implements a Gymnasium-compatible environment for autonomous parking
in CARLA simulator. The agent parks in one of three floor-plan geometries loaded
from pre-computed layout YAMLs (configs/layouts/). Localisation uncertainty comes
from the robot_localisation EKF node (via ROS 2 DDS), driven by noisy CARLA
sensors, weather conditions, and dynamic traffic - not from a simulated noise model.

The 18-dimensional observation comprises:
  - indices  0-5:  EKF filtered pose (x, y, yaw, vx, vy, vyaw)
  - indices  6-14: EKF covariance features (std_x, std_y, std_yaw,
                   cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw)
  - indices 15-17: target bay in ego body frame (dx, dy, dyaw)

CARLA ground truth is used only for reward computation and clearance checks,
not in the observation. This ensures sim-to-real transfer without retraining.
"""

import collections
import json
import logging
import math
import os
import random
import threading
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

from uncertainty_rl.envs.covariance_subscriber import _CovarianceSubscriber
from uncertainty_rl.utils.constants import (
    CLEARANCE_THRESHOLD,
    COVARIANCE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
    TARGET_POSE_DIM,
    TOTAL_OBS_DIM,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    _interpolate_cone_positions,
    zone_bbox,
)

logger = logging.getLogger(__name__)

# Trail length for debug overlays and vis state
_TRAJECTORY_MAXLEN = 50

# Vehicle types that overhang a standard 2.5 m bay - excluded from parking spawns.
_LARGE_VEHICLE_TYPES: Tuple[str, ...] = (
    "ambulance",
    "firetruck",
    "sprinter",
    "t2",
    "t2_2021",
    "carlacola",
    "cybertruck",
    "fusorosa",
    "bus",
)

# Micro/novelty vehicles excluded from parked car pool - unrealistically small
# for a standard parking bay and visually confusing during inspection.
_SMALL_VEHICLE_TYPES: Tuple[str, ...] = (
    "microlino",
    "isetta",
    "micro",
    "omafiets",
    "crossbike",
    "low_rider",
    "ninja",
    "yzf",
    "century",
    "harley",
    "kawasaki",
    "yamaha",
    "vespa",
    "zx125",
    "bike",
    "bicycle",
)

# Re-export geometry helpers so existing imports from this module still work
__all__ = [
    "CARLAParkingEnv",
    "_compute_relative_target_pose",
    "_interpolate_cone_positions",
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

    Observation space (18-dim when include_covariance=True, 9-dim when False):
      [0-5]  EKF pose: x, y, yaw, vx, vy, vyaw
      [6-14] EKF covariance features (omitted when include_covariance=False)
      [15-17] (or [6-8] when no covariance): target bay in ego body frame

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
        carla_conditions_config: Optional[Dict[str, Any]] = None,
        parking_scenarios_config: Optional[Dict[str, Any]] = None,
        include_covariance: bool = True,
        vis_output_path: Optional[str] = None,
        eval_mode: bool = False,
    ) -> None:
        """
        @brief Construct the CARLA parking environment.
        @param carla_host: CARLA server host address.
        @param carla_port: CARLA server port.
        @param town: CARLA town/map to load.
        @param max_steps: Maximum episode length.
        @param render_mode: Rendering mode ('human', 'rgb_array', or None).
        @param ros2_config: ROS 2 settings (covariance_topic, covariance_timeout).
        @param carla_sensors_config: Sensor noise parameters (imu, gnss subsections).
        @param carla_conditions_config: Weather and environmental settings.
        @param parking_scenarios_config: Parking lot configuration (floor_plans,
               cone spacing, bay occupancy, NPC counts).
        @param include_covariance: If True, 18-dim obs (pose + covariance + target).
               If False, 9-dim obs (pose + target, no covariance subscription).
        @param vis_output_path: Path for atomic vis_state.json writes. If None,
               visualisation writes are skipped.
        @param eval_mode: If True, OOD floor plans are included in sampling.
               If False (training), only non-OOD floor plans are used.
        """
        super().__init__()

        self.carla_host = carla_host
        self.carla_port = carla_port
        self.town = town
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._include_covariance = include_covariance
        self._eval_mode = eval_mode

        ros2_config = ros2_config or {}
        self._covariance_topic = ros2_config.get(
            "covariance_topic", "/ekf_uncertainty/covariance"
        )
        self._covariance_timeout = ros2_config.get("covariance_timeout", 10.0)

        self._sensors_config = carla_sensors_config or {}
        self._conditions_config = carla_conditions_config or {}

        scenarios = parking_scenarios_config or {}
        self._cone_spacing: float = scenarios.get("perimeter_cone_spacing", 2.0)
        self._entrance_half_width: float = scenarios.get("entrance_half_width", 4.0)
        self._bay_occupancy_rate: float = scenarios.get("bay_occupancy_rate", 0.70)
        self._num_patrol_max: int = scenarios.get("num_patrol_vehicles_max", 3)
        self._patrol_obstacle_distance: float = scenarios.get(
            "patrol_obstacle_stop_distance", 5.0
        )
        self._patrol_pedestrian_distance: float = scenarios.get(
            "patrol_pedestrian_stop_distance", 1.5
        )
        self._patrol_max_speed: float = scenarios.get("patrol_max_speed_ms", 3.0)
        self._patrol_heading_gain: float = scenarios.get("patrol_heading_gain", 0.8)
        self._patrol_stuck_speed_threshold: float = scenarios.get(
            "patrol_stuck_speed_threshold", 0.3
        )
        self._patrol_stuck_steps_max: int = scenarios.get("patrol_stuck_steps_max", 30)
        self._num_pedestrians_max: int = scenarios.get("num_pedestrians_max", 4)
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

        # Per-episode actor lists
        self._spawned_sensors: List[Any] = []
        self._spawned_cones: List[Any] = []
        self._spawned_static_vehicles: List[Any] = []
        self._patrol_npcs: List[Any] = []
        self._pedestrian_actors: List[Any] = []

        # The spawn transform chosen for this episode (set in _spawn_vehicle()).
        # Used by _spawn_perimeter_cones() to cut the correct entrance gap.
        self._chosen_spawn: Dict[str, float] = {}

        # Performance-optimisation caches - cleared in _cleanup_actors()
        # Static obstacle (x, y) positions: cones + static vehicles.  No
        # get_location() call needed for these since they never move.
        self._static_obstacle_positions: List[Tuple[float, float]] = []

        # Cached actor list for patrol obstacle proximity checks.  Rebuilt
        # once after all vehicles are spawned; avoids per-step world queries.
        self._all_vehicle_actors: List[Any] = []

        # Cached filtered blueprint lists - rebuilt once per reset.
        self._car_blueprints: List[Any] = []
        self._walker_blueprints: List[Any] = []
        # Easter-egg motorcycle blueprints (None when CARLA unavailable).
        self._ninja_bp: Optional[Any] = None
        self._yzf_bp: Optional[Any] = None

        # Pre-allocated observation buffer - reused every step to avoid
        # repeated small heap allocations.
        if self._include_covariance:
            _obs_dim = TOTAL_OBS_DIM
        else:
            _obs_dim = VEHICLE_STATE_DIM + TARGET_POSE_DIM
        self._obs_buffer: np.ndarray = np.zeros(_obs_dim, dtype=np.float32)

        # Target bay (world frame, set in reset)
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

        # Patrol NPC step counters
        self._patrol_waypoint_indices: List[int] = []
        # +1 = CCW (forward), -1 = CW (reverse). Randomised per episode.
        self._patrol_waypoint_directions: List[int] = []
        # Steps each NPC has been below stuck speed threshold while receiving throttle.
        self._patrol_stuck_counters: List[int] = []

        # Pedestrian step counters for heading re-randomisation and zone confinement
        self._pedestrian_heading_steps: List[int] = []
        self._pedestrian_headings: List[Tuple[float, float, float]] = []
        self._pedestrian_lifetime_steps: List[int] = []
        self._pedestrian_zones: List[Dict[str, float]] = []

        # Trajectory buffer for debug overlays (ring buffer of (x, y) tuples)
        self._trajectory_buffer: Deque[Tuple[float, float]] = collections.deque(
            maxlen=_TRAJECTORY_MAXLEN
        )

        # Visualisation state writer
        self._vis_output_path: Optional[Path] = (
            Path(vis_output_path) if vis_output_path else None
        )

        # Episode state
        self.steps = 0

        # Observation and action spaces
        if self._include_covariance:
            obs_dim = TOTAL_OBS_DIM  # 18: pose(6) + cov(9) + target(3)
        else:
            obs_dim = VEHICLE_STATE_DIM + TARGET_POSE_DIM  # 9: pose(6) + target(3)

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
    # ROS 2 initialisation
    # ------------------------------------------------------------------

    def _init_ros2(self) -> None:
        """
        @brief Initialise rclpy and start covariance subscriber in a daemon thread.

        Guards against double-initialisation when multiple env instances exist.
        When rclpy is unavailable (CI/tests), logs a warning and continues with
        zero uncertainty in state.
        """
        if not _ROS2_AVAILABLE:
            logger.warning(
                "rclpy not available. Covariance subscriber disabled. "
                "The environment will return zero uncertainty features. "
                "This is only acceptable for CI/unit tests, not for training."
            )
            return

        if not rclpy.ok():
            rclpy.init()

        node_name = f"covariance_subscriber_{id(self)}"
        self._cov_subscriber = _CovarianceSubscriber(
            covariance_topic=self._covariance_topic,
            node_name=node_name,
        )

        threading.Thread(
            target=rclpy.spin, args=(self._cov_subscriber,), daemon=True
        ).start()
        logger.info("ROS 2 covariance subscriber started in daemon thread.")

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
        @brief Fetch and filter blueprint lists once per reset.

        Stores results in self._car_blueprints and self._walker_blueprints so
        that _spawn_static_vehicles(), _spawn_npc_patrol(), _spawn_pedestrians(),
        and _respawn_pedestrian() can reuse the filtered lists without repeated
        world queries.

        Car blueprints: four-wheeled vehicles excluding _LARGE_VEHICLE_TYPES.
        Walker blueprints: all walker.pedestrian.* blueprints.
        """
        if self.world is None:
            return

        bp_lib = self.world.get_blueprint_library()

        vehicle_bps = bp_lib.filter("vehicle.*")
        self._car_blueprints = [
            bp
            for bp in vehicle_bps
            if int(bp.get_attribute("number_of_wheels").as_int()) == 4
            and not any(excl in bp.id.lower() for excl in _LARGE_VEHICLE_TYPES)
            and not any(excl in bp.id.lower() for excl in _SMALL_VEHICLE_TYPES)
        ]

        self._walker_blueprints = list(bp_lib.filter("walker.pedestrian.*"))

        # Easter-egg motorcycles: always parked in the two motorcycle bays.
        self._ninja_bp = bp_lib.find("vehicle.kawasaki.ninja")
        self._yzf_bp = bp_lib.find("vehicle.yamaha.yzf")

    def _spawn_perimeter_cones(self) -> None:
        """
        @brief Spawn static traffic cones along the lot perimeter polygon.

        Cones are placed using _interpolate_cone_positions() and physics is
        disabled so they don't move. Cones are appended to self._spawned_cones.
        """
        if self.world is None:
            return

        corners_raw = self._current_layout.get("corners", [])
        if not corners_raw:
            logger.warning("No perimeter corners found in layout, skipping cones.")
            return

        corners: List[Tuple[float, float]] = [
            (float(c["x"]), float(c["y"])) for c in corners_raw
        ]
        spawn_raw = self._current_layout.get("spawn_transform", {})
        # Only open a gap at the spawn chosen this episode - other entry points
        # stay walled off so the lot looks realistic from inside.
        chosen = self._chosen_spawn if self._chosen_spawn else spawn_raw
        entrance: Optional[Tuple[float, float]] = (
            (
                float(chosen["x"]),
                float(chosen["y"]),
            )
            if chosen
            else None
        )
        cone_positions = _interpolate_cone_positions(
            corners,
            self._cone_spacing,
            entrance_point=entrance,
            entrance_half_width=self._entrance_half_width,
        )

        bp_lib = self.world.get_blueprint_library()
        cone_bp = bp_lib.find("static.prop.constructioncone")
        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        for cx, cy in cone_positions:
            transform = carla.Transform(
                carla.Location(x=cx, y=cy, z=z),
                carla.Rotation(yaw=0.0),
            )
            cone = self.world.try_spawn_actor(cone_bp, transform)
            if cone is not None:
                cone.set_simulate_physics(False)
                self._spawned_cones.append(cone)
                # Cache static position so _check_clearance() needs no get_location()
                self._static_obstacle_positions.append((cx, cy))

        logger.debug(f"Spawned {len(self._spawned_cones)} perimeter cones.")

    def _spawn_obstacle_cones(self) -> None:
        """
        @brief Spawn static traffic cones around interior obstacle rectangles.

        Each obstacle in the layout YAML is a centre + half-extents rectangle.
        Cones are placed along the four sides at the same spacing used for
        perimeter cones. Physics is disabled so they act as static LiDAR targets.
        Spawned cones are appended to self._spawned_cones so they are cleaned up
        with the rest of the episode actors.
        """
        if self.world is None:
            return

        obstacles = self._current_layout.get("obstacles", [])
        if not obstacles:
            return

        bp_lib = self.world.get_blueprint_library()
        cone_bp = bp_lib.find("static.prop.constructioncone")
        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        for obs in obstacles:
            cx = float(obs["centre_x"])
            cy = float(obs["centre_y"])
            hw = float(obs["half_width"])
            hh = float(obs["half_height"])

            # Build cone positions around all four sides of the rectangle.
            cone_positions: List[Tuple[float, float]] = []
            # Bottom and top horizontal edges (y constant).
            for edge_y in (cy - hh, cy + hh):
                t = -hw
                while t <= hw + 1e-6:
                    cone_positions.append((cx + t, edge_y))
                    t += self._cone_spacing
            # Left and right vertical edges (x constant), excluding corners.
            for edge_x in (cx - hw, cx + hw):
                t = -hh + self._cone_spacing
                while t < hh - 1e-6:
                    cone_positions.append((edge_x, cy + t))
                    t += self._cone_spacing

            for px, py in cone_positions:
                transform = carla.Transform(
                    carla.Location(x=px, y=py, z=z),
                    carla.Rotation(yaw=0.0),
                )
                cone = self.world.try_spawn_actor(cone_bp, transform)
                if cone is not None:
                    cone.set_simulate_physics(False)
                    self._spawned_cones.append(cone)
                    self._static_obstacle_positions.append((px, py))

        logger.debug(
            f"Spawned obstacle cones for {len(obstacles)} interior obstacle(s)."
        )

    def _adjacent_bay_ids(self, target_id: str) -> List[str]:
        """
        @brief Return bay IDs adjacent (index +/-1, same type) to the target.

        Bay IDs follow the convention '<type>_<index>' (e.g. 'parallel_3').
        Adjacent bays are left empty each episode so the agent has clearance
        to manoeuvre into the target bay.

        @param target_id: Bay ID string of the selected target bay.
        @return List of adjacent bay ID strings (may be empty if target is an end bay).
        """
        try:
            bay_type, idx_str = target_id.rsplit("_", 1)
            idx = int(idx_str)
        except ValueError:
            return []
        return [f"{bay_type}_{idx - 1}", f"{bay_type}_{idx + 1}"]

    def _spawn_static_vehicles(self) -> None:
        """
        @brief Fill non-target bays with static parked vehicles at the configured
               occupancy rate.

        Target bay and its immediate neighbours (same type, index +/-1) are
        never filled. Keeping adjacent bays clear gives the agent realistic
        manoeuvring clearance. Static vehicles have physics disabled and act
        as obstacles for clearance checking.
        """
        if self.world is None:
            return

        bays = self._current_layout.get("bays", [])
        target_id = self._target_bay.get("bay_id", "")

        # Exclude the target bay and its immediate neighbours
        excluded_ids = {target_id} | set(self._adjacent_bay_ids(target_id))

        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.1
        # Spawn 2 m above the floor so the vehicle bounding box clears perimeter
        # cones whose tops reach ~1 m.  Physics is immediately disabled and the
        # actor is teleported back to the correct z.
        z_spawn = z + 2.0

        for bay in bays:
            if bay.get("id", bay.get("bay_id", "")) in excluded_ids:
                continue
            if bay.get("always_empty", False):
                continue
            if random.random() > self._bay_occupancy_rate:
                continue

            bp = random.choice(self._car_blueprints)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)

            bay_x = float(bay["x"])
            bay_y = float(bay["y"])
            yaw = float(bay.get("yaw_deg", math.degrees(bay.get("yaw", 0.0))))
            # Randomly reverse the parked car 50 % of the time - both nose-in
            # and nose-out orientations are valid in a real car park.
            if random.random() < 0.5:
                yaw = (yaw + 180.0) % 360.0
            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                actor.set_transform(
                    carla.Transform(
                        carla.Location(x=bay_x, y=bay_y, z=z),
                        carla.Rotation(yaw=yaw),
                    )
                )
                self._spawned_static_vehicles.append(actor)
                # Cache static position so _check_clearance() needs no get_location()
                self._static_obstacle_positions.append((bay_x, bay_y))

        # Easter egg: always spawn the Kawasaki Ninja and Yamaha YZF in their
        # dedicated motorcycle bays (bay_type="motorcycle", occupant field set).
        _OCCUPANT_BP = {
            "Kawasaki Ninja": self._ninja_bp,
            "Yamaha YZF-R": self._yzf_bp,
        }
        for bay in bays:
            if bay.get("bay_type") != "motorcycle":
                continue
            occupant = bay.get("occupant", "")
            bp = _OCCUPANT_BP.get(occupant)
            if bp is None:
                continue
            bay_x = float(bay["x"])
            bay_y = float(bay["y"])
            yaw = float(bay.get("yaw_deg", 0.0))
            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                actor.set_transform(
                    carla.Transform(
                        carla.Location(x=bay_x, y=bay_y, z=z),
                        carla.Rotation(yaw=yaw),
                    )
                )
                self._spawned_static_vehicles.append(actor)
                self._static_obstacle_positions.append((bay_x, bay_y))

        logger.debug(f"Spawned {len(self._spawned_static_vehicles)} static vehicles.")

    def _spawn_npc_patrol(self) -> None:
        """
        @brief Spawn scripted patrol vehicles that follow waypoints in the lot.

        Uses a proportional heading controller rather than the Traffic Manager,
        since the lot is off-road and TM requires CARLA road network.
        """
        if self.world is None:
            return

        waypoints_raw = self._current_layout.get("patrol_waypoints", [])
        if not waypoints_raw:
            return

        # Spawn at least 1 patrol vehicle so the agent encounters a moving obstacle.
        num_patrol = random.randint(1, max(1, self._num_patrol_max))

        waypoints: List[Tuple[float, float]] = [
            (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
        ]
        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.1

        for i in range(num_patrol):
            bp = random.choice(self._car_blueprints)
            # Randomise start waypoint so NPCs don't always begin at the same position.
            start_idx = random.randrange(len(waypoints))
            # Randomly choose patrol direction before spawn so the initial yaw matches.
            direction = random.choice([-1, 1])
            # The NPC's first target is the waypoint it will immediately drive toward.
            first_target_idx = (start_idx + direction) % len(waypoints)
            wp = waypoints[start_idx]
            first_target_wp = waypoints[first_target_idx]
            # Face toward first target so the NPC never drives away from it at spawn.
            spawn_yaw = math.degrees(
                math.atan2(
                    first_target_wp[1] - wp[1],
                    first_target_wp[0] - wp[0],
                )
            )
            transform = carla.Transform(
                carla.Location(x=wp[0], y=wp[1], z=z),
                carla.Rotation(yaw=spawn_yaw),
            )
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(True)
                self._patrol_npcs.append(actor)
                # Store first_target_idx as the current waypoint so _update_patrol_npcs
                # immediately drives toward it, consistent with the spawn orientation.
                self._patrol_waypoint_indices.append(first_target_idx)
                self._patrol_waypoint_directions.append(direction)
                self._patrol_stuck_counters.append(0)

        logger.debug(f"Spawned {len(self._patrol_npcs)} patrol NPC vehicles.")

    def _spawn_pedestrians(self) -> None:
        """
        @brief Spawn random-walk pedestrians inside the lot pedestrian zones.

        Uses WalkerControl with a random direction, re-randomised every
        pedestrian_heading_resample_steps steps.
        """
        if self.world is None:
            return

        zones_raw = self._current_layout.get("pedestrian_zones", [])
        if not zones_raw:
            return

        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        def _to_zone_dict(zone_raw: Dict[str, Any]) -> Dict[str, float]:
            """@brief Wrap zone_bbox tuple into the dict format used internally."""
            x_min, x_max, y_min, y_max = zone_bbox(zone_raw)
            return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        zones: List[Dict[str, float]] = [_to_zone_dict(zr) for zr in zones_raw]

        # Always fill every zone (one pedestrian each), up to the configured maximum.
        num_peds = min(self._num_pedestrians_max, len(zones))
        if num_peds == 0:
            return

        for ped_i in range(num_peds):
            # Assign zones round-robin so all zones are populated evenly
            zone = zones[ped_i % len(zones)]

            bp = random.choice(self._walker_blueprints)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            # Retry spawn with fresh random positions -- narrow zones mean the
            # first attempt may collide with an existing actor or parked car.
            walker = None
            for _ in range(5):
                px = random.uniform(zone["x_min"], zone["x_max"])
                py = random.uniform(zone["y_min"], zone["y_max"])
                transform = carla.Transform(
                    carla.Location(x=px, y=py, z=z),
                    carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                )
                walker = self.world.try_spawn_actor(bp, transform)
                if walker is not None:
                    break

            if walker is not None:
                heading_rad = random.uniform(0.0, 2.0 * math.pi)
                self._pedestrian_actors.append(walker)
                self._pedestrian_headings.append(
                    (math.cos(heading_rad), math.sin(heading_rad), 0.0)
                )
                self._pedestrian_heading_steps.append(0)
                self._pedestrian_lifetime_steps.append(0)
                self._pedestrian_zones.append(zone)

        logger.debug(f"Spawned {len(self._pedestrian_actors)} pedestrians.")

    # ------------------------------------------------------------------
    # Per-step NPC updates
    # ------------------------------------------------------------------

    def _update_patrol_npcs(self) -> None:
        """
        @brief Advance patrol NPC vehicles one step using a proportional heading
               controller.

        Proportional controller: steer = k_p * heading_error_to_next_waypoint.
        Wraps to next waypoint when within 3 m.
        """
        waypoints_raw = self._current_layout.get("patrol_waypoints", [])
        if not waypoints_raw:
            return

        waypoints: List[Tuple[float, float]] = [
            (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
        ]
        k_p = self._patrol_heading_gain

        # Use cached actor list; rebuilt once after spawn, avoids per-step world query
        all_vehicles: List[Any] = self._all_vehicle_actors

        for i, npc in enumerate(self._patrol_npcs):
            if not (npc is not None and npc.is_alive):
                continue

            wp_idx = self._patrol_waypoint_indices[i]
            wp_x, wp_y = waypoints[wp_idx]

            t = npc.get_transform()
            dx = wp_x - t.location.x
            dy = wp_y - t.location.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 3.0:
                # Advance to next waypoint in the NPC's chosen direction (cyclic)
                direction = self._patrol_waypoint_directions[i]
                wp_idx = (wp_idx + direction) % len(waypoints)
                self._patrol_waypoint_indices[i] = wp_idx
                wp_x, wp_y = waypoints[wp_idx]
                dx = wp_x - t.location.x
                dy = wp_y - t.location.y

            # Heading error to waypoint
            target_yaw = math.atan2(dy, dx)
            ego_yaw = math.radians(t.rotation.yaw)
            heading_error = math.atan2(
                math.sin(target_yaw - ego_yaw),
                math.cos(target_yaw - ego_yaw),
            )

            steer = float(np.clip(k_p * heading_error, -1.0, 1.0))

            # Obstacle proximity check - brake if any vehicle or pedestrian is ahead
            npc_yaw = math.radians(t.rotation.yaw)
            fwd_x = math.cos(npc_yaw)
            fwd_y = math.sin(npc_yaw)
            blocked = False
            for other in all_vehicles:
                if other.id == npc.id:
                    continue
                to_x = other.get_location().x - t.location.x
                to_y = other.get_location().y - t.location.y
                fwd_proj = to_x * fwd_x + to_y * fwd_y
                lat = abs(to_x * fwd_y - to_y * fwd_x)
                other_dist = math.sqrt(to_x * to_x + to_y * to_y)
                if (
                    0.0 < fwd_proj
                    and other_dist < self._patrol_obstacle_distance
                    and lat < 2.0
                ):
                    blocked = True
                    break

            if not blocked:
                # Brake if any pedestrian is within the stop radius (all directions)
                for walker in self._pedestrian_actors:
                    if walker is None or not walker.is_alive:
                        continue
                    to_x = walker.get_location().x - t.location.x
                    to_y = walker.get_location().y - t.location.y
                    walker_dist = math.sqrt(to_x * to_x + to_y * to_y)
                    if walker_dist < self._patrol_pedestrian_distance:
                        blocked = True
                        break

            v = npc.get_velocity()
            speed = math.sqrt(v.x * v.x + v.y * v.y)
            over_limit = speed > self._patrol_max_speed

            # Stuck detection: count steps below speed threshold while throttle applied.
            # If stuck too long, teleport to the next waypoint to recover.
            applying_throttle = not blocked and not over_limit
            if applying_throttle and speed < self._patrol_stuck_speed_threshold:
                self._patrol_stuck_counters[i] += 1
            else:
                self._patrol_stuck_counters[i] = 0

            if self._patrol_stuck_counters[i] >= self._patrol_stuck_steps_max:
                # Advance two waypoints ahead to clear whatever is blocking the NPC.
                direction = self._patrol_waypoint_directions[i]
                wp_idx = (wp_idx + 2 * direction) % len(waypoints)
                self._patrol_waypoint_indices[i] = wp_idx
                self._patrol_stuck_counters[i] = 0
                recover_x, recover_y = waypoints[wp_idx]
                next_recover_idx = (wp_idx + direction) % len(waypoints)
                next_recover_x, next_recover_y = waypoints[next_recover_idx]
                recover_yaw = math.degrees(
                    math.atan2(next_recover_y - recover_y, next_recover_x - recover_x)
                )
                npc.set_transform(
                    carla.Transform(
                        carla.Location(x=recover_x, y=recover_y, z=t.location.z),
                        carla.Rotation(yaw=recover_yaw),
                    )
                )
                logger.debug(
                    f"Patrol NPC {i} was stuck; teleported to waypoint {wp_idx}."
                )
                continue

            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = 0.0 if (blocked or over_limit) else 0.3
            control.brake = 1.0 if blocked else (0.3 if over_limit else 0.0)
            npc.apply_control(control)

    def _respawn_pedestrian(self, idx: int) -> None:
        """
        @brief Destroy and respawn pedestrian at index idx within its assigned zone.

        Called when a pedestrian exits its zone or exceeds its maximum lifetime.
        The pedestrian is destroyed, a new walker is spawned at a random point
        within the same zone, and the lifetime counter is reset.

        @param idx: Index into _pedestrian_actors / _pedestrian_zones.
        """
        if self.world is None:
            return
        zone = self._pedestrian_zones[idx]
        old = self._pedestrian_actors[idx]
        if old is not None and old.is_alive:
            old.destroy()

        z = float(self._current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        bp = random.choice(self._walker_blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        # Retry with fresh positions -- narrow zones can cause collision failures
        walker = None
        for _ in range(5):
            px = random.uniform(zone["x_min"], zone["x_max"])
            py = random.uniform(zone["y_min"], zone["y_max"])
            transform = carla.Transform(
                carla.Location(x=px, y=py, z=z),
                carla.Rotation(yaw=random.uniform(0.0, 360.0)),
            )
            walker = self.world.try_spawn_actor(bp, transform)
            if walker is not None:
                break

        self._pedestrian_actors[idx] = walker
        heading_rad = random.uniform(0.0, 2.0 * math.pi)
        self._pedestrian_headings[idx] = (
            math.cos(heading_rad),
            math.sin(heading_rad),
            0.0,
        )
        self._pedestrian_heading_steps[idx] = 0
        self._pedestrian_lifetime_steps[idx] = 0

    def _update_pedestrians(self) -> None:
        """
        @brief Advance pedestrians one step, re-randomise headings periodically,
               and handle zone boundaries and lifetime expiry.

        Boundary behaviour: when a pedestrian is within _BOUNDARY_MARGIN metres
        of any zone edge, its heading is replaced with a direction toward the
        zone centre. This avoids the oscillation produced by velocity reflection,
        where the walker overshoots the boundary by one physics step, the heading
        is negated, and on the next step the walker is still outside and is
        negated again -- producing a standing vibration on the wall. The inward-
        steering approach matches CARLA Scenario Runner, which assigns a new
        interior waypoint when an actor approaches a trigger-region boundary.

        Lifetime expiry: after pedestrian_max_lifetime_steps steps the walker
        is destroyed and respawned at a new random position within its zone.
        This provides episode variety without relying on boundary despawning.
        """
        # Activate inward correction when this close to any zone edge (metres)
        _BOUNDARY_MARGIN = 0.5

        for i, walker in enumerate(self._pedestrian_actors):
            if not (walker is not None and walker.is_alive):
                continue

            self._pedestrian_lifetime_steps[i] += 1
            self._pedestrian_heading_steps[i] += 1

            # Lifetime expiry: respawn at random position within zone
            if self._pedestrian_lifetime_steps[i] >= self._pedestrian_max_lifetime:
                self._respawn_pedestrian(i)
                continue

            loc = walker.get_location()
            zone = self._pedestrian_zones[i]

            near_boundary = (
                loc.x < zone["x_min"] + _BOUNDARY_MARGIN
                or loc.x > zone["x_max"] - _BOUNDARY_MARGIN
                or loc.y < zone["y_min"] + _BOUNDARY_MARGIN
                or loc.y > zone["y_max"] - _BOUNDARY_MARGIN
            )

            if near_boundary:
                # Steer toward zone centre so the walker moves back inward
                cx = (zone["x_min"] + zone["x_max"]) / 2.0
                cy = (zone["y_min"] + zone["y_max"]) / 2.0
                to_cx = cx - loc.x
                to_cy = cy - loc.y
                magnitude = math.sqrt(to_cx * to_cx + to_cy * to_cy)
                if magnitude > 1e-6:
                    to_cx, to_cy = to_cx / magnitude, to_cy / magnitude
                self._pedestrian_headings[i] = (to_cx, to_cy, 0.0)
                self._pedestrian_heading_steps[i] = 0
            elif self._pedestrian_heading_steps[i] >= self._pedestrian_resample_steps:
                # Periodic random direction change while comfortably inside zone
                heading_rad = random.uniform(0.0, 2.0 * math.pi)
                self._pedestrian_headings[i] = (
                    math.cos(heading_rad),
                    math.sin(heading_rad),
                    0.0,
                )
                self._pedestrian_heading_steps[i] = 0

            dx, dy, dz = self._pedestrian_headings[i]
            control = carla.WalkerControl()
            control.direction = carla.Vector3D(x=dx, y=dy, z=dz)
            control.speed = 1.2
            walker.apply_control(control)

    # ------------------------------------------------------------------
    # Clearance and reward
    # ------------------------------------------------------------------

    def _check_clearance(self) -> bool:
        """
        @brief Check whether the ego vehicle is within clearance threshold of any
               obstacle actor.

        Static obstacles (cones and parked vehicles) use pre-cached 2D positions
        from self._static_obstacle_positions, avoiding a get_location() RPC per
        actor per step.  Dynamic actors (patrol NPCs and pedestrians) still
        require a live get_location() call because they move each step.

        @return True if a clearance violation is detected (< CLEARANCE_THRESHOLD).
        """
        if self.vehicle is None:
            return False

        ego_loc = self.vehicle.get_location()
        ego_x = ego_loc.x
        ego_y = ego_loc.y

        # Static obstacles: use cached (x, y) -- no get_location() needed
        for ox, oy in self._static_obstacle_positions:
            dx = ego_x - ox
            dy = ego_y - oy
            if math.sqrt(dx * dx + dy * dy) < CLEARANCE_THRESHOLD:
                return True

        # Dynamic actors: patrol NPCs and pedestrians require live position
        for actor in self._patrol_npcs + self._pedestrian_actors:
            if actor is None or not actor.is_alive:
                continue
            other_loc = actor.get_location()
            if ego_loc.distance(other_loc) < CLEARANCE_THRESHOLD:
                return True

        return False

    def _compute_reward(self) -> Tuple[float, bool, bool]:
        """
        @brief Compute reward and termination flags for the current step.
        @return Tuple of (reward, terminated, success).

        Uses CARLA ground truth transform (not EKF pose) for position and
        orientation errors. Reward formula kept from original implementation;
        Task 9 rewrites it with potential-based shaping.

        Termination conditions (priority order):
          1. Clearance violation: < CLEARANCE_THRESHOLD to any obstacle
          2. Success: position < SUCCESS_THRESHOLD_POSITION,
             yaw < SUCCESS_THRESHOLD_ORIENTATION,
             velocity < SUCCESS_THRESHOLD_VELOCITY
          3. Out-of-bounds: > OUT_OF_BOUNDS_THRESHOLD from target
          4. Time limit handled externally via truncated flag in step()
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
        yaw_error_raw = yaw - target_yaw
        orientation_error = abs(
            math.atan2(math.sin(yaw_error_raw), math.cos(yaw_error_raw))
        )

        # Check clearance violation (collision penalty + termination)
        if self._check_clearance():
            return -10.0, True, False

        # Success condition
        success = (
            position_error < SUCCESS_THRESHOLD_POSITION
            and orientation_error < SUCCESS_THRESHOLD_ORIENTATION
            and speed < SUCCESS_THRESHOLD_VELOCITY
        )
        if success:
            return 100.0, True, True

        # Out-of-bounds
        if position_error > OUT_OF_BOUNDS_THRESHOLD:
            return -5.0, True, False

        # Shaping reward
        reward = -position_error - 0.5 * orientation_error - 0.1 * speed

        return float(reward), False, False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_state(self) -> np.ndarray:
        """
        @brief Build the 18-dim (or 9-dim) observation vector.
        @return Float32 array of shape (TOTAL_OBS_DIM,) or (VEHICLE_STATE_DIM
                + TARGET_POSE_DIM,).

        Indices 0-5: EKF pose (x, y, yaw, vx, vy, vyaw).
                     Currently sourced from CARLA ground truth as a placeholder
                     until the EKF ROS 2 bridge is integrated in Task 7.
        Indices 6-14: EKF covariance features (only when include_covariance=True).
        Indices 15-17 (or 6-8): relative target bay pose (dx, dy, dyaw).
        """
        if self.vehicle is None or self.world is None:
            obs_dim = (
                TOTAL_OBS_DIM
                if self._include_covariance
                else VEHICLE_STATE_DIM + TARGET_POSE_DIM
            )
            return np.zeros(obs_dim, dtype=np.float32)

        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        angular_vel = self.vehicle.get_angular_velocity()

        x = transform.location.x
        y = transform.location.y
        yaw = math.radians(transform.rotation.yaw)
        vx = velocity.x
        vy = velocity.y
        vyaw = math.radians(angular_vel.z)

        # Relative target pose in ego body frame
        dx, dy, dyaw = _compute_relative_target_pose(
            x,
            y,
            yaw,
            self._target_bay["x"],
            self._target_bay["y"],
            self._target_bay["yaw"],
        )

        if not self._include_covariance:
            # Fill pre-allocated 9-dim buffer in-place: [pose(6), target(3)]
            self._obs_buffer[0] = x
            self._obs_buffer[1] = y
            self._obs_buffer[2] = yaw
            self._obs_buffer[3] = vx
            self._obs_buffer[4] = vy
            self._obs_buffer[5] = vyaw
            self._obs_buffer[6] = dx
            self._obs_buffer[7] = dy
            self._obs_buffer[8] = dyaw
            return self._obs_buffer.copy()

        # EKF covariance features
        if self._cov_subscriber is not None:
            uncertainty = self._cov_subscriber.get_latest_uncertainty()
        else:
            uncertainty = None

        if uncertainty is None:
            uncertainty = np.zeros(COVARIANCE_FEATURES_DIM, dtype=np.float32)
        else:
            uncertainty = uncertainty.astype(np.float32)

        # Fill pre-allocated 18-dim buffer in-place: [pose(6), covariance(9), target(3)]
        self._obs_buffer[0] = x
        self._obs_buffer[1] = y
        self._obs_buffer[2] = yaw
        self._obs_buffer[3] = vx
        self._obs_buffer[4] = vy
        self._obs_buffer[5] = vyaw
        self._obs_buffer[6:15] = uncertainty
        self._obs_buffer[15] = dx
        self._obs_buffer[16] = dy
        self._obs_buffer[17] = dyaw
        return self._obs_buffer.copy()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _write_vis_state(self) -> None:
        """
        @brief Write the visualisation state JSON for the detachable 2D viewer.

        Atomic write: tmp file then os.replace (POSIX atomic rename).
        The visualiser process polls this file and redraws on change.
        If vis_output_path is None, this method is a no-op.
        """
        if self._vis_output_path is None or self.vehicle is None:
            return

        transform = self.vehicle.get_transform()
        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw

        actor_transforms = []
        for actor in self._patrol_npcs + self._spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                at = actor.get_transform()
                actor_transforms.append(
                    {
                        "x": at.location.x,
                        "y": at.location.y,
                        "yaw": at.rotation.yaw,
                        "type": "npc" if actor in self._patrol_npcs else "static",
                    }
                )

        pedestrian_transforms = []
        for walker in self._pedestrian_actors:
            if walker is not None and walker.is_alive:
                wt = walker.get_transform()
                pedestrian_transforms.append({"x": wt.location.x, "y": wt.location.y})

        state = {
            "ego": {"x": x, "y": y, "yaw": yaw},
            "trajectory": list(self._trajectory_buffer),
            "actors": actor_transforms,
            "pedestrians": pedestrian_transforms,
            "target_bay": self._target_bay,
            "floor_plan": self._current_floor_plan_name,
            "bays": self._current_layout.get("bays", []),
            "corners": self._current_layout.get("corners", []),
            "episode_step": self.steps,
        }

        try:
            json_str = json.dumps(state)
            output_path = self._vis_output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to .tmp then rename atomically
            tmp_path = output_path.with_suffix(".tmp")
            tmp_path.write_text(json_str)
            os.replace(str(tmp_path), str(output_path))
        except Exception as exc:
            # Non-fatal -- visualisation is optional
            logger.debug(f"Could not write vis state: {exc}")

    # ------------------------------------------------------------------
    # CARLA world management
    # ------------------------------------------------------------------

    def _connect_to_carla(self) -> None:
        """
        @brief Connect to CARLA and load the world.

        @note When town is "FlatPlane", loads configs/flat_plane.xodr via
              generate_opendrive_world() - a clean flat plane with no roads or
              buildings. Otherwise uses load_world() for named CARLA towns.
        """
        try:
            self.client = carla.Client(self.carla_host, self.carla_port)
            self.client.set_timeout(60.0)
            self.world = self.client.get_world()

            current_map_name = self.world.get_map().name.split("/")[-1]
            if self.town == "FlatPlane":
                if current_map_name != "FlatPlane":
                    xodr = Path("configs/flat_plane.xodr").read_text(encoding="utf-8")
                    logger.info("Loading flat_plane.xodr ...")
                    self.world = self.client.generate_opendrive_world(
                        xodr,
                        carla.OpendriveGenerationParameters(
                            vertex_distance=2.0,
                            max_road_length=600.0,
                            wall_height=0.0,
                            additional_width=300.0,
                            smooth_junctions=False,
                            enable_mesh_visibility=True,
                        ),
                    )
                    time.sleep(5.0)
                else:
                    logger.info("FlatPlane already loaded.")
            elif current_map_name != self.town:
                logger.info(f"Loading map: {self.town}")
                self.world = self.client.load_world(self.town)
                time.sleep(8.0)

        except Exception as exc:
            logger.error(f"Could not connect to CARLA: {exc}")
            self.client = None
            self.world = None

    def _unload_unnecessary_layers(self) -> None:
        """
        @brief Strip buildings, foliage, and props from the layered map to leave
               only the ground mesh, reducing scene complexity.
        """
        if self.world is None or carla is None:
            return

        layers_to_unload = [
            carla.MapLayer.Buildings,
            carla.MapLayer.Foliage,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
        ]
        for layer in layers_to_unload:
            try:
                self.world.unload_map_layer(layer)
            except Exception:
                pass  # Layer may not exist for all maps

    def _configure_weather(self) -> None:
        """
        @brief Randomise weather for the current episode.
        """
        if self.world is None:
            return

        presets = self._conditions_config.get("weather_presets", ["ClearNoon"])
        preset_name = random.choice(presets)

        weather = getattr(carla.WeatherParameters, preset_name, None)
        if weather is None:
            logger.warning(f"Unknown weather preset '{preset_name}', using ClearNoon.")
            weather = carla.WeatherParameters.ClearNoon

        fog_range = self._conditions_config.get("fog_density_range", [0.0, 0.0])
        fog_density = random.uniform(fog_range[0], fog_range[1])

        fog_dist_range = self._conditions_config.get(
            "fog_distance_range", [20.0, 100.0]
        )
        fog_distance = random.uniform(fog_dist_range[0], fog_dist_range[1])

        weather.fog_density = fog_density
        weather.fog_distance = fog_distance

        self.world.set_weather(weather)
        logger.info(
            f"Weather: {preset_name}, fog={fog_density:.1f}, "
            f"fog_dist={fog_distance:.1f}m"
        )

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
        # Store for _spawn_perimeter_cones() so only this entry gap is opened.
        self._chosen_spawn = chosen
        sx = float(chosen.get("x", 0.0))
        sy = float(chosen.get("y", 0.0))
        sz = float(chosen.get("z", default_z))
        syaw = float(chosen.get("yaw_deg", 0.0))

        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.bmw.grandtourer")[0]

        spawn_transform = carla.Transform(
            carla.Location(x=sx, y=sy, z=sz),
            carla.Rotation(yaw=syaw),
        )

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
        if self.vehicle is None:
            # Fall back to any valid spawn point on the map
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                self.vehicle = self.world.try_spawn_actor(
                    vehicle_bp, random.choice(spawn_points)
                )
            logger.warning("Chosen spawn point occupied, using fallback spawn.")

        if self.vehicle is not None:
            time.sleep(0.5)

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn noisy IMU sensor attached to ego vehicle.

        Sensor noise parameters come from carla_sensors_config. Sensor data
        flows through the CARLA ROS bridge into the robot_localisation EKF.

        @note GNSS is excluded from Suite A (unreliable indoors / at Lemonworx).
        """
        if self.vehicle is None or self.world is None:
            return

        bp_lib = self.world.get_blueprint_library()
        imu_config = self._sensors_config.get("imu", {})

        imu_bp = bp_lib.find("sensor.other.imu")
        for attr, default in [
            ("noise_accel_stddev_x", 0.1),
            ("noise_accel_stddev_y", 0.1),
            ("noise_accel_stddev_z", 0.1),
            ("noise_gyro_stddev_x", 0.01),
            ("noise_gyro_stddev_y", 0.01),
            ("noise_gyro_stddev_z", 0.01),
            ("sensor_tick", 0.05),
        ]:
            imu_bp.set_attribute(attr, str(imu_config.get(attr, default)))

        imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle,
        )
        self._spawned_sensors.append(imu_sensor)

    def _wait_for_covariance(self) -> None:
        """
        @brief Block until the first EKF covariance message arrives.

        Ticks the CARLA simulation while waiting. Raises RuntimeError on timeout.
        """
        if self._cov_subscriber is None:
            return

        start = time.monotonic()
        tick_interval = 0.05

        while not self._cov_subscriber.has_data:
            elapsed = time.monotonic() - start
            if elapsed > self._covariance_timeout:
                raise RuntimeError(
                    f"No covariance message received within "
                    f"{self._covariance_timeout}s. "
                    f"Ensure ros2-bridge container is healthy."
                )
            if self.world is not None:
                self.world.tick()
            time.sleep(tick_interval)

        logger.debug(
            f"First covariance received after {time.monotonic() - start:.2f}s."
        )

    def _cleanup_actors(self) -> None:
        """
        @brief Destroy all episode actors (sensors, cones, static vehicles, NPCs,
               pedestrians, ego vehicle).
        """
        for sensor in self._spawned_sensors:
            if sensor is not None and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._spawned_sensors.clear()

        for actor_list in [
            self._spawned_cones,
            self._spawned_static_vehicles,
            self._patrol_npcs,
            self._pedestrian_actors,
        ]:
            for actor in actor_list:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            actor_list.clear()

        self._patrol_waypoint_indices.clear()
        self._patrol_waypoint_directions.clear()
        self._patrol_stuck_counters.clear()
        self._pedestrian_heading_steps.clear()
        self._pedestrian_headings.clear()
        self._pedestrian_lifetime_steps.clear()
        self._pedestrian_zones.clear()

        # Clear per-episode performance caches
        self._static_obstacle_positions.clear()
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
        bay, spawns cones/static vehicles/patrol NPCs/pedestrians, randomises weather.
        """
        super().reset(seed=seed)

        self.steps = 0
        self._trajectory_buffer.clear()

        self._cleanup_actors()

        # Connect to CARLA on first reset
        if self.client is None:
            self._connect_to_carla()

        if self.world is None:
            obs_dim = (
                TOTAL_OBS_DIM
                if self._include_covariance
                else VEHICLE_STATE_DIM + TARGET_POSE_DIM
            )
            return np.zeros(obs_dim, dtype=np.float32), {}

        # Load floor plan and sample target bay
        if self._floor_plans_config:
            self._load_floor_plan()
            self._sample_target_bay()

        self._configure_weather()
        self._unload_unnecessary_layers()

        # Cache blueprint lists once before any spawning to avoid repeated
        # world queries inside each spawn method.
        self._cache_blueprints()

        self._spawn_vehicle()
        self._spawn_sensors()
        self._spawn_perimeter_cones()
        self._spawn_obstacle_cones()
        self._spawn_static_vehicles()
        self._spawn_npc_patrol()
        self._spawn_pedestrians()

        # Rebuild vehicle actor cache after all vehicles are spawned so
        # _update_patrol_npcs() can use it without a per-step world query.
        if self.world is not None:
            self._all_vehicle_actors = list(self.world.get_actors().filter("vehicle.*"))

        if self._include_covariance:
            self._wait_for_covariance()

        state = self._get_state()

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

        if self.vehicle is not None:
            control = carla.VehicleControl()
            control.steer = float(np.clip(action[0], -1.0, 1.0))
            control.throttle = float(np.clip(action[1], 0.0, 1.0))
            control.brake = float(np.clip(action[2], 0.0, 1.0))
            self.vehicle.apply_control(control)

            if self.world is not None:
                self._update_patrol_npcs()
                self._update_pedestrians()
                self.world.tick()

                # Update trajectory buffer for vis state writer
                t = self.vehicle.get_transform()
                self._trajectory_buffer.append((t.location.x, t.location.y))

        state = self._get_state()
        reward, terminated, success = self._compute_reward()

        truncated = self.steps >= self.max_steps

        self._write_vis_state()

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

        if self._cov_subscriber is not None:
            self._cov_subscriber.destroy_node()
            self._cov_subscriber = None

        self.client = None
        self.world = None

        super().close()
