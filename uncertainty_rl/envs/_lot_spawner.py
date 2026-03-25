"""
@file _lot_spawner.py
@brief Static lot actor spawner for CARLAParkingEnv.

Owns all per-episode static actor state: perimeter cones, interior obstacle
cones, and parked vehicles. CARLAParkingEnv holds a LotSpawner instance and
delegates static spawning and cleanup to it.

The spawner also maintains a list of static obstacle (x, y) positions that
is used by CARLAParkingEnv._get_obstacle_features() to classify LiDAR returns
as static vs dynamic.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.utils.geometry import (
    _interpolate_cone_positions,
)

logger = logging.getLogger(__name__)

# Vehicle types that overhang a standard 2.5 m bay.
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

# Micro/novelty vehicles excluded from parked car pool.
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


class LotSpawner:
    """
    @class LotSpawner
    @brief Manages static lot actors (cones and parked vehicles) for one episode.

    Instantiated once by CARLAParkingEnv and reused across episodes. Call
    cleanup() at episode reset before spawning new actors. After spawning,
    read static_obstacle_positions to get (x, y) pairs for obstacle
    classification in _get_obstacle_features().

    All CARLA world handles are passed as parameters -- this class does not
    store world references to avoid holding stale handles between episodes.
    """

    def __init__(
        self,
        cone_spacing: float,
        marker_blueprint: str,
        bay_occupancy_min: float,
        bay_occupancy_max: float,
        spawn_perimeter_cones: bool,
    ) -> None:
        """
        @brief Construct LotSpawner with fixed config parameters.

        @param cone_spacing: Spacing between perimeter/obstacle cone markers (metres).
        @param marker_blueprint: CARLA blueprint ID for cone markers (e.g.
               'static.prop.constructioncone').
        @param bay_occupancy_min: Minimum fraction of non-target bays to fill
               with parked cars [0, 1]. Resampled each episode.
        @param bay_occupancy_max: Maximum fraction of non-target bays to fill
               with parked cars [0, 1]. Resampled each episode.
        @param spawn_perimeter_cones: If True, spawn cones along the lot
               perimeter polygon. Set False when perimeter is already embedded
               in the xodr mesh or when running without Cartographer mapping.
        """
        self._cone_spacing = cone_spacing
        self._marker_blueprint = marker_blueprint
        self._bay_occupancy_min = bay_occupancy_min
        self._bay_occupancy_max = bay_occupancy_max
        self._spawn_perimeter_cones_flag = spawn_perimeter_cones

        # Per-episode occupancy rate; resampled in spawn_static_vehicles().
        self._bay_occupancy_rate: float = bay_occupancy_max

        # Per-episode actor lists
        self.spawned_cones: List[Any] = []
        self.spawned_static_vehicles: List[Any] = []

        # Cached (x, y) positions of all static objects in world frame.
        # Populated during spawning; used by _get_obstacle_features() to
        # classify LiDAR returns without per-step actor queries.
        self.static_obstacle_positions: List[Tuple[float, float]] = []

        # Cached blueprint lists -- populated once on first connect, never re-fetched.
        self._car_blueprints: List[Any] = []
        self._cone_bp: Optional[Any] = None
        # Easter-egg motorcycle blueprints (None when CARLA unavailable)
        self._ninja_bp: Optional[Any] = None
        self._yzf_bp: Optional[Any] = None

    # ------------------------------------------------------------------
    # Per-reset setup
    # ------------------------------------------------------------------

    def refresh_blueprints(self, world: Any) -> None:
        """
        @brief Fetch and filter blueprint lists from the CARLA world.

        Populates self._car_blueprints with four-wheeled vehicles excluding
        large and small/novelty types. Also caches easter-egg motorcycle BPs.
        Blueprints are static for the lifetime of the CARLA server, so the
        fetch is skipped on subsequent calls if already populated.

        @param world: Live carla.World handle.
        """
        if world is None:
            return

        # Blueprints do not change between episodes -- only fetch once.
        if self._car_blueprints:
            return

        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        self._car_blueprints = [
            bp
            for bp in vehicle_bps
            if int(bp.get_attribute("number_of_wheels").as_int()) == 4
            and not any(excl in bp.id.lower() for excl in _LARGE_VEHICLE_TYPES)
            and not any(excl in bp.id.lower() for excl in _SMALL_VEHICLE_TYPES)
        ]

        self._ninja_bp = bp_lib.find("vehicle.kawasaki.ninja")
        self._yzf_bp = bp_lib.find("vehicle.yamaha.yzf")
        self._cone_bp = bp_lib.find(self._marker_blueprint)

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def spawn_all(
        self,
        world: Any,
        current_layout: Dict[str, Any],
        target_bay: Dict[str, Any],
    ) -> None:
        """
        @brief Spawn all static actors for one episode.

        Spawns perimeter cones (if enabled), interior obstacle cones, and
        static parked vehicles. Resamples bay occupancy rate each call.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict.
        @param target_bay: Dict with at least 'bay_id' key for the target bay.
        """
        if world is None:
            return

        self._bay_occupancy_rate = random.uniform(
            self._bay_occupancy_min, self._bay_occupancy_max
        )

        if self._spawn_perimeter_cones_flag:
            self._spawn_perimeter_cones(world, current_layout)
        self._spawn_obstacle_cones(world, current_layout)
        self._spawn_static_vehicles(world, current_layout, target_bay)

    def cleanup(self) -> None:
        """
        @brief Destroy all spawned static actors and clear position caches.

        Call at the start of each episode reset before spawning new actors.
        """
        for actor_list in [self.spawned_cones, self.spawned_static_vehicles]:
            for actor in actor_list:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            actor_list.clear()

        self.static_obstacle_positions.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_perimeter_cones(
        self, world: Any, current_layout: Dict[str, Any]
    ) -> None:
        """
        @brief Spawn static markers along the lot perimeter polygon.

        Markers are placed using _interpolate_cone_positions() and physics is
        disabled so they don't move. The blueprint is configurable via
        parking_scenarios.perimeter_marker_blueprint in train_config.yaml.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'corners' key.
        """
        corners_raw = current_layout.get("corners", [])
        if not corners_raw:
            logger.warning("No perimeter corners found in layout, skipping cones.")
            return

        corners: List[Tuple[float, float]] = [
            (float(c["x"]), float(c["y"])) for c in corners_raw
        ]
        cone_positions = _interpolate_cone_positions(corners, self._cone_spacing)

        cone_bp = self._cone_bp
        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        for cx, cy, yaw_deg in cone_positions:
            transform = carla.Transform(
                carla.Location(x=cx, y=cy, z=z),
                carla.Rotation(yaw=yaw_deg),
            )
            cone = world.try_spawn_actor(cone_bp, transform)
            if cone is not None:
                cone.set_simulate_physics(False)
                self.spawned_cones.append(cone)
                self.static_obstacle_positions.append((cx, cy))

        logger.debug("Spawned %d perimeter markers.", len(self.spawned_cones))

    def _spawn_obstacle_cones(
        self, world: Any, current_layout: Dict[str, Any]
    ) -> None:
        """
        @brief Spawn static markers around interior obstacle rectangles.

        Each obstacle in the layout YAML is a centre + half-extents rectangle.
        Markers are placed along the four sides at the same spacing used for
        perimeter markers. Physics is disabled so they act as static LiDAR
        targets.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'obstacles' key.
        """
        obstacles = current_layout.get("obstacles", [])
        if not obstacles:
            return

        cone_bp = self._cone_bp
        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        for obs in obstacles:
            cx = float(obs["centre_x"])
            cy = float(obs["centre_y"])
            hw = float(obs["half_width"])
            hh = float(obs["half_height"])

            cone_positions: List[Tuple[float, float, float]] = []
            # Bottom and top horizontal edges (y constant, yaw=0)
            for edge_y in (cy - hh, cy + hh):
                t = -hw
                while t <= hw + 1e-6:
                    cone_positions.append((cx + t, edge_y, 0.0))
                    t += self._cone_spacing
            # Left and right vertical edges (x constant, yaw=90), excluding corners
            for edge_x in (cx - hw, cx + hw):
                t = -hh + self._cone_spacing
                while t < hh - 1e-6:
                    cone_positions.append((edge_x, cy + t, 90.0))
                    t += self._cone_spacing

            for px, py, yaw_deg in cone_positions:
                transform = carla.Transform(
                    carla.Location(x=px, y=py, z=z),
                    carla.Rotation(yaw=yaw_deg),
                )
                cone = world.try_spawn_actor(cone_bp, transform)
                if cone is not None:
                    cone.set_simulate_physics(False)
                    self.spawned_cones.append(cone)
                    self.static_obstacle_positions.append((px, py))

        logger.debug(
            "Spawned obstacle cones for %d interior obstacle(s).", len(obstacles)
        )

    def _spawn_static_vehicles(
        self,
        world: Any,
        current_layout: Dict[str, Any],
        target_bay: Dict[str, Any],
    ) -> None:
        """
        @brief Fill non-target bays with static parked vehicles.

        Target bay and its immediate neighbours (same type, index +/-1) are
        never filled. Keeping adjacent bays clear gives the agent realistic
        manoeuvring clearance. Static vehicles have physics disabled.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'bays' key.
        @param target_bay: Dict with 'bay_id' key for the selected target bay.
        """
        if not self._car_blueprints:
            logger.warning("No car blueprints cached -- call refresh_blueprints() first.")
            return

        bays = current_layout.get("bays", [])
        target_id = target_bay.get("bay_id", "")
        excluded_ids = {target_id} | set(self._adjacent_bay_ids(target_id))

        z = float(current_layout.get("origin", {}).get("z", 0.3))
        # Spawn 2 m above the floor so the bounding box clears perimeter cones
        # whose tops reach ~1 m. Physics is immediately disabled and the actor
        # is teleported back to ground level after the post-spawn tick.
        z_spawn = z + 2.0
        _pending_ground: List[Tuple[Any, float, float, float]] = []

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
            # Randomly reverse the parked car 50% of the time -- both nose-in
            # and nose-out orientations are valid in a real car park.
            if random.random() < 0.5:
                yaw = (yaw + 180.0) % 360.0

            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.spawned_static_vehicles.append(actor)
                self.static_obstacle_positions.append((bay_x, bay_y))
                _pending_ground.append((actor, bay_x, bay_y, yaw))

        # Easter egg: always spawn Kawasaki Ninja and Yamaha YZF in their
        # dedicated motorcycle bays (bay_type="motorcycle", occupant field set).
        _occupant_bp = {
            "Kawasaki Ninja": self._ninja_bp,
            "Yamaha YZF-R": self._yzf_bp,
        }
        for bay in bays:
            if bay.get("bay_type") != "motorcycle":
                continue
            occupant = bay.get("occupant", "")
            bp = _occupant_bp.get(occupant)
            if bp is None:
                continue
            bay_x = float(bay["x"])
            bay_y = float(bay["y"])
            yaw = float(bay.get("yaw_deg", 0.0))
            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                self.spawned_static_vehicles.append(actor)
                self.static_obstacle_positions.append((bay_x, bay_y))
                _pending_ground.append((actor, bay_x, bay_y, yaw))

        # Tick once so CARLA commits the physics-disabled state, then teleport
        # every static vehicle to ground level. Without the tick, set_transform
        # is ignored in synchronous mode and cars remain at z_spawn (floating).
        world.tick()
        for actor, ax, ay, actor_yaw in _pending_ground:
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=ax, y=ay, z=z),
                    carla.Rotation(yaw=actor_yaw),
                )
            )

        logger.debug("Spawned %d static vehicles.", len(self.spawned_static_vehicles))

    @staticmethod
    def _adjacent_bay_ids(target_id: str) -> List[str]:
        """
        @brief Return bay IDs adjacent (index +/-1, same type) to the target.

        Bay IDs follow the convention '<type>_<index>' (e.g. 'parallel_3').
        Adjacent bays are left empty so the agent has clearance to manoeuvre
        into the target bay.

        @param target_id: Bay ID string of the selected target bay.
        @return List of adjacent bay ID strings (may be empty for end bays).
        """
        try:
            bay_type, idx_str = target_id.rsplit("_", 1)
            idx = int(idx_str)
        except ValueError:
            return []
        return [f"{bay_type}_{idx - 1}", f"{bay_type}_{idx + 1}"]
