"""
@file _lot_spawner.py
@brief Static lot actor spawner for CARLAParkingEnv.

Owns all per-episode static actor state: perimeter cones, interior obstacle
cones, and parked vehicles. CARLAParkingEnv holds a LotSpawner instance and
delegates static spawning and cleanup to it.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.utils.geometry import _interpolate_cone_positions

logger = logging.getLogger(__name__)


def _bay_yaw_deg(bay: Dict[str, Any]) -> float:
    """
    @brief Extract bay heading in degrees from a layout YAML bay dict.

    Layout YAMLs may store heading as ``yaw_deg`` (degrees, preferred) or
    ``yaw`` (radians, legacy).  Returns 0.0 if neither key is present.

    @param bay: Single bay entry from the layout YAML.
    @return Heading in degrees.
    """
    if "yaw_deg" in bay:
        return float(bay["yaw_deg"])
    return math.degrees(float(bay.get("yaw", 0.0)))


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

# Emergency / special-purpose vehicles excluded from both parked and patrol pools.
# Police cars and taxis are visually implausible in a civilian parking lot scenario.
_EXCLUDED_VEHICLE_TYPES: Tuple[str, ...] = (
    "police",
    "taxi",
    "cab",
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
    cleanup() at episode reset before spawning new actors.

    All CARLA world handles are passed as parameters - this class does not
    store world references to avoid holding stale handles between episodes.
    """

    # Spawn height offsets above floor_contact_z
    _CONE_Z_OFFSET: float = 0.5
    _VEHICLE_Z_OFFSET: float = 1.0
    # Settle loop parameters
    _SETTLE_TICKS: int = 20
    _SETTLE_VZ_THRESHOLD: float = 0.01
    # Motorcycle occupant name -> blueprint attribute (matches YAML 'occupant' field)
    _MOTORCYCLE_OCCUPANTS: Tuple[str, ...] = ("Kawasaki Ninja", "Yamaha YZF-R")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        cone_spacing: float,
        marker_blueprint: str,
        bay_occupancy_min: float,
        bay_occupancy_max: float,
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
        """
        self._cone_spacing = cone_spacing
        self._marker_blueprint = marker_blueprint
        self._bay_occupancy_min = bay_occupancy_min
        self._bay_occupancy_max = bay_occupancy_max

        # Per-episode occupancy rate; resampled each episode in spawn_all().
        self._bay_occupancy_rate: float = bay_occupancy_max

        # Per-episode actor lists
        self.spawned_cones: List[Any] = []
        self.spawned_static_vehicles: List[Any] = []

        # Cones are layout-specific and fixed for the lifetime of a floor plan.
        # Caching avoids destroying and re-spawning ~60-80 actors every episode reset.
        self._cached_cones_layout: str = ""

        # Cached blueprint lists - populated once on first connect, never re-fetched.
        self._car_blueprints: List[Any] = []
        self._cone_bp: Optional[Any] = None
        # Dedicated motorcycle blueprints for fixed bays (Kawasaki Ninja, Yamaha YZF).
        self._ninja_bp: Optional[Any] = None
        self._yzf_bp: Optional[Any] = None

    # ------------------------------------------------------------------
    # Per-reset setup
    # ------------------------------------------------------------------

    def refresh_blueprints(self, world: Any) -> None:
        """
        @brief Fetch and filter blueprint lists from the CARLA world.

        Populates self._car_blueprints with four-wheeled vehicles excluding
        large, small/novelty, and special-purpose types. Also caches dedicated
        motorcycle blueprints for fixed bays. Blueprints are static for the
        lifetime of the CARLA server, so the fetch is skipped on subsequent
        calls if already populated.

        @param world: Live carla.World handle.
        """
        if world is None:
            return

        # Blueprints do not change between episodes - only fetch once.
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
            and not any(excl in bp.id.lower() for excl in _EXCLUDED_VEHICLE_TYPES)
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
        floor_contact_z: float = 0.3,
        layout_name: str = "",
    ) -> None:
        """
        @brief Spawn all static actors for one episode.

        Perimeter and obstacle cones are cached across episodes for the same
        floor plan - they are layout-fixed and do not need to be destroyed and
        re-spawned every reset. Only parked vehicles are re-randomised each
        episode.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict.
        @param target_bay: Dict with at least 'bay_id' key for the target bay.
        @param floor_contact_z: Ego vehicle CoM z after settling under gravity.
               Used as the drop height reference for cones and vehicles.
        @param layout_name: Floor plan name for cone cache invalidation.
               Pass the same value as self._current_floor_plan_name in the env.
        """
        if world is None:
            return

        self._bay_occupancy_rate = random.uniform(
            self._bay_occupancy_min, self._bay_occupancy_max
        )

        # Cones (layout-fixed, cached across episodes)
        cones_already_spawned = (
            layout_name
            and layout_name == self._cached_cones_layout
            and all(c.is_alive for c in self.spawned_cones)
        )

        if cones_already_spawned:
            logger.debug(
                "Reusing %d cached cone actors for layout '%s'.",
                len(self.spawned_cones),
                layout_name,
            )
        else:
            # Destroy any stale cones from a previous layout then re-spawn.
            for cone in self.spawned_cones:
                if cone is not None and cone.is_alive:
                    cone.destroy()
            self.spawned_cones.clear()

            cone_pending: List[Tuple[Any, float, float, float]] = []
            cone_pending.extend(
                self._spawn_perimeter_cones(world, current_layout, floor_contact_z)
            )
            cone_pending.extend(
                self._spawn_obstacle_cones(world, current_layout, floor_contact_z)
            )

            self._settle_pending(world, cone_pending)

            for actor, ax, ay, actor_yaw in cone_pending:
                if not actor.is_alive:
                    continue
                actor.set_simulate_physics(False)
                settled_z = actor.get_transform().location.z
                actor.set_transform(
                    carla.Transform(
                        carla.Location(x=ax, y=ay, z=settled_z),
                        carla.Rotation(yaw=actor_yaw),
                    )
                )
                self.spawned_cones.append(actor)

            self._cached_cones_layout = layout_name
            logger.debug(
                "Spawned and cached %d cone actors for layout '%s'.",
                len(self.spawned_cones),
                layout_name,
            )

        # Parked vehicles (re-randomised every episode)
        vehicle_pending = self._spawn_static_vehicles(
            world, current_layout, target_bay, floor_contact_z
        )

        # On FlatPlane the drop height is <0.1 m; vehicles settle in 2-4 ticks.
        self._settle_pending(world, vehicle_pending)

        for actor, ax, ay, actor_yaw in vehicle_pending:
            if not actor.is_alive:
                continue
            actor.set_simulate_physics(False)
            settled_z = actor.get_transform().location.z
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=ax, y=ay, z=settled_z),
                    carla.Rotation(yaw=actor_yaw),
                )
            )
            self.spawned_static_vehicles.append(actor)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        @brief Destroy per-episode parked vehicles and clear the vehicle list.

        Cones are kept alive and reused by the next episode if the floor plan
        has not changed. Call at the start of each episode reset.
        """
        for actor in self.spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.spawned_static_vehicles.clear()

    def cleanup_all(self) -> None:
        """
        @brief Destroy all static actors including cached cones.

        Call from CARLAParkingEnv.close() to fully clean up on shutdown.
        """
        for actor_list in [self.spawned_cones, self.spawned_static_vehicles]:
            for actor in actor_list:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            actor_list.clear()

        self._cached_cones_layout = ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _settle_pending(
        self,
        world: Any,
        pending: List[Tuple[Any, float, float, float]],
    ) -> None:
        """
        @brief Tick the world until all pending actors have settled under gravity.

        Actors in ``pending`` must have physics enabled. Ticks up to
        ``_SETTLE_TICKS`` times (1 s at 20 Hz), stopping early once every
        alive actor's vertical velocity drops below ``_SETTLE_VZ_THRESHOLD``.

        @param world: Live carla.World handle.
        @param pending: List of (actor, x, y, yaw) tuples awaiting settlement.
        """
        for _ in range(self._SETTLE_TICKS):
            world.tick()
            if all(
                abs(a.get_velocity().z) < self._SETTLE_VZ_THRESHOLD
                for a, _, _, _ in pending
                if a.is_alive
            ):
                break

    def _spawn_perimeter_cones(
        self, world: Any, current_layout: Dict[str, Any], floor_contact_z: float
    ) -> List[Tuple[Any, float, float, float]]:
        """
        @brief Spawn static markers along the lot perimeter polygon with physics ON.

        Returns pending (actor, x, y, yaw) tuples for the shared settle loop in
        spawn_all(). Physics is left ON so gravity drops each cone to the true
        ground surface; the caller freezes them after settling.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'corners' key.
        @param floor_contact_z: Ego CoM z - used as the spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned cone.
        """
        corners_raw = current_layout.get("corners", [])
        if not corners_raw:
            logger.warning("No perimeter corners found in layout, skipping cones.")
            return []

        corners: List[Tuple[float, float]] = [
            (float(c["x"]), float(c["y"])) for c in corners_raw
        ]
        cone_positions = _interpolate_cone_positions(corners, self._cone_spacing)

        # Spawn slightly above the ego CoM z so the cone clears the surface
        # before physics drops it flush to the ground.
        z = floor_contact_z + self._CONE_Z_OFFSET
        pending: List[Tuple[Any, float, float, float]] = []

        for cx, cy, yaw_deg in cone_positions:
            cone = world.try_spawn_actor(
                self._cone_bp,
                carla.Transform(
                    carla.Location(x=cx, y=cy, z=z),
                    carla.Rotation(yaw=yaw_deg),
                ),
            )
            if cone is not None:
                cone.set_simulate_physics(True)
                pending.append((cone, cx, cy, yaw_deg))

        logger.debug("Spawned %d perimeter markers (settling).", len(pending))
        return pending

    def _spawn_obstacle_cones(
        self, world: Any, current_layout: Dict[str, Any], floor_contact_z: float
    ) -> List[Tuple[Any, float, float, float]]:
        """
        @brief Spawn static markers around interior obstacle rectangles with physics ON.

        Returns pending (actor, x, y, yaw) tuples for the shared settle loop in
        spawn_all().

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'obstacles' key.
        @param floor_contact_z: Ego CoM z - used as the spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned cone.
        """
        obstacles = current_layout.get("obstacles", [])
        if not obstacles:
            return []

        z = floor_contact_z + self._CONE_Z_OFFSET
        pending: List[Tuple[Any, float, float, float]] = []

        for obs in obstacles:
            cx = float(obs["centre_x"])
            cy = float(obs["centre_y"])
            hw = float(obs["half_width"])
            hh = float(obs["half_height"])

            # Top/bottom edges: step along X; left/right edges: step along Y.
            # Corners are covered by the horizontal pass so the vertical pass
            # starts one spacing in from each corner to avoid overlaps.
            cone_positions: List[Tuple[float, float, float]] = []
            for ey in (cy - hh, cy + hh):
                t = -hw
                while t <= hw + 1e-6:
                    cone_positions.append((cx + t, ey, 0.0))
                    t += self._cone_spacing
            for ex in (cx - hw, cx + hw):
                t = -hh + self._cone_spacing
                while t < hh - 1e-6:
                    cone_positions.append((ex, cy + t, 90.0))
                    t += self._cone_spacing

            for px, py, yaw_deg in cone_positions:
                cone = world.try_spawn_actor(
                    self._cone_bp,
                    carla.Transform(
                        carla.Location(x=px, y=py, z=z),
                        carla.Rotation(yaw=yaw_deg),
                    ),
                )
                if cone is not None:
                    cone.set_simulate_physics(True)
                    pending.append((cone, px, py, yaw_deg))

        logger.debug(
            "Spawned obstacle cones for %d interior obstacle(s) (settling).",
            len(obstacles),
        )
        return pending

    def _spawn_static_vehicles(
        self,
        world: Any,
        current_layout: Dict[str, Any],
        target_bay: Dict[str, Any],
        floor_contact_z: float,
    ) -> List[Tuple[Any, float, float, float]]:
        """
        @brief Spawn static parked vehicles with physics ON and return pending list.

        The settle loop and freeze step are handled by spawn_all() so all static
        actors (cones + vehicles) settle in one shared tick loop.

        Target bay and its immediate neighbours are always left empty to give
        the agent clearance to manoeuvre in.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'bays' key.
        @param target_bay: Dict with 'bay_id' key for the selected target bay.
        @param floor_contact_z: Ego CoM z - used as spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned vehicle.
        """
        if not self._car_blueprints:
            logger.warning(
                "No car blueprints cached - call refresh_blueprints() first."
            )
            return []

        bays = current_layout.get("bays", [])
        target_id = target_bay.get("bay_id", "")
        excluded_ids = {target_id} | set(self._adjacent_bay_ids(target_id))

        z_spawn = floor_contact_z + self._VEHICLE_Z_OFFSET
        pending: List[Tuple[Any, float, float, float]] = []

        for bay in bays:
            # Layout YAMLs may use 'bay_id' or the shorter 'id' key.
            if bay.get("bay_id", bay.get("id", "")) in excluded_ids:
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
            # YAML stores heading as 'yaw_deg' (degrees) or 'yaw' (radians).
            yaw = _bay_yaw_deg(bay)
            if random.random() < 0.5:
                yaw = (yaw + 180.0) % 360.0

            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(True)
                pending.append((actor, bay_x, bay_y, yaw))

        # Motorcycle bays have a fixed occupant (Kawasaki Ninja or Yamaha YZF)
        # identified by the 'occupant' field in the layout YAML.
        _occupant_bp: Dict[str, Optional[Any]] = {
            self._MOTORCYCLE_OCCUPANTS[0]: self._ninja_bp,
            self._MOTORCYCLE_OCCUPANTS[1]: self._yzf_bp,
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
            yaw = _bay_yaw_deg(bay)
            transform = carla.Transform(
                carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                carla.Rotation(yaw=yaw),
            )
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(True)
                pending.append((actor, bay_x, bay_y, yaw))

        logger.debug("Spawned %d static vehicles (settling).", len(pending))
        return pending

    @staticmethod
    def _adjacent_bay_ids(target_id: str) -> List[str]:
        """
        @brief Return IDs of the bays immediately left/right of target_id.

        Bay IDs follow '<type>_<index>' (e.g. 'parallel_3'). Adjacent bays
        are kept empty so the agent has clearance to manoeuvre into the target.

        @param target_id: Bay ID of the selected target.
        @return List of adjacent IDs (empty list for end bays or bad IDs).
        """
        try:
            bay_type, idx_str = target_id.rsplit("_", 1)
            idx = int(idx_str)
        except ValueError:
            return []
        return [f"{bay_type}_{idx - 1}", f"{bay_type}_{idx + 1}"]
