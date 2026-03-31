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

        # Cones (perimeter + obstacle) are layout-specific and fixed for the
        # lifetime of a floor plan.  Caching them avoids destroying and
        # re-spawning ~60-80 actors every episode reset, which would otherwise
        # cause a large burst of bridge callbacks that temporarily starves
        # Cartographer of LiDAR data.
        self._cached_cones_layout: str = ""
        self._cached_cone_positions: List[Tuple[float, float]] = []

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
        floor_contact_z: float = 0.3,
        layout_name: str = "",
    ) -> None:
        """
        @brief Spawn all static actors for one episode.

        Perimeter and obstacle cones are cached across episodes for the same
        floor plan -- they are layout-fixed and do not need to be destroyed and
        re-spawned every reset.  Only parked vehicles are re-randomised each
        episode.  This avoids the large bridge callback burst that would
        otherwise temporarily starve Cartographer of LiDAR data.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict.
        @param target_bay: Dict with at least 'bay_id' key for the target bay.
        @param floor_contact_z: Ego vehicle CoM z after settling under gravity.
               Prop origins (cones) are placed at this z; vehicles settle by
               physics so this is only used as the drop height reference.
        @param layout_name: Floor plan name for cone cache invalidation.
               Pass the same value as self._current_floor_plan_name in the env.
        """
        if world is None:
            return

        self._bay_occupancy_rate = random.uniform(
            self._bay_occupancy_min, self._bay_occupancy_max
        )

        # --- Cones (layout-fixed, cached across episodes) ---
        cones_already_spawned = (
            layout_name
            and layout_name == self._cached_cones_layout
            and all(c.is_alive for c in self.spawned_cones)
        )

        if cones_already_spawned:
            # Reuse existing cone actors; rebuild static_obstacle_positions
            # from the cached positions so vehicle positions can be appended.
            self.static_obstacle_positions = list(self._cached_cone_positions)
            logger.debug("Reusing %d cached cone actors for layout '%s'.",
                         len(self.spawned_cones), layout_name)
        else:
            # Destroy any stale cones from a previous layout then re-spawn.
            for cone in self.spawned_cones:
                if cone is not None and cone.is_alive:
                    cone.destroy()
            self.spawned_cones.clear()

            cone_pending: List[Tuple[Any, float, float, float]] = []
            if self._spawn_perimeter_cones_flag:
                cone_pending.extend(
                    self._spawn_perimeter_cones(world, current_layout, floor_contact_z)
                )
            cone_pending.extend(
                self._spawn_obstacle_cones(world, current_layout, floor_contact_z)
            )

            # Settle cones first (they need fewer ticks than vehicles).
            for _ in range(60):
                world.tick()
                if all(
                    abs(a.get_velocity().z) < 0.01
                    for a, _, _, _ in cone_pending
                    if a.is_alive
                ):
                    break

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
                self.static_obstacle_positions.append((ax, ay))

            self._cached_cones_layout = layout_name
            self._cached_cone_positions = list(self.static_obstacle_positions)
            logger.debug("Spawned and cached %d cone actors for layout '%s'.",
                         len(self.spawned_cones), layout_name)

        # --- Parked vehicles (re-randomised every episode) ---
        vehicle_pending = self._spawn_static_vehicles(
            world, current_layout, target_bay, floor_contact_z
        )

        # Tick until all vehicles have settled (up to 60 ticks at 20 Hz = 3 s).
        for _ in range(60):
            world.tick()
            if all(
                abs(a.get_velocity().z) < 0.01
                for a, _, _, _ in vehicle_pending
                if a.is_alive
            ):
                break

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
            self.static_obstacle_positions.append((ax, ay))

    def cleanup(self) -> None:
        """
        @brief Destroy per-episode actors (parked vehicles only) and reset caches.

        Cones are kept alive and reused by the next episode if the floor plan
        has not changed.  Call at the start of each episode reset.
        """
        for actor in self.spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.spawned_static_vehicles.clear()
        self.static_obstacle_positions.clear()

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

        self.static_obstacle_positions.clear()
        self._cached_cones_layout = ""
        self._cached_cone_positions.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_perimeter_cones(
        self, world: Any, current_layout: Dict[str, Any], floor_contact_z: float
    ) -> List[Tuple[Any, float, float, float]]:
        """
        @brief Spawn static markers along the lot perimeter polygon with physics ON.

        Returns pending (actor, x, y, yaw) tuples for the shared settle loop in
        spawn_all().  Physics is left ON so gravity drops each cone to the true
        ground surface; the caller freezes them after settling.

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'corners' key.
        @param floor_contact_z: Ego CoM z -- used as the spawn height reference.
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

        cone_bp = self._cone_bp
        # Spawn slightly above the ego CoM z so the cone clears the surface
        # before physics drops it flush to the ground.
        z = floor_contact_z + 0.5
        pending: List[Tuple[Any, float, float, float]] = []

        for cx, cy, yaw_deg in cone_positions:
            cone = world.try_spawn_actor(
                cone_bp,
                carla.Transform(carla.Location(x=cx, y=cy, z=z), carla.Rotation(yaw=yaw_deg)),
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
        @param floor_contact_z: Ego CoM z -- used as the spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned cone.
        """
        obstacles = current_layout.get("obstacles", [])
        if not obstacles:
            return []

        cone_bp = self._cone_bp
        z = floor_contact_z + 0.5
        pending: List[Tuple[Any, float, float, float]] = []

        for obs in obstacles:
            cx = float(obs["centre_x"])
            cy = float(obs["centre_y"])
            hw = float(obs["half_width"])
            hh = float(obs["half_height"])

            # Top/bottom edges: step along X; left/right edges: step along Y
            # (corners are covered by the horizontal pass, so vertical pass
            # starts one spacing in from each corner to avoid overlaps).
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
                    cone_bp,
                    carla.Transform(carla.Location(x=px, y=py, z=z), carla.Rotation(yaw=yaw_deg)),
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

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'bays' key.
        @param target_bay: Dict with 'bay_id' key for the selected target bay.
        @param floor_contact_z: Ego CoM z -- used as spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned vehicle.
        """
        if not self._car_blueprints:
            logger.warning("No car blueprints cached -- call refresh_blueprints() first.")
            return []

        bays = current_layout.get("bays", [])
        target_id = target_bay.get("bay_id", "")
        excluded_ids = {target_id} | set(self._adjacent_bay_ids(target_id))

        z_spawn = floor_contact_z + 1.0
        pending: List[Tuple[Any, float, float, float]] = []

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
                actor.set_simulate_physics(True)
                pending.append((actor, bay_x, bay_y, yaw))

        logger.debug("Spawned %d static vehicles (settling).", len(pending))
        return pending

    @staticmethod
    def _adjacent_bay_ids(target_id: str) -> List[str]:
        """
        @brief Return IDs of the bays immediately left/right of target_id.

        Bay IDs follow '<type>_<index>' (e.g. 'parallel_3').  Adjacent bays
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
