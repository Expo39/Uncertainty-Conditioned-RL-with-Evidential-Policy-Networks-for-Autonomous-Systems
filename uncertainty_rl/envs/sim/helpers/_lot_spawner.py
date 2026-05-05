"""
@file _lot_spawner.py
@brief Static lot actor spawner for CARLAParkingEnv.

Owns all per-episode static actor state: perimeter cones, interior obstacle
cones, and parked vehicles. CARLAParkingEnv holds a LotSpawner instance and
delegates static spawning and cleanup to it.
"""

import itertools
import logging
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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

# Single compiled regex - one pass and one .search() call per blueprint
_EXCLUDED_BP_RE: re.Pattern = re.compile(
    "|".join(
        re.escape(s)
        for s in _LARGE_VEHICLE_TYPES + _SMALL_VEHICLE_TYPES + _EXCLUDED_VEHICLE_TYPES
    ),
    re.IGNORECASE,
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
        self._cached_cones_layout: str = ""

        # Cached blueprint lists - populated once on first connect, never re-fetched.
        self._car_blueprints: List[Any] = []
        self._cone_bp: Optional[Any] = None
        self._ninja_bp: Optional[Any] = None
        self._yzf_bp: Optional[Any] = None
        # Prebuilt occupant map - populated in refresh_blueprints().
        self._occupant_bp: Dict[str, Optional[Any]] = {}

    # ------------------------------------------------------------------
    # Per-reset setup
    # ------------------------------------------------------------------

    def refresh_blueprints(self, world: Any) -> None:
        """
        @brief Fetch and filter blueprint lists from the CARLA world.

        Populates self._car_blueprints with four-wheeled vehicles excluding
        large, small/novelty, and special-purpose types.
        Blueprints are static for the lifetime of the CARLA server, so the
        fetch is skipped on subsequent calls if already populated.

        @param world: Live carla.World handle.
        """
        if world is None or self._car_blueprints:
            return

        bp_lib = world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        self._car_blueprints = [
            bp
            for bp in vehicle_bps
            if int(bp.get_attribute("number_of_wheels").as_int()) == 4
            and not _EXCLUDED_BP_RE.search(bp.id)
        ]

        self._ninja_bp = bp_lib.find("vehicle.kawasaki.ninja")
        self._yzf_bp = bp_lib.find("vehicle.yamaha.yzf")
        self._cone_bp = bp_lib.find(self._marker_blueprint)
        self._occupant_bp = {
            self._MOTORCYCLE_OCCUPANTS[0]: self._ninja_bp,
            self._MOTORCYCLE_OCCUPANTS[1]: self._yzf_bp,
        }

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

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict.
        @param target_bay: Dict with at least 'bay_id' key for the target bay.
        @param floor_contact_z: Ego vehicle CoM z after settling under gravity.
        @param layout_name: Floor plan name for cone cache invalidation.
        """
        if world is None:
            return

        self._bay_occupancy_rate = random.uniform(
            self._bay_occupancy_min, self._bay_occupancy_max
        )

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
            for cone in self.spawned_cones:
                if cone is not None and cone.is_alive:
                    cone.destroy()
            self.spawned_cones.clear()

            cone_pending = list(itertools.chain(
                self._spawn_perimeter_cones(world, current_layout, floor_contact_z),
                self._spawn_obstacle_cones(world, current_layout, floor_contact_z),
            ))
            self._settle_pending(world, cone_pending)
            self._freeze_pending(cone_pending, self.spawned_cones)
            self._cached_cones_layout = layout_name
            logger.debug(
                "Spawned and cached %d cone actors for layout '%s'.",
                len(self.spawned_cones),
                layout_name,
            )

        vehicle_pending = self._spawn_static_vehicles(
            world, current_layout, target_bay, floor_contact_z
        )
        self._settle_pending(world, vehicle_pending)
        self._freeze_pending(vehicle_pending, self.spawned_static_vehicles)

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
        for actor in self.spawned_cones:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.spawned_cones.clear()
        for actor in self.spawned_static_vehicles:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.spawned_static_vehicles.clear()
        self._cached_cones_layout = ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_pending(
        self,
        pending: List[Tuple[Any, float, float, float]],
        target_list: List[Any],
    ) -> None:
        """
        @brief Freeze settled actors and append them to target_list.

        Shared post-settle step for both cones and vehicles. Disables physics,
        reads the settled z, then pin-transforms the actor in place.

        @param pending: List of (actor, x, y, yaw) from a spawn helper.
        @param target_list: Destination list (spawned_cones or spawned_static_vehicles).
        """
        for actor, ax, ay, actor_yaw in pending:
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
            target_list.append(actor)

    def _settle_pending(
        self,
        world: Any,
        pending: List[Tuple[Any, float, float, float]],
    ) -> None:
        """
        @brief Tick the world until all pending actors have settled under gravity.
        
        @param world: Live carla.World handle.
        @param pending: List of (actor, x, y, yaw) tuples awaiting settlement.
        """
        actors = [a for a, _, _, _ in pending]
        for _ in range(self._SETTLE_TICKS):
            world.tick()
            for a in actors:
                if a.is_alive and abs(a.get_velocity().z) >= self._SETTLE_VZ_THRESHOLD:
                    break
            else:
                break

    def _spawn_perimeter_cones(
        self, world: Any, current_layout: Dict[str, Any], floor_contact_z: float
    ) -> List[Tuple[Any, float, float, float]]:
        """
        @brief Spawn static markers along the lot perimeter polygon with physics ON.

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

        @param world: Live carla.World handle.
        @param current_layout: Parsed floor plan YAML dict with 'obstacles' key.
        @param floor_contact_z: Ego CoM z - used as the spawn height reference.
        @return List of (actor, x, y, yaw) for each successfully spawned cone.
        """
        obstacles = current_layout.get("obstacles", [])
        if not obstacles:
            return []

        z = floor_contact_z + self._CONE_Z_OFFSET
        spacing = self._cone_spacing
        pending: List[Tuple[Any, float, float, float]] = []

        for obs in obstacles:
            cx = float(obs["centre_x"])
            cy = float(obs["centre_y"])
            hw = float(obs["half_width"])
            hh = float(obs["half_height"])

            # Top/bottom edges step along X; left/right edges step along Y.
            # Corners are covered by the horizontal pass; vertical pass starts
            # one spacing in from each corner to avoid overlaps.
            h_steps = np.arange(-hw, hw + 1e-6, spacing)
            v_steps = np.arange(-hh + spacing, hh - 1e-6, spacing)
            n_h = len(h_steps)
            n_v = len(v_steps)

            xs = np.concatenate([
                cx + h_steps, cx + h_steps,
                np.full(n_v, cx - hw), np.full(n_v, cx + hw),
            ])
            ys = np.concatenate([
                np.full(n_h, cy - hh), np.full(n_h, cy + hh),
                cy + v_steps, cy + v_steps,
            ])
            yaws = np.concatenate([
                np.zeros(2 * n_h),
                np.full(2 * n_v, 90.0),
            ])

            for px, py, yaw_deg in zip(xs, ys, yaws):
                cone = world.try_spawn_actor(
                    self._cone_bp,
                    carla.Transform(
                        carla.Location(x=float(px), y=float(py), z=z),
                        carla.Rotation(yaw=float(yaw_deg)),
                    ),
                )
                if cone is not None:
                    cone.set_simulate_physics(True)
                    pending.append((cone, float(px), float(py), float(yaw_deg)))

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
        excluded_ids = {target_id}
        z_spawn = floor_contact_z + self._VEHICLE_Z_OFFSET
        pending: List[Tuple[Any, float, float, float]] = []

        for bay in bays:
            bay_type = bay.get("bay_type", "")
            bay_x = float(bay["x"])
            bay_y = float(bay["y"])

            if bay_type == "motorcycle":
                # Fixed occupant - skip exclusion / occupancy / colour randomisation.
                bp = self._occupant_bp.get(bay.get("occupant", ""))
                if bp is None:
                    continue
                yaw = _bay_yaw_deg(bay)
            else:
                if bay.get("bay_id", bay.get("id", "")) in excluded_ids:
                    continue
                if bay.get("always_empty", False):
                    continue
                if random.random() > self._bay_occupancy_rate:
                    continue
                bp = random.choice(self._car_blueprints)
                if bp.has_attribute("color"):
                    bp.set_attribute(
                        "color",
                        random.choice(bp.get_attribute("color").recommended_values),
                    )
                yaw = _bay_yaw_deg(bay)
                if random.random() < 0.5:
                    yaw = (yaw + 180.0) % 360.0

            actor = world.try_spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=bay_x, y=bay_y, z=z_spawn),
                    carla.Rotation(yaw=yaw),
                ),
            )
            if actor is not None:
                actor.set_simulate_physics(True)
                pending.append((actor, bay_x, bay_y, yaw))

        logger.debug("Spawned %d static vehicles (settling).", len(pending))
        return pending

