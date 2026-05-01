"""
@file _npc_controller.py
@brief NPC patrol vehicle and pedestrian controller for CARLAParkingEnv.

Owns all per-episode NPC state (patrol vehicles, pedestrians) and the spawn,
update, and cleanup logic for both. CARLAParkingEnv holds an NPCController
instance and delegates NPC lifecycle calls to it.
"""

import logging
import math
import random
from typing import Any, Dict, List, Set, Tuple

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.utils.geometry import zone_bbox

logger = logging.getLogger(__name__)


class NPCController:
    """
    @class NPCController
    @brief Manages patrol NPC vehicles and pedestrians for one parking episode.

    Instantiated once by CARLAParkingEnv and reused across episodes. Call
    cleanup() at the start of each episode reset, then spawn_patrol() and
    spawn_pedestrians() to populate the episode, then update_patrol() and
    update_pedestrians() once per world tick.
    """

    # Number of sectors to divide each zone into for respawn placement.
    _RESPAWN_N_SECTORS: int = 5
    # Distance threshold: ego_dist >= threshold -> all sectors available.
    # Each _threshold / n metres closer excludes one more sector
    # (nearest excluded first).
    _RESPAWN_SECTOR_THRESHOLD: float = 8.0
    # Pedestrian avoidance radii (metres) used in update_pedestrians()
    _EGO_AVOID_RADIUS: float = 5.0
    _PATROL_AVOID_RADIUS: float = 5.0
    # Zone boundary margin (metres): steer toward zone centre inside this band
    _BOUNDARY_MARGIN: float = 1.0
    # Zone clustering radius (metres): zones within this distance share one pedestrian
    _CLUSTER_RADIUS: float = 12.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_patrol_max: int,
        patrol_obstacle_distance: float,
        patrol_pedestrian_distance: float,
        patrol_max_speed: float,
        patrol_heading_gain: float,
        pedestrian_spawn_prob: float,
        pedestrian_speed: float,
        pedestrian_resample_steps: int,
        pedestrian_max_lifetime: int,
    ) -> None:
        """
        @brief Construct NPCController with fixed config parameters.

        @param num_patrol_max: Maximum number of patrol vehicles per episode.
        @param patrol_obstacle_distance: Stop distance (metres) for vehicle
               obstacle avoidance (centre-to-centre).
        @param patrol_pedestrian_distance: Stop distance (metres) for
               pedestrian avoidance (centre-to-centre).
        @param patrol_max_speed: Target cruise speed for patrol vehicles (m/s).
        @param patrol_heading_gain: Proportional gain k_p for heading controller.
        @param pedestrian_spawn_prob: Per-zone spawn probability [0, 1].
        @param pedestrian_speed: Walk speed (m/s).
        @param pedestrian_resample_steps: Steps between random heading changes.
        @param pedestrian_max_lifetime: Steps before a pedestrian is respawned.
        """
        self._num_patrol_max = num_patrol_max
        self._patrol_obstacle_distance = patrol_obstacle_distance
        self._patrol_pedestrian_distance = patrol_pedestrian_distance
        self._patrol_max_speed = patrol_max_speed
        self._patrol_heading_gain = patrol_heading_gain
        self._pedestrian_spawn_prob = pedestrian_spawn_prob
        self._pedestrian_speed = pedestrian_speed
        self._pedestrian_resample_steps = pedestrian_resample_steps
        self._pedestrian_max_lifetime = pedestrian_max_lifetime

        # Blueprint lists - refreshed each reset via refresh_blueprints()
        self._car_blueprints: List[Any] = []
        self._walker_blueprints: List[Any] = []

        # Cached all-vehicle list for patrol obstacle checks - rebuilt after
        # all vehicles are spawned via set_vehicle_cache()
        self._all_vehicle_actors: List[Any] = []

        # Per-episode patrol state
        self.patrol_npcs: List[Any] = []
        self.patrol_npc_ids: Set[int] = set()
        self._patrol_waypoint_indices: List[int] = []
        self._patrol_waypoint_directions: List[int] = []
        self._patrol_waypoints_cache: List[Tuple[float, float]] = []
        # Tracks which patrol NPCs are currently pinned via enable_constant_velocity
        # so disable_constant_velocity() is called before re-applying throttle.
        self._patrol_pinned: List[bool] = []

        # Per-episode pedestrian state
        self.pedestrian_actors: List[Any] = []
        self._pedestrian_headings: List[Tuple[float, float, float]] = []
        self._pedestrian_heading_steps: List[int] = []
        self._pedestrian_lifetime_steps: List[int] = []
        self._pedestrian_zones: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Per-reset setup
    # ------------------------------------------------------------------

    def refresh_blueprints(
        self,
        car_blueprints: List[Any],
        walker_blueprints: List[Any],
    ) -> None:
        """
        @brief Update cached blueprint lists before spawning.

        Call after CARLAParkingEnv._cache_blueprints() and before spawn_patrol()
        / spawn_pedestrians() so the controller uses the freshly filtered lists.

        @param car_blueprints: Filtered list of four-wheeled vehicle blueprints.
        @param walker_blueprints: Filtered list of pedestrian blueprints.
        """
        self._car_blueprints = car_blueprints
        self._walker_blueprints = walker_blueprints

    def set_vehicle_cache(self, all_vehicle_actors: List[Any]) -> None:
        """
        @brief Store the full vehicle actor list for patrol obstacle checks.

        Call after all vehicles (static + patrol) are spawned so update_patrol()
        can check inter-vehicle distances without a per-step world query.

        @param all_vehicle_actors: All live vehicle actors in the world.
        """
        self._all_vehicle_actors = all_vehicle_actors

    # ------------------------------------------------------------------
    # Spawning
    # ------------------------------------------------------------------

    def spawn_patrol(
        self,
        world: Any,
        vehicle: Any,
        current_layout: Dict[str, Any],
    ) -> None:
        """
        @brief Spawn scripted patrol vehicles that follow waypoints in the lot.

        Uses a proportional heading controller rather than the Traffic Manager,
        since the lot is off-road and TM requires CARLA road network.
        See documentation/detailed_notes/layout/patrol_paths.md for controller
        design rationale and waypoint derivation.

        @param world: Live CARLA world handle.
        @param vehicle: Ego vehicle actor (used for ego-safe spawn distance check).
        @param current_layout: Parsed floor plan YAML dict with patrol_waypoints.
        """
        if world is None:
            return

        waypoints_raw = current_layout.get("patrol_waypoints", [])
        if not waypoints_raw:
            return

        if self._num_patrol_max == 0:
            return
        num_patrol = self._num_patrol_max

        waypoints: List[Tuple[float, float]] = [
            (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
        ]
        # Drop a closing duplicate
        if len(waypoints) > 1 and waypoints[-1] == waypoints[0]:
            waypoints = waypoints[:-1]

        # Cache for update_patrol() so it does not re-parse the layout dict every step.
        self._patrol_waypoints_cache = waypoints
        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.1

        # Build a list of candidate start indices safely away from the ego spawn.
        ego_loc = vehicle.get_location() if vehicle is not None else None
        min_spawn_dist = self._patrol_obstacle_distance + 4.5
        safe_indices = list(range(len(waypoints)))
        if ego_loc is not None:
            safe_indices = [
                idx
                for idx in safe_indices
                if math.sqrt(
                    (waypoints[idx][0] - ego_loc.x) ** 2
                    + (waypoints[idx][1] - ego_loc.y) ** 2
                )
                >= min_spawn_dist
            ]
        if not safe_indices:
            safe_indices = list(range(len(waypoints)))

        for i in range(num_patrol):
            bp = random.choice(self._car_blueprints)
            start_idx = random.choice(safe_indices)
            direction = random.choice([-1, 1])
            first_target_idx = (start_idx + direction) % len(waypoints)
            wp = waypoints[start_idx]
            first_target_wp = waypoints[first_target_idx]
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
            actor = world.try_spawn_actor(bp, transform)
            if actor is not None:
                actor.set_simulate_physics(True)
                self.patrol_npcs.append(actor)
                self.patrol_npc_ids.add(actor.id)
                self._patrol_waypoint_indices.append(first_target_idx)
                self._patrol_waypoint_directions.append(direction)
                self._patrol_pinned.append(False)

        logger.debug("Spawned %d patrol NPC vehicles.", len(self.patrol_npcs))

    def spawn_pedestrians(
        self,
        world: Any,
        current_layout: Dict[str, Any],
    ) -> None:
        """
        @brief Spawn random-walk pedestrians inside the lot pedestrian zones.

        Uses WalkerControl with a random direction, re-randomised every
        pedestrian_heading_resample_steps steps.

        @param world: Live CARLA world handle.
        @param current_layout: Parsed floor plan YAML dict with pedestrian_zones.
        """
        if world is None:
            return

        zones_raw = current_layout.get("pedestrian_zones", [])
        if not zones_raw:
            return

        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.05

        zones: List[Dict[str, float]] = [
            dict(zip(("x_min", "x_max", "y_min", "y_max"), zone_bbox(zr)))
            for zr in zones_raw
        ]

        # Group zones into spatial clusters so that adjacent corridor segments
        # do not each spawn their own pedestrian. Two zones belong to the same
        # cluster when their centres are within CLUSTER_RADIUS metres of each
        # other (union-find via greedy single-linkage).
        cluster_id: List[int] = list(range(len(zones)))
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                cx_i = (zones[i]["x_min"] + zones[i]["x_max"]) / 2.0
                cy_i = (zones[i]["y_min"] + zones[i]["y_max"]) / 2.0
                cx_j = (zones[j]["x_min"] + zones[j]["x_max"]) / 2.0
                cy_j = (zones[j]["y_min"] + zones[j]["y_max"]) / 2.0
                dist = math.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                if dist <= self._CLUSTER_RADIUS:
                    # Merge j's cluster into i's cluster
                    old_id = cluster_id[j]
                    new_id = cluster_id[i]
                    for k in range(len(zones)):
                        if cluster_id[k] == old_id:
                            cluster_id[k] = new_id

        # Collect zones per cluster then roll spawn prob once per cluster.
        clusters: Dict[int, List[int]] = {}
        for idx, cid in enumerate(cluster_id):
            clusters.setdefault(cid, []).append(idx)

        for zone_indices in clusters.values():
            # One prob roll decides whether this cluster gets a pedestrian.
            if random.random() > self._pedestrian_spawn_prob:
                continue

            # Pick one zone from the cluster at random to spawn inside.
            zone = zones[random.choice(zone_indices)]

            bp = random.choice(self._walker_blueprints)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            walker = None
            for _ in range(5):
                px = random.uniform(zone["x_min"], zone["x_max"])
                py = random.uniform(zone["y_min"], zone["y_max"])
                transform = carla.Transform(
                    carla.Location(x=px, y=py, z=z),
                    carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                )
                walker = world.try_spawn_actor(bp, transform)
                if walker is not None:
                    break

            if walker is not None:
                heading_rad = random.uniform(0.0, 2.0 * math.pi)
                self.pedestrian_actors.append(walker)
                self._pedestrian_headings.append(
                    (math.cos(heading_rad), math.sin(heading_rad), 0.0)
                )
                self._pedestrian_heading_steps.append(0)
                self._pedestrian_lifetime_steps.append(0)
                self._pedestrian_zones.append(zone)

        logger.debug("Spawned %d pedestrians.", len(self.pedestrian_actors))

    # ------------------------------------------------------------------
    # Per-step updates
    # ------------------------------------------------------------------

    def update_patrol(
        self,
        vehicle: Any,
        steps: int,
    ) -> None:
        """
        @brief Advance patrol NPC vehicles one step via proportional heading
               controller.

        Wraps to next waypoint when within 3 m. Applies omnidirectional ego
        avoidance and forward-cone obstacle/pedestrian detection.
        Waypoints are read from self._patrol_waypoints_cache (populated by
        spawn_patrol()).

        @param vehicle: Ego vehicle actor.
        @param steps: Current episode step count (used for debug throttle).
        """
        waypoints = self._patrol_waypoints_cache
        if not waypoints:
            return
        k_p = self._patrol_heading_gain

        all_vehicles: List[Any] = self._all_vehicle_actors

        for i, npc in enumerate(self.patrol_npcs):
            if npc is None or not npc.is_alive:
                continue

            wp_idx = self._patrol_waypoint_indices[i]
            wp_x, wp_y = waypoints[wp_idx]

            t = npc.get_transform()
            dx = wp_x - t.location.x
            dy = wp_y - t.location.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 3.0:
                direction = self._patrol_waypoint_directions[i]
                wp_idx = (wp_idx + direction) % len(waypoints)
                self._patrol_waypoint_indices[i] = wp_idx
                wp_x, wp_y = waypoints[wp_idx]
                dx = wp_x - t.location.x
                dy = wp_y - t.location.y

            npc_yaw = math.radians(t.rotation.yaw)
            target_yaw = math.atan2(dy, dx)
            heading_error = math.atan2(
                math.sin(target_yaw - npc_yaw),
                math.cos(target_yaw - npc_yaw),
            )

            steer = max(-1.0, min(1.0, k_p * heading_error))

            fwd_x = math.cos(npc_yaw)
            fwd_y = math.sin(npc_yaw)
            blocked = False

            # Ego vehicle: angular cone stop.
            # Uses cos(heading_error) = fwd_proj / dist.
            if vehicle is not None and vehicle.is_alive:
                ego_to_x = vehicle.get_location().x - t.location.x
                ego_to_y = vehicle.get_location().y - t.location.y
                ego_dist = math.sqrt(ego_to_x * ego_to_x + ego_to_y * ego_to_y)
                if ego_dist < self._patrol_obstacle_distance:
                    ego_fwd_proj = ego_to_x * fwd_x + ego_to_y * fwd_y
                    # cos(60 deg) = 0.5 - stop for any ego within 60 deg of forward
                    if ego_fwd_proj / ego_dist > 0.5:
                        blocked = True

            for other in all_vehicles:
                if blocked:
                    break
                if other.id == npc.id:
                    continue
                if vehicle is not None and other.id == vehicle.id:
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
                for walker in self.pedestrian_actors:
                    if walker is None or not walker.is_alive:
                        continue
                    to_x = walker.get_location().x - t.location.x
                    to_y = walker.get_location().y - t.location.y
                    fwd_proj = to_x * fwd_x + to_y * fwd_y
                    lat = abs(to_x * fwd_y - to_y * fwd_x)
                    walker_dist = math.sqrt(to_x * to_x + to_y * to_y)
                    if (
                        fwd_proj > 0.0
                        and walker_dist < self._patrol_pedestrian_distance
                        and lat < 2.5
                    ):
                        blocked = True
                        break

            if blocked:
                # enable_constant_velocity bypasses physics entirely and pins
                # velocity to zero immediately.
                if not self._patrol_pinned[i]:
                    npc.enable_constant_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
                    self._patrol_pinned[i] = True
                if steps % 20 == 0:
                    logger.debug(
                        "[patrol] npc %d pinned (blocked)  wp=%d  step=%d",
                        npc.id,
                        self._patrol_waypoint_indices[i],
                        steps,
                    )
            else:
                if self._patrol_pinned[i]:
                    npc.disable_constant_velocity()
                    self._patrol_pinned[i] = False

                vel = npc.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2)
                speed_ratio = speed / max(self._patrol_max_speed, 0.1)
                throttle = max(0.1, min(1.0, 1.0 - speed_ratio))

                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = 0.0
                control.steer = steer
                npc.apply_control(control)

    def update_pedestrians(self, vehicle: Any) -> None:
        """
        @brief Advance pedestrians one step with avoidance, zone confinement,
               and periodic heading re-randomisation.

        @param vehicle: Ego vehicle actor (used for avoidance distance check).
        """
        ego_loc = (
            vehicle.get_location() if vehicle is not None and vehicle.is_alive else None
        )

        for i, walker in enumerate(self.pedestrian_actors):
            if walker is None or not walker.is_alive:
                continue

            self._pedestrian_lifetime_steps[i] += 1
            self._pedestrian_heading_steps[i] += 1

            if self._pedestrian_lifetime_steps[i] >= self._pedestrian_max_lifetime:
                self._respawn_pedestrian(i, vehicle)
                continue

            loc = walker.get_location()
            zone = self._pedestrian_zones[i]

            # Hard boundary enforcement: if the pedestrian has escaped the zone,
            # teleport it back to the nearest in-bounds point and point it toward
            # the zone centre so it does not immediately escape again.
            outside = (
                loc.x < zone["x_min"]
                or loc.x > zone["x_max"]
                or loc.y < zone["y_min"]
                or loc.y > zone["y_max"]
            )
            if outside:
                clamped_x = max(zone["x_min"], min(zone["x_max"], loc.x))
                clamped_y = max(zone["y_min"], min(zone["y_max"], loc.y))
                clamped_loc = carla.Location(
                    x=clamped_x, y=clamped_y, z=loc.z
                )
                walker.set_location(clamped_loc)
                loc = clamped_loc
                cx = (zone["x_min"] + zone["x_max"]) / 2.0
                cy = (zone["y_min"] + zone["y_max"]) / 2.0
                to_cx = cx - clamped_x
                to_cy = cy - clamped_y
                c_mag = math.sqrt(to_cx * to_cx + to_cy * to_cy)
                if c_mag > 1e-6:
                    self._pedestrian_headings[i] = (to_cx / c_mag, to_cy / c_mag, 0.0)
                self._pedestrian_heading_steps[i] = 0

            repulse_x = 0.0
            repulse_y = 0.0

            if ego_loc is not None:
                to_ego_x = ego_loc.x - loc.x
                to_ego_y = ego_loc.y - loc.y
                ego_dist = math.sqrt(to_ego_x * to_ego_x + to_ego_y * to_ego_y)
                if ego_dist < self._EGO_AVOID_RADIUS:
                    # Inverse-distance weighting: stronger repulsion when closer.
                    weight = 1.0 / max(ego_dist, 0.5)
                    repulse_x += -to_ego_x * weight
                    repulse_y += -to_ego_y * weight

            for patrol_npc in self.patrol_npcs:
                if patrol_npc is None or not patrol_npc.is_alive:
                    continue
                to_px = patrol_npc.get_location().x - loc.x
                to_py = patrol_npc.get_location().y - loc.y
                patrol_dist = math.sqrt(to_px * to_px + to_py * to_py)
                if patrol_dist < self._PATROL_AVOID_RADIUS:
                    weight = 1.0 / max(patrol_dist, 0.5)
                    repulse_x += -to_px * weight
                    repulse_y += -to_py * weight

            near_boundary = (
                loc.x < zone["x_min"] + self._BOUNDARY_MARGIN
                or loc.x > zone["x_max"] - self._BOUNDARY_MARGIN
                or loc.y < zone["y_min"] + self._BOUNDARY_MARGIN
                or loc.y > zone["y_max"] - self._BOUNDARY_MARGIN
            )

            # Priority: avoidance > boundary > random heading.
            # Avoidance runs first so the boundary logic can never override it
            # and push a pedestrian back toward the ego or a patrol vehicle.
            if repulse_x != 0.0 or repulse_y != 0.0:
                mag = math.sqrt(repulse_x * repulse_x + repulse_y * repulse_y)
                # If also near the boundary, blend in the centre-pull at 30 % so
                # the pedestrian both avoids the threat and does not escape the zone.
                if near_boundary:
                    cx = (zone["x_min"] + zone["x_max"]) / 2.0
                    cy = (zone["y_min"] + zone["y_max"]) / 2.0
                    to_cx = cx - loc.x
                    to_cy = cy - loc.y
                    c_mag = math.sqrt(to_cx * to_cx + to_cy * to_cy)
                    if c_mag > 1e-6:
                        blend = 0.3
                        nx = (1.0 - blend) * repulse_x / mag + blend * to_cx / c_mag
                        ny = (1.0 - blend) * repulse_y / mag + blend * to_cy / c_mag
                        n_mag = math.sqrt(nx * nx + ny * ny)
                        if n_mag > 1e-6:
                            nx, ny = nx / n_mag, ny / n_mag
                        self._pedestrian_headings[i] = (nx, ny, 0.0)
                    else:
                        self._pedestrian_headings[i] = (
                            repulse_x / mag,
                            repulse_y / mag,
                            0.0,
                        )
                else:
                    self._pedestrian_headings[i] = (
                        repulse_x / mag,
                        repulse_y / mag,
                        0.0,
                    )
                self._pedestrian_heading_steps[i] = 0
            elif near_boundary:
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
            control.speed = self._pedestrian_speed
            walker.apply_control(control)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        @brief Destroy all patrol and pedestrian actors and clear per-episode state.

        Call at the start of each episode reset, before spawning new actors.
        """
        for actor_list in [self.patrol_npcs, self.pedestrian_actors]:
            for actor in actor_list:
                if actor is not None and actor.is_alive:
                    actor.destroy()
            actor_list.clear()

        self.patrol_npc_ids.clear()
        self._patrol_waypoint_indices.clear()
        self._patrol_waypoint_directions.clear()
        self._patrol_waypoints_cache = []
        self._patrol_pinned.clear()
        self._pedestrian_headings.clear()
        self._pedestrian_heading_steps.clear()
        self._pedestrian_lifetime_steps.clear()
        self._pedestrian_zones.clear()
        self._all_vehicle_actors.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _respawn_pedestrian(self, idx: int, vehicle: Any) -> None:
        """
        @brief Destroy and respawn pedestrian at index idx within its zone.

        Called when a pedestrian exceeds its maximum lifetime. The walker is
        destroyed, a new one is spawned within an allowed set of sectors of the
        same zone, and the lifetime counter is reset.

        @param idx: Index into pedestrian_actors / _pedestrian_zones.
        @param vehicle: Ego vehicle actor (used for sector selection).
        """
        # _respawn_pedestrian needs world access - infer from current actors.
        # The zone dict and walker blueprints are all we need.
        zone = self._pedestrian_zones[idx]
        old = self.pedestrian_actors[idx]
        if old is not None and old.is_alive:
            world = old.get_world()
            old.destroy()
        else:
            # Cannot respawn without a world reference - mark as gone
            self.pedestrian_actors[idx] = None
            logger.debug(
                "[pedestrian] respawn idx=%d SKIPPED (no world reference)", idx
            )
            return

        z_origin = 0.3  # default floor z; exact value not critical for respawn
        z = z_origin + 0.05

        bp = random.choice(self._walker_blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        # Build sectors along the longest axis, ranked farthest-from-ego first.
        n = self._RESPAWN_N_SECTORS
        x_span = zone["x_max"] - zone["x_min"]
        y_span = zone["y_max"] - zone["y_min"]
        # Slice along the longer axis for maximum separation between sectors.
        slice_x = x_span >= y_span

        ego_loc = (
            vehicle.get_location() if vehicle is not None and vehicle.is_alive else None
        )

        # Each sector is a sub-rectangle of the zone.
        sectors: List[Tuple[float, float, float, float]] = []
        for s in range(n):
            if slice_x:
                s_lo = zone["x_min"] + s * x_span / n
                s_hi = zone["x_min"] + (s + 1) * x_span / n
                sectors.append((s_lo, s_hi, zone["y_min"], zone["y_max"]))
            else:
                s_lo = zone["y_min"] + s * y_span / n
                s_hi = zone["y_min"] + (s + 1) * y_span / n
                sectors.append((zone["x_min"], zone["x_max"], s_lo, s_hi))

        if ego_loc is not None:
            # Score each sector by distance from its centre to the ego.
            def _sector_dist(sec: Tuple[float, float, float, float]) -> float:
                cx = (sec[0] + sec[1]) / 2.0
                cy = (sec[2] + sec[3]) / 2.0
                return math.sqrt((cx - ego_loc.x) ** 2 + (cy - ego_loc.y) ** 2)

            sectors.sort(key=_sector_dist, reverse=True)  # farthest first

            # Exclude the closest sectors based on how close the ego is to the
            # nearest sector.
            nearest_dist = _sector_dist(sectors[-1])  # sectors sorted far->near
            excluded = int(
                (n - 1) * max(0.0, 1.0 - nearest_dist / self._RESPAWN_SECTOR_THRESHOLD)
            )
            allowed = max(1, n - excluded)
            sectors = sectors[:allowed]

        # Sample uniformly from the allowed sectors, then pick a random point
        # within the chosen sector.
        chosen = random.choice(sectors)

        walker = None
        for _ in range(5):
            px = random.uniform(chosen[0], chosen[1])
            py = random.uniform(chosen[2], chosen[3])
            transform = carla.Transform(
                carla.Location(x=px, y=py, z=z),
                carla.Rotation(yaw=random.uniform(0.0, 360.0)),
            )
            walker = world.try_spawn_actor(bp, transform)
            if walker is not None:
                break

        self.pedestrian_actors[idx] = walker
        if walker is None:
            logger.debug("[pedestrian] respawn idx=%d FAILED after 5 attempts", idx)
        else:
            logger.debug(
                "[pedestrian] respawn idx=%d ok  sectors_allowed=%d/%d",
                idx,
                len(sectors),
                n,
            )
        heading_rad = random.uniform(0.0, 2.0 * math.pi)
        self._pedestrian_headings[idx] = (
            math.cos(heading_rad),
            math.sin(heading_rad),
            0.0,
        )
        self._pedestrian_heading_steps[idx] = 0
        self._pedestrian_lifetime_steps[idx] = 0
