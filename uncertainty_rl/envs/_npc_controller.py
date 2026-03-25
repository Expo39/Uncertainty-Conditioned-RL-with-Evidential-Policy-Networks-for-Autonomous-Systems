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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

from uncertainty_rl.utils.geometry import zone_bbox
from uncertainty_rl.utils.logging import DebugLogger

logger = logging.getLogger(__name__)


class NPCController:
    """
    @class NPCController
    @brief Manages patrol NPC vehicles and pedestrians for one parking episode.

    Instantiated once by CARLAParkingEnv and reused across episodes. Call
    cleanup() at the start of each episode reset, then spawn_patrol() and
    spawn_pedestrians() to populate the episode, then update_patrol() and
    update_pedestrians() once per world tick.

    All CARLA world/vehicle handles are passed as parameters -- this class
    never stores them across calls to avoid holding stale references between
    episodes.
    """

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
        debug_logger: DebugLogger,
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
        @param debug_logger: Shared DebugLogger for per-event diagnostics.
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
        self._debug_logger = debug_logger

        # Blueprint lists -- refreshed each reset via refresh_blueprints()
        self._car_blueprints: List[Any] = []
        self._walker_blueprints: List[Any] = []

        # Cached all-vehicle list for patrol obstacle checks -- rebuilt after
        # all vehicles are spawned via set_vehicle_cache()
        self._all_vehicle_actors: List[Any] = []

        # Per-episode patrol state
        self.patrol_npcs: List[Any] = []
        self.patrol_npc_ids: set = set()
        self._patrol_waypoint_indices: List[int] = []
        self._patrol_waypoint_directions: List[int] = []
        self._patrol_waypoints_cache: List[Tuple[float, float]] = []

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
        # Drop a closing duplicate (e.g. rectangle layout repeats waypoint 0 at the end)
        # so the cyclic modulo wrap works correctly and first_target_idx is never the
        # same location as start_idx.
        if len(waypoints) > 1 and waypoints[-1] == waypoints[0]:
            waypoints = waypoints[:-1]

        # Cache for update_patrol() so it does not re-parse the layout dict every step.
        self._patrol_waypoints_cache = waypoints
        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.1

        # Build a list of candidate start indices safely away from the ego spawn.
        # The forward-cone check in update_patrol() is direction-dependent and
        # cannot prevent a patrol that spawns on top of the ego or approaches it
        # from behind before the first tick.
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

        def _to_zone_dict(zone_raw: Dict[str, Any]) -> Dict[str, float]:
            """@brief Wrap zone_bbox tuple into the dict format used internally."""
            x_min, x_max, y_min, y_max = zone_bbox(zone_raw)
            return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

        zones: List[Dict[str, float]] = [_to_zone_dict(zr) for zr in zones_raw]

        # Group zones into spatial clusters so that adjacent corridor segments
        # do not each spawn their own pedestrian.  Two zones belong to the same
        # cluster when their centres are within _CLUSTER_RADIUS metres of each
        # other (union-find via greedy single-linkage).
        _CLUSTER_RADIUS = 12.0

        def _zone_centre(z: Dict[str, float]) -> Tuple[float, float]:
            return (
                (z["x_min"] + z["x_max"]) / 2.0,
                (z["y_min"] + z["y_max"]) / 2.0,
            )

        cluster_id: List[int] = list(range(len(zones)))
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                cx_i, cy_i = _zone_centre(zones[i])
                cx_j, cy_j = _zone_centre(zones[j])
                dist = math.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                if dist <= _CLUSTER_RADIUS:
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
        current_layout: Dict[str, Any],
    ) -> None:
        """
        @brief Advance patrol NPC vehicles one step via proportional heading
               controller.

        Wraps to next waypoint when within 3 m. Applies omnidirectional ego
        avoidance and forward-cone obstacle/pedestrian detection.

        @param vehicle: Ego vehicle actor.
        @param steps: Current episode step count (used for debug throttle).
        @param current_layout: Parsed floor plan YAML dict with patrol_waypoints.
        """
        waypoints = self._patrol_waypoints_cache
        if not waypoints:
            return
        k_p = self._patrol_heading_gain

        all_vehicles: List[Any] = self._all_vehicle_actors

        for i, npc in enumerate(self.patrol_npcs):
            if not (npc is not None and npc.is_alive):
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

            target_yaw = math.atan2(dy, dx)
            ego_yaw = math.radians(t.rotation.yaw)
            heading_error = math.atan2(
                math.sin(target_yaw - ego_yaw),
                math.cos(target_yaw - ego_yaw),
            )

            steer = float(np.clip(k_p * heading_error, -1.0, 1.0))

            npc_yaw = math.radians(t.rotation.yaw)
            fwd_x = math.cos(npc_yaw)
            fwd_y = math.sin(npc_yaw)
            blocked = False

            # Ego vehicle: omnidirectional stop
            if vehicle is not None and vehicle.is_alive:
                ego_to_x = vehicle.get_location().x - t.location.x
                ego_to_y = vehicle.get_location().y - t.location.y
                ego_dist = math.sqrt(ego_to_x * ego_to_x + ego_to_y * ego_to_y)
                if ego_dist < self._patrol_obstacle_distance + 1.0:
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
                        and lat < 2.0
                    ):
                        blocked = True
                        break

            if blocked:
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
                npc.apply_control(control)
                # Log every 20 steps so patrol stalls are visible without spam
                if steps % 20 == 0:
                    self._debug_logger._logger.debug(
                        "[patrol] npc %d braking (blocked)  wp=%d  step=%d",
                        npc.id,
                        self._patrol_waypoint_indices[i],
                        steps,
                    )
            else:
                speed = math.sqrt(
                    npc.get_velocity().x ** 2 + npc.get_velocity().y ** 2
                )
                speed_ratio = speed / max(self._patrol_max_speed, 0.1)
                throttle = float(np.clip(1.0 - speed_ratio, 0.1, 1.0))

                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = 0.0
                control.steer = steer
                npc.apply_control(control)

    def update_pedestrians(self, vehicle: Any) -> None:
        """
        @brief Advance pedestrians one step with avoidance, zone confinement,
               and periodic heading re-randomisation.

        Priority order each step:
          1. Ego/patrol avoidance -- walk away from nearby threats.
          2. Zone boundary -- steer toward zone centre when near an edge.
          3. Periodic random re-heading every pedestrian_resample_steps steps.

        @param vehicle: Ego vehicle actor (used for avoidance distance check).
        """
        _EGO_AVOID_RADIUS = 2.5
        _BOUNDARY_MARGIN = 0.5

        ego_loc = (
            vehicle.get_location()
            if vehicle is not None and vehicle.is_alive
            else None
        )

        for i, walker in enumerate(self.pedestrian_actors):
            if not (walker is not None and walker.is_alive):
                continue

            self._pedestrian_lifetime_steps[i] += 1
            self._pedestrian_heading_steps[i] += 1

            if self._pedestrian_lifetime_steps[i] >= self._pedestrian_max_lifetime:
                self._respawn_pedestrian(i, vehicle)
                continue

            loc = walker.get_location()
            zone = self._pedestrian_zones[i]

            _PATROL_AVOID_RADIUS = 4.0
            repulse_x = 0.0
            repulse_y = 0.0

            if ego_loc is not None:
                to_ego_x = ego_loc.x - loc.x
                to_ego_y = ego_loc.y - loc.y
                ego_dist = math.sqrt(to_ego_x * to_ego_x + to_ego_y * to_ego_y)
                if ego_dist < _EGO_AVOID_RADIUS:
                    away_mag = ego_dist if ego_dist > 1e-6 else 1.0
                    repulse_x += -to_ego_x / away_mag
                    repulse_y += -to_ego_y / away_mag

            for patrol_npc in self.patrol_npcs:
                if not (patrol_npc is not None and patrol_npc.is_alive):
                    continue
                to_px = patrol_npc.get_location().x - loc.x
                to_py = patrol_npc.get_location().y - loc.y
                patrol_dist = math.sqrt(to_px * to_px + to_py * to_py)
                if patrol_dist < _PATROL_AVOID_RADIUS:
                    away_mag = patrol_dist if patrol_dist > 1e-6 else 1.0
                    repulse_x += -to_px / away_mag
                    repulse_y += -to_py / away_mag

            near_boundary = (
                loc.x < zone["x_min"] + _BOUNDARY_MARGIN
                or loc.x > zone["x_max"] - _BOUNDARY_MARGIN
                or loc.y < zone["y_min"] + _BOUNDARY_MARGIN
                or loc.y > zone["y_max"] - _BOUNDARY_MARGIN
            )

            # Priority: boundary > avoidance > random heading.
            # Boundary check runs first so avoidance can never push a
            # pedestrian out of its zone.
            if near_boundary:
                cx = (zone["x_min"] + zone["x_max"]) / 2.0
                cy = (zone["y_min"] + zone["y_max"]) / 2.0
                to_cx = cx - loc.x
                to_cy = cy - loc.y
                magnitude = math.sqrt(to_cx * to_cx + to_cy * to_cy)
                if magnitude > 1e-6:
                    to_cx, to_cy = to_cx / magnitude, to_cy / magnitude
                self._pedestrian_headings[i] = (to_cx, to_cy, 0.0)
                self._pedestrian_heading_steps[i] = 0
            elif repulse_x != 0.0 or repulse_y != 0.0:
                mag = math.sqrt(repulse_x * repulse_x + repulse_y * repulse_y)
                self._pedestrian_headings[i] = (
                    repulse_x / mag,
                    repulse_y / mag,
                    0.0,
                )
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
        destroyed, a new one is spawned at a random point within the same zone,
        and the lifetime counter is reset.

        @param idx: Index into pedestrian_actors / _pedestrian_zones.
        @param vehicle: Ego vehicle actor (unused directly; kept for future
               ego-proximity guard during respawn).
        """
        # _respawn_pedestrian needs world access -- infer from current actors.
        # The zone dict and walker blueprints are all we need.
        zone = self._pedestrian_zones[idx]
        old = self.pedestrian_actors[idx]
        if old is not None and old.is_alive:
            world = old.get_world()
            old.destroy()
        else:
            # Cannot respawn without a world reference -- mark as gone
            self.pedestrian_actors[idx] = None
            self._debug_logger._logger.debug(
                "[pedestrian] respawn idx=%d SKIPPED (no world reference)", idx
            )
            return

        z_origin = 0.3  # default floor z; exact value not critical for respawn
        z = z_origin + 0.05

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

        self.pedestrian_actors[idx] = walker
        if walker is None:
            self._debug_logger._logger.debug(
                "[pedestrian] respawn idx=%d FAILED after 5 attempts", idx
            )
        else:
            self._debug_logger._logger.debug(
                "[pedestrian] respawn idx=%d ok", idx
            )
        heading_rad = random.uniform(0.0, 2.0 * math.pi)
        self._pedestrian_headings[idx] = (
            math.cos(heading_rad),
            math.sin(heading_rad),
            0.0,
        )
        self._pedestrian_heading_steps[idx] = 0
        self._pedestrian_lifetime_steps[idx] = 0
