"""
@file _npc_controller.py
@brief NPC patrol vehicle and pedestrian controller for CARLAParkingEnv.

Owns all per-episode NPC state (patrol vehicles, pedestrians) and the spawn,
update, and cleanup logic for both. CARLAParkingEnv holds an NPCController
instance and delegates NPC lifecycle calls to it.
"""

import itertools
import logging
import math
import random
from typing import Any, Dict, List, Set, Tuple

import numpy as np

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

    _RESPAWN_N_SECTORS: int = 5
    _RESPAWN_SECTOR_THRESHOLD: float = 8.0
    _EGO_AVOID_RADIUS: float = 5.0
    _PATROL_AVOID_RADIUS: float = 5.0
    _BOUNDARY_MARGIN: float = 1.0
    _CLUSTER_RADIUS: float = 12.0

    # Precomputed squared avoidance thresholds.
    _EGO_AVOID_RADIUS_SQ: float = _EGO_AVOID_RADIUS ** 2
    _PATROL_AVOID_RADIUS_SQ: float = _PATROL_AVOID_RADIUS ** 2

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

        # Precomputed squared distance thresholds for update_patrol hot path.
        self._obs_dist_sq: float = patrol_obstacle_distance ** 2
        self._ped_dist_sq: float = patrol_pedestrian_distance ** 2

        # Blueprint lists - refreshed each reset via refresh_blueprints()
        self._car_blueprints: List[Any] = []
        self._walker_blueprints: List[Any] = []

        # Cached all-vehicle list for patrol obstacle checks.
        self._all_vehicle_actors: List[Any] = []

        # Per-episode patrol state
        self.patrol_npcs: List[Any] = []
        self.patrol_npc_ids: Set[int] = set()
        self._patrol_waypoint_indices: List[int] = []
        self._patrol_waypoint_directions: List[int] = []
        self._patrol_waypoints_cache: List[Tuple[float, float]] = []
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

        @param car_blueprints: Filtered list of four-wheeled vehicle blueprints.
        @param walker_blueprints: Filtered list of pedestrian blueprints.
        """
        self._car_blueprints = car_blueprints
        self._walker_blueprints = walker_blueprints

    def set_vehicle_cache(self, all_vehicle_actors: List[Any]) -> None:
        """
        @brief Store the full vehicle actor list for patrol obstacle checks.

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

        @param world: Live CARLA world handle.
        @param vehicle: Ego vehicle actor (used for ego-safe spawn distance check).
        @param current_layout: Parsed floor plan YAML dict with patrol_waypoints.
        """
        if world is None:
            return

        waypoints_raw = current_layout.get("patrol_waypoints", [])
        if not waypoints_raw or self._num_patrol_max == 0:
            return

        waypoints: List[Tuple[float, float]] = [
            (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
        ]
        if len(waypoints) > 1 and waypoints[-1] == waypoints[0]:
            waypoints = waypoints[:-1]

        self._patrol_waypoints_cache = waypoints
        z = float(current_layout.get("origin", {}).get("z", 0.3)) + 0.1

        ego_loc = vehicle.get_location() if vehicle is not None else None
        min_spawn_dist_sq = (self._patrol_obstacle_distance + 4.5) ** 2
        safe_indices = list(range(len(waypoints)))
        if ego_loc is not None:
            ex, ey = ego_loc.x, ego_loc.y
            safe_indices = [
                idx for idx in safe_indices
                if (waypoints[idx][0] - ex) ** 2 + (waypoints[idx][1] - ey) ** 2
                >= min_spawn_dist_sq
            ]
        if not safe_indices:
            safe_indices = list(range(len(waypoints)))

        n_waypoints = len(waypoints)
        for _ in range(self._num_patrol_max):
            bp = random.choice(self._car_blueprints)
            start_idx = random.choice(safe_indices)
            direction = random.choice([-1, 1])
            first_target_idx = (start_idx + direction) % n_waypoints
            wp = waypoints[start_idx]
            first_target_wp = waypoints[first_target_idx]
            spawn_yaw = math.degrees(
                math.atan2(
                    first_target_wp[1] - wp[1],
                    first_target_wp[0] - wp[0],
                )
            )
            actor = world.try_spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=wp[0], y=wp[1], z=z),
                    carla.Rotation(yaw=spawn_yaw),
                ),
            )
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

        n_zones = len(zones)

        # Build cluster assignment via union-find with path compression.
        parent = list(range(n_zones))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        if n_zones > 1:
            centres = np.array([
                [(z_["x_min"] + z_["x_max"]) * 0.5, (z_["y_min"] + z_["y_max"]) * 0.5]
                for z_ in zones
            ])
            diff = centres[:, None, :] - centres[None, :, :]
            sq_dist = (diff * diff).sum(axis=2)
            cluster_radius_sq = self._CLUSTER_RADIUS ** 2
            for i in range(n_zones):
                for j in range(i + 1, n_zones):
                    if sq_dist[i, j] <= cluster_radius_sq:
                        pi, pj = _find(i), _find(j)
                        if pi != pj:
                            parent[pi] = pj

        clusters: Dict[int, List[int]] = {}
        for idx in range(n_zones):
            clusters.setdefault(_find(idx), []).append(idx)

        for zone_indices in clusters.values():
            if random.random() > self._pedestrian_spawn_prob:
                continue

            zone = zones[random.choice(zone_indices)]
            bp = random.choice(self._walker_blueprints)
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")

            walker = None
            for _ in range(5):
                px = random.uniform(zone["x_min"], zone["x_max"])
                py = random.uniform(zone["y_min"], zone["y_max"])
                walker = world.try_spawn_actor(
                    bp,
                    carla.Transform(
                        carla.Location(x=px, y=py, z=z),
                        carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                    ),
                )
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

        @param vehicle: Ego vehicle actor.
        @param steps: Current episode step count (used for debug throttle).
        """
        waypoints = self._patrol_waypoints_cache
        if not waypoints:
            return

        k_p = self._patrol_heading_gain
        all_vehicles = self._all_vehicle_actors
        n_waypoints = len(waypoints)
        obs_dist_sq = self._obs_dist_sq
        ped_dist_sq = self._ped_dist_sq

        # Cache ego location and ID once.
        ego_loc = None
        ego_id = -1
        if vehicle is not None and vehicle.is_alive:
            ego_loc = vehicle.get_location()
            ego_id = vehicle.id

        for i, npc in enumerate(self.patrol_npcs):
            if npc is None or not npc.is_alive:
                continue

            npc_id = npc.id
            wp_idx = self._patrol_waypoint_indices[i]
            t = npc.get_transform()
            nx, ny = t.location.x, t.location.y

            wp_x, wp_y = waypoints[wp_idx]
            dx = wp_x - nx
            dy = wp_y - ny

            if dx * dx + dy * dy < 9.0:  # dist < 3.0 m
                direction = self._patrol_waypoint_directions[i]
                wp_idx = (wp_idx + direction) % n_waypoints
                self._patrol_waypoint_indices[i] = wp_idx
                wp_x, wp_y = waypoints[wp_idx]
                dx = wp_x - nx
                dy = wp_y - ny

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

            # Ego cone check: cos(60 deg) = 0.5.
            if ego_loc is not None:
                ex = ego_loc.x - nx
                ey = ego_loc.y - ny
                ego_dist_sq = ex * ex + ey * ey
                if ego_dist_sq < obs_dist_sq:
                    ego_fwd_proj = ex * fwd_x + ey * fwd_y
                    if ego_fwd_proj > 0.0 and ego_fwd_proj * ego_fwd_proj > 0.25 * ego_dist_sq:
                        blocked = True

            # Vehicles and pedestrians combined in one loop with one break path.
            if not blocked:
                for other, threshold_sq, lat_max in itertools.chain(
                    (
                        (ov, obs_dist_sq, 2.0)
                        for ov in all_vehicles
                        if ov.id != npc_id and ov.id != ego_id
                    ),
                    (
                        (w, ped_dist_sq, 2.5)
                        for w in self.pedestrian_actors
                        if w is not None and w.is_alive
                    ),
                ):
                    ol = other.get_location()
                    to_x = ol.x - nx
                    to_y = ol.y - ny
                    fwd_proj = to_x * fwd_x + to_y * fwd_y
                    if fwd_proj > 0.0 and to_x * to_x + to_y * to_y < threshold_sq:
                        if abs(to_x * fwd_y - to_y * fwd_x) < lat_max:
                            blocked = True
                            break

            if blocked:
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
                speed = math.hypot(vel.x, vel.y)
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
        ego_avoid_sq = self._EGO_AVOID_RADIUS_SQ
        patrol_avoid_sq = self._PATROL_AVOID_RADIUS_SQ

        for i, walker in enumerate(self.pedestrian_actors):
            if walker is None or not walker.is_alive:
                continue

            self._pedestrian_lifetime_steps[i] += 1
            self._pedestrian_heading_steps[i] += 1

            if self._pedestrian_lifetime_steps[i] >= self._pedestrian_max_lifetime:
                self._respawn_pedestrian(i, vehicle)
                continue

            loc = walker.get_location()
            lx, ly, lz = loc.x, loc.y, loc.z

            zone = self._pedestrian_zones[i]
            x_min = zone["x_min"]
            x_max = zone["x_max"]
            y_min = zone["y_min"]
            y_max = zone["y_max"]
            cx = (x_min + x_max) * 0.5
            cy = (y_min + y_max) * 0.5

            outside = lx < x_min or lx > x_max or ly < y_min or ly > y_max
            if outside:
                lx = max(x_min, min(x_max, lx))
                ly = max(y_min, min(y_max, ly))
                walker.set_location(carla.Location(x=lx, y=ly, z=lz))
                to_cx = cx - lx
                to_cy = cy - ly
                c_mag_sq = to_cx * to_cx + to_cy * to_cy
                if c_mag_sq > 1e-12:
                    inv_c = 1.0 / math.sqrt(c_mag_sq)
                    self._pedestrian_headings[i] = (to_cx * inv_c, to_cy * inv_c, 0.0)
                self._pedestrian_heading_steps[i] = 0

            near_boundary = (
                lx < x_min + self._BOUNDARY_MARGIN
                or lx > x_max - self._BOUNDARY_MARGIN
                or ly < y_min + self._BOUNDARY_MARGIN
                or ly > y_max - self._BOUNDARY_MARGIN
            )

            repulse_x = 0.0
            repulse_y = 0.0

            if ego_loc is not None:
                tex = ego_loc.x - lx
                tey = ego_loc.y - ly
                ego_dist_sq = tex * tex + tey * tey
                if ego_dist_sq < ego_avoid_sq:
                    weight = 1.0 / max(math.sqrt(ego_dist_sq), 0.5)
                    repulse_x -= tex * weight
                    repulse_y -= tey * weight

            for patrol_npc in self.patrol_npcs:
                if patrol_npc is None or not patrol_npc.is_alive:
                    continue
                pl = patrol_npc.get_location()
                tpx = pl.x - lx
                tpy = pl.y - ly
                pd_sq = tpx * tpx + tpy * tpy
                if pd_sq < patrol_avoid_sq:
                    weight = 1.0 / max(math.sqrt(pd_sq), 0.5)
                    repulse_x -= tpx * weight
                    repulse_y -= tpy * weight

            if repulse_x != 0.0 or repulse_y != 0.0:
                inv_mag = 1.0 / math.hypot(repulse_x, repulse_y)
                if near_boundary:
                    to_cx_n = cx - lx
                    to_cy_n = cy - ly
                    c_mag_sq = to_cx_n * to_cx_n + to_cy_n * to_cy_n
                    if c_mag_sq > 1e-12:
                        inv_c = 1.0 / math.sqrt(c_mag_sq)
                        blend = 0.3
                        nx_ = (1.0 - blend) * repulse_x * inv_mag + blend * to_cx_n * inv_c
                        ny_ = (1.0 - blend) * repulse_y * inv_mag + blend * to_cy_n * inv_c
                        n_mag = math.hypot(nx_, ny_)
                        if n_mag > 1e-6:
                            nx_, ny_ = nx_ / n_mag, ny_ / n_mag
                        self._pedestrian_headings[i] = (nx_, ny_, 0.0)
                    else:
                        self._pedestrian_headings[i] = (
                            repulse_x * inv_mag, repulse_y * inv_mag, 0.0
                        )
                else:
                    self._pedestrian_headings[i] = (
                        repulse_x * inv_mag, repulse_y * inv_mag, 0.0
                    )
                self._pedestrian_heading_steps[i] = 0
            elif near_boundary:
                to_cx_n = cx - lx
                to_cy_n = cy - ly
                mag_sq = to_cx_n * to_cx_n + to_cy_n * to_cy_n
                if mag_sq > 1e-12:
                    inv_m = 1.0 / math.sqrt(mag_sq)
                    to_cx_n *= inv_m
                    to_cy_n *= inv_m
                self._pedestrian_headings[i] = (to_cx_n, to_cy_n, 0.0)
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
        for actor in self.patrol_npcs:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.patrol_npcs.clear()
        for actor in self.pedestrian_actors:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.pedestrian_actors.clear()

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
        
        @param idx: Index into pedestrian_actors / _pedestrian_zones.
        @param vehicle: Ego vehicle actor (used for sector selection).
        """
        zone = self._pedestrian_zones[idx]
        old = self.pedestrian_actors[idx]
        if old is not None and old.is_alive:
            world = old.get_world()
            old.destroy()
        else:
            self.pedestrian_actors[idx] = None
            logger.debug(
                "[pedestrian] respawn idx=%d SKIPPED (no world reference)", idx
            )
            return

        z = 0.3 + 0.05

        bp = random.choice(self._walker_blueprints)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")

        n = self._RESPAWN_N_SECTORS
        x_span = zone["x_max"] - zone["x_min"]
        y_span = zone["y_max"] - zone["y_min"]
        slice_x = x_span >= y_span

        # Build sectors via linspace.
        if slice_x:
            edges = np.linspace(zone["x_min"], zone["x_max"], n + 1)
            sectors: List[Tuple[float, float, float, float]] = [
                (float(edges[s]), float(edges[s + 1]), zone["y_min"], zone["y_max"])
                for s in range(n)
            ]
        else:
            edges = np.linspace(zone["y_min"], zone["y_max"], n + 1)
            sectors = [
                (zone["x_min"], zone["x_max"], float(edges[s]), float(edges[s + 1]))
                for s in range(n)
            ]

        ego_loc = (
            vehicle.get_location() if vehicle is not None and vehicle.is_alive else None
        )

        if ego_loc is not None:
            ex, ey = ego_loc.x, ego_loc.y
            sectors.sort(
                key=lambda s: ((s[0] + s[1]) * 0.5 - ex) ** 2 + ((s[2] + s[3]) * 0.5 - ey) ** 2,
                reverse=True,
            )
            nearest_sq = (
                ((sectors[-1][0] + sectors[-1][1]) * 0.5 - ex) ** 2
                + ((sectors[-1][2] + sectors[-1][3]) * 0.5 - ey) ** 2
            )
            nearest_dist = math.sqrt(nearest_sq)
            excluded = int(
                (n - 1) * max(0.0, 1.0 - nearest_dist / self._RESPAWN_SECTOR_THRESHOLD)
            )
            sectors = sectors[: max(1, n - excluded)]

        chosen = random.choice(sectors)
        walker = None
        for _ in range(5):
            px = random.uniform(chosen[0], chosen[1])
            py = random.uniform(chosen[2], chosen[3])
            walker = world.try_spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=px, y=py, z=z),
                    carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                ),
            )
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
