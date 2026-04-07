"""
@file mapping_drive.py
@brief Autonomous waypoint driver for Cartographer SLAM mapping sessions.

Uses CARLAParkingEnv to spawn the ego vehicle, sensors, and perimeter cones,
then drives the patrol_waypoints loop at low speed so Cartographer accumulates
scans of the cone perimeter. Bay occupancy is zero (empty lot), no patrol NPCs,
no pedestrians -- just the ego and cones.

Drives the patrol loop in two phases (bidirectional mapping):
  1. Forward (CCW): loops/2 laps around the patrol waypoints in order.
  2. Reverse (CW):  loops/2 laps around the patrol waypoints in reverse.

The forward phase initialises Cartographer with consistent submaps. The reverse
phase revisits the same cone perimeter from opposing LiDAR angles, creating
inter-submap constraints that tighten the pose graph. The simulation resets
between phases so the vehicle always starts from spawn (0,0) facing the same
heading -- matching the initial conditions the trained model will see.

The --loops argument must be even (default 2) so the laps split equally between
the two directions.

After all laps the script exits -- then run make docker-save-map to serialise
the Cartographer state to a .pbstream file.

Usage::

    # Via make target (recommended):
    make docker-up
    make docker-map LAYOUT=rectangle

    # Directly (inside training container):
    python -m scripts.mapping_drive --layout rectangle

@author Antonio Galdes
"""

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Waypoint capture radius (m) -- advance to next waypoint when within this distance
_WAYPOINT_CAPTURE_RADIUS: float = 3.0

# Maximum drive speed during mapping (m/s) -- slow enough for Cartographer
# to accumulate high-quality submaps with many scans per location. At 2 m/s
# and 20 Hz, the vehicle moves 0.1m per scan -- well within the 0.5m SLAM
# search window, giving clean submap geometry.
_MAP_SPEED_MS: float = 2.0

# Proportional heading controller gain (mirrors patrol_heading_gain in
# train_config.yaml)
_K_P: float = 0.8


def get_mapping_waypoints(
    layout: str,
    asymmetric_corner: bool,
    layout_corners: Optional[List[Dict[str, float]]] = None,
    wall_inset: float = 3.0,
) -> List[Tuple[float, float]]:
    """
    @brief Get mapping waypoints for a given layout with dynamic wall inset.

    Waypoints start at spawn (0, 0) and trace the perimeter inset from walls.
    For asymmetric layouts, adds waypoints to trace around the notch with 1m inset.

    @param layout: Layout name (rectangle, trapezoid, irregular_a).
    @param asymmetric_corner: If True, use notch-avoiding waypoints (rectangle only).
    @param layout_corners: List of corner dicts {x, y} from layout YAML. If None,
                           will attempt to load from configs/layouts/{layout}.yaml.
    @param wall_inset: Distance inward from walls to place waypoints (metres).
    @return List of (x, y) waypoint tuples in world frame.
    """
    if layout_corners is None:
        # Load from YAML if not provided
        import yaml as yaml_module
        layout_file = f"configs/layouts/{layout}.yaml"
        with open(layout_file) as f:
            layout_data: Dict[str, Any] = yaml_module.safe_load(f)
        layout_corners = layout_data.get("corners", [])

    if not layout_corners:
        raise ValueError(f"No corners found for layout '{layout}'.")

    # Extract corner coordinates
    corners = [(c["x"], c["y"]) for c in layout_corners]

    # For rectangle: corners are (-3, -22.5), (62, -22.5), (62, 22.5), (-3, 22.5)
    x_min = min(c[0] for c in corners)
    x_max = max(c[0] for c in corners)
    y_min = min(c[1] for c in corners)
    y_max = max(c[1] for c in corners)

    # Inset from walls
    x_left = x_min + wall_inset
    x_right = x_max - wall_inset
    y_bottom = y_min + wall_inset
    y_top = y_max - wall_inset

    if asymmetric_corner and layout == "rectangle":
        # Asymmetric: trace around notch with wall_inset from notch edges.
        # Notch in local: x=26..32, y=0..8. In world: x=28..34, y=-14.5..(-22.5)
        x_notch_left = 28.0
        x_notch_right = 34.0
        y_notch_top = -14.5

        return [
            (0.0, y_bottom),                           # WP1: spawn on left wall (closest to 0,0)
            (x_left, y_bottom),                        # WP2: bottom-left corner
            (x_notch_left - wall_inset, y_bottom),     # WP3: at notch left edge (inset)
            (x_notch_left - wall_inset, y_notch_top + wall_inset),  # WP4: up notch left
            (x_notch_right + wall_inset, y_notch_top + wall_inset), # WP5: across notch top
            (x_notch_right + wall_inset, y_bottom),    # WP6: down notch right
            (x_right, y_bottom),                       # WP7: bottom-right corner
            (x_right, (y_top + y_bottom) / 2),         # WP8: right mid
            (x_right, y_top),                           # WP9: top-right corner
            ((x_left + x_right) / 2, y_top),           # WP10: top mid
            (x_left, y_top),                            # WP11: top-left
            (x_left, (y_top + y_bottom) / 2),          # WP12: left mid
        ]
    else:
        # Symmetric: rectangular loop starting from spawn on left wall.
        return [
            (0.0, y_bottom),                           # WP1: spawn on left wall (closest to 0,0)
            (x_left, y_bottom),                        # WP2: bottom-left corner
            ((x_left + x_right) / 2, y_bottom),       # WP3: bottom mid
            (x_right, y_bottom),                       # WP4: bottom-right corner
            (x_right, (y_top + y_bottom) / 2),        # WP5: right mid
            (x_right, y_top),                          # WP6: top-right corner
            ((x_left + x_right) / 2, y_top),          # WP7: top mid
            (x_left, y_top),                           # WP8: top-left corner
            (x_left, (y_top + y_bottom) / 2),         # WP9: left mid
        ]


def _compute_action(
    vehicle_transform: Any,
    vehicle_velocity: Any,
    target_x: float,
    target_y: float,
) -> np.ndarray:
    """
    @brief Compute a [steering, throttle, brake] action to drive toward a waypoint.
    @param vehicle_transform: CARLA Transform of the ego vehicle.
    @param vehicle_velocity: CARLA Vector3D velocity of the ego vehicle.
    @param target_x: Target waypoint X in world frame (metres).
    @param target_y: Target waypoint Y in world frame (metres).
    @return 3-dim action array [steering, throttle, brake].
    """
    dx = target_x - vehicle_transform.location.x
    dy = target_y - vehicle_transform.location.y

    target_yaw = math.atan2(dy, dx)
    ego_yaw = math.radians(vehicle_transform.rotation.yaw)
    heading_error = math.atan2(
        math.sin(target_yaw - ego_yaw),
        math.cos(target_yaw - ego_yaw),
    )

    steer = float(np.clip(_K_P * heading_error, -1.0, 1.0))

    speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2)
    over_speed = speed > _MAP_SPEED_MS

    throttle = 0.0 if over_speed else 0.6
    brake = 0.3 if over_speed else 0.0

    return np.array([steer, throttle, brake], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Drive the patrol-waypoint loop for Cartographer SLAM mapping."
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="rectangle",
        help="Layout name (e.g. rectangle, trapezoid, irregular_a).",
    )
    parser.add_argument(
        "--carla-host",
        type=str,
        default="carla-server",
        help="CARLA server hostname (default: carla-server).",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000).",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=2,
        help=(
            "Total number of full waypoint loops to drive (default: 2). "
            "Must be even when bidirectional (default): first half CCW, "
            "second half CW."
        ),
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        help=(
            "Drive all loops in one direction (CCW) only. Avoids a "
            "mid-drive reset that can misalign submaps between phases."
        ),
    )
    args = parser.parse_args()
    if not args.unidirectional and (args.loops < 2 or args.loops % 2 != 0):
        parser.error("--loops must be an even number >= 2.")
    return args


def main() -> None:
    """
    @brief Entry point: create env, drive the patrol loop, exit.
    """
    # Check CARLA availability before importing heavy dependencies
    try:
        import carla  # noqa: F401
        from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
    except ImportError:
        print("ERROR: carla package not found. Run inside the training container.")
        sys.exit(1)

    args = _parse_args()

    # -- Load env config for sensor and scenario settings --
    with open("configs/carla/env_config.yaml") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # -- Override parking scenarios for mapping: empty lot, cones on, no NPCs --
    scenarios = dict(config.get("parking_scenarios", {}))
    scenarios["spawn_perimeter_cones"] = True
    scenarios["bay_occupancy_min"] = 0.0
    scenarios["bay_occupancy_max"] = 0.0
    scenarios["num_patrol_vehicles_max"] = 0
    scenarios["pedestrian_spawn_probability"] = 0.0

    # Restrict to the requested layout only
    layout_file = f"configs/layouts/{args.layout}.yaml"
    scenarios["floor_plans"] = {args.layout: {"layout_file": layout_file, "ood": False}}

    # -- Create env (no covariance subscription -- Cartographer needs scans first) --
    # include_covariance=False skips EKF subscription and sync-mode setup in the
    # env constructor. We re-enable sync mode below because CARLA ray_cast sensors
    # only produce non-empty PointCloud2 data when world.tick() drives physics
    # (async mode returns 0-point scans regardless of real-time physics activity).
    env = CARLAParkingEnv(
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        town="FlatPlane",
        max_steps=999999,  # We control termination via waypoint count
        carla_sensors_config=config.get("carla_sensors"),
        parking_scenarios_config=scenarios,
        include_covariance=False,  # No EKF covariance during SLAM mapping
        include_obstacle_obs=False,  # Not needed for mapping
        sensor_suite=config.get("sensor_suite", "suite_a"),
        asymmetric_corner=config.get("asymmetric_corner", False),
    )

    print(f"Mapping drive: layout={args.layout}, loops={args.loops}")

    # -- Reset env (spawns ego, sensors, cones) --
    _obs, info = env.reset()
    print(f"Environment reset. Floor plan: {info.get('floor_plan', args.layout)}")

    # -- Enable synchronous mode so world.tick() drives LiDAR ray casting --
    # CARLA ray_cast sensors produce 0-point scans in async mode because the
    # rendering engine does not fire between ticks without an explicit tick call.
    # Synchronous mode ensures each env.step() -> world.tick() triggers a full
    # physics + sensor step, giving Cartographer dense (>20k point) scans.
    # Use mapping_timestep (default: half the training timestep) so the IMU
    # fires every tick while the LiDAR fires every 2 ticks, giving Cartographer
    # 2 IMU samples per scan for proper yaw interpolation.
    carla_timestep: float = float(config.get("mapping_timestep", 0.025))
    if env.world is not None:
        _map_settings = env.world.get_settings()
        if not _map_settings.synchronous_mode:
            print(
                f"Enabling synchronous mode for mapping (fixed_delta={carla_timestep}s) ..."
            )
            _map_settings.synchronous_mode = True
            _map_settings.fixed_delta_seconds = carla_timestep
            env.world.apply_settings(_map_settings)
            print("Synchronous mode enabled. LiDAR will produce dense scans.")

    # -- Load layout and get mapping waypoints --
    with open(layout_file) as f:
        layout_data: Dict[str, Any] = yaml.safe_load(f)

    # Define wall inset for waypoint positioning (metres)
    wall_inset = 3.0

    waypoints = get_mapping_waypoints(
        args.layout,
        config.get("asymmetric_corner", False),
        layout_corners=layout_data.get("corners"),
        wall_inset=wall_inset,
    )

    n_waypoints = len(waypoints)
    waypoints_reversed = list(reversed(waypoints))

    total_waypoints = args.loops * n_waypoints

    if args.unidirectional:
        print(
            f"Patrol path: {n_waypoints} waypoints x {args.loops} loops "
            f"(CCW only) = {total_waypoints} total"
        )
        phases: List[Tuple[str, List[Tuple[float, float]], int]] = [
            ("CCW (forward)", waypoints, args.loops),
        ]
    else:
        half_loops = args.loops // 2
        print(
            f"Patrol path: {n_waypoints} waypoints x {args.loops} loops "
            f"({half_loops} CCW + {half_loops} CW) = {total_waypoints} total"
        )
        phases = [
            ("CCW (forward)", waypoints, half_loops),
            ("CW (reverse)", waypoints_reversed, half_loops),
        ]

    step_count = 0
    global_wp_idx = 0

    print("Starting mapping drive ...")
    for phase_idx, (phase_name, phase_waypoints, phase_loops) in enumerate(phases):
        # Reset the sim at the start of each phase so the vehicle always
        # begins from spawn (0,0) facing the same heading. This mirrors
        # training episodes and avoids a messy U-turn between phases.
        if phase_idx > 0:
            print(
                f"\n  >>> Resetting simulation for {phase_name} phase ...\n"
            )
            _obs, info = env.reset()
            print(
                f"  Environment reset. "
                f"Floor plan: {info.get('floor_plan', args.layout)}"
            )

        phase_total = phase_loops * n_waypoints
        phase_wp_idx = 0

        wp_order = " -> ".join(
            f"({x:.0f},{y:.0f})" for x, y in phase_waypoints
        )
        print(
            f"\n{'=' * 60}\n"
            f"  Phase: {phase_name} -- {phase_loops} laps x "
            f"{n_waypoints} waypoints\n"
            f"  Route: {wp_order}\n"
            f"{'=' * 60}"
        )

        while phase_wp_idx < phase_total:
            target_x, target_y = phase_waypoints[phase_wp_idx % n_waypoints]

            # Compute steering from current vehicle state
            transform = env.vehicle.get_transform()
            velocity = env.vehicle.get_velocity()
            action = _compute_action(transform, velocity, target_x, target_y)

            # Advance simulation
            env.step(action)
            step_count += 1

            # Re-read post-step position for waypoint capture check
            transform = env.vehicle.get_transform()
            dx = target_x - transform.location.x
            dy = target_y - transform.location.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < _WAYPOINT_CAPTURE_RADIUS:
                phase_wp_idx += 1
                global_wp_idx += 1
                loop_num = (phase_wp_idx - 1) // n_waypoints + 1
                wp_in_loop = ((phase_wp_idx - 1) % n_waypoints) + 1
                velocity = env.vehicle.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                print(
                    f"  [{global_wp_idx}/{total_waypoints}] "
                    f"{phase_name} lap {loop_num}/{phase_loops}, "
                    f"wp {wp_in_loop}/{n_waypoints} "
                    f"-> ({target_x:.1f},{target_y:.1f}) "
                    f"(step {step_count}, speed={speed:.1f} m/s)"
                )

        print(f"  {phase_name} phase complete.")

    # -- Stop the vehicle but do NOT close the env --
    # env.close() destroys the ego vehicle and sensors, which can cause the
    # ROS bridge to crash before the Makefile can serialise the .pbstream.
    # The process exit handles cleanup; CARLA garbage-collects orphaned actors.
    env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))

    if args.unidirectional:
        lap_summary = f"{args.loops} CCW laps"
    else:
        half = args.loops // 2
        lap_summary = f"{half} CCW + {half} CW laps"

    print(
        f"\nMapping drive complete after {step_count} steps "
        f"({lap_summary}). "
        "Keeping env alive for pbstream serialisation."
    )


if __name__ == "__main__":
    main()
