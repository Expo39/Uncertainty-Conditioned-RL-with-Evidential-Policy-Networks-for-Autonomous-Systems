"""
@file mapping_drive.py
@brief Autonomous waypoint driver for Cartographer SLAM mapping sessions.

Uses CARLAParkingEnv to spawn the ego vehicle, sensors, and perimeter cones,
then drives the patrol_waypoints loop at low speed so Cartographer accumulates
scans of the cone perimeter. Bay occupancy is zero (empty lot), no patrol NPCs,
no pedestrians -- just the ego and cones.

Runs the loop twice: the first lap initialises Cartographer and the second lap
closes the loop constraint for a consistent pose-graph. After both loops the
script exits -- then run make docker-save-map to serialise the Cartographer
state to a .pbstream file.

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
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

try:
    import carla  # noqa: F401 -- needed at runtime inside training container
except ImportError:
    print("ERROR: carla package not found. Run inside the training container.")
    sys.exit(1)

from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

# Waypoint capture radius (m) -- advance to next waypoint when within this distance
_WAYPOINT_CAPTURE_RADIUS: float = 3.0

# Maximum drive speed during mapping (m/s) -- fast enough to finish quickly,
# slow enough for Cartographer scan quality at 10 Hz rotation
_MAP_SPEED_MS: float = 5.0

# Proportional heading controller gain (mirrors patrol_heading_gain in
# train_config.yaml)
_K_P: float = 0.8


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
        help="Number of full waypoint loops to drive (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    """
    @brief Entry point: create env, drive the patrol loop, exit.
    """
    args = _parse_args()

    # -- Load train config for sensor settings --
    with open("configs/train_config.yaml") as f:
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
    )

    print(f"Mapping drive: layout={args.layout}, loops={args.loops}")

    # -- Reset env (spawns ego, sensors, cones) --
    _obs, info = env.reset()
    print(f"Environment reset. Floor plan: {info.get('floor_plan', args.layout)}")

    # -- Load patrol waypoints from layout YAML --
    with open(layout_file) as f:
        layout: Dict[str, Any] = yaml.safe_load(f)

    waypoints_raw = layout.get("patrol_waypoints", [])
    if not waypoints_raw:
        raise ValueError(
            f"Layout '{args.layout}' has no patrol_waypoints. "
            "Regenerate layout YAMLs with `make generate-layouts`."
        )
    waypoints: List[Tuple[float, float]] = [
        (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
    ]
    if len(waypoints) > 1 and waypoints[-1] == waypoints[0]:
        waypoints = waypoints[:-1]

    n_waypoints = len(waypoints)
    total_waypoints = args.loops * n_waypoints
    print(
        f"Patrol path: {n_waypoints} waypoints x {args.loops} loops "
        f"= {total_waypoints} total"
    )

    # -- Drive the patrol loop --
    wp_idx = 0
    step_count = 0

    print("Starting mapping drive ...")
    while wp_idx < total_waypoints:
        target_x, target_y = waypoints[wp_idx % n_waypoints]

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
            wp_idx += 1
            loop_num = (wp_idx - 1) // n_waypoints + 1
            wp_in_loop = ((wp_idx - 1) % n_waypoints) + 1
            velocity = env.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            print(
                f"  [{wp_idx}/{total_waypoints}] "
                f"loop {loop_num}/{args.loops}, "
                f"wp {wp_in_loop}/{n_waypoints} "
                f"(step {step_count}, speed={speed:.1f} m/s)"
            )

    # -- Stop the vehicle but do NOT close the env --
    # env.close() destroys the ego vehicle and sensors, which can cause the
    # ROS bridge to crash before the Makefile can serialise the .pbstream.
    # The process exit handles cleanup; CARLA garbage-collects orphaned actors.
    env.step(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    print(
        f"Mapping drive complete after {step_count} steps. "
        "Keeping env alive for pbstream serialisation."
    )


if __name__ == "__main__":
    main()
