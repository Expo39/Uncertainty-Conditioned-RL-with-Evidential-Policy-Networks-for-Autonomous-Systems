"""
@file mapping_drive.py
@brief Autonomous waypoint driver for Cartographer SLAM mapping sessions.

Connects to CARLA, spawns the ego vehicle at the floor plan's primary spawn
point, and drives the patrol_waypoints loop from the layout YAML at low speed.
The 2D LiDAR (30 m range) covers all perimeter cones from the interior aisles.
Runs the loop twice: the first lap initialises Cartographer and the second lap
closes the loop constraint for a consistent pose-graph. After both loops the
script saves a trajectory PNG and exits -- then run make docker-save-map to
serialise the Cartographer state to a .pbstream file.

Usage::

    # Via make target (recommended):
    make docker-up
    make docker-mapping-drive LAYOUT=rectangle
    make docker-save-map LAYOUT=rectangle

    # Directly:
    python scripts/mapping_drive.py --layout rectangle \
        --output-dir outputs/maps/rectangle

@author Antonio Galdes
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import carla
except ImportError:
    print("ERROR: carla package not found. Run inside the training container.")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for container use
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

# Project root for resolving layout YAML paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Proportional heading controller gain (mirrors patrol_heading_gain in
# train_config.yaml)
_K_P: float = 0.8

# Maximum drive speed during mapping (m/s) -- low speed for scan quality
_MAP_SPEED_MS: float = 3.0

# Waypoint capture radius: advance to next waypoint when within this distance (m)
_WAYPOINT_CAPTURE_RADIUS: float = 3.0

# Trajectory sample interval (seconds)
_TRAJ_SAMPLE_INTERVAL: float = 0.5


def _load_layout(layout_name: str) -> Dict[str, Any]:
    """
    @brief Load a pre-computed world-frame layout YAML from configs/layouts/.
    @param layout_name: Layout name without extension (e.g. 'rectangle').
    @return Parsed layout dictionary.
    @raises FileNotFoundError: If the layout YAML does not exist.
    """
    yaml_path = _PROJECT_ROOT / "configs" / "layouts" / f"{layout_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Layout YAML not found: {yaml_path}. "
            f"Run `make generate-layouts LAYOUT={layout_name}` first."
        )
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _connect_carla(host: str, port: int, timeout: float = 30.0) -> "carla.World":
    """
    @brief Connect to the CARLA server and return the world handle.
    @param host: CARLA server hostname.
    @param port: CARLA server port.
    @param timeout: Connection timeout in seconds.
    @return CARLA world object.
    @raises RuntimeError: If the connection fails.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    try:
        world = client.get_world()
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not connect to CARLA at {host}:{port}. "
            "Is the carla-server container running?"
        ) from exc
    return world


def _spawn_ego(
    world: "carla.World",
    spawn_x: float,
    spawn_y: float,
    spawn_z: float,
    spawn_yaw_deg: float,
) -> "carla.Actor":
    """
    @brief Spawn the ego vehicle at the given world-frame transform.
    @param world: CARLA world handle.
    @param spawn_x: Spawn position X (metres).
    @param spawn_y: Spawn position Y (metres).
    @param spawn_z: Spawn position Z (metres, CARLA frame).
    @param spawn_yaw_deg: Spawn yaw in degrees.
    @return Spawned vehicle actor.
    @raises RuntimeError: If no suitable blueprint or spawn slot is available.
    """
    bp_library = world.get_blueprint_library()
    # Use the BMW Grand Tourer as the canonical ego vehicle (matches training)
    ego_bp = bp_library.find("vehicle.bmw.grandtourer")
    if ego_bp is None:
        # Fallback to any car blueprint
        cars = bp_library.filter("vehicle.audi.*")
        if not cars:
            raise RuntimeError("No suitable vehicle blueprint found in CARLA.")
        ego_bp = cars[0]

    ego_bp.set_attribute("role_name", "mapping_driver")

    transform = carla.Transform(
        carla.Location(x=spawn_x, y=spawn_y, z=spawn_z + 0.5),
        carla.Rotation(yaw=spawn_yaw_deg),
    )

    vehicle = world.try_spawn_actor(ego_bp, transform)
    if vehicle is None:
        raise RuntimeError(
            f"Failed to spawn ego vehicle at ({spawn_x:.1f}, {spawn_y:.1f}). "
            "Spawn point may be blocked."
        )
    return vehicle


def _steer_to_waypoint(
    vehicle: "carla.Actor",
    target_x: float,
    target_y: float,
    max_speed_ms: float,
) -> None:
    """
    @brief Apply a proportional heading-error control step towards (target_x, target_y).
    @param vehicle: CARLA vehicle actor to control.
    @param target_x: Target waypoint X in world frame (metres).
    @param target_y: Target waypoint Y in world frame (metres).
    @param max_speed_ms: Speed cap in m/s.
    """
    t = vehicle.get_transform()
    dx = target_x - t.location.x
    dy = target_y - t.location.y

    target_yaw = math.atan2(dy, dx)
    ego_yaw = math.radians(t.rotation.yaw)
    heading_error = math.atan2(
        math.sin(target_yaw - ego_yaw),
        math.cos(target_yaw - ego_yaw),
    )

    steer = float(max(-1.0, min(1.0, _K_P * heading_error)))

    v = vehicle.get_velocity()
    speed = math.sqrt(v.x * v.x + v.y * v.y)
    over_limit = speed > max_speed_ms

    control = carla.VehicleControl()
    control.steer = steer
    control.throttle = 0.0 if over_limit else 0.3
    control.brake = 0.3 if over_limit else 0.0
    vehicle.apply_control(control)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    """
    @brief Euclidean distance between two 2D points.
    @param ax: Point A x.
    @param ay: Point A y.
    @param bx: Point B x.
    @param by: Point B y.
    @return Distance in metres.
    """
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def _save_trajectory_png(
    layout: Dict[str, Any],
    trajectory: List[Tuple[float, float]],
    waypoints: List[Tuple[float, float]],
    output_path: Path,
    layout_name: str,
) -> None:
    """
    @brief Save a bird's-eye PNG showing cone positions and the driven trajectory.

    Plots the exact cone positions that Cartographer will see (computed from
    the perimeter corners at the configured spacing), the perimeter outline for
    spatial orientation, and the driven trajectory. No bay outlines, patrol
    waypoint markers, or legend.

    @param layout: World-frame layout dictionary.
    @param trajectory: List of (x, y) positions sampled during the drive.
    @param waypoints: Kept for API compatibility; not used.
    @param output_path: Full output PNG file path.
    @param layout_name: Layout name for the title.
    """
    if not _MPL_AVAILABLE:
        print("matplotlib not available -- skipping trajectory PNG.")
        return

    sys.path.insert(0, str(_PROJECT_ROOT))
    from scripts.colours import HEX_EGO
    from uncertainty_rl.utils.geometry import _interpolate_cone_positions

    cone_spacing: float = float(layout.get("cone_spacing", 2.0))
    corners_raw = layout.get("corners", [])
    corners: List[Tuple[float, float]] = [
        (float(c["x"]), float(c["y"])) for c in corners_raw
    ]
    cone_positions = _interpolate_cone_positions(corners, cone_spacing)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # -- Perimeter outline for spatial orientation --
    if corners:
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]
        ax.plot(xs, ys, color="#444444", linewidth=1.0)

    # -- Cone positions (what Cartographer sees) --
    if cone_positions:
        cpx = [p[0] for p in cone_positions]
        cpy = [p[1] for p in cone_positions]
        ax.scatter(cpx, cpy, s=8, color="white", zorder=3, alpha=0.9)

    # -- Driven trajectory --
    if trajectory:
        tx = [p[0] for p in trajectory]
        ty = [p[1] for p in trajectory]
        ax.plot(tx, ty, color=HEX_EGO, linewidth=1.8, zorder=5)

    ax.set_title(
        f"Cartographer SLAM map -- {layout_name} (cone spacing: {cone_spacing:.1f} m)",
        color="white",
        fontsize=12,
    )
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Y (m)", color="white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close(fig)
    print(f"Saved trajectory PNG: {output_path}")


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_PROJECT_ROOT / "outputs" / "maps",
        help="Directory for trajectory PNG output.",
    )
    return parser.parse_args()


def main() -> None:
    """
    @brief Entry point: connect to CARLA, drive the patrol loop, save PNG, exit.
    """
    args = _parse_args()

    # -- Load layout YAML --
    print(f"Loading layout: {args.layout}")
    layout = _load_layout(args.layout)

    spawn_info = layout.get("spawn", {})
    spawn_x = float(spawn_info.get("x", 0.0))
    spawn_y = float(spawn_info.get("y", 0.0))
    spawn_z = float(spawn_info.get("z", 0.3))
    spawn_yaw = float(spawn_info.get("yaw_deg", 0.0))

    waypoints_raw = layout.get("patrol_waypoints", [])
    if not waypoints_raw:
        raise ValueError(
            f"Layout '{args.layout}' has no patrol_waypoints. "
            "Regenerate layout YAMLs with `make generate-layouts`."
        )
    waypoints: List[Tuple[float, float]] = [
        (float(wp["x"]), float(wp["y"])) for wp in waypoints_raw
    ]
    # Remove duplicate last waypoint if layout generation closed the loop
    if len(waypoints) > 1 and waypoints[-1] == waypoints[0]:
        waypoints = waypoints[:-1]

    n_waypoints = len(waypoints)
    total_waypoints = args.loops * n_waypoints
    print(
        f"Patrol path: {n_waypoints} waypoints, {args.loops} loops "
        f"= {total_waypoints} waypoints total"
    )

    # -- Connect to CARLA --
    print(f"Connecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    world = _connect_carla(args.carla_host, args.carla_port)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    print("Connected. Spawning ego vehicle ...")

    vehicle: Optional["carla.Actor"] = None
    try:
        vehicle = _spawn_ego(world, spawn_x, spawn_y, spawn_z, spawn_yaw)
        print(
            f"Ego spawned at ({spawn_x:.1f}, {spawn_y:.1f}) " f"yaw={spawn_yaw:.1f} deg"
        )

        # Allow the vehicle to settle before driving
        for _ in range(20):
            world.tick()
        time.sleep(0.5)

        # -- Drive loop --
        trajectory: List[Tuple[float, float]] = []
        wp_global_idx = 0  # counts total waypoints reached across all loops
        current_wp_idx = 0  # index into waypoints[]
        last_sample_time = time.monotonic()

        print("Starting mapping drive ...")
        while wp_global_idx < total_waypoints:
            world.tick()

            # Sample trajectory
            now = time.monotonic()
            if now - last_sample_time >= _TRAJ_SAMPLE_INTERVAL:
                t = vehicle.get_transform()
                trajectory.append((t.location.x, t.location.y))
                last_sample_time = now

            target_x, target_y = waypoints[current_wp_idx]
            _steer_to_waypoint(vehicle, target_x, target_y, _MAP_SPEED_MS)

            t = vehicle.get_transform()
            dist = _dist(t.location.x, t.location.y, target_x, target_y)

            if dist < _WAYPOINT_CAPTURE_RADIUS:
                wp_global_idx += 1
                loop_num = wp_global_idx // n_waypoints + 1
                wp_in_loop = wp_global_idx % n_waypoints
                wp_pos = wp_in_loop + 1 if wp_in_loop > 0 else n_waypoints
                print(
                    f"[waypoint {wp_global_idx}/{total_waypoints} reached] "
                    f"loop {min(loop_num, args.loops)}/{args.loops}, "
                    f"wp {wp_pos}/{n_waypoints}"
                )
                current_wp_idx = (current_wp_idx + 1) % n_waypoints

        # Bring to a stop
        stop = carla.VehicleControl()
        stop.brake = 1.0
        vehicle.apply_control(stop)
        for _ in range(20):
            world.tick()

        # Final trajectory sample
        t = vehicle.get_transform()
        trajectory.append((t.location.x, t.location.y))

        print(
            "Mapping drive complete. Run `make docker-save-map` to serialise .pbstream."
        )

    finally:
        # Restore async mode and destroy vehicle
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    # -- Save trajectory PNG --
    output_dir = Path(args.output_dir) / args.layout
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "mapping_trajectory.png"
    _save_trajectory_png(layout, trajectory, waypoints, png_path, args.layout)


if __name__ == "__main__":
    main()
