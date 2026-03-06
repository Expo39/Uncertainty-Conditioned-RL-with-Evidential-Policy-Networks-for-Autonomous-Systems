"""
@file explore_map.py
@brief Interactive CARLA map exploration tool for identifying parking scenario coordinates.

Connects to a running CARLA instance, loads a specified town, and iterates through
all actor spawn points whilst moving the spectator camera to each one. Coordinates
are printed to the terminal so candidate parking areas can be recorded for use in
train_config.yaml parking_scenarios.

Usage (from host, with carla-server container running):
    python scripts/explore_map.py [--host HOST] [--port PORT] [--town TOWN]
    python scripts/explore_map.py --town Town10HD --pause 3.0

The script prints each spawn point's index, x, y, yaw and pauses between each so
you can observe the view in RustDesk and note down interesting locations.

Press Ctrl+C at any point to stop cycling and exit.
"""

import argparse
import sys
import time
from typing import List, Optional

# CARLA Python API -- available inside the training container or via the egg on the host.
try:
    import carla
except ImportError:
    print(
        "ERROR: carla module not found. "
        "Run this script inside the training container:\n"
        "  make docker-shell\n"
        "  python scripts/explore_map.py --town Town10HD"
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace with host, port, town, pause, and start_index.
    """
    parser = argparse.ArgumentParser(
        description="Explore CARLA map spawn points for parking scenario design."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="carla-server",
        help="CARLA server hostname (default: carla-server, the Docker service name).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000).",
    )
    parser.add_argument(
        "--town",
        type=str,
        default="Town10HD",
        help="CARLA town to load (default: Town10HD).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=3.0,
        help="Seconds to pause at each spawn point (default: 3.0).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Spawn point index to start from (default: 0).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size between spawn points shown (default: 1, i.e. every point).",
    )
    return parser.parse_args()


def connect(host: str, port: int, timeout: float = 10.0) -> carla.Client:
    """
    @brief Connect to the CARLA server.
    @param host: Server hostname or IP.
    @param port: Server port number.
    @param timeout: Connection timeout in seconds.
    @return Connected carla.Client instance.
    @raises RuntimeError if the connection cannot be established within the timeout.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    try:
        server_version = client.get_server_version()
        print(f"Connected to CARLA {server_version} at {host}:{port}")
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not connect to CARLA at {host}:{port}. "
            "Ensure the carla-server container is running:\n"
            "  make docker-up"
        ) from exc
    return client


def load_town(client: carla.Client, town: str) -> carla.World:
    """
    @brief Load the specified CARLA town, reloading if already loaded.
    @param client: Connected CARLA client.
    @param town: Town name string (e.g. 'Town10HD').
    @return carla.World instance for the loaded town.
    """
    world = client.get_world()
    current_map = world.get_map().name

    # Map names from CARLA include the path prefix, e.g. '/Game/Carla/Maps/Town10HD'
    if town in current_map:
        print(f"Map '{current_map}' already loaded -- skipping reload.")
    else:
        print(f"Loading {town} (this may take 30-60 seconds)...")
        world = client.load_world(town)
        # Allow the world to settle after loading
        time.sleep(2.0)
        print(f"Loaded: {world.get_map().name}")

    return world


def set_synchronous_mode(world: carla.World, enabled: bool) -> None:
    """
    @brief Enable or disable synchronous mode on the world.
    @param world: Active carla.World instance.
    @param enabled: True to enable synchronous mode, False to disable.
    """
    settings = world.get_settings()
    settings.synchronous_mode = enabled
    settings.fixed_delta_seconds = 0.05 if enabled else None
    world.apply_settings(settings)


def get_spawn_transforms(world: carla.World) -> List[carla.Transform]:
    """
    @brief Retrieve all recommended spawn transforms from the current map.
    @param world: Active carla.World instance.
    @return List of carla.Transform objects representing spawn locations.
    """
    spawn_points = world.get_map().get_spawn_points()
    return spawn_points


def move_spectator(
    world: carla.World,
    transform: carla.Transform,
    height_offset: float = 10.0,
    pitch_deg: float = -60.0,
) -> None:
    """
    @brief Move the spectator camera above and looking down at a transform.
    @param world: Active carla.World instance.
    @param transform: Target ground-level transform.
    @param height_offset: Metres above the target location to position the camera.
    @param pitch_deg: Camera pitch in degrees (negative = looking down).
    """
    spectator = world.get_spectator()
    spectator_transform = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + height_offset,
        ),
        carla.Rotation(pitch=pitch_deg, yaw=transform.rotation.yaw, roll=0.0),
    )
    spectator.set_transform(spectator_transform)


def print_spawn_point(
    index: int,
    total: int,
    transform: carla.Transform,
) -> None:
    """
    @brief Print spawn point information to the terminal.
    @param index: Spawn point index.
    @param total: Total number of spawn points.
    @param transform: Transform for this spawn point.
    """
    loc = transform.location
    rot = transform.rotation
    print(
        f"  [{index:3d}/{total-1}]  "
        f"x={loc.x:8.2f}  y={loc.y:8.2f}  z={loc.z:6.2f}  "
        f"yaw={rot.yaw:7.2f} deg"
    )


def print_candidate_yaml(transforms: List[carla.Transform], indices: List[int]) -> None:
    """
    @brief Print a YAML snippet for manually noted candidate spawn points.
    @param transforms: Full list of all spawn transforms.
    @param indices: List of indices the user found interesting.
    @note Copy the relevant entries into configs/train_config.yaml.
    """
    if not indices:
        return

    print("\n--- YAML snippet for train_config.yaml (parking_scenarios) ---")
    for i in indices:
        t = transforms[i]
        loc = t.location
        rot = t.rotation
        print(f"  # Spawn point {i}")
        print(f"  - spawn: [x: {loc.x:.2f}, y: {loc.y:.2f}, yaw: {rot.yaw:.2f}]")
        print(f"    target: [x: ???, y: ???, yaw: ???]  # set target manually")
    print("--- end snippet ---\n")


def run_exploration(
    world: carla.World,
    pause: float,
    start: int,
    step: int,
) -> Optional[List[int]]:
    """
    @brief Cycle through spawn points, moving the spectator and printing coordinates.
    @param world: Active carla.World instance.
    @param pause: Seconds to pause at each spawn point.
    @param start: Index of the first spawn point to show.
    @param step: Step size between consecutive spawn points shown.
    @return List of indices manually marked by the user, or None if interrupted early.
    """
    spawn_points = get_spawn_transforms(world)
    total = len(spawn_points)
    print(f"\nTotal spawn points in map: {total}")
    print(f"Starting from index {start}, step={step}, pause={pause}s per point.")
    print("Watch the CARLA window in RustDesk. Press Ctrl+C to stop.\n")
    print(
        f"{'Index':>8}  {'x':>10}  {'y':>10}  {'z':>8}  {'yaw':>10}"
    )
    print("-" * 60)

    interesting: List[int] = []
    indices = list(range(start, total, step))

    try:
        for idx in indices:
            transform = spawn_points[idx]
            move_spectator(world, transform)
            print_spawn_point(idx, total, transform)
            time.sleep(pause)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    print(f"\nExploration complete. Reviewed {len(indices)} spawn points.")
    print(
        "Note down the indices of candidate parking areas and add them to "
        "configs/train_config.yaml under 'parking_scenarios'."
    )
    return interesting


def main() -> None:
    """
    @brief Entry point for the map exploration script.
    """
    args = parse_args()

    client = connect(args.host, args.port)
    world = load_town(client, args.town)

    # Use async mode so we can drive the spectator without ticking
    set_synchronous_mode(world, enabled=False)

    run_exploration(
        world=world,
        pause=args.pause,
        start=args.start,
        step=args.step,
    )


if __name__ == "__main__":
    main()
