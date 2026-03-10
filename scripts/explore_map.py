"""
@file explore_map.py
@brief Interactive CARLA map exploration tool for identifying parking scenario coordinates.

Connects to a running CARLA instance, loads a specified town, and iterates through
all actor spawn points whilst moving the spectator camera to each one. Coordinates
are printed to the terminal so candidate parking areas can be recorded for use in
train_config.yaml parking_scenarios.

Three modes:

1. Interactive exploration (default):
   Cycles through spawn points with the spectator camera overhead.
   Press ENTER within the pause window to mark a point as a candidate.
   Press Ctrl+C to stop early.

2. Dump mode (--dump):
   Prints ALL spawn points for one or more towns and exits immediately.
   No CARLA display required. Useful for offline analysis.
   Use --towns to specify multiple maps: --towns Town03 Town05 Town10HD

3. Mark mode (--mark):
   Fly the CARLA spectator camera freely. Press ENTER to record the current
   spectator (x, y, z, yaw) as a lot origin candidate. Press Ctrl+C to finish.
   Prints a YAML snippet on exit. Use to record 3 world-frame origins for the
   3 floor plan configs (rectangle, trapezoid, irregular_a).
   Usage: python scripts/explore_map.py --mark --town Town05_Opt

Usage (inside the training container):
    python scripts/explore_map.py --town Town10HD --pause 3.0
    python scripts/explore_map.py --town Town05 --start 10 --step 2 --pause 5.0
    python scripts/explore_map.py --dump --towns Town03 Town05 Town10HD
    python scripts/explore_map.py --mark --town Town05_Opt
    make docker-explore-map-mark
"""

import argparse
import select
import sys
import time
from typing import List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace with all exploration and dump options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Explore CARLA map spawn points for parking scenario design.\n"
            "Use --dump to print all spawn points for multiple maps without\n"
            "needing a display (useful for finding flat open areas offline)."
        )
    )
    parser.add_argument(
        "--host",
        type=str,
        default="carla-server",
        help="CARLA server hostname (default: carla-server).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000).",
    )
    # --- interactive mode ---
    parser.add_argument(
        "--town",
        type=str,
        default="Town10HD",
        help="CARLA town to load for interactive exploration (default: Town10HD).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=3.0,
        help="Seconds to pause at each spawn point in interactive mode (default: 3.0).",
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
        help="Step size between spawn points shown (default: 1).",
    )
    # --- dump mode ---
    parser.add_argument(
        "--dump",
        action="store_true",
        help=(
            "Dump all spawn points for one or more towns and exit. "
            "No display required. Use --towns to specify maps."
        ),
    )
    parser.add_argument(
        "--towns",
        nargs="+",
        default=["Town01", "Town02", "Town03", "Town04", "Town05", "Town10HD"],
        help=(
            "Towns to dump in --dump mode "
            "(default: Town01 Town02 Town03 Town04 Town05 Town10HD)."
        ),
    )
    # --- mark mode ---
    parser.add_argument(
        "--mark",
        action="store_true",
        help=(
            "Mark mode: print the live spectator position every second. "
            "Press ENTER to record it as a lot origin candidate. "
            "Press Ctrl+C to stop and print a YAML snippet. "
            "Use this to record 3 origins for configs/layouts/*.yaml."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CARLA connection helpers
# ---------------------------------------------------------------------------


def connect(host: str, port: int, timeout: float = 30.0) -> carla.Client:
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
            "Ensure the carla-server container is running (make docker-up)."
        ) from exc
    return client


def load_town(client: carla.Client, town: str) -> carla.World:
    """
    @brief Load the specified CARLA town, skipping reload if already active.
    @param client: Connected CARLA client.
    @param town: Town name string (e.g. 'Town10HD').
    @return carla.World instance for the loaded town.
    """
    world = client.get_world()
    current_map = world.get_map().name

    # Map names from the CARLA server include a path prefix, e.g. '/Game/Carla/Maps/Town10HD'
    if town in current_map:
        print(f"Map '{current_map}' already loaded.")
    else:
        print(f"Loading {town} (this may take 30–60 seconds)…")
        world = client.load_world(town)
        time.sleep(3.0)  # let assets settle before querying spawn points
        print(f"Loaded: {world.get_map().name}")

    return world


def set_asynchronous_mode(world: carla.World) -> carla.WorldSettings:
    """
    @brief Switch the world to asynchronous mode and return the original settings.
    @param world: Active carla.World instance.
    @return Original carla.WorldSettings for later restoration.
    """
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    return original


def restore_settings(world: carla.World, original: carla.WorldSettings) -> None:
    """
    @brief Restore previously snapshotted world settings.
    @param world: Active carla.World instance.
    @param original: Settings captured before this script modified them.
    """
    world.apply_settings(original)


# ---------------------------------------------------------------------------
# Spectator
# ---------------------------------------------------------------------------


def move_spectator(
    world: carla.World,
    transform: carla.Transform,
    height_offset: float = 20.0,
    pitch_deg: float = -70.0,
) -> None:
    """
    @brief Position the spectator camera above a spawn point looking down.
    @param world: Active carla.World instance.
    @param transform: Ground-level spawn transform to look at.
    @param height_offset: Metres above the location to position the camera.
    @param pitch_deg: Camera pitch in degrees (negative = looking down).
    """
    spectator = world.get_spectator()
    overhead = carla.Transform(
        carla.Location(
            x=transform.location.x,
            y=transform.location.y,
            z=transform.location.z + height_offset,
        ),
        carla.Rotation(pitch=pitch_deg, yaw=transform.rotation.yaw, roll=0.0),
    )
    spectator.set_transform(overhead)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_header() -> None:
    """
    @brief Print the column header for spawn point output.
    """
    print(f"\n{'Index':>8}  {'x':>10}  {'y':>10}  {'z':>8}  {'yaw (deg)':>10}")
    print("-" * 58)


def print_spawn_point(index: int, total: int, transform: carla.Transform) -> None:
    """
    @brief Print one spawn point's coordinates to the terminal.
    @param index: Spawn point index.
    @param total: Total number of spawn points in the map.
    @param transform: Transform for this spawn point.
    """
    loc = transform.location
    rot = transform.rotation
    print(
        f"  [{index:3d}/{total - 1}]  "
        f"x={loc.x:8.2f}  y={loc.y:8.2f}  z={loc.z:6.2f}  "
        f"yaw={rot.yaw:7.2f}"
    )


def print_yaml_snippet(
    town: str, candidates: List[Tuple[int, carla.Transform]]
) -> None:
    """
    @brief Print a valid YAML block for candidate spawn points to copy into train_config.yaml.
    @param town: Name of the CARLA town these candidates are from.
    @param candidates: List of (index, transform) tuples marked during exploration.
    """
    if not candidates:
        return

    print(f"\n# --- Copy relevant entries into configs/train_config.yaml ---")
    print(f"map: {town}")
    print("parking_scenarios:")
    for idx, t in candidates:
        loc = t.location
        rot = t.rotation
        print(f"  # spawn point index {idx}")
        print(f"  - spawn_x:   {loc.x:.2f}")
        print(f"    spawn_y:   {loc.y:.2f}")
        print(f"    spawn_yaw: {rot.yaw:.2f}")
        print(f"    target_x:   0.0  # TODO: measure from map")
        print(f"    target_y:   0.0  # TODO: measure from map")
        print(f"    target_yaw: 0.0  # TODO: measure from map")
    print("# ---")


def _stdin_ready(timeout: float) -> bool:
    """
    @brief Non-blocking check for stdin input within a timeout window.
    @param timeout: Seconds to wait for input.
    @return True if the user pressed Enter within the timeout, False otherwise.
    @note Falls back to False on non-Unix systems or non-interactive terminals.
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            sys.stdin.readline()  # consume the newline
            return True
        return False
    except (OSError, ValueError):
        # Non-interactive terminal or Windows — just sleep and skip marking
        time.sleep(timeout)
        return False


# ---------------------------------------------------------------------------
# Interactive exploration mode
# ---------------------------------------------------------------------------


def run_exploration(
    world: carla.World,
    town: str,
    pause: float,
    start: int,
    step: int,
) -> List[Tuple[int, carla.Transform]]:
    """
    @brief Cycle through spawn points, moving the spectator and printing coordinates.

    At each spawn point the script waits `pause` seconds. Pressing ENTER during
    the wait marks the point as a candidate. Non-interactive terminals skip the prompt.

    @param world: Active carla.World instance.
    @param town: Town name string for YAML output labelling.
    @param pause: Seconds to pause at each spawn point.
    @param start: Index of the first spawn point to show.
    @param step: Step size between consecutive spawn points.
    @return List of (index, transform) tuples the user marked as candidates.
    """
    spawn_points = world.get_map().get_spawn_points()
    total = len(spawn_points)
    indices = list(range(start, total, step))

    print(f"\nTotal spawn points in map: {total}")
    print(f"Showing indices {start}–{total - 1}, step={step}, pause={pause}s each.")
    print("Press ENTER at any point during the pause to mark as a candidate.")
    print("Press Ctrl+C to stop early.\n")
    print_header()

    candidates: List[Tuple[int, carla.Transform]] = []
    reviewed = 0

    try:
        for idx in indices:
            transform = spawn_points[idx]
            move_spectator(world, transform)
            print_spawn_point(idx, total, transform)
            print(
                f"         → Press ENTER to mark #{idx} as candidate (waiting {pause}s)…",
                end="",
                flush=True,
            )

            marked = _stdin_ready(pause)
            if marked:
                candidates.append((idx, transform))
                print(f" ✓ MARKED")
            else:
                print()  # newline after the waiting prompt

            reviewed += 1

    except KeyboardInterrupt:
        print("\n\nStopped early.")

    print(f"\nReviewed {reviewed} of {len(indices)} spawn points.")
    print(f"Marked {len(candidates)} candidate(s).")

    if candidates:
        print_yaml_snippet(town, candidates)
    else:
        print(
            "\nNo candidates marked. Re-run with smaller --step and longer --pause\n"
            "to revisit interesting areas, or use --dump to see all coordinates."
        )

    return candidates


# ---------------------------------------------------------------------------
# Dump mode — no display required
# ---------------------------------------------------------------------------


def run_dump(client: carla.Client, towns: List[str]) -> None:
    """
    @brief Print all spawn points for each specified town and exit.

    Spawn points are sorted by z then x to make flat open clusters easy to spot.
    Towns that fail to load are skipped with a warning.

    @param client: Connected CARLA client.
    @param towns: List of town name strings to query.
    """
    for town in towns:
        print(f"\n{'=' * 60}")
        print(f"  {town}")
        print(f"{'=' * 60}")
        try:
            world = load_town(client, town)
            spawn_points = world.get_map().get_spawn_points()
            total = len(spawn_points)
            print(f"  {total} spawn points (sorted by z asc, then x asc)\n")
            print(f"  {'Idx':>5}  {'x':>10}  {'y':>10}  {'z':>8}  {'yaw':>8}")
            print(f"  {'-' * 50}")

            # Sort by z (flat areas first), then x for readability
            indexed = sorted(
                enumerate(spawn_points),
                key=lambda pair: (round(pair[1].location.z, 1), pair[1].location.x),
            )
            for idx, t in indexed:
                loc = t.location
                rot = t.rotation
                print(
                    f"  {idx:5d}  "
                    f"x={loc.x:8.2f}  y={loc.y:8.2f}  "
                    f"z={loc.z:6.2f}  yaw={rot.yaw:6.2f}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: Could not load {town}: {exc}")

    print(
        "\nTip: look for clusters of spawn points with z ≈ 0 within ~50 m of each other —\n"
        "     those indicate flat open areas suitable for parking bay placement."
    )


# ---------------------------------------------------------------------------
# Mark mode -- record lot origins by flying the spectator freely
# ---------------------------------------------------------------------------


def _print_mark_yaml_snippet(
    town: str,
    marked: List[Tuple[str, carla.Transform]],
) -> None:
    """
    @brief Print a YAML snippet for the recorded lot origins.

    The snippet shows the origin fields expected by configs/layouts/*.yaml.
    Paste the relevant values into each layout file, then re-run
    'make generate-layouts'.

    @param town: Town name for reference comment.
    @param marked: List of (label, spectator_transform) tuples recorded by the user.
    """
    if not marked:
        return

    print(f"\n# --- Recorded lot origins (paste into configs/layouts/*.yaml) ---")
    print(f"# Town: {town}")
    for label, t in marked:
        loc = t.location
        rot = t.rotation
        print(f"# {label}")
        print(f"origin:")
        print(f"  x: {loc.x:.2f}")
        print(f"  y: {loc.y:.2f}")
        print(f"  z: {loc.z:.2f}")
        print(f"  heading_deg: {rot.yaw:.2f}")
        print()
    print("# ---")
    print("# After editing configs/layouts/*.yaml, run: make generate-layouts")


def run_mark_mode(
    world: carla.World,
    town: str,
) -> None:
    """
    @brief Mark mode: print live spectator position, record origins on ENTER.

    Fly the CARLA spectator camera to each desired lot position. Press ENTER
    to record the current (x, y, z, yaw) as a lot origin candidate. The
    spectator z is typically 0-50 m above ground; the printed origin uses the
    ground z (0.3) rather than camera height.

    Press Ctrl+C to finish and print the YAML snippet.

    @param world: Active carla.World instance.
    @param town: Town name string for YAML output labelling.
    """
    print(f"\nMark mode active on {town}.")
    print("Fly the spectator camera to each desired lot position.")
    print(
        "Press ENTER to record the spectator (x, y) as a lot origin (z fixed to 0.30)."
    )
    print("Press Ctrl+C when done.\n")

    marked: List[Tuple[str, carla.Transform]] = []
    count = 0

    try:
        while True:
            spectator = world.get_spectator()
            t = spectator.get_transform()
            loc = t.location
            rot = t.rotation
            # Print live position (overwrite line)
            print(
                f"\r  Spectator: x={loc.x:8.2f}  y={loc.y:8.2f}  z={loc.z:6.2f}"
                f"  yaw={rot.yaw:7.2f}   [ENTER to record, Ctrl+C to finish]",
                end="",
                flush=True,
            )

            marked_this = _stdin_ready(0.5)
            if marked_this:
                count += 1
                label = f"lot_{count}"
                # Use ground z, not spectator camera height
                ground_transform = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=0.3),
                    carla.Rotation(pitch=0.0, yaw=rot.yaw, roll=0.0),
                )
                marked.append((label, ground_transform))
                print(
                    f"\n  [+] Recorded {label}: x={loc.x:.2f}  y={loc.y:.2f}  yaw={rot.yaw:.2f}"
                )

    except KeyboardInterrupt:
        print("\n\nStopped.")

    print(f"\nRecorded {len(marked)} origin(s).")
    _print_mark_yaml_snippet(town, marked)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    @brief Entry point: dispatches to dump mode, mark mode, or interactive exploration.
    """
    args = parse_args()
    client = connect(args.host, args.port)

    if args.dump:
        run_dump(client, args.towns)
        return

    # Both mark mode and interactive exploration need a loaded town
    world = load_town(client, args.town)
    original_settings = set_asynchronous_mode(world)

    try:
        if args.mark:
            run_mark_mode(world=world, town=args.town)
        else:
            run_exploration(
                world=world,
                town=args.town,
                pause=args.pause,
                start=args.start,
                step=args.step,
            )
    finally:
        restore_settings(world, original_settings)


if __name__ == "__main__":
    main()
