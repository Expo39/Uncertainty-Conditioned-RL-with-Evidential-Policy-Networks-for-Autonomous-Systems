"""
@file inspect_layout.py
@brief Interactive CARLA layout inspector.

Spawns a parking lot layout in windowed CARLA with full debug overlays:
  - Perimeter cones (physical props)
  - Bay outlines coloured by type (perpendicular=blue, angled=orange, parallel=green)
  - Target bay highlighted in bright green with "TARGET" label
  - Spawn point dot (yellow)
  - Extra spawn point dots (orange)
  - Pedestrian zone outlines (purple)
  - Patrol waypoint path (red dots + connecting lines)
Holds the scene for --duration seconds. Use to verify layout before training.

Usage:
  make docker-inspect LAYOUT=trapezoid

@note Runs in synchronous CARLA mode (world.tick() every second).
      The debug overlays use long life_time so they persist without stepping.
"""

import argparse
import math
import sys
import time
from typing import Any, Dict, List, Tuple

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found. Run inside the training container.")
    sys.exit(1)

from uncertainty_rl.envs.carla_parking import CARLAParkingEnv


# ---------------------------------------------------------------------------
# Debug overlay helpers (inspect-only, not used during training)
# ---------------------------------------------------------------------------


def _draw_inspect_overlays(
    world: Any,
    layout: Dict[str, Any],
    target_bay_id: str,
    duration: float,
) -> None:
    """
    @brief Draw all layout debug overlays with long life_time for static display.

    All overlays persist for duration+10 seconds so they don't need to be
    redrawn every tick. Covers bays, spawn points, pedestrian zones, and
    patrol waypoints.

    @param world: Active carla.World instance.
    @param layout: Loaded layout dict from the YAML file.
    @param target_bay_id: ID of the selected target bay (highlighted green).
    @param duration: Scene duration in seconds (used to set life_time).
    """
    debug = world.debug
    life = duration + 10.0
    z = float(layout.get("origin_z", 0.0)) + 0.15

    # Bay outlines coloured by type
    type_colours = {
        "perpendicular": carla.Color(r=0, g=0, b=220),
        "angled": carla.Color(r=220, g=100, b=0),
        "parallel": carla.Color(r=0, g=160, b=0),
    }

    for bay in layout.get("bays", []):
        bay_type = bay.get("bay_type", "perpendicular")
        is_target = bay.get("id", bay.get("bay_id", "")) == target_bay_id
        colour = carla.Color(r=0, g=255, b=0) if is_target else type_colours.get(
            bay_type, carla.Color(r=120, g=120, b=120)
        )
        thickness = 0.10 if is_target else 0.05

        bx = float(bay["x"])
        by = float(bay["y"])
        width = float(bay.get("width", 2.5))
        depth = float(bay.get("depth", 5.0))
        yaw_deg = float(bay.get("yaw_deg", math.degrees(bay.get("yaw", 0.0))))

        box = carla.BoundingBox(
            carla.Location(x=bx, y=by, z=z),
            carla.Vector3D(x=depth / 2.0, y=width / 2.0, z=0.05),
        )
        debug.draw_box(
            box,
            carla.Rotation(yaw=yaw_deg),
            thickness=thickness,
            color=colour,
            life_time=life,
        )

        if is_target:
            debug.draw_string(
                carla.Location(x=bx, y=by, z=z + 1.5),
                "TARGET",
                color=carla.Color(r=0, g=255, b=0),
                life_time=life,
            )

    # Spawn point -- yellow dot
    spawn = layout.get("spawn_transform", {})
    sx = float(spawn.get("x", 0.0))
    sy = float(spawn.get("y", 0.0))
    debug.draw_point(
        carla.Location(x=sx, y=sy, z=z + 0.3),
        size=0.2,
        color=carla.Color(r=255, g=255, b=0),
        life_time=life,
    )
    debug.draw_string(
        carla.Location(x=sx, y=sy, z=z + 1.0),
        "SPAWN",
        color=carla.Color(r=255, g=255, b=0),
        life_time=life,
    )

    # Extra spawn points -- orange dots
    for i, extra in enumerate(layout.get("extra_spawn_transforms", [])):
        ex = float(extra.get("x", 0.0))
        ey = float(extra.get("y", 0.0))
        debug.draw_point(
            carla.Location(x=ex, y=ey, z=z + 0.3),
            size=0.15,
            color=carla.Color(r=255, g=140, b=0),
            life_time=life,
        )
        debug.draw_string(
            carla.Location(x=ex, y=ey, z=z + 1.0),
            f"SPAWN{i + 2}",
            color=carla.Color(r=255, g=140, b=0),
            life_time=life,
        )

    # Pedestrian zones -- purple outlines
    def _zone_bbox(zone_raw: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """@brief Convert zone dict to (x_min, x_max, y_min, y_max)."""
        if "x_min" in zone_raw:
            return (
                float(zone_raw["x_min"]),
                float(zone_raw["x_max"]),
                float(zone_raw["y_min"]),
                float(zone_raw["y_max"]),
            )
        cx = float(zone_raw["centre_x"])
        cy = float(zone_raw["centre_y"])
        hw = float(zone_raw["half_width"])
        hh = float(zone_raw["half_height"])
        return cx - hw, cx + hw, cy - hh, cy + hh

    purple = carla.Color(r=180, g=0, b=220)
    for zone_raw in layout.get("pedestrian_zones", []):
        x_min, x_max, y_min, y_max = _zone_bbox(zone_raw)
        corners = [
            carla.Location(x=x_min, y=y_min, z=z),
            carla.Location(x=x_max, y=y_min, z=z),
            carla.Location(x=x_max, y=y_max, z=z),
            carla.Location(x=x_min, y=y_max, z=z),
        ]
        for j in range(4):
            debug.draw_line(
                corners[j],
                corners[(j + 1) % 4],
                thickness=0.05,
                color=purple,
                life_time=life,
            )

    # Patrol waypoints -- red dots connected by lines
    waypoints: List[Tuple[float, float]] = [
        (float(wp["x"]), float(wp["y"]))
        for wp in layout.get("patrol_waypoints", [])
    ]
    red = carla.Color(r=220, g=0, b=0)
    for i, (wx, wy) in enumerate(waypoints):
        debug.draw_point(
            carla.Location(x=wx, y=wy, z=z + 0.2),
            size=0.12,
            color=red,
            life_time=life,
        )
        if len(waypoints) > 1:
            nx, ny = waypoints[(i + 1) % len(waypoints)]
            debug.draw_line(
                carla.Location(x=wx, y=wy, z=z + 0.2),
                carla.Location(x=nx, y=ny, z=z + 0.2),
                thickness=0.04,
                color=red,
                life_time=life,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """@brief Parse arguments and spawn a layout in windowed CARLA for inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect a parking lot layout in CARLA with full debug overlays."
    )
    parser.add_argument(
        "--layout",
        default="trapezoid",
        choices=["rectangle", "trapezoid", "irregular_a"],
        help="Floor plan to spawn (default: trapezoid).",
    )
    parser.add_argument(
        "--host", default="carla-server-demo", help="CARLA server hostname."
    )
    parser.add_argument("--port", type=int, default=2100, help="CARLA server port.")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Seconds to hold the scene (default: 300).",
    )
    args = parser.parse_args()

    # Inspect mode
    scenarios = {
        "floor_plans": {
            args.layout: {
                "weight": 1.0,
                "always_empty": [],
                "layout_file": f"configs/layouts/{args.layout}.yaml",
            }
        },
        "num_patrol_vehicles_max": 1,
        "num_pedestrians_max": 2,
        "bay_occupancy_rate": 0.6,
    }
    env = CARLAParkingEnv(
        carla_host=args.host,
        carla_port=args.port,
        town="FlatPlane",
        parking_scenarios_config=scenarios,
        include_covariance=False,
    )

    print(f"Spawning '{args.layout}' layout in CARLA at {args.host}:{args.port} ...")
    env.reset()

    if env.world is None:
        print("ERROR: Could not connect to CARLA.")
        env.close()
        sys.exit(1)

    # Position spectator above vehicle spawn point
    if env._current_layout:
        spawn = env._current_layout.get("spawn_transform", {})
        sx = float(spawn.get("x", 0.0))
        sy = float(spawn.get("y", 0.0))
        sz = float(spawn.get("z", 0.3))
        spectator = env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=sx, y=sy, z=sz + 80.0),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        )
        print(f"Spectator at ({sx:.0f}, {sy:.0f}, {sz + 80.0:.0f}) -- bird's-eye view.")

    # Draw all overlays with long life_time
    target_bay_id = env._target_bay.get("bay_id", "")
    _draw_inspect_overlays(env.world, env._current_layout, target_bay_id, float(args.duration))
    print("Debug overlays drawn:")
    print("  blue=perpendicular bays, orange=angled bays, green=parallel bays")
    print("  bright green=TARGET bay, yellow=spawn, purple=pedestrian zones, red=patrol path")

    # Tick at 20 Hz so patrol vehicles and pedestrians move smoothly.
    tick_hz = 20
    total_ticks = args.duration * tick_hz
    overlay_redraw_ticks = 30 * tick_hz  # redraw every 30 s
    log_ticks = 10 * tick_hz             # print progress every 10 s

    print(f"Scene live for {args.duration}s. Press Ctrl+C to exit early.")
    try:
        for i in range(total_ticks):
            env.world.tick()
            env._update_patrol_npcs()
            env._update_pedestrians()
            time.sleep(1.0 / tick_hz)
            if (i + 1) % overlay_redraw_ticks == 0:
                _draw_inspect_overlays(
                    env.world, env._current_layout, target_bay_id, float(args.duration)
                )
            if (i + 1) % log_ticks == 0:
                elapsed = (i + 1) // tick_hz
                print(f"  {elapsed}/{args.duration} s elapsed ...")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
