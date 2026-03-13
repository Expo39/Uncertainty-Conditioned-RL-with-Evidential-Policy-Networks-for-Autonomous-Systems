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

import yaml

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found. Run inside the training container.")
    sys.exit(1)

from scripts.layouts.colours import BAY_HEX, HEX_PATROL_PATH, HEX_PEDESTRIAN_ZONE, HEX_TARGET_BAY, hex_to_carla_color
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
from uncertainty_rl.utils.geometry import zone_bbox


# ---------------------------------------------------------------------------
# Debug overlay helpers (inspect-only, not used during training)
# ---------------------------------------------------------------------------


def _draw_inspect_overlays(
    world: Any,
    layout: Dict[str, Any],
    target_bay_id: str,
    life_time: float = 0.1,
) -> None:
    """
    @brief Draw all layout debug overlays.

    Designed to be called every tick with a short life_time (0.1 s = 2 frames
    at 20 Hz). Using a long life_time causes CARLA's server-side debug buffer
    to overflow -- older entries are evicted and the overlay becomes patchy.
    Redrawing each tick with a short life_time keeps the buffer small and all
    geometry visible at all times.

    @param world: Active carla.World instance.
    @param layout: Loaded layout dict from the YAML file.
    @param target_bay_id: ID of the selected target bay (highlighted green).
    @param life_time: How long each primitive persists (seconds). Default 0.1.
    """
    debug = world.debug
    life = life_time
    z = float(layout.get("origin", {}).get("z", 0.3)) + 0.15
    # draw_line ignores color in CARLA 0.9.16 -- use draw_point at 0.5 m spacing instead.
    _DOT_SPACING = 0.5

    # Bay outlines coloured by type
    type_colours = {k: hex_to_carla_color(v) for k, v in BAY_HEX.items()}
    _target_colour = hex_to_carla_color(HEX_TARGET_BAY)
    _ped_colour = hex_to_carla_color(HEX_PEDESTRIAN_ZONE)
    _patrol_colour = hex_to_carla_color(HEX_PATROL_PATH)

    for bay_idx, bay in enumerate(layout.get("bays", [])):
        bay_type = bay.get("bay_type", "perpendicular")
        is_target = bay.get("id", bay.get("bay_id", "")) == target_bay_id
        colour = _target_colour if is_target else type_colours.get(
            bay_type, carla.Color(r=120, g=120, b=120)
        )
        thickness = 0.10 if is_target else 0.05

        bx = float(bay["x"])
        by = float(bay["y"])
        width = float(bay.get("width", 2.5))
        depth = float(bay.get("depth", 5.0))
        yaw_rad = math.radians(
            float(bay.get("yaw_deg", math.degrees(bay.get("yaw", 0.0))))
        )

        # Compute the four corners in world space by rotating the local half-extents.
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        hd = depth / 2.0
        hw = width / 2.0
        local = [(hd, hw), (hd, -hw), (-hd, -hw), (-hd, hw)]
        corners_world = [
            carla.Location(
                x=bx + lx * cos_y - ly * sin_y,
                y=by + lx * sin_y + ly * cos_y,
                z=z,
            )
            for lx, ly in local
        ]
        pt_size = 0.08 if is_target else 0.05
        # Draw target bay slightly higher so it renders on top of other bay outlines.
        pt_z = z + 0.05 if is_target else z
        for j in range(4):
            a = corners_world[j]
            b = corners_world[(j + 1) % 4]
            edge_dx = b.x - a.x
            edge_dy = b.y - a.y
            edge_len = math.sqrt(edge_dx * edge_dx + edge_dy * edge_dy)
            n_pts = max(2, int(math.ceil(edge_len / _DOT_SPACING)))
            for s in range(n_pts):
                t = s / (n_pts - 1)
                debug.draw_point(
                    carla.Location(x=a.x + t * edge_dx, y=a.y + t * edge_dy, z=pt_z),
                    size=pt_size,
                    color=colour,
                    life_time=life,
                )

        # Label: use bay colour for normal bays, target green for the target bay.
        label = "T" if is_target else str(bay_idx)
        label_colour = _target_colour if is_target else colour
        debug.draw_string(
            carla.Location(x=bx, y=by, z=z + 1.5),
            label,
            color=label_colour,
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

    # Pedestrian zones
    for zone_idx, zone_raw in enumerate(layout.get("pedestrian_zones", [])):
        x_min, x_max, y_min, y_max = zone_bbox(zone_raw)
        zone_corners = [
            carla.Location(x=x_min, y=y_min, z=z),
            carla.Location(x=x_max, y=y_min, z=z),
            carla.Location(x=x_max, y=y_max, z=z),
            carla.Location(x=x_min, y=y_max, z=z),
        ]
        for j in range(4):
            a = zone_corners[j]
            b = zone_corners[(j + 1) % 4]
            edge_dx = b.x - a.x
            edge_dy = b.y - a.y
            edge_len = math.sqrt(edge_dx * edge_dx + edge_dy * edge_dy)
            n_pts = max(2, int(math.ceil(edge_len / _DOT_SPACING)))
            for s in range(n_pts):
                t = s / (n_pts - 1)
                debug.draw_point(
                    carla.Location(x=a.x + t * edge_dx, y=a.y + t * edge_dy, z=z),
                    size=0.05,
                    color=_ped_colour,
                    life_time=life,
                )
        # Label at zone centre
        zone_cx = (x_min + x_max) / 2.0
        zone_cy = (y_min + y_max) / 2.0
        debug.draw_string(
            carla.Location(x=zone_cx, y=zone_cy, z=z + 1.5),
            f"PED {zone_idx}",
            color=_ped_colour,
            life_time=life,
        )

    # Patrol waypoints -- red dots connected by segmented lines.
    # Lines are split into 4 m segments so CARLA's midpoint-based cull never
    # drops a segment whose midpoint is close to the spectator.
    waypoints: List[Tuple[float, float]] = [
        (float(wp["x"]), float(wp["y"]))
        for wp in layout.get("patrol_waypoints", [])
    ]
    for i, (wx, wy) in enumerate(waypoints):
        debug.draw_point(
            carla.Location(x=wx, y=wy, z=z + 0.2),
            size=0.12,
            color=_patrol_colour,
            life_time=life,
        )
        if len(waypoints) > 1:
            nx, ny = waypoints[(i + 1) % len(waypoints)]
            seg_dx = nx - wx
            seg_dy = ny - wy
            seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)
            n_pts = max(2, int(math.ceil(seg_len / _DOT_SPACING)))
            for s in range(n_pts):
                t = s / (n_pts - 1)
                debug.draw_point(
                    carla.Location(x=wx + t * seg_dx, y=wy + t * seg_dy, z=z + 0.2),
                    size=0.04,
                    color=_patrol_colour,
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

    # Load parking_scenarios from train_config.yaml so inspection matches training
    # conditions exactly (same pedestrian count, bay occupancy, patrol vehicles, etc.).
    with open("configs/train_config.yaml", "r") as _f:
        _train_cfg = yaml.safe_load(_f)
    scenarios = dict(_train_cfg.get("parking_scenarios", {}))
    # Override floor_plans to show only the requested layout.
    scenarios["floor_plans"] = {
        args.layout: {
            "weight": 1.0,
            "always_empty": [],
            "layout_file": f"configs/layouts/{args.layout}.yaml",
        }
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

    # Position spectator above lot centroid so all debug geometry stays in view.
    # CARLA culls debug geometry beyond ~60 m from the spectator camera.
    if env._current_layout:
        corners = env._current_layout.get("corners", [])
        if corners:
            xs = [float(c["x"]) for c in corners]
            ys = [float(c["y"]) for c in corners]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            # Set height so the full lot fits in view. CARLA's spectator FOV is
            # roughly 90 deg, so height ~= max(span_x, span_y) * 1.1 ensures all
            # corners remain within the ~60 m debug-geometry cull radius.
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            cam_z_offset = max(span * 1.1, 80.0)
        else:
            spawn = env._current_layout.get("spawn_transform", {})
            cx = float(spawn.get("x", 0.0))
            cy = float(spawn.get("y", 0.0))
            cam_z_offset = 80.0
        sz = float(env._current_layout.get("origin", {}).get("z", 0.3))
        spectator = env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cx, y=cy, z=sz + cam_z_offset),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
        )
        print(f"Spectator at bbox centre ({cx:.0f}, {cy:.0f}, {sz + cam_z_offset:.0f}) -- bird's-eye view.")

    target_bay_id = env._target_bay.get("bay_id", "")
    print("Debug overlays active (redrawn every tick):")
    print("  blue=perpendicular bays, yellow=angled bays, violet=parallel bays")
    print("  bright green=TARGET bay, yellow=spawn, turquoise=pedestrian zones, red=patrol path")

    # Tick at 20 Hz. Overlays are redrawn every 3 s with life_time=3.5 s so
    # they persist between redraws without flickering, yet the buffer never
    # accumulates enough entries to overflow and drop older primitives.
    tick_hz = 20
    total_ticks = args.duration * tick_hz
    overlay_redraw_ticks = 3 * tick_hz   # redraw every 3 s
    log_ticks = 10 * tick_hz

    print(f"Scene live for {args.duration}s. Press Ctrl+C to exit early.")
    try:
        for i in range(total_ticks):
            env.world.tick()
            env._update_patrol_npcs()
            env._update_pedestrians()
            if i % overlay_redraw_ticks == 0:
                _draw_inspect_overlays(
                    env.world, env._current_layout, target_bay_id, life_time=3.5
                )
            time.sleep(1.0 / tick_hz)
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
