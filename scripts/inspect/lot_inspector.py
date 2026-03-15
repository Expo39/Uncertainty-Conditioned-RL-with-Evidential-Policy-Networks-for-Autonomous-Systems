"""
@file lot_inspector.py
@brief Entry point for the unified CARLA parking lot and sensor inspector.

Parses command-line arguments, builds a :class:`CARLAParkingEnv`, and dispatches
to the appropriate inspector class from :mod:`scripts.inspect._inspectors`.

Class hierarchy (defined in ``_inspectors.py``):
  _Inspector          -- CARLA connection, world tick loop, spectator placement
    LayoutInspector   -- lot bay outlines, spawn/patrol/pedestrian overlays
      SensorInspector -- sensor mount dots + LiDAR/camera FOV arcs on top of layout
    LiveInspector     -- real spawned sensors: LiDAR debug dots or camera spectator view

Drawing helpers (free functions) are in :mod:`scripts.inspect._drawing`.

Usage (via Make targets):
  make docker-inspect INSPECT_LAYOUT=rectangle          # Layout only
  make docker-inspect-sensors INSPECT_SUITE=suite_a     # Sensors on layout
  make docker-inspect-sensors INSPECT_SUITE=suite_b INSPECT_LAYOUT=trapezoid
  make docker-inspect-live INSPECT_SUITE=suite_a        # Live LiDAR feed
  make docker-inspect-live INSPECT_SUITE=suite_c        # Live camera view (default)
  make docker-inspect-live INSPECT_SUITE=suite_c INSPECT_SENSOR=lidar  # Override to LiDAR

Or directly:
  python -m scripts.inspect.lot_inspector --mode layout   --layout trapezoid
  python -m scripts.inspect.lot_inspector --mode sensors  --suite suite_a --layout rectangle
  python -m scripts.inspect.lot_inspector --mode sensors  --suite suite_c --layout irregular_a
  python -m scripts.inspect.lot_inspector --mode live     --suite suite_a
  python -m scripts.inspect.lot_inspector --mode live     --suite suite_c
  python -m scripts.inspect.lot_inspector --mode live     --suite suite_c --sensor lidar

Arguments:
  --mode       layout | sensors | live (default: sensors)
  --layout     rectangle | trapezoid | irregular_a (default: rectangle)
  --suite      suite_a | suite_b | suite_c (default: suite_a, sensors/live modes only)
  --view       birds_eye | side | front (default: birds_eye, sensors mode only)
  --sensor     lidar | camera (live mode, suite_c only; suite_c defaults to camera)
  --host       CARLA server hostname (default: carla-server-demo)
  --port       CARLA server port (default: 2100)
  --duration   Seconds to hold the scene (default: 300)

@note Runs in synchronous CARLA mode.  Requires the full Docker stack.
@note CARLA 0.9.16 draw_line ignores colour -- overlays use draw_point at small
      spacing to simulate coloured lines.
"""

import argparse
import sys
from typing import Any, Dict, Optional

import yaml

try:
    import carla  # noqa: F401 -- imported for the sys.exit guard below
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect._inspectors import (
    _Inspector,
    LayoutInspector,
    LiveInspector,
    SensorInspector,
)
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv


# ===========================================================================
# Entry point
# ===========================================================================


def _build_env(
    host: str,
    port: int,
    layout: str,
    train_cfg: Dict[str, Any],
    sensors_cfg: Optional[Dict[str, Any]] = None,
    suite: Optional[str] = None,
    full_lot: bool = True,
) -> CARLAParkingEnv:
    """
    @brief Build and reset a CARLAParkingEnv configured for the inspector.

    @param host: CARLA server hostname.
    @param port: CARLA server port.
    @param layout: Floor plan name ('rectangle', 'trapezoid', 'irregular_a').
    @param train_cfg: Full train_config.yaml dict.
    @param sensors_cfg: Optional sensors config dict (for sensor mode).
    @param suite: Sensor suite name (for sensor mode).
    @param full_lot: If True spawn full lot (bays, NPCs, cones).  If False,
                     suppress NPCs/cones to keep scene clean for sensor mode.
    @return Pre-reset CARLAParkingEnv instance.
    """
    scenarios = dict(train_cfg.get("parking_scenarios", {}))
    scenarios["floor_plans"] = {
        layout: {
            "weight": 1.0,
            "always_empty": [],
            "layout_file": f"configs/layouts/{layout}.yaml",
        }
    }
    if not full_lot:
        scenarios["bay_occupancy_rate"] = 0.0
        scenarios["num_patrol_vehicles_max"] = 0
        scenarios["num_pedestrians_max"] = 0
        scenarios["perimeter_cone_spacing"] = 9999.0

    env = CARLAParkingEnv(
        carla_host=host,
        carla_port=port,
        town="FlatPlane",
        parking_scenarios_config=scenarios,
        carla_sensors_config=sensors_cfg,
        sensor_suite=suite,
        include_covariance=False,
        include_obstacle_obs=False,
    )
    return env


def _resolve_live_sensor(suite: str, sensor_arg: str) -> str:
    """
    @brief Resolve which sensor to display in live mode.

    suite_a / suite_b always use lidar (no camera available).
    suite_c defaults to camera unless the user explicitly passes --sensor lidar.

    @param suite: Sensor suite name ('suite_a', 'suite_b', 'suite_c').
    @param sensor_arg: Value of the --sensor CLI argument.
    @return 'camera' or 'lidar'.
    """
    if suite == "suite_c" and sensor_arg != "lidar":
        return "camera"
    return "lidar"


def main() -> None:
    """@brief Parse arguments and run the selected inspector mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified CARLA parking lot + sensor inspector.  "
            "Shows sensors in the context of the actual parking lot layout."
        )
    )
    parser.add_argument(
        "--mode",
        default="sensors",
        choices=["layout", "sensors", "live"],
        help=(
            "Inspector mode: 'layout' = lot geometry only; "
            "'sensors' = sensors on lot; "
            "'live' = real spawned sensors with live output "
            "(LiDAR debug dots, or camera spectator view for suite_c).  "
            "Default: sensors."
        ),
    )
    parser.add_argument(
        "--layout",
        default="rectangle",
        choices=["rectangle", "trapezoid", "irregular_a"],
        help="Floor plan to spawn (default: rectangle).",
    )
    parser.add_argument(
        "--suite",
        default="suite_a",
        choices=["suite_a", "suite_b", "suite_c"],
        help="Sensor suite to visualise (sensors mode only, default: suite_a).",
    )
    parser.add_argument(
        "--view",
        default="birds_eye",
        choices=["birds_eye", "side", "front"],
        help=(
            "Spectator view for sensors mode only: "
            "'birds_eye' = top-down showing FOV arcs against the lot (default); "
            "'side' = left-profile showing sensor mount heights; "
            "'front' = front-profile showing sensor lateral positions.  "
            "Not used in live mode."
        ),
    )
    parser.add_argument(
        "--sensor",
        default="lidar",
        choices=["lidar", "camera"],
        help=(
            "Active sensor for live mode, suite_c only: "
            "'lidar' = birds-eye + red LiDAR debug dots; "
            "'camera' = CARLA spectator locked to camera mount (no dots).  "
            "suite_c defaults to camera; suite_a/b always use lidar.  "
            "Default: lidar."
        ),
    )
    parser.add_argument(
        "--zoom",
        default="close",
        choices=["close", "wide"],
        help=(
            "Camera height (birds_eye only): "
            "'close' = 80 m above vehicle (lot detail, default); "
            "'wide' = high enough to see the full FOV arc boundary."
        ),
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

    with open("configs/train_config.yaml", "r") as _f:
        train_cfg = yaml.safe_load(_f)

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    print(f"Mode: {args.mode}  |  Layout: {args.layout}", end="")
    if args.mode == "sensors":
        print(f"  |  Suite: {args.suite}  |  View: {args.view}", end="")
    elif args.mode == "live":
        active_sensor = _resolve_live_sensor(args.suite, args.sensor)
        print(f"  |  Suite: {args.suite}  |  Sensor: {active_sensor}", end="")
    print()

    if args.mode == "layout":
        env = _build_env(args.host, args.port, args.layout, train_cfg, full_lot=True)
        env.reset()
        if env.world is None:
            print("ERROR: Could not connect to CARLA.")
            env.close()
            sys.exit(1)

        inspector: _Inspector = LayoutInspector(env, args.duration)
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Debug overlays:")
        print("  blue=perpendicular bays | yellow=angled bays | violet=parallel bays")
        print(
            "  bright green=TARGET | yellow=SPAWN"
            " | turquoise=pedestrian zones | red=patrol path"
        )

    elif args.mode == "sensors":
        sensors_cfg = dict(train_cfg.get("carla_sensors", {}))
        sensors_cfg["sensor_suite"] = args.suite

        env = _build_env(
            args.host, args.port, args.layout, train_cfg,
            sensors_cfg=sensors_cfg,
            suite=args.suite,
            # Side/front views: ego only. Birds-eye: full lot for context.
            full_lot=(args.view == "birds_eye"),
        )
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        inspector = SensorInspector(
            env, args.duration, args.suite, train_cfg, args.view, args.zoom
        )
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Layout overlays:")
        print("  blue=perpendicular | yellow=angled | violet=parallel | green=TARGET")
        print("Sensor overlays:")
        print("  yellow=IMU | cyan=2D LiDAR | green=3D LiDAR | orange=RGB camera")
        print("FOV arcs:")
        print("  light-blue arc = 270 deg (Suite A) | green ring = 360 deg (Suite B/C)")
        print("  orange arc = 90 deg camera (Suite C)")

    else:  # live
        sensors_cfg = dict(train_cfg.get("carla_sensors", {}))
        sensors_cfg["sensor_suite"] = args.suite

        live_sensor = _resolve_live_sensor(args.suite, args.sensor)

        env = _build_env(
            args.host, args.port, args.layout, train_cfg,
            sensors_cfg=sensors_cfg,
            suite=args.suite,
            full_lot=True,  # Spawn lot so LiDAR / camera has scene context
        )
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        inspector = LiveInspector(
            env, args.duration, args.suite, train_cfg, live_sensor
        )
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Live sensor mode:")
        if live_sensor == "camera":
            print("  RGB camera -- CARLA spectator locked to camera mount position.")
        else:
            print("  LiDAR hit points -> red debug dots in CARLA world.")

    try:
        inspector.run()
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
