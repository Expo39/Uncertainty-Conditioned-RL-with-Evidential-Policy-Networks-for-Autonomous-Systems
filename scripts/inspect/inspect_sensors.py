"""
@file inspect_sensors.py
@brief CARLA sensor-placement inspector for all sensor suites.

Spawns the ego vehicle in CARLA with the selected sensor suite and draws debug
overlays showing:
  - Sensor attach points as labelled coloured dots at mount positions
  - LiDAR 270 deg FOV arc overlay (Suite A) or 360 deg ring (Suite B/C)
  - Spectator positioned in bird's-eye view above the vehicle

Use this to visually confirm that sensors are mounted at the correct positions
before running a full training or evaluation session.

Usage:
  make docker-inspect-sensors [INSPECT_SUITE=suite_a]
  make docker-inspect-sensors INSPECT_SUITE=suite_b
  make docker-inspect-sensors INSPECT_SUITE=suite_c

@note Runs in synchronous CARLA mode. Requires the full Docker stack:
      make docker-up (or the demo CARLA server used by docker-inspect).
@note CARLA 0.9.16 draw_line ignores colour -- sensor overlays use
      draw_point at small spacing to simulate lines.
"""

import argparse
import math
import sys
import time
from typing import Any, Dict

import yaml

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found. Run inside the training container.")
    sys.exit(1)

from scripts.colours import (
    HEX_SENSOR_CAMERA,
    HEX_SENSOR_FOV,
    HEX_SENSOR_FOV_BLIND,
    HEX_SENSOR_IMU,
    HEX_SENSOR_LIDAR_2D,
    HEX_SENSOR_LIDAR_3D,
    hex_to_carla_color,
)
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

# ---------------------------------------------------------------------------
# Colour constants for sensor types (converted from scripts.colours hex palette)
# ---------------------------------------------------------------------------

_COL_IMU = hex_to_carla_color(HEX_SENSOR_IMU)
_COL_LIDAR_2D = hex_to_carla_color(HEX_SENSOR_LIDAR_2D)
_COL_LIDAR_3D = hex_to_carla_color(HEX_SENSOR_LIDAR_3D)
_COL_CAMERA = hex_to_carla_color(HEX_SENSOR_CAMERA)
_COL_FOV = hex_to_carla_color(HEX_SENSOR_FOV)
_COL_FOV_BLIND = hex_to_carla_color(HEX_SENSOR_FOV_BLIND)

_DOT_SPACING = 0.2  # metres between dots when simulating lines


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------


def _draw_sensor_point(
    debug: Any,
    x: float,
    y: float,
    z: float,
    label: str,
    colour: Any,
    life_time: float,
) -> None:
    """
    @brief Draw a labelled dot at a sensor mount position.
    @param debug: carla.DebugHelper instance.
    @param x: World x coordinate.
    @param y: World y coordinate.
    @param z: World z coordinate.
    @param label: Text label rendered above the dot.
    @param colour: carla.Color for the dot and label.
    @param life_time: Overlay lifetime in seconds.
    """
    debug.draw_point(
        carla.Location(x=x, y=y, z=z),
        size=0.15,
        color=colour,
        life_time=life_time,
    )
    debug.draw_string(
        carla.Location(x=x, y=y, z=z + 0.6),
        label,
        color=colour,
        life_time=life_time,
    )


def _draw_arc(
    debug: Any,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    radius: float,
    angle_min_rad: float,
    angle_max_rad: float,
    colour: Any,
    life_time: float,
    dot_size: float = 0.04,
) -> None:
    """
    @brief Draw an arc (or full circle) of dots in the XY plane.

    Used to visualise LiDAR FOV coverage. Points are placed at `_DOT_SPACING`
    arc-length intervals along the arc at `radius` metres from `origin`.

    @param debug: carla.DebugHelper instance.
    @param origin_x: Arc centre X (world frame).
    @param origin_y: Arc centre Y (world frame).
    @param origin_z: Arc centre Z (world frame).
    @param radius: Arc radius in metres.
    @param angle_min_rad: Start angle in radians (0 = forward/+X axis in CARLA).
    @param angle_max_rad: End angle in radians.
    @param colour: carla.Color for the dots.
    @param life_time: Overlay lifetime in seconds.
    @param dot_size: Radius of each dot marker.
    """
    arc_len = abs(angle_max_rad - angle_min_rad) * radius
    n_pts = max(4, int(math.ceil(arc_len / _DOT_SPACING)))
    for i in range(n_pts + 1):
        t = i / n_pts
        angle = angle_min_rad + t * (angle_max_rad - angle_min_rad)
        px = origin_x + radius * math.cos(angle)
        py = origin_y + radius * math.sin(angle)
        debug.draw_point(
            carla.Location(x=px, y=py, z=origin_z),
            size=dot_size,
            color=colour,
            life_time=life_time,
        )

    # Draw radial boundary lines from origin to arc ends
    for angle in (angle_min_rad, angle_max_rad):
        px_end = origin_x + radius * math.cos(angle)
        py_end = origin_y + radius * math.sin(angle)
        n_radial = max(2, int(math.ceil(radius / _DOT_SPACING)))
        for i in range(n_radial + 1):
            t = i / n_radial
            debug.draw_point(
                carla.Location(
                    x=origin_x + t * (px_end - origin_x),
                    y=origin_y + t * (py_end - origin_y),
                    z=origin_z,
                ),
                size=dot_size * 0.7,
                color=colour,
                life_time=life_time,
            )


def _draw_sensor_overlays(
    env: CARLAParkingEnv,
    suite: str,
    train_cfg: Dict[str, Any],
    life_time: float,
) -> None:
    """
    @brief Draw all sensor mount overlays for the current vehicle pose.

    Reads mount positions from train_cfg (carla_sensors section) and draws:
      - IMU: yellow dot at centre-of-mass height
      - 2D LiDAR (Suite A): cyan dot at bumper + 270 deg FOV arc at 5 m radius
      - 3D LiDAR (Suite B/C): green dot at roof + full 360 deg ring at 5 m radius
      - RGB camera (Suite C): orange dot at windscreen + 30 deg half-cone arc

    Mount positions are in the vehicle body frame, transformed to world frame
    using the vehicle's current transform.

    @param env: Active CARLAParkingEnv instance (vehicle must be spawned).
    @param suite: Sensor suite name ('suite_a', 'suite_b', 'suite_c').
    @param train_cfg: Loaded train_config.yaml dict (for mount positions).
    @param life_time: Overlay lifetime in seconds.
    """
    if env.vehicle is None or env.world is None:
        return

    debug = env.world.debug
    sensors_cfg = train_cfg.get("carla_sensors", {})
    transform = env.vehicle.get_transform()
    vx = transform.location.x
    vy = transform.location.y
    vz = transform.location.z
    yaw_rad = math.radians(transform.rotation.yaw)

    def to_world(lx: float, ly: float, lz: float) -> carla.Location:
        """Convert vehicle-body-frame offset to world frame."""
        wx = vx + lx * math.cos(yaw_rad) - ly * math.sin(yaw_rad)
        wy = vy + lx * math.sin(yaw_rad) + ly * math.cos(yaw_rad)
        wz = vz + lz
        return carla.Location(x=wx, y=wy, z=wz)

    # --- IMU (all suites) ---
    imu_mount = sensors_cfg.get("imu", {}).get("mount", {})
    imu_loc = to_world(
        float(imu_mount.get("x", 0.0)),
        float(imu_mount.get("y", 0.0)),
        float(imu_mount.get("z", 0.3)),
    )
    _draw_sensor_point(
        debug, imu_loc.x, imu_loc.y, imu_loc.z, "IMU", _COL_IMU, life_time
    )

    # --- Suite A: 2D LiDAR ---
    if suite == "suite_a":
        lidar_mount = sensors_cfg.get("lidar", {}).get("mount", {})
        lx = float(lidar_mount.get("x", 2.4))
        ly = float(lidar_mount.get("y", 0.0))
        lz = float(lidar_mount.get("z", 0.3))
        lidar_loc = to_world(lx, ly, lz)
        _draw_sensor_point(
            debug,
            lidar_loc.x, lidar_loc.y, lidar_loc.z,
            "2D LiDAR", _COL_LIDAR_2D, life_time,
        )
        # 270 deg FOV arc centred forward. In CARLA world frame, forward is along
        # the vehicle's yaw direction. -135 to +135 deg relative to vehicle heading.
        fov_half = math.radians(135.0)
        _draw_arc(
            debug,
            lidar_loc.x, lidar_loc.y, lidar_loc.z + 0.05,
            radius=5.0,
            angle_min_rad=yaw_rad - fov_half,
            angle_max_rad=yaw_rad + fov_half,
            colour=_COL_FOV,
            life_time=life_time,
        )
        # Small dot at sensor to mark the 90 deg blind sector (rear)
        blind_arc_min = yaw_rad + fov_half
        blind_arc_max = yaw_rad + math.radians(360.0) - fov_half
        _draw_arc(
            debug,
            lidar_loc.x, lidar_loc.y, lidar_loc.z + 0.05,
            radius=5.0,
            angle_min_rad=blind_arc_min,
            angle_max_rad=blind_arc_max,
            colour=_COL_FOV_BLIND,
            life_time=life_time,
            dot_size=0.03,
        )

    # --- Suite B / C: 3D LiDAR ---
    if suite in ("suite_b", "suite_c"):
        lidar3d_mount = sensors_cfg.get("lidar_3d", {}).get("mount", {})
        lx = float(lidar3d_mount.get("x", 0.0))
        ly = float(lidar3d_mount.get("y", 0.0))
        lz = float(lidar3d_mount.get("z", 1.5))
        lidar3d_loc = to_world(lx, ly, lz)
        _draw_sensor_point(
            debug,
            lidar3d_loc.x, lidar3d_loc.y, lidar3d_loc.z,
            "3D LiDAR", _COL_LIDAR_3D, life_time,
        )
        # Full 360 deg ring at 5 m radius
        _draw_arc(
            debug,
            lidar3d_loc.x, lidar3d_loc.y, lidar3d_loc.z,
            radius=5.0,
            angle_min_rad=0.0,
            angle_max_rad=2.0 * math.pi,
            colour=_COL_LIDAR_3D,
            life_time=life_time,
        )

    # --- Suite C: RGB camera ---
    if suite == "suite_c":
        cam_mount = sensors_cfg.get("camera_rgb", {}).get("mount", {})
        cx = float(cam_mount.get("x", 2.0))
        cy_local = float(cam_mount.get("y", 0.0))
        cz = float(cam_mount.get("z", 1.2))
        cam_loc = to_world(cx, cy_local, cz)
        _draw_sensor_point(
            debug, cam_loc.x, cam_loc.y, cam_loc.z, "RGB CAM", _COL_CAMERA, life_time
        )
        # Camera FOV cone: 90 deg horizontal, centred on vehicle heading
        cam_fov_half = math.radians(45.0)
        _draw_arc(
            debug,
            cam_loc.x, cam_loc.y, cam_loc.z,
            radius=4.0,
            angle_min_rad=yaw_rad - cam_fov_half,
            angle_max_rad=yaw_rad + cam_fov_half,
            colour=_COL_CAMERA,
            life_time=life_time,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """@brief Parse arguments and run the sensor-placement inspector."""
    parser = argparse.ArgumentParser(
        description="Inspect sensor mount positions in CARLA with debug overlays."
    )
    parser.add_argument(
        "--suite",
        default="suite_a",
        choices=["suite_a", "suite_b", "suite_c"],
        help="Sensor suite to visualise (default: suite_a).",
    )
    parser.add_argument(
        "--host", default="carla-server-demo", help="CARLA server hostname."
    )
    parser.add_argument("--port", type=int, default=2100, help="CARLA server port.")
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Seconds to hold the scene (default: 120).",
    )
    args = parser.parse_args()

    with open("configs/train_config.yaml", "r") as _f:
        train_cfg = yaml.safe_load(_f)

    sensors_cfg = train_cfg.get("carla_sensors", {})
    # Inject sensor_suite so _spawn_sensors() picks the right suite.
    sensors_cfg["sensor_suite"] = args.suite

    # Use the rectangle layout for a clean minimal scene.
    scenarios = dict(train_cfg.get("parking_scenarios", {}))
    scenarios["floor_plans"] = {
        "rectangle": {
            "weight": 1.0,
            "always_empty": [],
            "layout_file": "configs/layouts/rectangle.yaml",
        }
    }

    env = CARLAParkingEnv(
        carla_host=args.host,
        carla_port=args.port,
        town="FlatPlane",
        parking_scenarios_config=scenarios,
        carla_sensors_config=sensors_cfg,
        sensor_suite=args.suite,
        include_covariance=False,
        include_obstacle_obs=False,
    )

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    env.reset()

    if env.world is None or env.vehicle is None:
        print("ERROR: Could not connect to CARLA or spawn vehicle.")
        env.close()
        sys.exit(1)

    # Position spectator above the vehicle for a clear bird's-eye view.
    spectator = env.world.get_spectator()
    vt = env.vehicle.get_transform()
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=vt.location.x, y=vt.location.y, z=vt.location.z + 12.0),
            carla.Rotation(pitch=-90.0, yaw=vt.rotation.yaw, roll=0.0),
        )
    )

    print(f"Suite: {args.suite}")
    print(
        "Sensor colours: yellow=IMU  cyan=2D LiDAR  green=3D LiDAR  orange=RGB camera"
    )
    print(
        "FOV arcs: cyan=270 deg (Suite A)  green=360 deg (Suite B/C)"
        "  orange=90 deg (camera)"
    )
    print(f"Scene live for {args.duration}s. Press Ctrl+C to exit early.")

    tick_hz = 20
    total_ticks = args.duration * tick_hz
    overlay_redraw_ticks = 3 * tick_hz
    log_ticks = 30 * tick_hz

    try:
        for i in range(total_ticks):
            env.world.tick()
            if i % overlay_redraw_ticks == 0:
                _draw_sensor_overlays(env, args.suite, train_cfg, life_time=3.5)
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
