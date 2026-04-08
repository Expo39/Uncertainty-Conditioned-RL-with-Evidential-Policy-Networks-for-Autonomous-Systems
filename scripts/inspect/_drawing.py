"""
@file _drawing.py
@brief Internal CARLA debug-geometry drawing helpers for the lot inspector.

Contains all free functions that place coloured dots on the CARLA debug overlay:
  - _draw_dotted_segment  -- dotted line between two world-frame points
  - _draw_layout_overlays -- bay outlines, spawn points, patrol path, pedestrian zones
  - _draw_sensor_dot      -- labelled dot at a sensor mount position
  - _draw_fov_arc         -- bold arc boundary for a sensor FOV wedge or ring
  - _draw_sensor_overlays -- composite sensor overlay (all mounts + FOV arcs)

All colour constants are derived here from ``scripts.colours`` and exported so
that ``_inspectors.py`` can use them directly without re-importing.

@note This module is internal -- import via ``scripts.inspect._drawing``.
@note CARLA 0.9.16 ``draw_line`` ignores colour; all lines are simulated with
      closely-spaced ``draw_point`` calls via ``_draw_dotted_segment``.
"""

import math
import sys
from typing import Any, Dict, List, Tuple

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.colours import (
    BAY_HEX,
    HEX_LOT,
    HEX_PATROL_PATH,
    HEX_PEDESTRIAN_ZONE,
    HEX_SENSOR_CAMERA,
    HEX_SENSOR_FOV_BLIND,
    HEX_SENSOR_FOV_LIDAR,
    HEX_SENSOR_IMU,
    HEX_SENSOR_LIDAR_2D,
    HEX_SENSOR_LIDAR_3D,
    HEX_TARGET_BAY,
    hex_to_carla_color,
)
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
from uncertainty_rl.utils.geometry import zone_bbox

# ---------------------------------------------------------------------------
# Dot-drawing constants
# ---------------------------------------------------------------------------

_DOT_SPACING: float = 0.4  # metres between adjacent dot centres (layout lines)
_ARC_SPACING: float = 0.3  # metres between dot centres on FOV arcs

# ---------------------------------------------------------------------------
# Derived colour constants
# ---------------------------------------------------------------------------

_COL_IMU = hex_to_carla_color(HEX_SENSOR_IMU)
_COL_LIDAR_2D = hex_to_carla_color(HEX_SENSOR_LIDAR_2D)
_COL_LIDAR_3D = hex_to_carla_color(HEX_SENSOR_LIDAR_3D)
_COL_CAMERA = hex_to_carla_color(HEX_SENSOR_CAMERA)
_COL_FOV_LIDAR = hex_to_carla_color(HEX_SENSOR_FOV_LIDAR)
_COL_FOV_BLIND = hex_to_carla_color(HEX_SENSOR_FOV_BLIND)
_COL_TARGET = hex_to_carla_color(HEX_TARGET_BAY)
_COL_PED = hex_to_carla_color(HEX_PEDESTRIAN_ZONE)
_COL_PATROL = hex_to_carla_color(HEX_PATROL_PATH)
_COL_LOT = hex_to_carla_color(HEX_LOT)


# ===========================================================================
# Layout overlay drawing functions
# ===========================================================================


def _draw_dotted_segment(
    debug: Any,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    z: float,
    colour: Any,
    dot_size: float,
    life_time: float,
) -> None:
    """
    @brief Draw a dotted line segment between two world-frame points.
    @param debug: carla.DebugHelper.
    @param ax: Segment start X.
    @param ay: Segment start Y.
    @param bx: Segment end X.
    @param by: Segment end Y.
    @param z: World Z coordinate for all dots.
    @param colour: carla.Color.
    @param dot_size: Dot marker radius.
    @param life_time: Primitive lifetime in seconds.
    """
    dx = bx - ax
    dy = by - ay
    length = math.sqrt(dx * dx + dy * dy)
    n_pts = max(2, int(math.ceil(length / _DOT_SPACING)))
    for s in range(n_pts):
        t = s / (n_pts - 1)
        debug.draw_point(
            carla.Location(x=ax + t * dx, y=ay + t * dy, z=z),
            size=dot_size,
            color=colour,
            life_time=life_time,
        )


def _draw_layout_overlays(
    world: Any,
    layout: Dict[str, Any],
    target_bay_id: str,
    life_time: float,
) -> None:
    """
    @brief Draw all lot geometry overlays (bays, spawn, patrol, pedestrian zones).

    @param world: Active carla.World.
    @param layout: Loaded layout dict from the YAML file.
    @param target_bay_id: ID of the current target bay (highlighted bright green).
    @param life_time: Primitive lifetime in seconds.
    """
    debug = world.debug
    z = float(layout.get("origin", {}).get("z", 0.3)) + 0.15

    type_colours = {k: hex_to_carla_color(v) for k, v in BAY_HEX.items()}

    # --- Bay outlines ---
    for bay_idx, bay in enumerate(layout.get("bays", [])):
        bay_type = bay.get("bay_type", "perpendicular")
        is_target = bay.get("id", bay.get("bay_id", "")) == target_bay_id
        colour = (
            _COL_TARGET
            if is_target
            else type_colours.get(bay_type, carla.Color(r=120, g=120, b=120))
        )
        bx = float(bay["x"])
        by = float(bay["y"])
        width = float(bay.get("width", 2.5))
        depth = float(bay.get("depth", 5.0))
        yaw_rad = math.radians(
            float(bay.get("yaw_deg", math.degrees(float(bay.get("yaw", 0.0)))))
        )
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        hd = depth / 2.0
        hw = width / 2.0
        local: List[Tuple[float, float]] = [(hd, hw), (hd, -hw), (-hd, -hw), (-hd, hw)]
        corners_world = [
            (bx + lx * cos_y - ly * sin_y, by + lx * sin_y + ly * cos_y)
            for lx, ly in local
        ]
        pt_size = 0.08 if is_target else 0.05
        pt_z = z + 0.05 if is_target else z
        for j in range(4):
            ax, ay = corners_world[j]
            bxc, byc = corners_world[(j + 1) % 4]
            _draw_dotted_segment(
                debug, ax, ay, bxc, byc, pt_z, colour, pt_size, life_time
            )

        label = "TARGET" if is_target else str(bay_idx)
        debug.draw_string(
            carla.Location(x=bx, y=by, z=z + 1.5),
            label,
            color=colour,
            life_time=life_time,
        )

    # --- Lot perimeter boundary ---
    # Always drawn as a dotted line so lot geometry is visible in the inspector
    # regardless of whether spawn_perimeter_cones is true or false.
    lot_corners_raw = layout.get("corners", [])
    if lot_corners_raw:
        lot_corners: List[Tuple[float, float]] = [
            (float(c["x"]), float(c["y"])) for c in lot_corners_raw
        ]
        n_corners = len(lot_corners)
        for j in range(n_corners):
            ax, ay = lot_corners[j]
            bxc, byc = lot_corners[(j + 1) % n_corners]
            _draw_dotted_segment(
                debug, ax, ay, bxc, byc, z + 0.1, _COL_LOT, 0.06, life_time
            )

    # --- Spawn point ---
    spawn = layout.get("spawn_transform", {})
    sx = float(spawn.get("x", 0.0))
    sy = float(spawn.get("y", 0.0))
    _col_spawn = carla.Color(r=255, g=255, b=0)
    debug.draw_point(
        carla.Location(x=sx, y=sy, z=z + 0.3),
        size=0.2,
        color=_col_spawn,
        life_time=life_time,
    )
    debug.draw_string(
        carla.Location(x=sx, y=sy, z=z + 1.0),
        "SPAWN",
        color=_col_spawn,
        life_time=life_time,
    )

    # --- Extra spawn points ---
    _col_extra = carla.Color(r=255, g=140, b=0)
    for i, extra in enumerate(layout.get("extra_spawn_transforms", [])):
        ex = float(extra.get("x", 0.0))
        ey = float(extra.get("y", 0.0))
        debug.draw_point(
            carla.Location(x=ex, y=ey, z=z + 0.3),
            size=0.15,
            color=_col_extra,
            life_time=life_time,
        )
        debug.draw_string(
            carla.Location(x=ex, y=ey, z=z + 1.0),
            f"SPAWN{i + 2}",
            color=_col_extra,
            life_time=life_time,
        )

    # --- Pedestrian zones ---
    for zone_idx, zone_raw in enumerate(layout.get("pedestrian_zones", [])):
        x_min, x_max, y_min, y_max = zone_bbox(zone_raw)
        zc = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
        for j in range(4):
            ax, ay = zc[j]
            bxc, byc = zc[(j + 1) % 4]
            _draw_dotted_segment(debug, ax, ay, bxc, byc, z, _COL_PED, 0.05, life_time)
        zone_cx = (x_min + x_max) / 2.0
        zone_cy = (y_min + y_max) / 2.0
        debug.draw_string(
            carla.Location(x=zone_cx, y=zone_cy, z=z + 1.5),
            f"PED {zone_idx}",
            color=_COL_PED,
            life_time=life_time,
        )

    # --- Patrol waypoints ---
    waypoints: List[Tuple[float, float]] = [
        (float(wp["x"]), float(wp["y"])) for wp in layout.get("patrol_waypoints", [])
    ]
    for i, (wx, wy) in enumerate(waypoints):
        debug.draw_point(
            carla.Location(x=wx, y=wy, z=z + 0.2),
            size=0.12,
            color=_COL_PATROL,
            life_time=life_time,
        )
        if len(waypoints) > 1:
            nx, ny = waypoints[(i + 1) % len(waypoints)]
            _draw_dotted_segment(
                debug, wx, wy, nx, ny, z + 0.2, _COL_PATROL, 0.04, life_time
            )


# ===========================================================================
# Sensor overlay drawing functions
# ===========================================================================


def _draw_sensor_dot(
    debug: Any,
    loc: Any,
    label: str,
    colour: Any,
    life_time: float,
    drop_line: bool = False,
    ground_z: float = 0.3,
) -> None:
    """
    @brief Draw a labelled dot at a sensor mount position.
    @param debug: carla.DebugHelper.
    @param loc: carla.Location of the sensor mount.
    @param label: Text label rendered above the dot.
    @param colour: carla.Color.
    @param life_time: Primitive lifetime in seconds.
    @param drop_line: If True, draw a vertical dotted line from mount to ground.
                      Useful in side-profile view to show mount height clearly.
    @param ground_z: Ground Z for the drop line base (default 0.3).
    """
    debug.draw_point(loc, size=0.35, color=colour, life_time=life_time)
    if drop_line and loc.z > ground_z + 0.1:
        _col_white = carla.Color(r=255, g=255, b=255)
        n_pts = max(2, int(math.ceil((loc.z - ground_z) / 0.12)))
        for i in range(n_pts + 1):
            t = i / n_pts
            debug.draw_point(
                carla.Location(x=loc.x, y=loc.y, z=ground_z + t * (loc.z - ground_z)),
                size=0.12,
                color=_col_white,
                life_time=life_time,
            )
    debug.draw_string(
        carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.7),
        label,
        color=colour,
        life_time=life_time,
    )


def _draw_fov_arc(
    debug: Any,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    radius: float,
    angle_min_rad: float,
    angle_max_rad: float,
    colour: Any,
    life_time: float,
    dot_size: float = 0.22,
    draw_radials: bool = True,
) -> None:
    """
    @brief Draw a single bold arc boundary for a sensor FOV wedge or ring.

    Places dots at ``_ARC_SPACING`` intervals along the arc at ``radius``.
    Radial boundary lines run from the origin to each arc endpoint.

    @param debug: carla.DebugHelper.
    @param origin_x: Arc centre X (world frame).
    @param origin_y: Arc centre Y (world frame).
    @param origin_z: Arc centre Z (world frame).
    @param radius: Arc radius in metres.
    @param angle_min_rad: Start angle in radians (0 = +X axis = forward in CARLA).
    @param angle_max_rad: End angle in radians.
    @param colour: carla.Color.
    @param life_time: Primitive lifetime in seconds.
    @param dot_size: Dot marker radius.  Larger = more visible against ground.
    @param draw_radials: If True, draw radial lines from origin to arc endpoints.
    """
    arc_span = angle_max_rad - angle_min_rad
    arc_len = abs(arc_span) * radius
    n_pts = max(4, int(math.ceil(arc_len / _ARC_SPACING)))
    for i in range(n_pts + 1):
        t = i / n_pts
        angle = angle_min_rad + t * arc_span
        debug.draw_point(
            carla.Location(
                x=origin_x + radius * math.cos(angle),
                y=origin_y + radius * math.sin(angle),
                z=origin_z,
            ),
            size=dot_size,
            color=colour,
            life_time=life_time,
        )

    if draw_radials:
        for angle in (angle_min_rad, angle_max_rad):
            px_end = origin_x + radius * math.cos(angle)
            py_end = origin_y + radius * math.sin(angle)
            _draw_dotted_segment(
                debug,
                origin_x,
                origin_y,
                px_end,
                py_end,
                origin_z,
                colour,
                dot_size * 0.7,
                life_time,
            )


def _draw_sensor_overlays(
    env: CARLAParkingEnv,
    train_cfg: Dict[str, Any],
    life_time: float,
    side_view: bool = False,
) -> None:
    """
    @brief Draw sensor mount dots and FOV arcs for the current vehicle pose.

    Reads mount positions from ``train_cfg`` (``carla_sensors`` section) and draws:
      - IMU: yellow dot at centre-of-mass height
      - GNSS: magenta dot at roof antenna mount
      - 2D LiDAR: cyan dot at front bumper + 270 deg FOV arc + faint 90 deg blind sector

    All positions are read from ``carla_sensors.<sensor>.mount`` in ``train_cfg``
    and transformed from vehicle body frame to world frame using the current
    vehicle transform.

    @param env: Active CARLAParkingEnv (vehicle must be spawned).
    @param train_cfg: Loaded train_config.yaml dict.
    @param life_time: Primitive lifetime in seconds.
    @param side_view: If True, draw vertical drop lines from each sensor mount
                      to the ground to show mount heights clearly.
    """
    if env.vehicle is None or env.world is None:
        return

    debug = env.world.debug
    sensors_cfg = train_cfg.get("carla_sensors", {})
    vt = env.vehicle.get_transform()
    vx = vt.location.x
    vy = vt.location.y
    vz = vt.location.z
    ground_z = vz + 0.05
    yaw_rad = math.radians(vt.rotation.yaw)

    def _to_world(lx: float, ly: float, lz: float) -> Any:
        """@brief Transform vehicle body-frame offset to world-frame carla.Location."""
        wx = vx + lx * math.cos(yaw_rad) - ly * math.sin(yaw_rad)
        wy = vy + lx * math.sin(yaw_rad) + ly * math.cos(yaw_rad)
        return carla.Location(x=wx, y=wy, z=vz + lz)

    # ---- IMU ---------------------------------------------------------------
    imu_m = sensors_cfg.get("imu", {}).get("mount", {})
    imu_loc = _to_world(
        float(imu_m.get("x", 0.0)),
        float(imu_m.get("y", 0.0)),
        float(imu_m.get("z", 0.3)),
    )
    _draw_sensor_dot(
        debug,
        imu_loc,
        "IMU",
        _COL_IMU,
        life_time,
        drop_line=side_view,
        ground_z=ground_z,
    )

    # ---- 2D LiDAR (obstacle detection) ------------------------------------
    lid_m = sensors_cfg.get("lidar", {}).get("mount", {})
    lx = float(lid_m.get("x", 2.4))
    ly = float(lid_m.get("y", 0.0))
    lz = float(lid_m.get("z", 0.5))
    lidar_loc = _to_world(lx, ly, lz)
    _draw_sensor_dot(
        debug,
        lidar_loc,
        "2D LiDAR",
        _COL_LIDAR_2D,
        life_time,
        drop_line=side_view,
        ground_z=ground_z,
    )

    if not side_view:
        lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))
        fov_half = math.radians(135.0)
        _draw_fov_arc(
            debug,
            lidar_loc.x,
            lidar_loc.y,
            lidar_loc.z + 0.05,
            radius=lidar_range,
            angle_min_rad=yaw_rad - fov_half,
            angle_max_rad=yaw_rad + fov_half,
            colour=_COL_FOV_LIDAR,
            life_time=life_time,
            dot_size=0.05,
            draw_radials=True,
        )
        _draw_fov_arc(
            debug,
            lidar_loc.x,
            lidar_loc.y,
            lidar_loc.z + 0.05,
            radius=lidar_range,
            angle_min_rad=yaw_rad + fov_half,
            angle_max_rad=yaw_rad + math.radians(360.0) - fov_half,
            colour=_COL_FOV_BLIND,
            life_time=life_time,
            dot_size=0.03,
            draw_radials=False,
        )

