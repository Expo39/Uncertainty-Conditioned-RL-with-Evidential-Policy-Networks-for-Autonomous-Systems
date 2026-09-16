"""
@file _drawing.py
@brief Internal CARLA debug-geometry drawing helpers for the lot inspector.
"""

import math
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.colours import (
    BAY_HEX,
    HEX_LOT,
    HEX_OOB_BOUNDARY,
    HEX_PATROL_PATH,
    HEX_PEDESTRIAN_ZONE,
    HEX_SENSOR_CAMERA,
    HEX_SENSOR_FOV_BLIND,
    HEX_SENSOR_FOV_LIDAR,
    HEX_SENSOR_GNSS,
    HEX_SENSOR_IMU,
    HEX_SENSOR_LIDAR_2D,
    HEX_SENSOR_LIDAR_3D,
    HEX_TARGET_BAY,
    hex_to_carla_color,
)
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv
from uncertainty_rl.utils.geometry import inflate_polygon, zone_bbox

# ---------------------------------------------------------------------------
# Dot-drawing constants
# ---------------------------------------------------------------------------

_DOT_SPACING: float = 0.4  # metres between adjacent dot centres (layout lines)
_ARC_SPACING: float = 0.3  # metres between dot centres on FOV arcs

# ---------------------------------------------------------------------------
# Derived colour constants
# ---------------------------------------------------------------------------

_COL_IMU = hex_to_carla_color(HEX_SENSOR_IMU)
_COL_GNSS = hex_to_carla_color(HEX_SENSOR_GNSS)
_COL_LIDAR_2D = hex_to_carla_color(HEX_SENSOR_LIDAR_2D)
_COL_LIDAR_3D = hex_to_carla_color(HEX_SENSOR_LIDAR_3D)
_COL_CAMERA = hex_to_carla_color(HEX_SENSOR_CAMERA)
_COL_FOV_LIDAR = hex_to_carla_color(HEX_SENSOR_FOV_LIDAR)
_COL_FOV_BLIND = hex_to_carla_color(HEX_SENSOR_FOV_BLIND)
_COL_TARGET = hex_to_carla_color(HEX_TARGET_BAY)
_COL_PED = hex_to_carla_color(HEX_PEDESTRIAN_ZONE)
_COL_PATROL = hex_to_carla_color(HEX_PATROL_PATH)
_COL_LOT = hex_to_carla_color(HEX_LOT)
_COL_OOB = hex_to_carla_color(HEX_OOB_BOUNDARY)
_COL_SPAWN = carla.Color(r=255, g=255, b=0)
_COL_EXTRA_SPAWN = carla.Color(r=255, g=140, b=0)
_COL_WHITE = carla.Color(r=255, g=255, b=255)
_COL_GREY = carla.Color(r=120, g=120, b=120)

# Bay-type colours
_BAY_TYPE_COLOURS: Dict[str, Any] = {
    k: hex_to_carla_color(v) for k, v in BAY_HEX.items()
}

# GNSS sensor keys -> display labels (extend if a second antenna is added).
_GNSS_LABELS: Dict[str, str] = {
    "gnss": "GNSS",
}


# ---------------------------------------------------------------------------
# Layout overlay drawing functions
# ---------------------------------------------------------------------------


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
    length = math.hypot(dx, dy)
    n_pts = max(2, int(math.ceil(length / _DOT_SPACING)))
    inv_n = 1.0 / (n_pts - 1)
    draw_point = debug.draw_point
    Location = carla.Location
    for s in range(n_pts):
        t = s * inv_n
        draw_point(
            Location(x=ax + t * dx, y=ay + t * dy, z=z),
            size=dot_size,
            color=colour,
            life_time=life_time,
        )


def _draw_layout_overlays(
    world: Any,
    layout: Dict[str, Any],
    target_bay_id: str,
    life_time: float,
    show_patrol: bool = True,
    show_pedestrians: bool = True,
    show_spawns: bool = False,
    oob_inflation_margin: Optional[float] = None,
) -> None:
    """
    @brief Draw all lot geometry overlays (bays, spawn, patrol, pedestrian zones).

    The patrol path and pedestrian zones are drawn only when the corresponding
    flags are set, so the overlay reflects the environment the agent actually
    trains in. The soft out-of-bounds boundary is drawn when a margin is given.

    @param world: Active carla.World.
    @param layout: Loaded layout dict from the YAML file.
    @param target_bay_id: ID of the current target bay (highlighted bright green).
    @param life_time: Primitive lifetime in seconds.
    @param show_patrol: Draw the patrol waypoints and path.
    @param show_pedestrians: Draw the pedestrian zones.
    @param show_spawns: Draw the spawn point and any extra spawn points. Off by
           default: only one spawn is in use (use_extra_spawns is false in every
           stage), so the markers label start positions that are never taken.
    @param oob_inflation_margin: When not None, draw the lot polygon inflated by
           this many metres as the soft out-of-bounds boundary.
    """
    debug = world.debug
    draw_point = debug.draw_point
    draw_string = debug.draw_string
    Location = carla.Location
    z = float(layout.get("origin", {}).get("z", 0.3)) + 0.15

    # Bay outlines
    for bay_idx, bay in enumerate(layout.get("bays", [])):
        bay_type = bay.get("bay_type", "perpendicular")
        is_target = bay.get("id", bay.get("bay_id", "")) == target_bay_id
        colour = (
            _COL_TARGET if is_target else _BAY_TYPE_COLOURS.get(bay_type, _COL_GREY)
        )
        bx = float(bay["x"])
        by = float(bay["y"])
        width = float(bay.get("width", 2.5))
        depth = float(bay.get("depth", 5.0))

        # Prefer yaw_deg; fall back to yaw (already in radians).
        if "yaw_deg" in bay:
            yaw_rad = math.radians(float(bay["yaw_deg"]))
        else:
            yaw_rad = float(bay.get("yaw", 0.0))

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
        draw_string(
            Location(x=bx, y=by, z=z + 1.5),
            label,
            color=colour,
            life_time=life_time,
        )

    # Lot perimeter boundary
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

        # Soft out-of-bounds boundary (lot polygon inflated by the margin)
        if oob_inflation_margin is not None:
            oob_corners = inflate_polygon(lot_corners, oob_inflation_margin)
            n_oob = len(oob_corners)
            for j in range(n_oob):
                ax, ay = oob_corners[j]
                bxc, byc = oob_corners[(j + 1) % n_oob]
                _draw_dotted_segment(
                    debug, ax, ay, bxc, byc, z + 0.1, _COL_OOB, 0.06, life_time
                )

    # Spawn points. The extra spawns exist in the layout YAML but are only used
    # when use_extra_spawns is on (it is off in every stage), so drawing them
    # advertises start positions the episode will never use.
    if show_spawns:
        spawn = layout.get("spawn_transform", {})
        sx = float(spawn.get("x", 0.0))
        sy = float(spawn.get("y", 0.0))
        draw_point(
            Location(x=sx, y=sy, z=z + 0.3),
            size=0.2,
            color=_COL_SPAWN,
            life_time=life_time,
        )
        draw_string(
            Location(x=sx, y=sy, z=z + 1.0),
            "SPAWN",
            color=_COL_SPAWN,
            life_time=life_time,
        )

        for i, extra in enumerate(layout.get("extra_spawn_transforms", [])):
            ex = float(extra.get("x", 0.0))
            ey = float(extra.get("y", 0.0))
            draw_point(
                Location(x=ex, y=ey, z=z + 0.3),
                size=0.15,
                color=_COL_EXTRA_SPAWN,
                life_time=life_time,
            )
            draw_string(
                Location(x=ex, y=ey, z=z + 1.0),
                f"SPAWN{i + 2}",
                color=_COL_EXTRA_SPAWN,
                life_time=life_time,
            )

    # Pedestrian zones
    if show_pedestrians:
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
                _draw_dotted_segment(
                    debug, ax, ay, bxc, byc, z, _COL_PED, 0.05, life_time
                )
            zone_cx = (x_min + x_max) * 0.5
            zone_cy = (y_min + y_max) * 0.5
            draw_string(
                Location(x=zone_cx, y=zone_cy, z=z + 1.5),
                f"PED {zone_idx}",
                color=_COL_PED,
                life_time=life_time,
            )

    # Patrol waypoints
    if show_patrol:
        waypoints: List[Tuple[float, float]] = [
            (float(wp["x"]), float(wp["y"]))
            for wp in layout.get("patrol_waypoints", [])
        ]
        n_wp = len(waypoints)
        for i, (wx, wy) in enumerate(waypoints):
            draw_point(
                Location(x=wx, y=wy, z=z + 0.2),
                size=0.12,
                color=_COL_PATROL,
                life_time=life_time,
            )
            if n_wp > 1:
                nx, ny = waypoints[(i + 1) % n_wp]
                _draw_dotted_segment(
                    debug, wx, wy, nx, ny, z + 0.2, _COL_PATROL, 0.04, life_time
                )


# ---------------------------------------------------------------------------
# Sensor overlay drawing functions
# ---------------------------------------------------------------------------


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
    draw_point = debug.draw_point
    Location = carla.Location
    draw_point(loc, size=0.35, color=colour, life_time=life_time)
    if drop_line and loc.z > ground_z + 0.1:
        height = loc.z - ground_z
        n_pts = max(2, int(math.ceil(height / 0.12)))
        inv_n = 1.0 / n_pts
        lx, ly = loc.x, loc.y
        for i in range(n_pts + 1):
            t = i * inv_n
            draw_point(
                Location(x=lx, y=ly, z=ground_z + t * height),
                size=0.12,
                color=_COL_WHITE,
                life_time=life_time,
            )
    debug.draw_string(
        Location(x=loc.x, y=loc.y, z=loc.z + 0.7),
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

    # Rotation-recurrence: advance (c, s) by step_angle each iteration.
    step = arc_span / n_pts
    dc = math.cos(step)
    ds = math.sin(step)
    c = math.cos(angle_min_rad)
    s = math.sin(angle_min_rad)

    draw_point = debug.draw_point
    Location = carla.Location

    for _ in range(n_pts + 1):
        draw_point(
            Location(
                x=origin_x + radius * c,
                y=origin_y + radius * s,
                z=origin_z,
            ),
            size=dot_size,
            color=colour,
            life_time=life_time,
        )
        # Advance by step_angle via 2x2 rotation multiply.
        c, s = c * dc - s * ds, s * dc + c * ds

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
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    def _to_world(lx: float, ly: float, lz: float) -> Any:
        """@brief Transform vehicle body-frame offset to world-frame carla.Location."""
        return carla.Location(
            x=vx + lx * cos_yaw - ly * sin_yaw,
            y=vy + lx * sin_yaw + ly * cos_yaw,
            z=vz + lz,
        )

    # IMU
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

    # 2D LiDAR (obstacle detection)
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
        arc_z = lidar_loc.z + 0.05
        lx_w, ly_w = lidar_loc.x, lidar_loc.y
        _draw_fov_arc(
            debug,
            lx_w,
            ly_w,
            arc_z,
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
            lx_w,
            ly_w,
            arc_z,
            radius=lidar_range,
            angle_min_rad=yaw_rad + fov_half,
            angle_max_rad=yaw_rad + math.radians(360.0) - fov_half,
            colour=_COL_FOV_BLIND,
            life_time=life_time,
            dot_size=0.03,
            draw_radials=False,
        )

    # GNSS antenna
    _gnss_locs: List[Any] = []
    for sensor_key, label in _GNSS_LABELS.items():
        if sensor_key not in sensors_cfg:
            continue
        m = sensors_cfg[sensor_key].get("mount", {})
        loc = _to_world(
            float(m.get("x", 0.0)),
            float(m.get("y", 0.0)),
            float(m.get("z", 1.6)),
        )
        _gnss_locs.append(loc)
        _draw_sensor_dot(
            debug,
            loc,
            label,
            _COL_GNSS,
            life_time,
            drop_line=side_view,
            ground_z=ground_z,
        )
    # Draw baseline line between front and rear antennas when both are present.
    if len(_gnss_locs) == 2:
        _draw_dotted_segment(
            debug,
            _gnss_locs[0].x,
            _gnss_locs[0].y,
            _gnss_locs[1].x,
            _gnss_locs[1].y,
            _gnss_locs[0].z,
            _COL_GNSS,
            0.06,
            life_time,
        )
