"""
@file lot_inspector.py
@brief Unified CARLA parking lot and sensor inspector.

Combines layout debug overlays and sensor FOV overlays into a single class
hierarchy so sensors can be seen in the context of the actual parking lot they
will operate in.

Class hierarchy:
  _Inspector          -- CARLA connection, world tick loop, spectator placement
    LayoutInspector   -- lot bay outlines, spawn/patrol/pedestrian overlays
      SensorInspector -- sensor mount dots + LiDAR/camera FOV arcs on top of layout

Usage (via Make targets):
  make docker-inspect INSPECT_LAYOUT=rectangle          # Layout only
  make docker-inspect-sensors INSPECT_SUITE=suite_a     # Sensors on layout
  make docker-inspect-sensors INSPECT_SUITE=suite_b INSPECT_LAYOUT=trapezoid

Or directly:
  python -m scripts.inspect.inspect --mode layout   --layout trapezoid
  python -m scripts.inspect.inspect --mode sensors  --suite suite_a --layout rectangle
  python -m scripts.inspect.inspect --mode sensors  --suite suite_c --layout irregular_a

Arguments:
  --mode       layout | sensors (default: sensors)
  --layout     rectangle | trapezoid | irregular_a (default: rectangle)
  --suite      suite_a | suite_b | suite_c (default: suite_a, sensors mode only)
  --view       birds_eye | side (default: birds_eye, sensors mode only)
  --host       CARLA server hostname (default: carla-server-demo)
  --port       CARLA server port (default: 2100)
  --duration   Seconds to hold the scene (default: 300)

@note Runs in synchronous CARLA mode.  Requires the full Docker stack.
@note CARLA 0.9.16 draw_line ignores colour -- overlays use draw_point at small
      spacing to simulate coloured lines.
"""

import argparse
import math
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.colours import (
    BAY_HEX,
    HEX_PATROL_PATH,
    HEX_PEDESTRIAN_ZONE,
    HEX_SENSOR_CAMERA,
    HEX_SENSOR_FOV_BLIND,
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

_DOT_SPACING: float = 0.4   # metres between adjacent dot centres (layout lines)
_ARC_SPACING: float = 0.3   # metres between dot centres on FOV arcs

# ---------------------------------------------------------------------------
# Derived colour constants
# ---------------------------------------------------------------------------

_COL_IMU = hex_to_carla_color(HEX_SENSOR_IMU)
_COL_LIDAR_2D = hex_to_carla_color(HEX_SENSOR_LIDAR_2D)
_COL_LIDAR_3D = hex_to_carla_color(HEX_SENSOR_LIDAR_3D)
_COL_CAMERA = hex_to_carla_color(HEX_SENSOR_CAMERA)
_COL_FOV_BLIND = hex_to_carla_color(HEX_SENSOR_FOV_BLIND)
_COL_TARGET = hex_to_carla_color(HEX_TARGET_BAY)
_COL_PED = hex_to_carla_color(HEX_PEDESTRIAN_ZONE)
_COL_PATROL = hex_to_carla_color(HEX_PATROL_PATH)


# ===========================================================================
# Base class: CARLA connection + tick loop
# ===========================================================================


class _Inspector:
    """
    @class _Inspector
    @brief Base inspector: CARLA connection, tick loop, spectator helper.

    Subclasses override :meth:`_draw_overlays` to add their debug geometry.
    The :meth:`run` method drives the synchronous tick loop, redrawing overlays
    every ``_REDRAW_INTERVAL`` seconds.

    @note The env reset must be called before :meth:`run`.
    """

    _TICK_HZ: int = 20
    _REDRAW_INTERVAL: int = 3  # seconds between overlay redraws
    _OVERLAY_LIFE: float = 3.5  # seconds an overlay primitive persists

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
    ) -> None:
        """
        @brief Construct the inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Total scene duration in seconds.
        """
        self._env = env
        self._duration = duration

    # ------------------------------------------------------------------
    # Spectator helpers
    # ------------------------------------------------------------------

    def _place_spectator_birds_eye(
        self,
        cx: float,
        cy: float,
        cz: float,
        height: float,
        yaw: float = 0.0,
    ) -> None:
        """
        @brief Position spectator in a top-down birds-eye view.
        @param cx: Centroid X (world frame).
        @param cy: Centroid Y (world frame).
        @param cz: Ground Z at centroid.
        @param height: Camera height above cz.
        @param yaw: Camera yaw in degrees (default 0 = north).
        """
        if self._env.world is None:
            return
        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cx, y=cy, z=cz + height),
                carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
            )
        )

    def _place_spectator_side(
        self,
        vt: Any,
    ) -> None:
        """
        @brief Position spectator to the left of the vehicle (side profile).
        @param vt: Vehicle carla.Transform.
        """
        if self._env.world is None:
            return
        yaw_rad = math.radians(vt.rotation.yaw)
        offset = 8.0
        side_x = vt.location.x - offset * math.sin(yaw_rad)
        side_y = vt.location.y + offset * math.cos(yaw_rad)
        look_yaw = math.degrees(
            math.atan2(vt.location.y - side_y, vt.location.x - side_x)
        )
        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=side_x, y=side_y, z=vt.location.z + 1.0),
                carla.Rotation(pitch=0.0, yaw=look_yaw, roll=0.0),
            )
        )

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw debug overlays.  Override in subclasses.
        @param life_time: Primitive lifetime in seconds.
        """

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """@brief Run the synchronous tick loop for ``self._duration`` seconds."""
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        tick_hz = self._TICK_HZ
        total_ticks = self._duration * tick_hz
        redraw_ticks = self._REDRAW_INTERVAL * tick_hz
        log_ticks = 30 * tick_hz

        print(f"Scene live for {self._duration}s.  Press Ctrl+C to exit early.")
        try:
            for i in range(total_ticks):
                self._env.world.tick()
                self._env._update_patrol_npcs()
                self._env._update_pedestrians()
                if i % redraw_ticks == 0:
                    self._draw_overlays(life_time=self._OVERLAY_LIFE)
                time.sleep(1.0 / tick_hz)
                if (i + 1) % log_ticks == 0:
                    elapsed = (i + 1) // tick_hz
                    print(f"  {elapsed}/{self._duration} s elapsed ...")
        except KeyboardInterrupt:
            print("Interrupted.")


# ===========================================================================
# Layout inspector
# ===========================================================================


class LayoutInspector(_Inspector):
    """
    @class LayoutInspector
    @brief Draws parking lot geometry overlays (bays, spawn, patrol, pedestrian zones).

    Positions the spectator in a birds-eye view centred above the lot bounding box.
    The spectator height is chosen so that the full lot fits within CARLA's
    ~60 m debug-geometry cull radius.
    """

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
    ) -> None:
        """
        @brief Construct the layout inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Scene duration in seconds.
        """
        super().__init__(env, duration)

    # ------------------------------------------------------------------
    # Spectator placement
    # ------------------------------------------------------------------

    def place_spectator(self) -> None:
        """
        @brief Position spectator above the lot centroid.

        Uses the lot bounding box corners from ``env._current_layout`` to compute
        a centroid and a camera height that keeps all corners within the 60 m cull
        radius.
        """
        if self._env.world is None or not self._env._current_layout:
            return

        layout = self._env._current_layout
        corners = layout.get("corners", [])
        if corners:
            xs = [float(c["x"]) for c in corners]
            ys = [float(c["y"]) for c in corners]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            span = max(max(xs) - min(xs), max(ys) - min(ys))
            cam_z = max(span * 1.1, 80.0)
        else:
            spawn = layout.get("spawn_transform", {})
            cx = float(spawn.get("x", 0.0))
            cy = float(spawn.get("y", 0.0))
            cam_z = 80.0

        sz = float(layout.get("origin", {}).get("z", 0.3))
        self._place_spectator_birds_eye(cx, cy, sz, cam_z)
        print(
            f"Spectator at lot centroid ({cx:.0f}, {cy:.0f}, {sz + cam_z:.0f})"
            " -- birds-eye view."
        )

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw bay outlines, spawn points, pedestrian zones, patrol path.
        @param life_time: Primitive lifetime in seconds.
        """
        if self._env.world is None or not self._env._current_layout:
            return
        _draw_layout_overlays(
            self._env.world,
            self._env._current_layout,
            self._env._target_bay.get("bay_id", ""),
            life_time,
        )


# ===========================================================================
# Sensor inspector (extends layout inspector)
# ===========================================================================


class SensorInspector(LayoutInspector):
    """
    @class SensorInspector
    @brief Adds sensor mount overlays and FOV arcs on top of the layout.

    Inherits all lot geometry overlays from :class:`LayoutInspector`.  In
    addition, it draws labelled dots at each sensor mount position and FOV arcs
    (or rings) at the actual sensor range distances.

    The spectator is placed either birds-eye (default, shows FOV arcs against
    the lot) or side profile (shows mount heights on the vehicle body).
    """

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
        suite: str,
        train_cfg: Dict[str, Any],
        view: str = "birds_eye",
    ) -> None:
        """
        @brief Construct the sensor inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Scene duration in seconds.
        @param suite: Sensor suite ('suite_a', 'suite_b', 'suite_c').
        @param train_cfg: Loaded train_config.yaml dict (for mount positions).
        @param view: Spectator view ('birds_eye' or 'side').
        """
        super().__init__(env, duration)
        self._suite = suite
        self._train_cfg = train_cfg
        self._view = view

    def place_spectator(self) -> None:
        """
        @brief Position spectator based on --view flag.

        birds_eye: above the lot centroid (height chosen to fit the larger of
          the lot span or the LiDAR range so FOV arcs remain visible).
        side: 8 m to the left of the ego vehicle, at vehicle height.
        """
        if self._env.world is None:
            return

        if self._view == "side" and self._env.vehicle is not None:
            self._place_spectator_side(self._env.vehicle.get_transform())
            print("Spectator: side profile view.")
            return

        # birds_eye: centre on the vehicle so the full FOV arc stays within CARLA's
        # ~60 m debug-geometry cull radius.  Camera height = lidar_range * 1.3 so
        # the arc boundary (radius metres away) is always within that cull sphere.
        if self._env.vehicle is None:
            return
        vt = self._env.vehicle.get_transform()
        cx, cy, sz = vt.location.x, vt.location.y, vt.location.z

        sensors_cfg = self._train_cfg.get("carla_sensors", {})
        if self._suite in ("suite_b", "suite_c"):
            lidar_range = float(sensors_cfg.get("lidar_3d", {}).get("range", 100.0))
        else:
            lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))
        # Height = lidar_range * 2.2 ensures the arc (lidar_range metres away in
        # the XY plane) is within CARLA's ~60 m debug-geometry cull sphere even
        # when the spectator is directly above the sensor origin.
        cam_z = max(lidar_range * 2.2, 60.0)

        yaw = vt.rotation.yaw
        self._place_spectator_birds_eye(cx, cy, sz, cam_z, yaw=yaw)
        print(
            f"Spectator at ({cx:.0f}, {cy:.0f}, {sz + cam_z:.0f})"
            " -- birds-eye view."
        )

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw layout overlays then sensor overlays on top.
        @param life_time: Primitive lifetime in seconds.
        """
        # Layout geometry first (bays, spawn, patrol, pedestrian zones)
        super()._draw_overlays(life_time)
        # Sensor overlays second (rendered above layout geometry)
        if self._env.vehicle is not None and self._env.world is not None:
            _draw_sensor_overlays(
                self._env,
                self._suite,
                self._train_cfg,
                life_time,
            )


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
            (x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max),
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
) -> None:
    """
    @brief Draw a labelled dot at a sensor mount position.
    @param debug: carla.DebugHelper.
    @param loc: carla.Location of the sensor mount.
    @param label: Text label rendered above the dot.
    @param colour: carla.Color.
    @param life_time: Primitive lifetime in seconds.
    """
    debug.draw_point(loc, size=0.18, color=colour, life_time=life_time)
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
                origin_x, origin_y, px_end, py_end,
                origin_z,
                colour,
                dot_size * 0.7,
                life_time,
            )


def _draw_sensor_overlays(
    env: CARLAParkingEnv,
    suite: str,
    train_cfg: Dict[str, Any],
    life_time: float,
) -> None:
    """
    @brief Draw sensor mount dots and FOV arcs for the current vehicle pose.

    Reads mount positions from ``train_cfg`` (``carla_sensors`` section) and draws:
      - IMU: yellow dot at centre-of-mass height (all suites)
      - Suite A: cyan dot at front bumper + 270 deg FOV arc + faint 90 deg blind sector
      - Suite B/C: green dot at roof + full 360 deg ring
      - Suite C: orange dot at windscreen + 90 deg camera cone arc

    All positions are read from ``carla_sensors.<sensor>.mount`` in ``train_cfg``
    and transformed from vehicle body frame to world frame using the current
    vehicle transform.

    @param env: Active CARLAParkingEnv (vehicle must be spawned).
    @param suite: Sensor suite ('suite_a', 'suite_b', 'suite_c').
    @param train_cfg: Loaded train_config.yaml dict.
    @param life_time: Primitive lifetime in seconds.
    """
    if env.vehicle is None or env.world is None:
        return

    debug = env.world.debug
    sensors_cfg = train_cfg.get("carla_sensors", {})
    vt = env.vehicle.get_transform()
    vx = vt.location.x
    vy = vt.location.y
    vz = vt.location.z
    yaw_rad = math.radians(vt.rotation.yaw)

    def _to_world(lx: float, ly: float, lz: float) -> Any:
        """@brief Transform vehicle body-frame offset to world-frame carla.Location."""
        wx = vx + lx * math.cos(yaw_rad) - ly * math.sin(yaw_rad)
        wy = vy + lx * math.sin(yaw_rad) + ly * math.cos(yaw_rad)
        return carla.Location(x=wx, y=wy, z=vz + lz)

    # ---- IMU (all suites) ------------------------------------------------
    imu_m = sensors_cfg.get("imu", {}).get("mount", {})
    imu_loc = _to_world(
        float(imu_m.get("x", 0.0)),
        float(imu_m.get("y", 0.0)),
        float(imu_m.get("z", 0.3)),
    )
    _draw_sensor_dot(debug, imu_loc, "IMU", _COL_IMU, life_time)

    # ---- Suite A: 2D LiDAR -----------------------------------------------
    if suite == "suite_a":
        lid_m = sensors_cfg.get("lidar", {}).get("mount", {})
        lx = float(lid_m.get("x", 2.4))
        ly = float(lid_m.get("y", 0.0))
        lz = float(lid_m.get("z", 0.3))
        lidar_loc = _to_world(lx, ly, lz)
        _draw_sensor_dot(debug, lidar_loc, "2D LiDAR", _COL_LIDAR_2D, life_time)

        lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))
        fov_half = math.radians(135.0)  # 270 deg total = 135 deg either side of forward

        # 270 deg coverage arc (orange, bold).
        # carla.Color constructed directly: hex conversion shifts #FF5000 to yellow.
        _draw_fov_arc(
            debug,
            lidar_loc.x, lidar_loc.y, lidar_loc.z + 0.05,
            radius=lidar_range,
            angle_min_rad=yaw_rad - fov_half,
            angle_max_rad=yaw_rad + fov_half,
            colour=carla.Color(r=255, g=100, b=0),
            life_time=life_time,
            dot_size=0.22,
            draw_radials=True,
        )
        # 90 deg blind sector arc (dark grey, thin)
        _draw_fov_arc(
            debug,
            lidar_loc.x, lidar_loc.y, lidar_loc.z + 0.05,
            radius=lidar_range,
            angle_min_rad=yaw_rad + fov_half,
            angle_max_rad=yaw_rad + math.radians(360.0) - fov_half,
            colour=_COL_FOV_BLIND,
            life_time=life_time,
            dot_size=0.08,
            draw_radials=False,
        )

    # ---- Suite B / C: 3D LiDAR ------------------------------------------
    if suite in ("suite_b", "suite_c"):
        lid3_m = sensors_cfg.get("lidar_3d", {}).get("mount", {})
        lx = float(lid3_m.get("x", 0.0))
        ly = float(lid3_m.get("y", 0.0))
        lz = float(lid3_m.get("z", 1.5))
        lidar3d_loc = _to_world(lx, ly, lz)
        _draw_sensor_dot(debug, lidar3d_loc, "3D LiDAR", _COL_LIDAR_3D, life_time)

        lidar3d_range = float(sensors_cfg.get("lidar_3d", {}).get("range", 100.0))
        _draw_fov_arc(
            debug,
            lidar3d_loc.x, lidar3d_loc.y, lidar3d_loc.z,
            radius=lidar3d_range,
            angle_min_rad=0.0,
            angle_max_rad=2.0 * math.pi,
            colour=_COL_LIDAR_3D,
            life_time=life_time,
            dot_size=0.22,
            draw_radials=False,
        )

    # ---- Suite C: RGB camera ---------------------------------------------
    if suite == "suite_c":
        cam_m = sensors_cfg.get("camera_rgb", {}).get("mount", {})
        cx = float(cam_m.get("x", 2.0))
        cy_l = float(cam_m.get("y", 0.0))
        cz = float(cam_m.get("z", 1.2))
        cam_loc = _to_world(cx, cy_l, cz)
        _draw_sensor_dot(debug, cam_loc, "RGB CAM", _COL_CAMERA, life_time)

        cam_fov_deg = float(sensors_cfg.get("camera_rgb", {}).get("fov", 90.0))
        cam_fov_half = math.radians(cam_fov_deg / 2.0)
        _draw_fov_arc(
            debug,
            cam_loc.x, cam_loc.y, cam_loc.z,
            radius=30.0,
            angle_min_rad=yaw_rad - cam_fov_half,
            angle_max_rad=yaw_rad + cam_fov_half,
            colour=_COL_CAMERA,
            life_time=life_time,
            dot_size=0.22,
            draw_radials=True,
        )


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
        choices=["layout", "sensors"],
        help=(
            "Inspector mode: 'layout' = lot geometry only; "
            "'sensors' = sensors on lot (default: sensors)."
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
        choices=["birds_eye", "side"],
        help=(
            "Spectator view (sensors mode only): "
            "'birds_eye' shows FOV arcs against the lot; "
            "'side' shows sensor mount heights on the vehicle body.  "
            "Default: birds_eye."
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

    else:
        sensors_cfg = dict(train_cfg.get("carla_sensors", {}))
        sensors_cfg["sensor_suite"] = args.suite

        env = _build_env(
            args.host, args.port, args.layout, train_cfg,
            sensors_cfg=sensors_cfg,
            suite=args.suite,
            full_lot=True,   # show the real lot environment sensors will operate in
        )
        env.reset()
        if env.world is None or env.vehicle is None:
            print("ERROR: Could not connect to CARLA or spawn vehicle.")
            env.close()
            sys.exit(1)

        inspector = SensorInspector(
            env, args.duration, args.suite, train_cfg, args.view
        )
        inspector.place_spectator()  # type: ignore[attr-defined]
        print("Layout overlays:")
        print("  blue=perpendicular | yellow=angled | violet=parallel | green=TARGET")
        print("Sensor overlays:")
        print("  yellow=IMU | cyan=2D LiDAR | green=3D LiDAR | orange=RGB camera")
        print("FOV arcs:")
        print("  light-blue arc = 270 deg (Suite A) | green ring = 360 deg (Suite B/C)")
        print("  orange arc = 90 deg camera (Suite C)")

    try:
        inspector.run()
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
