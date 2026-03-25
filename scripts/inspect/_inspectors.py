"""
@file _inspectors.py
@brief Internal inspector class hierarchy for the CARLA lot inspector.

Defines the four inspector classes that drive CARLA synchronous tick loops
and render debug overlays:

  _Inspector        -- base class: CARLA connection, tick loop, spectator helpers
    LayoutInspector -- lot bay outlines, spawn/patrol/pedestrian overlays
      SensorInspector -- sensor mount dots + FOV arcs on top of layout
    LiveInspector   -- real spawned sensors: LiDAR debug dots or camera spectator view

Drawing helpers are imported from :mod:`scripts.inspect._drawing`.

@note This module is internal -- import via ``scripts.inspect._inspectors``.
@note Requires the full Docker stack (CARLA server + training container).
"""

import math
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect._drawing import (
    _draw_layout_overlays,
    _draw_sensor_overlays,
)
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv

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

    def _place_spectator_side(self, vt: Any) -> None:
        """
        @brief Position spectator to the left of the vehicle (side profile).

        Placed 11 m to the side and 2 m above the vehicle origin, pitched
        slightly down (-8 deg) so the full vehicle body is visible without
        clipping into it.

        @param vt: Vehicle carla.Transform.
        """
        if self._env.world is None:
            return
        yaw_rad = math.radians(vt.rotation.yaw)
        offset = 11.0
        side_x = vt.location.x - offset * math.sin(yaw_rad)
        side_y = vt.location.y + offset * math.cos(yaw_rad)
        look_yaw = math.degrees(
            math.atan2(vt.location.y - side_y, vt.location.x - side_x)
        )
        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=side_x, y=side_y, z=vt.location.z + 2.0),
                carla.Rotation(pitch=-8.0, yaw=look_yaw, roll=0.0),
            )
        )

    def _place_spectator_front(self, vt: Any) -> None:
        """
        @brief Position spectator in front of and above the vehicle, angled down
               to look through the windscreen into the cabin.
        @param vt: Vehicle carla.Transform.
        """
        if self._env.world is None:
            return
        yaw_rad = math.radians(vt.rotation.yaw)
        offset = 5.0
        front_x = vt.location.x + offset * math.cos(yaw_rad)
        front_y = vt.location.y + offset * math.sin(yaw_rad)
        look_yaw = math.degrees(
            math.atan2(vt.location.y - front_y, vt.location.x - front_x)
        )
        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=front_x, y=front_y, z=vt.location.z + 2.5),
                carla.Rotation(pitch=-20.0, yaw=look_yaw, roll=0.0),
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
        zoom: str = "close",
    ) -> None:
        """
        @brief Construct the sensor inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Scene duration in seconds.
        @param suite: Sensor suite ('suite_a', 'suite_b', 'suite_c').
        @param train_cfg: Loaded train_config.yaml dict (for mount positions).
        @param view: Spectator view ('birds_eye' or 'side').
        @param zoom: Camera height mode ('close' = lot detail, 'wide' = full FOV arc).
        """
        super().__init__(env, duration)
        self._suite = suite
        self._train_cfg = train_cfg
        self._view = view
        self._zoom = zoom

    def place_spectator(self) -> None:
        """
        @brief Position spectator based on --view and --zoom flags.

        birds_eye + close: 80 m above vehicle -- lot detail clearly visible,
          arc may be clipped for long-range suites (suite_b/c).
        birds_eye + wide: high enough to see the full FOV arc boundary.
        side: 8 m to the left of the ego vehicle, at vehicle height.
        """
        if self._env.world is None:
            return

        if self._view in ("side", "front") and self._env.vehicle is not None:
            vt = self._env.vehicle.get_transform()
            if self._view == "side":
                self._place_spectator_side(vt)
                print("Spectator: side profile view.")
            else:
                self._place_spectator_front(vt)
                print("Spectator: front profile view.")
            return

        if self._env.vehicle is None:
            return
        vt = self._env.vehicle.get_transform()
        cx, cy, sz = vt.location.x, vt.location.y, vt.location.z

        sensors_cfg = self._train_cfg.get("carla_sensors", {})
        if self._suite in ("suite_b", "suite_c"):
            lidar_range = float(sensors_cfg.get("lidar_3d", {}).get("range", 100.0))
        else:
            lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))

        if self._zoom == "wide":
            # High enough so the full arc boundary stays within CARLA's cull sphere.
            cam_z = max(lidar_range * 2.2, 80.0)
        else:
            # Close view: fixed 80 m shows lot detail clearly regardless of suite.
            cam_z = 80.0

        yaw = vt.rotation.yaw
        self._place_spectator_birds_eye(cx, cy, sz, cam_z, yaw=yaw)
        print(
            f"Spectator at ({cx:.0f}, {cy:.0f}, {sz + cam_z:.0f})"
            f" -- birds-eye {self._zoom} view."
        )

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw layout overlays then sensor overlays on top.
        @param life_time: Primitive lifetime in seconds.
        """
        # Layout geometry only in birds-eye view -- side/front views show only
        # the vehicle body and sensor mounts.
        if self._view == "birds_eye":
            super()._draw_overlays(life_time)
        # Sensor overlays second (rendered above layout geometry)
        if self._env.vehicle is not None and self._env.world is not None:
            _draw_sensor_overlays(
                self._env,
                self._suite,
                self._train_cfg,
                life_time,
                side_view=(self._view in ("side", "front")),
            )


# ===========================================================================
# Live inspector (real spawned sensors, LiDAR debug dots or camera spectator view)
# ===========================================================================


class LiveInspector(_Inspector):
    """
    @class LiveInspector
    @brief Spawns real CARLA sensors on the ego vehicle and shows live output.

    Inherits from :class:`_Inspector` directly -- no layout overlays are drawn
    because the focus is on raw sensor data, not lot geometry annotations.

    Behaviour by suite and sensor mode:
      - suite_a / suite_b: always LiDAR mode -- scan points drawn as small red
        debug dots in the CARLA world frame at ~20 Hz, spectator birds-eye.
      - suite_c (default / sensor='camera'): spectator locked to the RGB camera
        mount position and orientation -- shows exactly what the camera sees
        directly in the CARLA window.  No LiDAR dots drawn.
      - suite_c (sensor='lidar'): override to birds-eye + LiDAR dots, same as
        suite_a/b.

    All sensor mount positions are read from ``train_cfg["carla_sensors"]``.
    Sensors are attached via ``world.spawn_actor()`` with ``attach_to=vehicle``.

    @note Requires the full Docker stack (CARLA server + training container).
    @warning Sensors are destroyed in a ``try/finally`` block -- if CARLA
             crashes mid-session the actors may linger until the server restarts.
    """

    _LIDAR_DOT_SIZE: float = 0.08
    _LIDAR_LIFE: float = 0.5  # Long enough to persist until next scan at 10 Hz

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
        suite: str,
        train_cfg: Dict[str, Any],
        sensor: str = "lidar",
    ) -> None:
        """
        @brief Construct the live inspector.
        @param env: Pre-reset CARLAParkingEnv instance (vehicle must be spawned).
        @param duration: Total session duration in seconds.
        @param suite: Sensor suite ('suite_a', 'suite_b', 'suite_c').
        @param train_cfg: Loaded train_config.yaml dict (for mount positions and
                         blueprint attributes).
        @param sensor: Active sensor to display ('lidar' or 'camera').
                       Ignored for suite_a/b (always lidar).
                       For suite_c: 'camera' moves spectator to camera mount;
                       'lidar' shows birds-eye + debug dots.
        """
        super().__init__(env, duration)
        self._suite = suite
        self._train_cfg = train_cfg
        # suite_a/b always use lidar mode regardless of sensor arg
        self._sensor = sensor if suite == "suite_c" else "lidar"

        # Spawned sensor actors -- destroyed on exit
        self._sensors: List[Any] = []

    # ------------------------------------------------------------------
    # Sensor spawning
    # ------------------------------------------------------------------

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn only the sensor needed for the active display mode.

        In lidar mode: spawns the LiDAR (2D for suite_a, 3D for suite_b/c).
        In camera mode (suite_c only): spawns the RGB camera only -- no LiDAR
          is spawned so no debug dots appear and the CARLA viewport is clean.

        Reads blueprint attributes and mount positions from
        ``self._train_cfg["carla_sensors"]``.
        """
        if self._env.vehicle is None or self._env.world is None:
            return

        world = self._env.world
        bp_lib = world.get_blueprint_library()
        sensors_cfg = self._train_cfg.get("carla_sensors", {})

        if self._sensor == "camera":
            # Camera mode: spawn RGB camera only
            cam_cfg = sensors_cfg.get("camera_rgb", {})
            cam_bp = bp_lib.find("sensor.camera.rgb")

            for attr_name in ("image_size_x", "image_size_y", "fov", "sensor_tick"):
                if attr_name in cam_cfg:
                    cam_bp.set_attribute(attr_name, str(cam_cfg[attr_name]))

            cam_mount = cam_cfg.get("mount", {})
            cam_transform = carla.Transform(
                carla.Location(
                    x=float(cam_mount.get("x", 0.2)),
                    y=float(cam_mount.get("y", 0.0)),
                    z=float(cam_mount.get("z", 1.4)),
                ),
                carla.Rotation(
                    pitch=float(cam_mount.get("pitch", 0.0)),
                    yaw=float(cam_mount.get("yaw", 0.0)),
                    roll=float(cam_mount.get("roll", 0.0)),
                ),
            )
            cam_actor = world.spawn_actor(
                cam_bp,
                cam_transform,
                attach_to=self._env.vehicle,
            )
            # No callback needed -- spectator provides the view directly
            cam_actor.listen(lambda _: None)
            self._sensors.append(cam_actor)
            return

        # Lidar mode: spawn LiDAR only
        lidar_key = "lidar" if self._suite == "suite_a" else "lidar_3d"
        lidar_cfg = sensors_cfg.get(lidar_key, {})
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")

        for attr_name in (
            "channels",
            "range",
            "points_per_second",
            "rotation_frequency",
            "upper_fov",
            "lower_fov",
            "sensor_tick",
        ):
            if attr_name in lidar_cfg:
                lidar_bp.set_attribute(attr_name, str(lidar_cfg[attr_name]))

        mount = lidar_cfg.get("mount", {})
        lidar_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 0.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 0.5)),
            ),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        lidar_actor = world.spawn_actor(
            lidar_bp,
            lidar_transform,
            attach_to=self._env.vehicle,
        )
        lidar_actor.listen(self._on_lidar)
        self._sensors.append(lidar_actor)

    # ------------------------------------------------------------------
    # Sensor data callbacks
    # ------------------------------------------------------------------

    def _on_lidar(self, lidar_data: Any) -> None:
        """
        @brief LiDAR data callback -- draws each hit point as a red debug dot.

        CARLA LiDAR points are in sensor-local frame.  They are transformed to
        world frame via the sensor transform before drawing so the dots appear
        at the correct positions in the CARLA world.

        @param lidar_data: carla.LidarMeasurement received from the sensor.
        """
        if self._env.world is None:
            return

        debug = self._env.world.debug
        dot_colour = carla.Color(r=255, g=0, b=0)

        # lidar_data.transform is the sensor world-frame transform at scan time.
        sensor_transform = lidar_data.transform
        n_pts = len(lidar_data)
        if not hasattr(self, "_lidar_logged"):
            print(f"  LiDAR callback firing: {n_pts} points per scan.")
            self._lidar_logged = True  # type: ignore[attr-defined]

        for detection in lidar_data:
            # Transform point from sensor frame to world frame.
            world_point = sensor_transform.transform(detection.point)
            debug.draw_point(
                world_point,
                size=self._LIDAR_DOT_SIZE,
                color=dot_colour,
                life_time=self._LIDAR_LIFE,
            )

    # ------------------------------------------------------------------
    # Spectator placement
    # ------------------------------------------------------------------

    def place_spectator(self) -> None:
        """
        @brief Position the spectator according to suite and sensor mode.

        lidar mode: birds-eye view 80 m above the ego vehicle.
        camera mode (suite_c only): spectator locked to the RGB camera mount
          position and orientation so the CARLA viewport shows exactly what
          the camera sees.
        """
        if self._env.vehicle is None or self._env.world is None:
            return

        if self._sensor == "camera":
            self._place_spectator_camera()
            print("Spectator: RGB camera mount (suite_c camera view).")
        else:
            vt = self._env.vehicle.get_transform()
            cx, cy, sz = vt.location.x, vt.location.y, vt.location.z
            self._place_spectator_birds_eye(cx, cy, sz, 80.0)
            print("Spectator: birds-eye view (80 m).")

    def _place_spectator_camera(self) -> None:
        """
        @brief Move the CARLA spectator to the RGB camera mount position.

        Reads mount x/y/z and pitch from ``train_cfg["carla_sensors"]["camera_rgb"]``
        and transforms the local mount offset into world frame using the vehicle
        transform.  The spectator yaw matches the vehicle heading so the view
        faces forward.
        """
        if self._env.vehicle is None or self._env.world is None:
            return

        sensors_cfg = self._train_cfg.get("carla_sensors", {})
        cam_mount = sensors_cfg.get("camera_rgb", {}).get("mount", {})
        mx = float(cam_mount.get("x", 0.2))
        my = float(cam_mount.get("y", 0.0))
        mz = float(cam_mount.get("z", 1.4))
        pitch = float(cam_mount.get("pitch", 0.0))

        vt = self._env.vehicle.get_transform()
        yaw_rad = math.radians(vt.rotation.yaw)

        # Rotate mount offset into world frame
        wx = vt.location.x + mx * math.cos(yaw_rad) - my * math.sin(yaw_rad)
        wy = vt.location.y + mx * math.sin(yaw_rad) + my * math.cos(yaw_rad)
        wz = vt.location.z + mz

        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=wx, y=wy, z=wz),
                carla.Rotation(pitch=pitch, yaw=vt.rotation.yaw, roll=0.0),
            )
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Run the live sensor session at ~20 Hz.

        Spawns only the active sensor, then drives the synchronous CARLA tick
        loop.  On each tick:
          1. World is ticked (triggers sensor callbacks via listener threads).
          2. Spectator is locked to the vehicle (birds-eye for lidar mode,
             camera mount position for camera mode).

        Sensors are destroyed in the ``finally`` block regardless of how the
        loop exits (Ctrl+C or normal timeout).
        """
        if self._env.world is None or self._env.vehicle is None:
            print("ERROR: CARLA world or vehicle not available.  Aborting.")
            return

        self._spawn_sensors()

        tick_hz = self._TICK_HZ
        total_ticks = self._duration * tick_hz
        log_ticks = 30 * tick_hz

        print(
            f"Live sensor session for {self._duration}s.  Press Ctrl+C to exit early."
        )
        if self._sensor == "camera":
            print("  RGB camera view -- CARLA spectator locked to camera mount.")
        else:
            print("  LiDAR hit points rendered as red debug dots in the CARLA world.")

        try:
            for i in range(total_ticks):
                self._env.world.tick()
                self._env._update_patrol_npcs()
                self._env._update_pedestrians()

                # Keep spectator locked to the vehicle each tick.
                if self._env.vehicle is not None:
                    if self._sensor == "camera":
                        self._place_spectator_camera()
                    else:
                        vt = self._env.vehicle.get_transform()
                        self._place_spectator_birds_eye(
                            vt.location.x, vt.location.y, vt.location.z, 80.0
                        )

                time.sleep(1.0 / tick_hz)

                if (i + 1) % log_ticks == 0:
                    elapsed = (i + 1) // tick_hz
                    print(f"  {elapsed}/{self._duration} s elapsed ...")

        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            print("Destroying spawned sensors ...")
            for sensor in self._sensors:
                if sensor.is_alive:
                    sensor.destroy()
            self._sensors.clear()


# ===========================================================================
# Dry-run inspector -- full training pipeline, random actions, no model
# ===========================================================================


class DryRunInspector(_Inspector):
    """
    @class DryRunInspector
    @brief Runs the full training environment (reset + step loop) with random
           actions, no model.  Spectator follows the ego vehicle.

    This mode uses the full pipeline: CARLA + ROS bridge + EKF covariance in the
    observation.  It is identical to what training would see, but actions are
    sampled uniformly at random from the action space.  Use this to verify:
      - The environment resets and steps without errors
      - Covariance features are non-zero (EKF is live)
      - NPCs, pedestrians, and static actors spawn correctly each episode
      - The spectator view looks exactly as training will see

    @note Requires the full Docker stack: carla-server + ros2-bridge + training.
    @note include_covariance must be True on the env for covariance to appear.
    """

    _LOG_INTERVAL: int = 50  # steps between console obs prints

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
        n_episodes: Optional[int] = None,
        dryrun_action: Optional[List[float]] = None,
    ) -> None:
        """
        @brief Construct the dry-run inspector.
        @param env: Pre-reset CARLAParkingEnv with include_covariance=True.
        @param duration: Maximum wall-clock seconds to run (across all episodes).
        @param n_episodes: If set, stop after this many episodes regardless of
               duration.  If None, runs until duration expires.
        @param dryrun_action: Fixed [steer, throttle, brake] action to apply each
               step.  If None, falls back to action_space.sample() (random).
               Set via inspect.dryrun_action in train_config.yaml.
        """
        super().__init__(env, duration)
        self._n_episodes = n_episodes
        self._dryrun_action: Optional[np.ndarray] = (
            np.array(dryrun_action, dtype=np.float32)
            if dryrun_action is not None
            else None
        )

    def place_spectator(self) -> None:
        """
        @brief Position spectator 20 m behind and above the ego vehicle.

        Called once after the first reset.  The run loop updates it every step.
        """
        if self._env.vehicle is None or self._env.world is None:
            return
        self._follow_vehicle()

    def _follow_vehicle(self) -> None:
        """
        @brief Lock the spectator 20 m behind and 10 m above the ego vehicle.

        Gives a clear view of the vehicle in context of the lot.
        """
        if self._env.vehicle is None or self._env.world is None:
            return
        vt = self._env.vehicle.get_transform()
        yaw_rad = math.radians(vt.rotation.yaw)
        # 20 m behind the vehicle
        bx = vt.location.x - 20.0 * math.cos(yaw_rad)
        by = vt.location.y - 20.0 * math.sin(yaw_rad)
        bz = vt.location.z + 10.0
        look_yaw = vt.rotation.yaw
        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=bx, y=by, z=bz),
                carla.Rotation(pitch=-25.0, yaw=look_yaw, roll=0.0),
            )
        )

    def _print_obs(self, obs: Any, step: int, episode: int) -> None:
        """
        @brief Print key observation values to the console for diagnosis.

        Prints EKF pose (obs[0:3]), covariance diagonal (obs[6:9] = std_x,
        std_y, std_yaw), and obstacle features (obs[18:21]) if present.

        @param obs: Observation array from env.step() or env.reset().
        @param step: Current step within the episode.
        @param episode: Current episode index.
        """
        if obs is None or len(obs) < 6:
            return
        x, y, yaw = obs[0], obs[1], obs[2]
        line = f"  ep={episode:3d}  step={step:4d}  pos=({x:7.2f},{y:7.2f})  yaw={yaw:6.3f}"
        if self._env.vehicle is not None:
            t = self._env.vehicle.get_transform()
            v = self._env.vehicle.get_velocity()
            spd = math.sqrt(v.x ** 2 + v.y ** 2)
            line += f"  carla=({t.location.x:7.2f},{t.location.y:7.2f})  spd={spd:.2f}"
        if len(obs) >= 15:
            std_x, std_y, std_yaw = obs[6], obs[7], obs[8]
            line += f"  std=({std_x:.4f},{std_y:.4f},{std_yaw:.4f})"
        if len(obs) >= 21:
            nd, nb, nt = obs[18], obs[19], obs[20]
            line += f"  obs=({nd:.2f},{nb:.2f},{nt:.0f})"
        print(line)

    def run(self) -> None:
        """
        @brief Run the dry-run episode loop.

        Each episode: reset() -> step loop with random actions -> on
        termination/truncation reset again.  Spectator follows the ego
        vehicle every step.  Key observation values are printed every
        _LOG_INTERVAL steps so the EKF signal can be visually verified.
        """
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        deadline = time.monotonic() + self._duration
        episode = 0
        total_steps = 0

        print(
            f"Dry-run: random actions for up to {self._duration}s"
            + (f" / {self._n_episodes} episodes." if self._n_episodes else ".")
        )
        print("  Columns: pos=(x,y)  yaw  std=(std_x,std_y,std_yaw)  obs=(dist,bearing,type)")

        try:
            while time.monotonic() < deadline:
                if self._n_episodes is not None and episode >= self._n_episodes:
                    break

                obs, _ = self._env.reset()
                episode += 1
                step = 0
                print(f"\n--- Episode {episode} ---")
                self._follow_vehicle()
                self._print_obs(obs, step, episode)

                terminated = truncated = False
                while not (terminated or truncated):
                    if time.monotonic() >= deadline:
                        break
                    action = (
                        self._dryrun_action
                        if self._dryrun_action is not None
                        else self._env.action_space.sample()
                    )
                    obs, reward, terminated, truncated, info = self._env.step(action)
                    step += 1
                    total_steps += 1
                    self._follow_vehicle()
                    if step % self._LOG_INTERVAL == 0:
                        self._print_obs(obs, step, episode)

                reason = info.get("termination_reason", "truncated" if truncated else "terminated")
                print(
                    f"  Episode {episode} ended: {reason}"
                    f"  steps={step}  total_steps={total_steps}"
                )

        except KeyboardInterrupt:
            print("Interrupted.")

        print(f"\nDry-run complete: {episode} episodes, {total_steps} steps.")


# ===========================================================================
# Z-check inspector -- spawn one of each actor type, confirm shared z-axis
# ===========================================================================


class ZCheckInspector(_Inspector):
    """
    @class ZCheckInspector
    @brief Spawns one of each actor type in a row and reports their actual z coords.

    Spawns six actors in a straight line spaced _ACTOR_SPACING metres apart,
    anchored at the lot's own ego spawn_transform so they land on the same
    ground surface used during training.  The slot layout is:

      0: static_parked_car  1: motorcycle  2: ego_vehicle (centre)
      3: patrol_npc         4: cone        5: pedestrian

    All z offsets and physics settings mirror the real spawners exactly:
      ego_vehicle       : z_ground + 0.0, physics off (mirrors _spawn_vehicle)
      static_parked_car : z_ground + 2.0, physics off, tick, teleport to z_ground
      motorcycle        : same as static_parked_car
      patrol_npc        : z_ground + 0.1, physics on, no teleport
      cone              : z_ground + 0.05, physics off
      pedestrian        : z_ground + 0.05, physics off

    z_ground is read from env._current_layout["origin"]["z"] -- the same
    expression used by LotSpawner and NPCController.  No z constants are
    hardcoded in this class.
    """

    # Metres between consecutive actors along the row X axis.
    _ACTOR_SPACING: float = 5.0

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
    ) -> None:
        """
        @brief Construct the z-check inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Seconds to hold the scene before auto-exit.
        """
        super().__init__(env, duration)
        self._spawned: List[Any] = []

    # ------------------------------------------------------------------
    # Actor spawning
    # ------------------------------------------------------------------

    def _spawn_actors(self) -> None:
        """
        @brief Spawn six reference actors on the true ground plane.

        The patrol NPC is spawned first with physics ON and allowed to settle
        under gravity.  Its lowest point (CoM.z - bbox.extent.z) is the true
        ground contact z for this map tile -- this is z_ground.

        Every other actor is then spawned with physics OFF from birth and
        teleported so its CoM sits at z_ground + bbox.extent.z.  Physics is
        set OFF before the first tick so CARLA never runs physics on them,
        avoiding the 0.9.16 snap-back that teleports actors to their spawn
        position when physics is toggled OFF after running.

        Slot layout (each _ACTOR_SPACING metres apart along X):
          slot 0: static_parked_car  -- physics OFF, CoM = z_ground + bbox.z
          slot 1: motorcycle         -- physics OFF, CoM = z_ground + bbox.z
          slot 2: ego_vehicle        -- physics OFF, CoM = z_ground + bbox.z
          slot 3: patrol_npc         -- physics ON, DEFINES z_ground
          slot 4: cone               -- physics OFF, CoM = z_ground + bbox.z
          slot 5: pedestrian         -- physics OFF, CoM = z_ground + bbox.z
        """
        if self._env.world is None or not self._env._current_layout:
            return

        world = self._env.world
        bp_lib = world.get_blueprint_library()
        layout = self._env._current_layout

        z_yaml = float(layout.get("origin", {}).get("z", 0.3))
        spawn = layout.get("spawn_transform", {})
        row_x = float(spawn.get("x", float(layout.get("origin", {}).get("x", 0.0))))
        row_y = float(spawn.get("y", float(layout.get("origin", {}).get("y", 0.0))))

        spacing = self._ACTOR_SPACING
        car_bps = [
            bp
            for bp in bp_lib.filter("vehicle.*")
            if int(bp.get_attribute("number_of_wheels").as_int()) == 4
        ]

        ego_bp = bp_lib.find("vehicle.bmw.grandtourer")
        if ego_bp is None and car_bps:
            ego_bp = car_bps[0]
        static_bp = car_bps[0] if car_bps else None
        patrol_bp = car_bps[1] if len(car_bps) > 1 else static_bp
        ninja_bp = bp_lib.find("vehicle.kawasaki.ninja")

        if ego_bp is not None and ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "zcheck_ego")

        # ------------------------------------------------------------------
        # Step 1: spawn patrol NPC with physics ON, wait for it to settle,
        # then read z_ground = CoM.z - bbox.extent.z (true floor contact).
        # ------------------------------------------------------------------
        z_ground = z_yaml  # fallback if patrol spawn fails
        x_patrol = row_x + 3 * spacing
        patrol_actor: Optional[Any] = None
        if patrol_bp is not None:
            patrol_actor = world.try_spawn_actor(
                patrol_bp,
                carla.Transform(
                    carla.Location(x=x_patrol, y=row_y, z=z_yaml + 1.0),
                    carla.Rotation(yaw=0.0),
                ),
            )
        if patrol_actor is not None:
            patrol_actor.set_simulate_physics(True)
            for _ in range(60):
                world.tick()
                if abs(patrol_actor.get_velocity().z) < 0.01:
                    break
            # Read the settled transform before freezing.
            settled_t = patrol_actor.get_transform()
            z_ground = settled_t.location.z - patrol_actor.bounding_box.extent.z
            print(
                f"  Patrol NPC settled.  CoM z={settled_t.location.z:.4f}"
                f"  ground z={z_ground:.4f}  (YAML was {z_yaml:.3f})"
            )
            # Freeze at the settled position so it does not drift during the
            # scene.  Physics OFF before the tick commits the disabled state;
            # set_transform then locks it in place.
            patrol_actor.set_simulate_physics(False)
            world.tick()
            patrol_actor.set_transform(settled_t)
            world.tick()
            self._spawned.append(("patrol_npc", patrol_actor))

        # ------------------------------------------------------------------
        # Step 2: spawn all other actors physics OFF, CoM at z_ground + bbox.z.
        # Spawn high, set physics OFF immediately (before any tick), tick once
        # to commit the disabled state, then teleport to correct CoM height.
        # ------------------------------------------------------------------
        def _place_static(bp: Any, x: float, label: str) -> None:
            if bp is None:
                return
            actor = world.try_spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=x, y=row_y, z=z_yaml + 2.0),
                    carla.Rotation(yaw=0.0),
                ),
            )
            if actor is None:
                return
            actor.set_simulate_physics(False)
            world.tick()
            z_com = z_ground + actor.bounding_box.extent.z
            actor.set_transform(carla.Transform(
                carla.Location(x=x, y=row_y, z=z_com),
                carla.Rotation(yaw=0.0),
            ))
            world.tick()
            self._spawned.append((label, actor))

        _place_static(static_bp, row_x, "static_parked_car")
        _place_static(ninja_bp, row_x + spacing, "motorcycle")
        _place_static(ego_bp, row_x + 2 * spacing, "ego_vehicle")

        cone_bp = bp_lib.find("static.prop.constructioncone")
        _place_static(cone_bp, row_x + 4 * spacing, "cone")

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walker_bp = walker_bps[0] if walker_bps else None
        if walker_bp is not None and walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")
        _place_static(walker_bp, row_x + 5 * spacing, "pedestrian")

    def _print_z_report(self) -> None:
        """
        @brief Read back each actor's actual z and compare against the true floor.

        Uses the ego vehicle's settled z as the true ground-plane reference --
        the same value that CARLAParkingEnv._floor_z is set to and that all
        physics-disabled spawners now use.  The layout YAML origin.z (0.3) is
        only the requested spawn z, not the physical ground surface (FlatPlane
        actual floor is ~0.399 after gravity settling).

        Tolerance is 0.05 m -- tight enough to catch genuine floaters while
        forgiving sub-centimetre physics noise.
        """
        if not self._env._current_layout:
            return

        z_yaml = float(self._env._current_layout.get("origin", {}).get("z", 0.3))

        # Use the ego vehicle's settled z as the true floor reference -- this
        # is exactly what _spawn_vehicle() stores in env._floor_z.
        ego_actor = next(
            (a for lbl, a in self._spawned if lbl == "ego_vehicle" and a.is_alive),
            None,
        )
        if ego_actor is not None:
            z_floor = (
                ego_actor.get_transform().location.z
                - ego_actor.bounding_box.extent.z
            )
            floor_note = f"ego wheel contact z = {z_floor:.4f}"
        else:
            z_floor = z_yaml
            floor_note = f"layout YAML z = {z_yaml:.4f} (no ego actor to confirm)"

        # Expected z = z_floor for vehicles (CoM at floor level after settling).
        # Cones and pedestrians sit 0.05 m above the floor CoM reference.
        expected_offsets: Dict[str, float] = {
            "ego_vehicle": 0.0,
            "static_parked_car": 0.0,
            "motorcycle": 0.0,
            "patrol_npc": 0.0,
            "cone": 0.05,
            "pedestrian": 0.05,
        }
        # Pedestrian transform origin is the torso centre, not feet -- always higher.
        skip_mismatch = {"pedestrian"}
        _TOLERANCE = 0.05

        print(f"\n--- Z-axis report (true floor: {floor_note}, YAML z = {z_yaml:.3f}) ---")
        print(f"  {'Actor type':<22}  {'expected_z':>10}  {'actual_z':>10}  {'delta':>8}")
        print("  " + "-" * 60)

        all_ok = True
        for label, actor in self._spawned:
            if not actor.is_alive:
                print(f"  {label:<22}  (destroyed)")
                continue
            actual_z = actor.get_transform().location.z
            expected_z = z_floor + expected_offsets.get(label, 0.0)
            delta = actual_z - expected_z
            if label == "pedestrian":
                flag = "  (torso origin, feet on ground)"
            elif abs(delta) > _TOLERANCE and label not in skip_mismatch:
                flag = "  <-- MISMATCH"
                all_ok = False
            else:
                flag = ""
            print(f"  {label:<22}  {expected_z:>10.4f}  {actual_z:>10.4f}  {delta:>+8.4f}{flag}")

        print("  " + "-" * 60)
        if all_ok:
            print("  All actors within 0.05 m of true floor. No misalignment detected.")
        else:
            print("  WARNING: one or more actors deviate > 0.05 m from true floor z.")
        print()

    # ------------------------------------------------------------------
    # Spectator
    # ------------------------------------------------------------------

    def place_spectator(self) -> None:
        """
        @brief Position spectator in a side view centred on the ego vehicle slot.

        Derives position from the ego actor in self._spawned (slot 2 in the row)
        so the spectator is correctly anchored after _spawn_actors completes.
        """
        if self._env.world is None:
            return

        ego_actor = next(
            (a for label, a in self._spawned if label == "ego_vehicle"), None
        )
        if ego_actor is None or not ego_actor.is_alive:
            return

        vt = ego_actor.get_transform()
        cx = vt.location.x
        cy = vt.location.y
        cz = vt.location.z

        # Stand well back so perspective distortion is minimal across the row.
        standoff = 10.0

        spectator = self._env.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cx, y=cy - standoff, z=cz + 3.0),
                carla.Rotation(pitch=-3.0, yaw=90.0, roll=0.0),
            )
        )
        print(
            f"Spectator at ({cx:.0f}, {cy - standoff:.0f}, {cz + 3.0:.1f})"
            " -- side view centred on ego vehicle."
        )

    # ------------------------------------------------------------------
    # Floor indicator
    # ------------------------------------------------------------------

    def _draw_floor_indicator(self, life_time: float) -> None:
        """
        @brief Draw a bright green horizontal line at the true floor z.

        The line spans the full width of the actor row so the side-view
        spectator can immediately see whether wheels are touching the floor
        or floating above it.  Redrawn on every _REDRAW_INTERVAL cycle so
        it appears continuous.

        @param life_time: Debug primitive lifetime in seconds.
        """
        if self._env.world is None:
            return

        ego_actor = next(
            (a for lbl, a in self._spawned if lbl == "ego_vehicle" and a.is_alive),
            None,
        )
        if ego_actor is None:
            return

        # True floor z = ego wheel contact z (CoM z minus bbox half-height).
        z_floor = (
            ego_actor.get_transform().location.z
            - ego_actor.bounding_box.extent.z
        )
        ego_t = ego_actor.get_transform()
        row_y = ego_t.location.y

        # Row spans from slot 0 to slot 5: 5 * spacing metres wide, plus margin.
        x_start = ego_t.location.x - 2 * self._ACTOR_SPACING - 3.0
        x_end = ego_t.location.x + 3 * self._ACTOR_SPACING + 3.0

        # Draw as closely spaced points to simulate a solid line.
        step = 0.3
        x = x_start
        green = carla.Color(r=0, g=220, b=60)
        debug = self._env.world.debug
        while x <= x_end:
            debug.draw_point(
                carla.Location(x=x, y=row_y, z=z_floor),
                size=0.05,
                color=green,
                life_time=life_time,
            )
            x += step

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Spawn reference row, print z report, hold scene, then destroy actors.

        Clears actors spawned by env.reset() first so only the six
        reference actors appear in the scene.
        """
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        # env.reset() already spawned the ego vehicle and lot actors.
        # Destroy them all so only the z-check reference row is in the scene.
        self._env._cleanup_actors()
        self._env.world.tick()

        self._spawn_actors()
        self.place_spectator()
        self._print_z_report()

        redraw_ticks = self._REDRAW_INTERVAL * self._TICK_HZ
        print(f"Scene live for {self._duration}s.  Press Ctrl+C to exit early.")
        print("  Green line = true floor z reference.")
        try:
            for i in range(self._duration * self._TICK_HZ):
                self._env.world.tick()
                if i % redraw_ticks == 0:
                    self._draw_floor_indicator(life_time=self._OVERLAY_LIFE)
                time.sleep(1.0 / self._TICK_HZ)
                if (i + 1) % (30 * self._TICK_HZ) == 0:
                    elapsed = (i + 1) // self._TICK_HZ
                    print(f"  {elapsed}/{self._duration} s elapsed ...")
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            print("Destroying z-check actors ...")
            for _, actor in self._spawned:
                if actor.is_alive:
                    actor.destroy()
            self._spawned.clear()
