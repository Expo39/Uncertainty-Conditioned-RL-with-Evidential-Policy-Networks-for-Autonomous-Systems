"""
@file _inspectors.py
@brief Internal inspector class hierarchy for the CARLA lot inspector.

Defines the inspector classes that drive CARLA synchronous tick loops
and render debug overlays:

  _Inspector        -- base class: CARLA connection, tick loop, spectator helpers
    LayoutInspector -- lot bay outlines, spawn/patrol/pedestrian overlays
      SensorInspector -- sensor mount dots + FOV arcs on layout
    LiveInspector   -- real spawned sensors: LiDAR dots or camera view
    DryRunInspector -- full training pipeline, random actions, no model

Drawing helpers are imported from :mod:`scripts.inspect._drawing`.

@note This module is internal -- import via ``scripts.inspect._inspectors``.
@note Requires the full Docker stack (CARLA server + training container).
"""

import math
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import carla
except ImportError:
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect._drawing import _draw_layout_overlays, _draw_sensor_overlays
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
        train_cfg: Dict[str, Any],
        view: str = "birds_eye",
        zoom: str = "close",
    ) -> None:
        """
        @brief Construct the sensor inspector.
        @param env: Pre-reset CARLAParkingEnv instance.
        @param duration: Scene duration in seconds.
        @param train_cfg: Loaded train_config.yaml dict (for mount positions).
        @param view: Spectator view ('birds_eye', 'side', or 'front').
        @param zoom: Camera height mode ('close' = lot detail, 'wide' = full FOV arc).
        """
        super().__init__(env, duration)
        self._train_cfg = train_cfg
        self._view = view
        self._zoom = zoom

    def place_spectator(self) -> None:
        """
        @brief Position spectator based on --view and --zoom flags.

        birds_eye + close: 80 m above vehicle -- lot detail clearly visible.
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
        lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))

        if self._zoom == "wide":
            # High enough so the full arc boundary stays within CARLA's cull sphere.
            cam_z = max(lidar_range * 2.2, 80.0)
        else:
            # Close view: fixed 80 m shows lot detail clearly.
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
    @brief Spawns a real 2D LiDAR on the ego vehicle and shows live scan dots.

    Inherits from :class:`_Inspector` directly -- no layout overlays are drawn
    because the focus is on raw sensor data, not lot geometry annotations.

    LiDAR scan points are drawn as small red debug dots in the CARLA world frame
    at ~20 Hz, with the spectator in birds-eye view above the vehicle.

    Sensor mount positions are read from ``train_cfg["carla_sensors"]``.
    The sensor is attached via ``world.spawn_actor()`` with ``attach_to=vehicle``.

    @note Requires the full Docker stack (CARLA server + training container).
    @warning The sensor is destroyed in a ``try/finally`` block -- if CARLA
             crashes mid-session the actor may linger until the server restarts.
    """

    _LIDAR_DOT_SIZE: float = 0.08
    _LIDAR_LIFE: float = 0.5  # Long enough to persist until next scan at 10 Hz

    def __init__(
        self,
        env: CARLAParkingEnv,
        duration: int,
        train_cfg: Dict[str, Any],
    ) -> None:
        """
        @brief Construct the live inspector.
        @param env: Pre-reset CARLAParkingEnv instance (vehicle must be spawned).
        @param duration: Total session duration in seconds.
        @param train_cfg: Loaded train_config.yaml dict (for mount positions and
                         blueprint attributes).
        """
        super().__init__(env, duration)
        self._train_cfg = train_cfg

        # Spawned sensor actors -- destroyed on exit
        self._sensors: List[Any] = []

    # ------------------------------------------------------------------
    # Sensor spawning
    # ------------------------------------------------------------------

    def _spawn_sensors(self) -> None:
        """
        @brief Spawn the 2D LiDAR sensor on the ego vehicle.

        Reads blueprint attributes and mount position from
        ``self._train_cfg["carla_sensors"]["lidar"]``.
        """
        if self._env.vehicle is None or self._env.world is None:
            return

        world = self._env.world
        bp_lib = world.get_blueprint_library()
        sensors_cfg = self._train_cfg.get("carla_sensors", {})

        lidar_cfg = sensors_cfg.get("lidar", {})
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
        @brief Position the spectator in birds-eye view 80 m above the ego vehicle.
        """
        if self._env.vehicle is None or self._env.world is None:
            return

        vt = self._env.vehicle.get_transform()
        cx, cy, sz = vt.location.x, vt.location.y, vt.location.z
        self._place_spectator_birds_eye(cx, cy, sz, 80.0)
        print("Spectator: birds-eye view (80 m).")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Run the live sensor session at ~20 Hz.

        Spawns the 2D LiDAR, then drives the synchronous CARLA tick loop.
        On each tick:
          1. World is ticked (triggers sensor callbacks via listener threads).
          2. Spectator is locked to the vehicle in birds-eye view.

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
        print("  LiDAR hit points rendered as red debug dots in the CARLA world.")

        try:
            for i in range(total_ticks):
                self._env.world.tick()
                self._env._update_patrol_npcs()
                self._env._update_pedestrians()

                # Keep spectator locked to the vehicle each tick.
                if self._env.vehicle is not None:
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
# Keyboard controller for manual dryrun mode
# ===========================================================================


class KeyboardController:
    """
    @class KeyboardController
    @brief Latching TTY keyboard input for manual dryrun control.

    Reads raw keypresses from stdin in a background daemon thread using
    Python's tty/termios modules -- no X11 interaction, no display dependency.

    TTY terminals have no key-up events, so a timeout-based approach breaks
    when holding two keys simultaneously (the idle axis times out). Instead
    each key latches its axis on until the opposite key or a neutral press
    cancels it:

      Up    -- throttle on  (longitudinal = +1.0); cancels brake
      Down  -- brake on     (longitudinal = -1.0); cancels throttle
      Left  -- steer left   (steer = -1.0);  cancels right steer
      Right -- steer right  (steer = +1.0);  cancels left steer
      Space -- zero both axes (full stop + straight)

    Steer and longitudinal are independent so steering while driving works.

    @note Requires the container to be run with -it. The Makefile
          docker-inspect-dryrun MANUAL=true target handles this automatically.
    """

    def __init__(self) -> None:
        """@brief Initialise controller with zeroed latched state."""
        import tty  # noqa: F401 -- verify stdlib available before starting thread
        import termios  # noqa: F401

        self._steer: float = 0.0
        self._longitudinal: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # Step sizes for incremental control.
    # Steer: 0.1 per press (10 presses = full lock).
    # Throttle: 0.1 per press (10 presses = full throttle).
    # Brake: 0.2 per press (5 presses = full brake -- braking is faster).
    _STEER_STEP: float = 0.1
    _THROTTLE_STEP: float = 0.1
    _BRAKE_STEP: float = 0.2

    def start(self) -> None:
        """@brief Start background stdin reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print(
            "  Keyboard control (incremental): Up=+throttle  Down=+brake"
            "  Left/Right=steer  Space=stop  Ctrl+C=quit"
        )

    def stop(self) -> None:
        """@brief Signal the reader thread to stop."""
        self._running = False

    def get_action(self) -> np.ndarray:
        """
        @brief Return the current latched [steer, longitudinal] action.
        @return numpy array of shape (2,) with values in [-1.0, 1.0].
        """
        with self._lock:
            return np.array([self._steer, self._longitudinal], dtype=np.float32)

    def _read_loop(self) -> None:
        """@brief Read raw escape sequences from stdin and update latched state."""
        import os
        import select
        import signal
        import sys
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            print(
                "  WARNING: stdin is not a TTY -- keyboard control disabled. "
                "Run with MANUAL=true so the container gets an interactive stdin."
            )
            return

        try:
            tty.setraw(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    # Ctrl+C -- propagate SIGINT to the main thread
                    os.kill(os.getpid(), signal.SIGINT)
                    break
                if ch == " ":
                    with self._lock:
                        self._steer = 0.0
                        self._longitudinal = 0.0
                elif ch == "\x1b":
                    rest = sys.stdin.read(2)
                    seq = ch + rest
                    with self._lock:
                        if seq == "\x1b[A":    # Up -- increment throttle
                            self._longitudinal = min(
                                1.0, self._longitudinal + self._THROTTLE_STEP
                            )
                        elif seq == "\x1b[B":  # Down -- increment brake
                            self._longitudinal = max(
                                -1.0, self._longitudinal - self._BRAKE_STEP
                            )
                        elif seq == "\x1b[D":  # Left -- increment left steer
                            self._steer = max(
                                -1.0, self._steer - self._STEER_STEP
                            )
                        elif seq == "\x1b[C":  # Right -- increment right steer
                            self._steer = min(
                                1.0, self._steer + self._STEER_STEP
                            )
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except termios.error:
                pass


# Dry-run inspector -- full training pipeline, random actions, no model
# ===========================================================================


class DryRunInspector(_Inspector):
    """
    @class DryRunInspector
    @brief Runs the full training environment (reset + step loop) with a constant
           forward action or keyboard control, no model.  Spectator follows the ego.

    Views (set via --inspect-view):
      third_person -- 20 m behind + 10 m above, follows vehicle heading
      side         -- 15 m to the left, at ego z, level
      back         -- 20 m behind, at ego z, level
      front        -- 15 m ahead, at ego z, looking back
      free         -- placed once at z=0, not updated; CARLA's own controls work

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
        initial_view: str = "third_person",
        termination_pause: float = 3.0,
        manual: bool = False,
    ) -> None:
        """
        @brief Construct the dry-run inspector.
        @param env: Pre-reset CARLAParkingEnv with include_covariance=True.
        @param duration: Maximum wall-clock seconds to run (across all episodes).
        @param n_episodes: Stop after this many episodes; None = run until duration.
        @param dryrun_action: Fixed [steer, longitudinal] to apply each step.
               None = action_space.sample(). Ignored when manual=True.
               Set via inspect.dryrun_action in YAML.
        @param initial_view: One of 'third_person', 'side', 'back', 'front', 'free'.
        @param termination_pause: Seconds to hold scene after episode ends.
        @param manual: If True, use keyboard arrow keys instead of random/fixed action.
        """
        super().__init__(env, duration)
        self._n_episodes = n_episodes
        self._dryrun_action: Optional[np.ndarray] = (
            np.array(dryrun_action, dtype=np.float32)
            if dryrun_action is not None
            else None
        )
        self._view: str = (
            initial_view
            if initial_view in ("third_person", "side", "back", "front", "free")
            else "third_person"
        )
        self._termination_pause = termination_pause
        self._keyboard: Optional[KeyboardController] = (
            KeyboardController() if manual else None
        )

    # ------------------------------------------------------------------
    # Spectator placement
    # ------------------------------------------------------------------

    def place_spectator(self) -> None:
        """
        @brief Position spectator for the active view.

        'free' mode: placed once at z=0 beside the vehicle, not touched again.
        All other views: delegates to _update_spectator().
        """
        if self._env.vehicle is None or self._env.world is None:
            return
        if self._view == "free":
            vt = self._env.vehicle.get_transform()
            yaw_rad = math.radians(vt.rotation.yaw)
            sx = vt.location.x - 20.0 * math.sin(yaw_rad)
            sy = vt.location.y + 20.0 * math.cos(yaw_rad)
            look_yaw = math.degrees(math.atan2(vt.location.y - sy, vt.location.x - sx))
            self._env.world.get_spectator().set_transform(
                carla.Transform(
                    carla.Location(x=sx, y=sy, z=0.0),
                    carla.Rotation(pitch=10.0, yaw=look_yaw, roll=0.0),
                )
            )
            print(
                "  Free view: spectator at z=0, pitched up. "
                "Use CARLA controls to fly."
            )
            return
        self._update_spectator()

    def _update_spectator(self) -> None:
        """
        @brief Move spectator to follow the ego vehicle.  No-op in 'free' mode.
        """
        if self._env.vehicle is None or self._env.world is None or self._view == "free":
            return
        vt = self._env.vehicle.get_transform()
        yaw_rad = math.radians(vt.rotation.yaw)
        spectator = self._env.world.get_spectator()

        if self._view == "third_person":
            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=vt.location.x - 20.0 * math.cos(yaw_rad),
                        y=vt.location.y - 20.0 * math.sin(yaw_rad),
                        z=vt.location.z + 10.0,
                    ),
                    carla.Rotation(pitch=-25.0, yaw=vt.rotation.yaw),
                )
            )
        elif self._view == "side":
            sx = vt.location.x - 15.0 * math.sin(yaw_rad)
            sy = vt.location.y + 15.0 * math.cos(yaw_rad)
            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=sx, y=sy, z=vt.location.z),
                    carla.Rotation(
                        pitch=0.0,
                        yaw=math.degrees(
                            math.atan2(vt.location.y - sy, vt.location.x - sx)
                        ),
                    ),
                )
            )
        elif self._view == "back":
            spectator.set_transform(
                carla.Transform(
                    carla.Location(
                        x=vt.location.x - 20.0 * math.cos(yaw_rad),
                        y=vt.location.y - 20.0 * math.sin(yaw_rad),
                        z=vt.location.z,
                    ),
                    carla.Rotation(pitch=0.0, yaw=vt.rotation.yaw),
                )
            )
        else:  # front
            fx = vt.location.x + 15.0 * math.cos(yaw_rad)
            fy = vt.location.y + 15.0 * math.sin(yaw_rad)
            spectator.set_transform(
                carla.Transform(
                    carla.Location(x=fx, y=fy, z=vt.location.z),
                    carla.Rotation(
                        pitch=0.0,
                        yaw=math.degrees(
                            math.atan2(vt.location.y - fy, vt.location.x - fx)
                        ),
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Observation logging
    # ------------------------------------------------------------------

    def _print_obs(self, obs: Any, step: int, episode: int) -> None:
        """
        @brief Print key observation values to the console for diagnosis.

        Colour scheme:
          WHITE  -- values that enter the model (obs indices 0-19)
          YELLOW -- ground truth (CARLA pose, GT target bay world coords)
          RED    -- derived/diagnostic values not fed to the model
                    (raw EKF odom pose, odom-frame target, recon check)

        @param obs: Observation array from env.step() or env.reset().
        @param step: Current step within the episode.
        @param episode: Current episode index.
        """
        _WHITE = "\033[97m"
        _YELLOW = "\033[33m"
        _RED = "\033[31m"
        _RESET = "\033[0m"

        if obs is None or len(obs) < 3:
            return

        W = _WHITE
        Y = _YELLOW
        R = _RED
        X = _RESET
        lines = []

        # Header
        lines.append(f"--- ep={episode}  step={step} " + "-" * 40)

        # WHITE -- velocity (indices 0-2)
        lines.append(
            W + f"vel  vx={obs[0]:+.2f}  vy={obs[1]:+.2f}  vyaw={math.degrees(obs[2]):+.1f}deg/s" + X
        )

        # YELLOW -- CARLA ground truth
        if self._env.vehicle is not None:
            t = self._env.vehicle.get_transform()
            v = self._env.vehicle.get_velocity()
            spd = math.sqrt(v.x**2 + v.y**2)
            lines.append(
                Y + f"GT   pos=({t.location.x:.2f},{t.location.y:.2f})"
                f"  yaw={t.rotation.yaw:+.1f}deg  spd={spd:.2f}m/s" + X
            )

        # RED -- EKF odom pose AND projected into world frame for direct GT comparison
        if self._env._cov_subscriber is not None:
            ekf_pose = self._env._cov_subscriber.get_latest_pose()
            if ekf_pose is not None:
                ekf_x = float(ekf_pose[0])
                ekf_y = float(ekf_pose[1])
                ekf_yaw = float(ekf_pose[2])
                lines.append(
                    R + f"EKF(odom)   x={ekf_x:.2f}  y={ekf_y:.2f}"
                    f"  yaw={math.degrees(ekf_yaw):+.1f}deg" + X
                )
                # Project EKF odom pose into world frame using the calibrated
                # odom->world transform so it can be compared directly to GT.
                tx, ty, cos_r, sin_r, r = self._env._ekf_odom_offset
                wx = cos_r * ekf_x - sin_r * ekf_y + tx
                wy = sin_r * ekf_x + cos_r * ekf_y + ty
                wyaw_rad = math.atan2(
                    math.sin(ekf_yaw + r), math.cos(ekf_yaw + r)
                )
                wyaw = math.degrees(wyaw_rad)
                lines.append(
                    R + f"EKF(world)  x={wx:.2f}  y={wy:.2f}"
                    f"  yaw={wyaw:+.1f}deg  (compare to GT above)" + X
                )

        # WHITE -- EKF covariance split over two lines (indices 3-11)
        if len(obs) >= 12:
            lines.append(
                W + f"cov  std=({obs[3]:.3f},{obs[4]:.3f},{obs[5]:.3f})"
                f"  diag=({obs[6]:.4f},{obs[7]:.4f},{obs[8]:.4f})" + X
            )
            lines.append(
                W + f"     off=({obs[9]:.4f},{obs[10]:.4f},{obs[11]:.4f})" + X
            )

        # WHITE -- target in ego body frame (indices 12-14)
        if len(obs) >= 15:
            lines.append(
                W + f"tgt  dx={obs[12]:+.2f}m  dy={obs[13]:+.2f}m"
                f"  dyaw={math.degrees(obs[14]):+.1f}deg" + X
            )

        # YELLOW -- GT target world; RED -- odom projection + vehicle recon
        gt = self._env._target_bay
        odom_t = self._env._target_bay_odom
        tx, ty, cos_r, sin_r, r = self._env._ekf_odom_offset

        # Vehicle reconstruction error: transform current EKF pose to
        # world frame and compare to GT vehicle position. This measures
        # actual localisation + transform accuracy (unlike the bay
        # round-trip which is always zero by construction).
        err_m = 0.0
        if self._env._cov_subscriber is not None:
            ekf_pose = self._env._cov_subscriber.get_latest_pose()
            if ekf_pose is not None and self._env.vehicle is not None:
                ekf_x = float(ekf_pose[0])
                ekf_y = float(ekf_pose[1])
                recon_veh_wx = cos_r * ekf_x - sin_r * ekf_y + tx
                recon_veh_wy = sin_r * ekf_x + cos_r * ekf_y + ty
                gt_veh = self._env.vehicle.get_transform()
                err_m = math.sqrt(
                    (recon_veh_wx - gt_veh.location.x) ** 2
                    + (recon_veh_wy - gt_veh.location.y) ** 2
                )

        lines.append(
            Y + f"bay  world=({gt['x']:.2f},{gt['y']:.2f})"
            f"  yaw={math.degrees(gt['yaw']):+.1f}deg" + X
        )
        lines.append(
            R + f"     odom=({odom_t['x']:.2f},{odom_t['y']:.2f})"
            f"  yaw={math.degrees(odom_t['yaw']):+.1f}deg"
            f"  recon_err={err_m:.3f}m  r={math.degrees(r):+.1f}deg" + X
        )

        # WHITE -- obstacle clearance (indices 15-19)
        if len(obs) >= 20:
            lines.append(
                W + f"obs  L={obs[15]:.2f}m({math.degrees(obs[16]):+.1f}deg)"
                f"  R={obs[17]:.2f}m({math.degrees(obs[18]):+.1f}deg)"
                f"  F={obs[19]:.2f}m" + X
            )

        print("\n" + "\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw target bay highlight and lot geometry in the CARLA window.

        Reuses _draw_layout_overlays so the target bay appears in bright green
        with a "TARGET" label, matching the layout inspector style.

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

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Run the dry-run episode loop.

        Each episode: reset() -> step loop -> on termination hold for
        termination_pause seconds so the final state is visible in CARLA.
        Spectator follows the ego every step.  Key obs printed every
        _LOG_INTERVAL steps.
        """
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        if self._keyboard is not None:
            self._keyboard.start()

        deadline = time.monotonic() + self._duration
        episode = 0
        total_steps = 0
        _latest_obs = None

        mode_desc = (
            "keyboard control"
            if self._keyboard is not None
            else f"constant action {self._dryrun_action.tolist()}"
        )
        print(
            f"Dry-run: {mode_desc} for up to {self._duration}s"
            + (f" / {self._n_episodes} episodes." if self._n_episodes else ".")
        )
        print(
            "\nModel inputs per step (20-dim obs):"
            "\n  [0-2]   vel: vx vy vyaw"
            "\n  [3-11]  cov: std(x,y,yaw)  diag(xx,yy,yawyaw)  off(xy,xyaw,yyaw)"
            "\n  [12-14] tgt: dx dy dyaw (ego-relative)"
            "\n  [15-19] obs: L(dist,bear)  R(dist,bear)  F(dist)"
            "\n  bay/EKF lines are diagnostic only (not fed to model)"
            "\n  recon_err should be < 0.05m"
        )

        try:
            while time.monotonic() < deadline:
                if self._n_episodes is not None and episode >= self._n_episodes:
                    break

                obs, _ = self._env.reset()
                episode += 1
                step = 0
                print(f"\n--- Episode {episode}  [view: {self._view}] ---")
                self._update_spectator()
                self._draw_overlays(life_time=self._OVERLAY_LIFE)
                self._print_obs(obs, step, episode)

                terminated = truncated = False
                while not (terminated or truncated):
                    if time.monotonic() >= deadline:
                        break
                    if self._keyboard is not None:
                        action = self._keyboard.get_action()
                    else:
                        action = self._dryrun_action
                    obs, reward, terminated, truncated, info = self._env.step(action)
                    step += 1
                    total_steps += 1
                    # Pace the dryrun at real-time speed. Each env.step() advances
                    # the simulation by action_repeat * carla_timestep seconds.
                    # Without this sleep the loop runs as fast as CARLA can tick,
                    # which is much faster than wall-clock time on a GPU machine.
                    time.sleep(
                        self._env._action_repeat * self._env._carla_timestep
                    )
                    self._update_spectator()

                    # Inspector-level GT proximity check: end episode the moment
                    # CARLA GT position is within 5 m of the GT target bay.
                    _gt_done = False
                    if self._env.vehicle is not None:
                        _vt = self._env.vehicle.get_transform()
                        _tgt = self._env._target_bay
                        _gt_dist = math.sqrt(
                            (_vt.location.x - _tgt["x"]) ** 2
                            + (_vt.location.y - _tgt["y"]) ** 2
                        )
                        if _gt_dist <= 0.5:
                            print(
                                f"\n  [GT proximity] {_gt_dist:.2f}m from target "
                                f"-- ending episode early"
                            )
                            self._print_obs(obs, step, episode)
                            _gt_done = True
                            terminated = True

                    if not _gt_done and step % self._LOG_INTERVAL == 0:
                        self._draw_overlays(life_time=self._OVERLAY_LIFE)
                        self._print_obs(obs, step, episode)
                    _latest_obs = obs

                reason = (
                    "gt_proximity"
                    if _gt_done
                    else info.get(
                        "termination_reason",
                        "truncated" if truncated else "terminated",
                    )
                )
                print(
                    f"  Episode {episode} ended: {reason}"
                    f"  steps={step}  total_steps={total_steps}"
                )
                if not _gt_done:
                    print("--- Final observation ---")
                    self._print_obs(obs, step, episode)

                # Hold the scene so the final state can be inspected in CARLA.
                if self._termination_pause > 0:
                    print(
                        f"  Pausing {self._termination_pause:.1f}s "
                        "(Ctrl+C to skip) ..."
                    )
                    pause_end = time.monotonic() + self._termination_pause
                    while time.monotonic() < pause_end:
                        self._env.world.tick()
                        self._update_spectator()
                        time.sleep(1.0 / self._TICK_HZ)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            if _latest_obs is not None:
                print("--- Final observation at interruption ---")
                self._print_obs(_latest_obs, step, episode)
        finally:
            if self._keyboard is not None:
                self._keyboard.stop()

        print(f"\nDry-run complete: {episode} episodes, {total_steps} steps.")
