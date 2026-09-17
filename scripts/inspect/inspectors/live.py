"""
@file live.py
@brief LiveInspector: real spawned 2D LiDAR with live scan dots in the CARLA world.
"""

import time
from typing import Any, Dict, List

try:
    import carla
except ImportError:
    import sys

    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from scripts.inspect._drawing import _draw_layout_overlays
from scripts.inspect.inspectors.base import _Inspector
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv


class LiveInspector(_Inspector):
    """
    @class LiveInspector
    @brief Spawns a real 2D LiDAR on the ego vehicle and shows live scan dots.

    LiDAR points are drawn as red debug dots in the CARLA world frame at ~20 Hz.
    Spectator is in birds-eye view above the vehicle.
    """

    _LIDAR_DOT_SIZE: float = 0.08
    _LIDAR_LIFE: float = 0.5  # long enough to persist until next scan at 10 Hz

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
        self._sensors: List[Any] = []
        self._lidar_logged: bool = False

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

    def _on_lidar(self, lidar_data: Any) -> None:
        """
        @brief LiDAR data callback - draws each hit point as a red debug dot.

        Points are in sensor-local frame and are transformed to world frame via
        the sensor transform before drawing.

        @param lidar_data: carla.LidarMeasurement received from the sensor.
        """
        if self._env.world is None:
            return

        debug = self._env.world.debug
        dot_colour = carla.Color(r=255, g=0, b=0)
        sensor_transform = lidar_data.transform

        if not self._lidar_logged:
            print(f"  LiDAR callback firing: {len(lidar_data)} points per scan.")
            self._lidar_logged = True

        # Bind hot callables to locals.
        draw_point = debug.draw_point
        transform_pt = sensor_transform.transform
        dot_size = self._LIDAR_DOT_SIZE
        life = self._LIDAR_LIFE
        for detection in lidar_data:
            draw_point(
                transform_pt(detection.point),
                size=dot_size,
                color=dot_colour,
                life_time=life,
            )

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

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw the lot geometry so the scan is read against the bays.
        @param life_time: Primitive lifetime in seconds.
        """
        if self._env.world is None or not self._env._current_layout:
            return
        _draw_layout_overlays(
            self._env.world,
            self._env._current_layout,
            self._env._target_bay.get("bay_id", ""),
            life_time,
            show_patrol=self._env._num_patrol_max > 0,
            show_pedestrians=self._env._pedestrian_spawn_prob > 0.0,
            oob_inflation_margin=self._env._oob_inflation_margin,
        )

    def run(self) -> None:
        """
        @brief Run the live sensor session at ~20 Hz.

        Spawns the 2D LiDAR, then drives the synchronous CARLA tick loop.
        Sensors are destroyed in the ``finally`` block regardless of how the loop exits.
        """
        if self._env.world is None or self._env.vehicle is None:
            print("ERROR: CARLA world or vehicle not available.  Aborting.")
            return

        self._spawn_sensors()

        tick_hz = self._TICK_HZ
        tick_period = 1.0 / tick_hz
        total_ticks = self._duration * tick_hz
        log_ticks = 30 * tick_hz

        # Bind hot-path callables to locals.
        world_tick = self._env.world.tick
        update_patrol = self._env._update_patrol_npcs
        update_peds = self._env._update_pedestrians
        place_birds_eye = self._place_spectator_birds_eye
        draw_overlays = self._draw_overlays
        vehicle = self._env.vehicle
        # Redraw the lot overlays on the same cadence the other inspectors use,
        # so the scan is read against the bay geometry rather than bare tarmac.
        redraw_ticks = self._REDRAW_INTERVAL * tick_hz

        print(
            f"Live sensor session for {self._duration}s.  Press Ctrl+C to exit early."
        )
        print("  LiDAR hit points rendered as red debug dots in the CARLA world.")

        try:
            for i in range(total_ticks):
                world_tick()
                update_patrol()
                update_peds()

                if vehicle is not None:
                    vt = vehicle.get_transform()
                    place_birds_eye(vt.location.x, vt.location.y, vt.location.z, 80.0)

                if i % redraw_ticks == 0:
                    draw_overlays(life_time=self._OVERLAY_LIFE)

                time.sleep(tick_period)

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
