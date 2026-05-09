"""
@file sensor.py
@brief SensorInspector: sensor mount dots and FOV arcs on the layout.
"""

from typing import Any, Dict

from scripts.inspect._drawing import _draw_sensor_overlays
from scripts.inspect.inspectors.layout import LayoutInspector
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv


class SensorInspector(LayoutInspector):
    """
    @class SensorInspector
    @brief Adds sensor mount overlays and FOV arcs on top of the layout.

    Inherits all lot geometry overlays from :class:`LayoutInspector`.
    The spectator is placed birds-eye (default) or side/front profile.
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

        birds_eye + close: 80 m above vehicle - lot detail clearly visible.
        birds_eye + wide: high enough to see the full FOV arc boundary.
        side/front: delegates to the corresponding _place_spectator_* helper.
        """
        if self._env.world is None:
            return

        if self._env.vehicle is None:
            return
        vt = self._env.vehicle.get_transform()

        if self._view in ("side", "front"):
            if self._view == "side":
                self._place_spectator_side(vt)
                print("Spectator: side profile view.")
            else:
                self._place_spectator_front(vt)
                print("Spectator: front profile view.")
            return

        cx, cy, sz = vt.location.x, vt.location.y, vt.location.z

        sensors_cfg = self._train_cfg.get("carla_sensors", {})
        lidar_range = float(sensors_cfg.get("lidar", {}).get("range", 30.0))

        if self._zoom == "wide":
            cam_z = max(lidar_range * 2.2, 80.0)
        else:
            cam_z = 80.0

        yaw = vt.rotation.yaw
        self._place_spectator_birds_eye(cx, cy, sz, cam_z, yaw=yaw)
        print(
            f"Spectator at ({cx:.0f}, {cy:.0f}, {sz + cam_z:.0f})"
            f" - birds-eye {self._zoom} view."
        )

    def _draw_overlays(self, life_time: float) -> None:
        """
        @brief Draw layout overlays then sensor overlays on top.
        @param life_time: Primitive lifetime in seconds.
        """
        # Layout geometry only in birds-eye view - side/front views show only
        # the vehicle body and sensor mounts.
        if self._view == "birds_eye":
            super()._draw_overlays(life_time)
        if self._env.vehicle is not None and self._env.world is not None:
            _draw_sensor_overlays(
                self._env,
                self._train_cfg,
                life_time,
                side_view=(self._view in ("side", "front")),
            )
