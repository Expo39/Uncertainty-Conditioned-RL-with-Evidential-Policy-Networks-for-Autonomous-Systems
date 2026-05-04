"""
@file base.py
@brief Base inspector class and shared helpers for the lot inspector.
"""

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import carla
except ImportError:
    import sys
    print("ERROR: carla Python package not found.  Run inside the training container.")
    sys.exit(1)

from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv

_GNSS_NOISE_CONFIG_PATH = Path("/workspace/outputs/gnss_noise_config.json")


def _read_live_tier() -> str:
    """
    @brief Read the current GNSS tier name from gnss_noise_config.json.
    @return Tier name string, or 'unknown' if the file is absent or unreadable.
    """
    try:
        with open(_GNSS_NOISE_CONFIG_PATH) as _f:
            return str(json.load(_f).get("tier_name", "unknown"))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Base class: CARLA connection + tick loop
# ---------------------------------------------------------------------------


class _Inspector:
    """
    @class _Inspector
    @brief Base inspector: CARLA connection, tick loop, spectator helper.

    Subclasses override :meth:`_draw_overlays` to add their debug geometry.
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
        @param vt: Vehicle carla.Transform.
        """
        if self._env.world is None:
            return
        vx = vt.location.x
        vy = vt.location.y
        vz = vt.location.z
        yaw_rad = math.radians(vt.rotation.yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        offset = 11.0
        side_x = vx - offset * sin_y
        side_y = vy + offset * cos_y
        look_yaw = math.degrees(math.atan2(vy - side_y, vx - side_x))
        self._env.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=side_x, y=side_y, z=vz + 2.0),
                carla.Rotation(pitch=-8.0, yaw=look_yaw, roll=0.0),
            )
        )

    def _place_spectator_front(self, vt: Any) -> None:
        """
        @brief Position spectator in front of and above the vehicle, angled down.
        @param vt: Vehicle carla.Transform.
        """
        if self._env.world is None:
            return
        vx = vt.location.x
        vy = vt.location.y
        vz = vt.location.z
        yaw_rad = math.radians(vt.rotation.yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        offset = 5.0
        front_x = vx + offset * cos_y
        front_y = vy + offset * sin_y
        look_yaw = math.degrees(math.atan2(vy - front_y, vx - front_x))
        self._env.world.get_spectator().set_transform(
            carla.Transform(
                carla.Location(x=front_x, y=front_y, z=vz + 2.5),
                carla.Rotation(pitch=-20.0, yaw=look_yaw, roll=0.0),
            )
        )

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """@brief Run the synchronous tick loop for ``self._duration`` seconds."""
        if self._env.world is None:
            print("ERROR: CARLA world not available.  Aborting.")
            return

        tick_hz = self._TICK_HZ
        tick_period = 1.0 / tick_hz
        total_ticks = self._duration * tick_hz
        redraw_ticks = self._REDRAW_INTERVAL * tick_hz
        log_ticks = 30 * tick_hz

        # Bind hot-path callables to locals.
        world_tick = self._env.world.tick
        update_patrol = self._env._update_patrol_npcs
        update_peds = self._env._update_pedestrians
        draw_overlays = self._draw_overlays
        overlay_life = self._OVERLAY_LIFE

        print(f"Scene live for {self._duration}s.  Press Ctrl+C to exit early.")
        try:
            for i in range(total_ticks):
                world_tick()
                update_patrol()
                update_peds()
                if i % redraw_ticks == 0:
                    draw_overlays(life_time=overlay_life)
                time.sleep(tick_period)
                if (i + 1) % log_ticks == 0:
                    elapsed = (i + 1) // tick_hz
                    print(f"  {elapsed}/{self._duration} s elapsed ...")
        except KeyboardInterrupt:
            print("Interrupted.")
