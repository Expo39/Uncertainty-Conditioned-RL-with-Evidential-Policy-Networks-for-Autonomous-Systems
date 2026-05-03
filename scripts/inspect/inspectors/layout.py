"""
@file layout.py
@brief LayoutInspector: parking lot geometry overlays (bays, spawn, patrol, pedestrian zones).
"""

from scripts.inspect.inspectors.base import _Inspector
from scripts.inspect._drawing import _draw_layout_overlays
from uncertainty_rl.envs.sim.carla_parking import CARLAParkingEnv


class LayoutInspector(_Inspector):
    """
    @class LayoutInspector
    @brief Draws parking lot geometry overlays (bays, spawn, patrol, pedestrian zones).

    Positions the spectator in birds-eye view centred above the lot bounding box.
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
            " - birds-eye view."
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
