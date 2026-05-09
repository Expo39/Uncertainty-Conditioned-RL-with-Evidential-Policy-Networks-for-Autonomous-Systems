"""
@file visualiser.py
@brief Live 2D bird's-eye visualiser for CARLA parking training and evaluation.

Tails outputs/vis_history.jsonl and renders each frame as it arrives using
Pygame. Creates outputs/.vis_active so the environment starts writing frames;
removing it (on window close) stops the env writing.
"""

import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from scripts.colours import (
    BAY_HEX,
    HEX_CONE,
    HEX_EGO,
    HEX_LOT,
    HEX_PATROL_VEHICLE,
    HEX_PEDESTRIAN_ZONE,
    HEX_STATIC_VEHICLE,
    HEX_TARGET_BAY,
    hex_to_rgb,
)

# ---------------------------------------------------------------------------
# Colour palette (converted once at import time)
# ---------------------------------------------------------------------------

_C_LOT = hex_to_rgb(HEX_LOT)
_C_LOT_EDGE = (153, 153, 153)
_C_TARGET_BAY = hex_to_rgb(HEX_TARGET_BAY)
_C_PERP_BAY = hex_to_rgb(BAY_HEX["perpendicular"])
_C_ANGLED_BAY = hex_to_rgb(BAY_HEX["angled"])
_C_PARALLEL_BAY = hex_to_rgb(BAY_HEX["parallel"])
_C_STATIC_VEHICLE = hex_to_rgb(HEX_STATIC_VEHICLE)
_C_PATROL_VEHICLE = hex_to_rgb(HEX_PATROL_VEHICLE)
_C_PEDESTRIAN = hex_to_rgb(HEX_PEDESTRIAN_ZONE)
_C_EGO = hex_to_rgb(HEX_EGO)
_C_TRAIL = (*hex_to_rgb(HEX_EGO), 100)
_C_BG = (248, 248, 248)
_C_HUD_BG = (30, 30, 30, 180)
_C_HUD_TEXT = (212, 212, 212)
_C_CONE = hex_to_rgb(HEX_CONE)

# ---------------------------------------------------------------------------
# Bay colour/linewidth lookup
# ---------------------------------------------------------------------------

_BAY_STYLE: Dict[str, Tuple[Any, int]] = {
    "angled": (_C_ANGLED_BAY, 1),
    "parallel": (_C_PARALLEL_BAY, 1),
    "perpendicular": (_C_PERP_BAY, 1),
}
_BAY_STYLE_DEFAULT: Tuple[Any, int] = (_C_PERP_BAY, 1)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_LEGEND_W = 180
_MAP_W = 900
_WINDOW_W = _MAP_W + _LEGEND_W
_WINDOW_H = 900
_FPS_CAP = 120
_MARGIN_PX = 40
_HUD_FONT_SIZE = 14
_LABEL_FONT_SIZE = 13
_EGO_HALF_L = 2.25
_EGO_HALF_W = 1.0
_NPC_HALF_L = 2.35
_NPC_HALF_W = 1.05
_TRAIL_MAX_POINTS = 500
_POLL_SLEEP = 0.02

_DEFAULT_HISTORY_FILE = Path("outputs/vis_history.jsonl")
_SIGNAL_FILE = Path("outputs/.vis_active")

# Legend fonts are initialised lazily on first _draw_legend call (requires
# pygame.font.init() to have run first).
_legend_font: Optional[pygame.font.Font] = None
_legend_title_font: Optional[pygame.font.Font] = None


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rot_corners(
    cx: float, cy: float, half_l: float, half_w: float, yaw_deg: float
) -> np.ndarray:
    """
    @brief Compute the four corners of a rotated rectangle in world coords.
    @param cx: Centre x (metres).
    @param cy: Centre y (metres).
    @param half_l: Half-length.
    @param half_w: Half-width.
    @param yaw_deg: Yaw in degrees (CARLA convention).
    @return Array of shape (4, 2).
    """
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (R @ local.T).T + np.array([cx, cy])


def _world_to_screen(
    pts: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> List[Tuple[int, int]]:
    """
    @brief Convert world-space points to Pygame screen pixels.
    @param pts: Array of shape (N, 2) in metres.
    @param origin: World coordinate mapped to the viewport top-left (after margin).
    @param scale: Pixels per metre.
    @return List of (px, py) integer tuples.
    """
    # Layout coordinates are in CARLA's left-handed frame (Y increases
    # rightward). Screen Y increases downward, which matches CARLA Y
    # direction in a bird's-eye view, so no Y-flip is needed.
    shifted = pts - origin
    px = (shifted[:, 0] * scale + _MARGIN_PX).astype(int)
    py = (shifted[:, 1] * scale + _MARGIN_PX).astype(int)
    return list(zip(px, py))


def _w2s(x: float, y: float, origin: np.ndarray, scale: float) -> Tuple[int, int]:
    """@brief Convert a single world point to a screen pixel."""
    ox, oy = origin
    return (int((x - ox) * scale + _MARGIN_PX), int((y - oy) * scale + _MARGIN_PX))


# ---------------------------------------------------------------------------
# Static scene surface
# ---------------------------------------------------------------------------


def _build_static_surface(
    state: Dict[str, Any],
    origin: np.ndarray,
    scale: float,
) -> pygame.Surface:
    """
    @brief Render the static scene elements into a surface (rebuilt once per episode).

    Includes lot boundary, bay outlines, target bay arrows, and static parked vehicles.

    @param state: Any frame from the episode (static data is identical across frames).
    @param origin: World origin for the viewport.
    @param scale: Pixels per metre.
    @return Opaque Surface with static scene painted on it.
    """
    surf = pygame.Surface((_MAP_W, _WINDOW_H))
    surf.fill(_C_BG)

    # Lot boundary
    corners_raw = state.get("corners", [])
    if corners_raw:
        pts = np.array([[c["x"], c["y"]] for c in corners_raw])
        spts = _world_to_screen(pts, origin, scale)
        if len(spts) >= 3:
            pygame.draw.polygon(surf, _C_LOT, spts)
            pygame.draw.polygon(surf, _C_LOT_EDGE, spts, 2)

    # Bay outlines
    target_id = state.get("target_bay", {}).get("bay_id", "")
    for bay in state.get("bays", []):
        bay_id = bay.get("id", bay.get("bay_id", ""))
        is_target = bay_id == target_id

        if is_target:
            colour: Any = _C_TARGET_BAY
            lw = 3
        else:
            colour, lw = _BAY_STYLE.get(
                bay.get("bay_type", "perpendicular"), _BAY_STYLE_DEFAULT
            )

        half_d = float(bay.get("depth", 5.0)) / 2.0
        half_w = float(bay.get("width", 2.5)) / 2.0
        yaw_deg = float(bay.get("yaw_deg", bay.get("yaw", 0.0)))
        bx, by = float(bay["x"]), float(bay["y"])
        spts = _world_to_screen(
            _rot_corners(bx, by, half_d, half_w, yaw_deg), origin, scale
        )
        pygame.draw.polygon(surf, colour, spts, lw)

        # Heading arrows for target bay (nose-in and nose-out)
        if is_target:
            arrow_len = half_d * 0.8
            yaw_r = math.radians(yaw_deg)
            cos_y, sin_y = math.cos(yaw_r), math.sin(yaw_r)
            s0 = _w2s(bx, by, origin, scale)
            for cx_dir, cy_dir in ((cos_y, sin_y), (-cos_y, -sin_y)):
                s1 = _w2s(
                    bx + arrow_len * cx_dir, by + arrow_len * cy_dir, origin, scale
                )
                pygame.draw.line(surf, _C_TARGET_BAY, s0, s1, 2)
                dx, dy = s1[0] - s0[0], s1[1] - s0[1]
                length = math.hypot(dx, dy) or 1.0
                ux, uy = dx / length, dy / length
                left = (int(s1[0] - ux * 8 + uy * 5), int(s1[1] - uy * 8 - ux * 5))
                right = (int(s1[0] - ux * 8 - uy * 5), int(s1[1] - uy * 8 + ux * 5))
                pygame.draw.polygon(surf, _C_TARGET_BAY, [s1, left, right])

    # Static parked vehicles
    for actor in state.get("actors", []):
        if actor.get("type", "static") != "npc":
            spts = _world_to_screen(
                _rot_corners(
                    actor["x"],
                    actor["y"],
                    _NPC_HALF_L,
                    _NPC_HALF_W,
                    actor.get("yaw", 0.0),
                ),
                origin,
                scale,
            )
            pygame.draw.polygon(surf, _C_STATIC_VEHICLE, spts)

    return surf


def _draw_legend(screen: pygame.Surface) -> None:
    """
    @brief Draw the colour legend in the right-hand panel.
    @param screen: Main Pygame surface.
    """
    global _legend_font, _legend_title_font
    if _legend_font is None:
        _legend_font = pygame.font.SysFont("monospace", _LABEL_FONT_SIZE)
        _legend_title_font = pygame.font.SysFont(
            "monospace", _LABEL_FONT_SIZE, bold=True
        )
    font = _legend_font
    title_font = _legend_title_font

    pygame.draw.rect(
        screen,
        (235, 235, 235),
        pygame.Rect(_MAP_W, 0, _LEGEND_W, _WINDOW_H),
    )
    pygame.draw.line(screen, (180, 180, 180), (_MAP_W, 0), (_MAP_W, _WINDOW_H), 2)

    entries = [
        (_C_TARGET_BAY, "Target bay"),
        (_C_PERP_BAY, "Perpendicular"),
        (_C_ANGLED_BAY, "Angled"),
        (_C_PARALLEL_BAY, "Parallel"),
        (_C_EGO, "Ego"),
        (_C_PATROL_VEHICLE, "Patrol NPC"),
        (_C_STATIC_VEHICLE, "Parked"),
        (_C_PEDESTRIAN, "Pedestrian"),
    ]

    x0, y0 = _MAP_W + 10, 16
    screen.blit(title_font.render("Legend", True, (40, 40, 40)), (x0, y0))
    y0 += _LABEL_FONT_SIZE + 8

    for colour, label in entries:
        pygame.draw.rect(screen, colour, (x0, y0 + 2, 14, 14))
        screen.blit(font.render(label, True, (50, 50, 50)), (x0 + 20, y0))
        y0 += 22


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------


class LiveVisualiser:
    """
    @class LiveVisualiser
    @brief Live 2D bird's-eye Pygame visualiser for CARLA parking.

    Tails outputs/vis_history.jsonl and renders each frame as it arrives.
    The static scene (lot, bays, parked vehicles) is rebuilt once per episode.

    Controls:
        F        - toggle fullscreen
        ESC / Q  - exit
    """

    def __init__(self, history_file: Optional[Path] = None) -> None:
        """
        @brief Initialise Pygame window and state.
        @param history_file: Path to vis_history.jsonl. Defaults to
               outputs/vis_history.jsonl.
        """
        self._history_file = history_file or _DEFAULT_HISTORY_FILE
        self._signal_file = _SIGNAL_FILE

        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H))
        pygame.display.set_caption(
            "CARLA Parking Visualiser  |  F fullscreen  |  ESC quit"
        )
        self._clock = pygame.time.Clock()
        self._hud_font = pygame.font.SysFont("monospace", _HUD_FONT_SIZE)

        self._static_surf: Optional[pygame.Surface] = None
        self._static_episode_id: Optional[int] = None

        self._origin = np.zeros(2)
        self._scale = 1.0

        # Accumulated ego trail - grows every frame, cleared on episode reset.
        self._vis_trail: List[Tuple[float, float]] = []
        self._vis_trail_episode_id: Optional[int] = None

        self._file_offset: int = 0
        self._exit_requested: bool = False
        self._fullscreen: bool = False
        self._ever_received_frame: bool = False
        self._trail_surf: pygame.Surface = pygame.Surface(
            (_MAP_W, _WINDOW_H), pygame.SRCALPHA
        )

        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._signal_file.touch()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """@brief Open the Pygame window and enter the main loop."""
        self._exit_requested = False

        def _sigint(_sig: int, _frame: Any) -> None:
            self._exit_requested = True

        signal.signal(signal.SIGINT, _sigint)

        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self._signal_file.unlink(missing_ok=True)
            except OSError:
                pass
            pygame.quit()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        @brief Read new JSONL frames and render them as fast as they arrive.

        Polls the JSONL file each iteration. When no new data is available
        the loop sleeps briefly to avoid busy-waiting.
        """
        while not self._exit_requested:
            self._handle_events()

            frames = self._read_new_frames()
            if frames:
                self._ever_received_frame = True
                # Render only the latest frame to avoid falling behind
                self._draw_frame(frames[-1])
            elif not self._ever_received_frame:
                self._draw_waiting()
                time.sleep(_POLL_SLEEP)
            else:
                # Data stream temporarily dry - hold last frame, don't flicker
                time.sleep(_POLL_SLEEP)

            self._clock.tick(_FPS_CAP)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        """@brief Process Pygame event queue."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._exit_requested = True
                elif event.key == pygame.K_f:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    self._screen = pygame.display.set_mode(
                        (_WINDOW_W, _WINDOW_H), flags
                    )
                    self._static_episode_id = None  # Force static surface rebuild
                    self._vis_trail_episode_id = None  # Force trail reset

    # ------------------------------------------------------------------
    # JSONL ingestion
    # ------------------------------------------------------------------

    def _read_new_frames(self) -> List[Dict[str, Any]]:
        """
        @brief Read all complete JSONL lines written since the last call.
        @return List of parsed frame dicts in arrival order.
        """
        if not self._history_file.exists():
            return []
        frames: List[Dict[str, Any]] = []
        try:
            with open(self._history_file, "rb") as f:
                if self._file_offset > self._history_file.stat().st_size:
                    self._file_offset = 0
                f.seek(self._file_offset)
                chunk = f.read()
                if not chunk:
                    return []
                last_nl = chunk.rfind(b"\n")
                if last_nl == -1:
                    return []
                self._file_offset += last_nl + 1
                try:
                    text = chunk[: last_nl + 1].decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk[: last_nl + 1].decode("utf-8", errors="replace")
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frames.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return frames

    # ------------------------------------------------------------------
    # Viewport
    # ------------------------------------------------------------------

    def _compute_viewport(self, state: Dict[str, Any]) -> None:
        """
        @brief Compute origin and scale so the lot fits inside the map viewport.
        @param state: Any frame from the episode.
        """
        corners_raw = state.get("corners", [])
        if corners_raw:
            pts = np.array([[c["x"], c["y"]] for c in corners_raw])
        else:
            ego = state.get("ego", {})
            cx, cy = float(ego.get("x", 0.0)), float(ego.get("y", 0.0))
            pts = np.array([[cx - 30, cy - 30], [cx + 30, cy + 30]])

        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = np.where((mx - mn) < 1.0, 1.0, mx - mn)
        usable_w = _MAP_W - 2 * _MARGIN_PX
        usable_h = _WINDOW_H - 2 * _MARGIN_PX
        self._scale = min(usable_w / span[0], usable_h / span[1])
        self._origin = mn

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_waiting(self) -> None:
        """@brief Render a waiting splash when no data has arrived yet."""
        self._screen.fill(_C_BG)
        _draw_legend(self._screen)
        font = pygame.font.SysFont("monospace", 18)
        txt = font.render("Waiting for data...", True, (80, 80, 80))
        self._screen.blit(
            txt,
            ((_MAP_W - txt.get_width()) // 2, (_WINDOW_H - txt.get_height()) // 2),
        )
        pygame.display.flip()

    def _draw_frame(self, state: Dict[str, Any]) -> None:
        """
        @brief Render a single frame: rebuild static surface if episode changed,
               then blit static + dynamic layers.
        @param state: Parsed frame dict from vis_history.jsonl.
        """
        episode_id = state.get("episode_id")

        if episode_id != self._static_episode_id:
            self._static_episode_id = episode_id
            self._compute_viewport(state)
            self._static_surf = _build_static_surface(state, self._origin, self._scale)

        assert self._static_surf is not None
        self._screen.blit(self._static_surf, (0, 0))
        _draw_legend(self._screen)

        origin, scale = self._origin, self._scale

        # Accumulated ego trail: clear on episode reset, then append current position.
        if episode_id != self._vis_trail_episode_id:
            self._vis_trail = []
            self._vis_trail_episode_id = episode_id
        ego = state.get("ego", {})
        if ego:
            self._vis_trail.append((float(ego["x"]), float(ego["y"])))
        trail_len = len(self._vis_trail)
        if trail_len > 1:
            trail_pts = (
                self._vis_trail[-_TRAIL_MAX_POINTS:]
                if trail_len > _TRAIL_MAX_POINTS
                else self._vis_trail
            )
            spts = _world_to_screen(np.array(trail_pts), origin, scale)
            self._trail_surf.fill((0, 0, 0, 0))
            pygame.draw.lines(self._trail_surf, _C_TRAIL, False, spts, 2)
            self._screen.blit(self._trail_surf, (0, 0))

        # Patrol NPC vehicles
        for actor in state.get("actors", []):
            if actor.get("type", "static") == "npc":
                pygame.draw.polygon(
                    self._screen,
                    _C_PATROL_VEHICLE,
                    _world_to_screen(
                        _rot_corners(
                            actor["x"],
                            actor["y"],
                            _NPC_HALF_L,
                            _NPC_HALF_W,
                            actor.get("yaw", 0.0),
                        ),
                        origin,
                        scale,
                    ),
                )

        # Pedestrians
        ped_r = max(2, int(0.4 * scale))
        for ped in state.get("pedestrians", []):
            pygame.draw.circle(
                self._screen,
                _C_PEDESTRIAN,
                _w2s(ped["x"], ped["y"], origin, scale),
                ped_r,
            )

        # Ego vehicle
        if ego:
            ex, ey, eyaw = float(ego["x"]), float(ego["y"]), float(ego["yaw"])
            pygame.draw.polygon(
                self._screen,
                _C_EGO,
                _world_to_screen(
                    _rot_corners(ex, ey, _EGO_HALF_L, _EGO_HALF_W, eyaw), origin, scale
                ),
            )
            yaw_r = math.radians(eyaw)
            cos_y, sin_y = math.cos(yaw_r), math.sin(yaw_r)
            s0 = _w2s(ex, ey, origin, scale)
            s1 = _w2s(ex + 3.5 * cos_y, ey + 3.5 * sin_y, origin, scale)
            pygame.draw.line(self._screen, _C_EGO, s0, s1, 2)
            dx, dy = s1[0] - s0[0], s1[1] - s0[1]
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            left = (int(s1[0] - ux * 8 + uy * 5), int(s1[1] - uy * 8 - ux * 5))
            right = (int(s1[0] - ux * 8 - uy * 5), int(s1[1] - uy * 8 + ux * 5))
            pygame.draw.polygon(self._screen, _C_EGO, [s1, left, right])

        # HUD
        hud = (
            f"Floor: {state.get('floor_plan', '?')}  "
            f"Ep: {episode_id}  "
            f"Step: {state.get('episode_step', '?')}  "
            f"t={state.get('sim_time', 0.0):.2f}s"
        )
        self._draw_hud(hud, y_offset=8)

        dbg = state.get("debug")
        if dbg:
            self._draw_hud(
                f"err={dbg.get('pos_err', 0.0):.2f}m "
                f"yaw={dbg.get('yaw_err_deg', 0.0):.1f}deg "
                f"spd={dbg.get('speed', 0.0):.2f}m/s "
                f"rwd={dbg.get('reward', 0.0):.3f} | "
                f"cov={dbg.get('cov_rms', 0.0):.3f} "
                f"drift={dbg.get('ekf_drift', 0.0):.2f}m | "
                f"act=[{dbg.get('steer', 0.0):.2f} "
                f"{dbg.get('throttle', 0.0):.2f} "
                f"{dbg.get('brake', 0.0):.2f}]",
                y_offset=34,
            )

        pygame.display.flip()

    def _draw_hud(self, text: str, y_offset: int = 8) -> None:
        """
        @brief Render a semi-transparent HUD bar at the top of the screen.
        @param text: Text to display.
        @param y_offset: Vertical position from the top of the window (pixels).
        """
        txt_surf = self._hud_font.render(text, True, _C_HUD_TEXT)
        bar = pygame.Surface(
            (txt_surf.get_width() + 16, txt_surf.get_height() + 8), pygame.SRCALPHA
        )
        bar.fill(_C_HUD_BG)
        bar.blit(txt_surf, (8, 4))
        self._screen.blit(bar, (8, y_offset))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="2D bird's-eye Pygame visualiser for CARLA parking."
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Path to vis_history.jsonl (default: outputs/vis_history.jsonl).",
    )
    _args = parser.parse_args()
    LiveVisualiser(history_file=_args.history_file).run()
