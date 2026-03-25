"""
@file visualiser.py
@brief Live 2D bird's-eye visualiser for CARLA parking training and evaluation.

Tails outputs/vis_history.jsonl in real-time, rendering each frame as a
top-down diagram using Pygame for zero-overhead rendering. Creates a signal
file (outputs/.vis_active) so the environment knows to write frames. Closing
the window removes the signal file and the env stops writing.

Layers drawn (back to front):
  1. Lot boundary polygon (light grey fill)
  2. Bay outlines by type: perpendicular=blue, angled=yellow, parallel=violet
  3. Target bay (bright green, thick outline + heading arrows)
  4. Static parked vehicles (orange rectangles)
  5. Patrol NPC vehicles (red rectangles)  -- dynamic
  6. Pedestrians (teal circles)            -- dynamic
  7. Ego trajectory trail (faded cyan)     -- dynamic
  8. Ego vehicle (cyan rectangle + heading arrow) -- dynamic
  9. HUD overlay (episode info, navigation hint)

Usage:
    make visualise               # Live window during training
"""

import json
import math
import signal
import sys
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
# Colours (convert hex palette to Pygame RGB tuples once at import time)
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
_C_TRAIL = (*hex_to_rgb(HEX_EGO), 100)   # RGBA with alpha for trail
_C_BG = (248, 248, 248)
_C_HUD_BG = (30, 30, 30, 180)            # RGBA semi-transparent HUD
_C_HUD_TEXT = (212, 212, 212)
_C_CONE = hex_to_rgb(HEX_CONE)

# ---------------------------------------------------------------------------
# Window and rendering constants
# ---------------------------------------------------------------------------

_LEGEND_W = 180         # Width of the right-hand legend panel (pixels)
_MAP_W = 900            # Width of the map viewport (pixels)
_WINDOW_W = _MAP_W + _LEGEND_W  # Total window width
_WINDOW_H = 900
_FPS_CAP = 120          # Pygame FPS cap -- well above CARLA sim rate
_MARGIN_PX = 40         # Pixel margin inside the map viewport
_HUD_FONT_SIZE = 14
_LABEL_FONT_SIZE = 13

# Vehicle dimensions in metres
_EGO_HALF_L = 2.25
_EGO_HALF_W = 1.0
_NPC_HALF_L = 2.35
_NPC_HALF_W = 1.05

# Trail
_TRAIL_MAX_POINTS = 500   # Cap to avoid slow poly-line draws

# Episode buffering
_MIN_EPISODE_FRAMES = 40

# JSONL / signal paths
_DEFAULT_HISTORY_FILE = Path("outputs/vis_history.jsonl")
_SIGNAL_FILE = Path("outputs/.vis_active")

# Poll sleep when no new data (seconds) -- keeps CPU usage low
_POLL_SLEEP = 0.02


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
    @param yaw_deg: Yaw in degrees (CARLA convention, y-up).
    @return Array of shape (4, 2).
    """
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    local = np.array([
        [ half_l,  half_w],
        [-half_l,  half_w],
        [-half_l, -half_w],
        [ half_l, -half_w],
    ])
    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (R @ local.T).T + np.array([cx, cy])


def _world_to_screen(
    pts: np.ndarray,
    origin: np.ndarray,
    scale: float,
) -> List[Tuple[int, int]]:
    """
    @brief Convert world-space points to Pygame screen pixels.

    World x -> screen right, world y -> screen up (flipped because Pygame
    y=0 is at top).

    @param pts: Array of shape (N, 2) in metres.
    @param origin: World coordinate that maps to the top-left of the viewport
                   after margin is applied.
    @param scale: Pixels per metre.
    @return List of (px, py) integer tuples.
    """
    shifted = pts - origin
    px = (shifted[:, 0] * scale + _MARGIN_PX).astype(int)
    # Flip y: world up = screen up means we negate y before scaling
    py = (_WINDOW_H - _MARGIN_PX - shifted[:, 1] * scale).astype(int)
    return list(zip(px.tolist(), py.tolist()))


def _w2s_single(
    x: float, y: float, origin: np.ndarray, scale: float
) -> Tuple[int, int]:
    """@brief Convert a single world point to screen pixel."""
    pts = np.array([[x, y]])
    return _world_to_screen(pts, origin, scale)[0]


# ---------------------------------------------------------------------------
# Static scene surface builder
# ---------------------------------------------------------------------------


def _build_static_surface(
    state: Dict[str, Any],
    origin: np.ndarray,
    scale: float,
) -> pygame.Surface:
    """
    @brief Render the parts of the scene that do not change within an episode
           into a dedicated Surface. This surface is blitted each frame instead
           of redrawing all geometry from scratch.

    Includes: lot boundary, bay outlines, target bay arrows, static parked
    vehicles, legend.

    @param state: First frame of the episode (or any frame -- static data
                  is identical across frames).
    @param origin: World origin for the viewport.
    @param scale: Pixels per metre.
    @return Opaque Surface with the static scene painted on it.
    """
    # Only covers the map viewport -- legend panel is drawn separately.
    surf = pygame.Surface((_MAP_W, _WINDOW_H))
    surf.fill(_C_BG)

    # -- Lot boundary ---------------------------------------------------------
    corners_raw = state.get("corners", [])
    if corners_raw:
        lot_pts = np.array([[c["x"], c["y"]] for c in corners_raw])
        screen_pts = _world_to_screen(lot_pts, origin, scale)
        if len(screen_pts) >= 3:
            pygame.draw.polygon(surf, _C_LOT, screen_pts)
            pygame.draw.polygon(surf, _C_LOT_EDGE, screen_pts, 2)

    # -- Bay outlines ---------------------------------------------------------
    target_bay = state.get("target_bay", {})
    target_id = target_bay.get("bay_id", "")
    for bay in state.get("bays", []):
        bay_id = bay.get("id", bay.get("bay_id", ""))
        is_target = bay_id == target_id
        bay_type = bay.get("bay_type", "perpendicular")
        if is_target:
            colour = _C_TARGET_BAY
            lw = 3
        elif bay_type == "angled":
            colour = _C_ANGLED_BAY
            lw = 1
        elif bay_type == "parallel":
            colour = _C_PARALLEL_BAY
            lw = 1
        else:
            colour = _C_PERP_BAY
            lw = 1

        half_d = float(bay.get("depth", 5.0)) / 2.0
        half_w = float(bay.get("width", 2.5)) / 2.0
        yaw_deg = float(bay.get("yaw_deg", bay.get("yaw", 0.0)))
        corners = _rot_corners(float(bay["x"]), float(bay["y"]), half_d, half_w, yaw_deg)
        spts = _world_to_screen(corners, origin, scale)
        if len(spts) >= 3:
            pygame.draw.polygon(surf, colour, spts, lw)

        # Heading arrows for target bay (both directions)
        if is_target:
            bx, by = float(bay["x"]), float(bay["y"])
            arrow_len = half_d * 0.8
            for yaw_r in (math.radians(yaw_deg), math.radians(yaw_deg + 180.0)):
                tip = (bx + arrow_len * math.cos(yaw_r),
                       by + arrow_len * math.sin(yaw_r))
                s0 = _w2s_single(bx, by, origin, scale)
                s1 = _w2s_single(tip[0], tip[1], origin, scale)
                pygame.draw.line(surf, _C_TARGET_BAY, s0, s1, 2)
                # Arrowhead
                dx, dy = s1[0] - s0[0], s1[1] - s0[1]
                length = math.hypot(dx, dy) or 1.0
                ux, uy = dx / length, dy / length
                left = (int(s1[0] - ux * 8 + uy * 5),
                        int(s1[1] - uy * 8 - ux * 5))
                right = (int(s1[0] - ux * 8 - uy * 5),
                         int(s1[1] - uy * 8 + ux * 5))
                pygame.draw.polygon(surf, _C_TARGET_BAY, [s1, left, right])

    # -- Static parked vehicles -----------------------------------------------
    for actor in state.get("actors", []):
        if actor.get("type", "static") != "npc":
            corners = _rot_corners(
                actor["x"], actor["y"], _NPC_HALF_L, _NPC_HALF_W,
                actor.get("yaw", 0.0)
            )
            spts = _world_to_screen(corners, origin, scale)
            if len(spts) >= 3:
                pygame.draw.polygon(surf, _C_STATIC_VEHICLE, spts)

    return surf


def _draw_legend(screen: pygame.Surface) -> None:
    """
    @brief Draw the colour legend in the dedicated right-hand panel.

    The panel occupies x=[_MAP_W, _WINDOW_W] and is drawn directly onto the
    main screen surface so it is never overwritten by map blits.
    """
    font = pygame.font.SysFont("monospace", _LABEL_FONT_SIZE)
    title_font = pygame.font.SysFont("monospace", _LABEL_FONT_SIZE, bold=True)

    # Panel background
    panel_rect = pygame.Rect(_MAP_W, 0, _LEGEND_W, _WINDOW_H)
    pygame.draw.rect(screen, (235, 235, 235), panel_rect)
    # Divider line
    pygame.draw.line(screen, (180, 180, 180), (_MAP_W, 0), (_MAP_W, _WINDOW_H), 2)

    entries = [
        (_C_TARGET_BAY,     "Target bay"),
        (_C_PERP_BAY,       "Perpendicular"),
        (_C_ANGLED_BAY,     "Angled"),
        (_C_PARALLEL_BAY,   "Parallel"),
        (_C_EGO,            "Ego"),
        (_C_PATROL_VEHICLE, "Patrol NPC"),
        (_C_STATIC_VEHICLE, "Parked"),
        (_C_PEDESTRIAN,     "Pedestrian"),
    ]

    x0 = _MAP_W + 10
    y0 = 16
    title = title_font.render("Legend", True, (40, 40, 40))
    screen.blit(title, (x0, y0))
    y0 += title.get_height() + 8

    for colour, label in entries:
        pygame.draw.rect(screen, colour, (x0, y0 + 2, 14, 14))
        txt = font.render(label, True, (50, 50, 50))
        screen.blit(txt, (x0 + 20, y0))
        y0 += 22


# ---------------------------------------------------------------------------
# Main visualiser class
# ---------------------------------------------------------------------------


class LiveVisualiser:
    """
    @class LiveVisualiser
    @brief Live 2D bird's-eye Pygame visualiser for CARLA parking.

    Tails outputs/vis_history.jsonl written by the training env and renders
    each frame with Pygame for real-time performance. The static scene (lot,
    bays, parked vehicles) is pre-rendered to a surface once per episode;
    only dynamic actors (ego, patrol NPCs, pedestrians, trail) are redrawn
    each frame via blitting.

    Controls:
        right arrow  -- next episode in history
        left arrow   -- previous episode in history
        ESC / Q      -- exit
    """

    def __init__(self, history_file: Optional[Path] = None) -> None:
        """
        @brief Initialise Pygame window and internal state.
        @param history_file: Path to vis_history.jsonl. Defaults to
               outputs/vis_history.jsonl.
        """
        self._history_file = history_file or _DEFAULT_HISTORY_FILE
        self._signal_file = _SIGNAL_FILE

        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H))
        pygame.display.set_caption(
            "CARLA Parking Visualiser  |  <- -> navigate  |  F fullscreen  |  ESC quit"
        )
        self._clock = pygame.time.Clock()
        self._hud_font = pygame.font.SysFont("monospace", _HUD_FONT_SIZE)

        # Static scene surface (rebuilt once per episode)
        self._static_surf: Optional[pygame.Surface] = None
        self._static_episode_id: Optional[int] = None
        # Trail surface (alpha-blended, rebuilt each frame)
        self._trail_surf = pygame.Surface((_WINDOW_W, _WINDOW_H), pygame.SRCALPHA)

        # Viewport transform (set when the first frame of an episode arrives)
        self._origin = np.zeros(2)
        self._scale = 1.0

        # JSONL state
        self._file_offset: int = 0
        self._current_buf: List[Dict[str, Any]] = []
        self._current_episode_id: Optional[int] = None

        # Episode history
        self._episode_history: List[List[Dict[str, Any]]] = []
        self._play_index: int = -1
        self._skip_requested: bool = False
        self._user_navigated: bool = False

        self._exit_requested: bool = False
        self._fullscreen: bool = False

        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._signal_file.touch()

    # ------------------------------------------------------------------
    # Signal / cleanup
    # ------------------------------------------------------------------

    def _remove_signal(self) -> None:
        """@brief Remove the vis_active signal file."""
        try:
            self._signal_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _delete_history(self) -> None:
        """@brief Delete vis_history.jsonl on exit."""
        try:
            self._history_file.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Public entry point
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
            self._remove_signal()
            self._delete_history()
            pygame.quit()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """
        @brief Ingest JSONL frames and render them as fast as Pygame allows.

        In live mode (no manual navigation) the visualiser always shows the
        latest frame of the latest episode -- no replay delay. When the user
        presses left to browse history, full frame-by-frame playback runs for
        the selected episode.
        """
        while not self._exit_requested:
            self._handle_events()
            self._ingest_new_frames()

            n_hist = len(self._episode_history)

            if n_hist == 0:
                self._draw_waiting()
                self._clock.tick(_FPS_CAP)
                time.sleep(_POLL_SLEEP)
                continue

            # Determine which episode to show
            if self._skip_requested:
                self._skip_requested = False
            else:
                if self._play_index == -1:
                    self._play_index = 0
                elif self._user_navigated:
                    self._draw_paused()
                    self._clock.tick(_FPS_CAP)
                    time.sleep(_POLL_SLEEP)
                    continue
                else:
                    self._play_index = n_hist - 1

            episode = self._episode_history[self._play_index]
            n = len(episode)
            dt = episode[0].get("carla_timestep", 0.05) if episode else 0.05
            play_eid = episode[0].get("episode_id", "?") if episode else "?"
            first_eid = self._episode_history[0][0].get("episode_id", "?")
            last_eid = self._episode_history[-1][0].get("episode_id", "?")

            # In auto-advance mode: show only the last frame instantly.
            # In manual-browse mode: play every frame.
            frames_to_show = (
                [episode[-1]]
                if not self._user_navigated and self._play_index == n_hist - 1
                else episode
            )

            for frame in frames_to_show:
                if self._exit_requested or self._skip_requested:
                    break
                self._handle_events()
                self._ingest_new_frames()
                t0 = time.monotonic()
                self._draw_frame(frame)
                self._clock.tick(_FPS_CAP)
                # Real-time throttle during manual replay: sleep the remainder
                # of the sim timestep so playback matches wall-clock speed.
                # In auto-advance (live) mode we skip this so the latest frame
                # appears immediately.
                if self._user_navigated:
                    elapsed = time.monotonic() - t0
                    remaining = dt - elapsed
                    if remaining > 0.002:
                        time.sleep(remaining)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        """@brief Process Pygame event queue (key presses, window close)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit_requested = True
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

    def _handle_key(self, key: int) -> None:
        """@brief Handle a single keydown event."""
        n = len(self._episode_history)
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self._exit_requested = True
            return
        if key == pygame.K_f:
            self._fullscreen = not self._fullscreen
            flags = pygame.FULLSCREEN if self._fullscreen else 0
            self._screen = pygame.display.set_mode((_WINDOW_W, _WINDOW_H), flags)
            # Force static scene rebuild for the new surface
            self._static_episode_id = None
            return
        if n == 0:
            return
        if key == pygame.K_RIGHT:
            new_idx = self._play_index + 1 if self._play_index + 1 < n else None
        elif key == pygame.K_LEFT:
            new_idx = self._play_index - 1 if self._play_index - 1 >= 0 else None
        else:
            return
        if new_idx is not None and new_idx != self._play_index:
            self._play_index = new_idx
            self._skip_requested = True
            # Re-enable auto-advance when the user reaches the latest episode.
            self._user_navigated = new_idx < n - 1

    # ------------------------------------------------------------------
    # JSONL ingestion
    # ------------------------------------------------------------------

    def _ingest_new_frames(self) -> None:
        """@brief Read new JSONL lines and sort into episode buffers."""
        try:
            for frame in self._read_new_frames():
                eid = frame.get("episode_id")
                if eid != self._current_episode_id:
                    if self._current_buf:
                        self._flush_episode_buf()
                    if (
                        self._current_episode_id is not None
                        and eid is not None
                        and eid < self._current_episode_id
                    ):
                        self._episode_history.clear()
                        self._play_index = -1
                        self._skip_requested = False
                        self._user_navigated = False
                        print(
                            f"[visualiser] episode id reset "
                            f"({self._current_episode_id} -> {eid}) "
                            f"-- new env instance, session cleared"
                        )
                    self._current_buf = []
                    self._current_episode_id = eid
                self._current_buf.append(frame)

                # Flush immediately when the episode ends so the visualiser
                # can display it without waiting for the next episode to start.
                if frame.get("end_reason"):
                    self._flush_episode_buf()
                    self._current_buf = []
                    self._current_episode_id = None
        except Exception:
            pass

    def _flush_episode_buf(self) -> None:
        """@brief Commit the current buffer to history if it is long enough."""
        buf = self._current_buf
        n = len(buf)
        eid = self._current_episode_id
        last = buf[-1] if buf else {}
        end_reason = last.get("end_reason", "unknown")
        steps = last.get("episode_step", "?")
        sim_t = last.get("sim_time", 0.0)

        if n < _MIN_EPISODE_FRAMES:
            print(
                f"[visualiser] SKIP  ep {eid}  {n} frames  "
                f"steps={steps}  t={sim_t:.1f}s  end={end_reason}  "
                f"(need >={_MIN_EPISODE_FRAMES} frames)"
            )
            return

        self._episode_history.append(buf)
        print(
            f"[visualiser] KEEP  ep {eid}  {n} frames  "
            f"steps={steps}  t={sim_t:.1f}s  end={end_reason}"
        )

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
                size = self._history_file.stat().st_size
                if self._file_offset > size:
                    self._file_offset = 0
                f.seek(self._file_offset)
                chunk = f.read()
                if not chunk:
                    return []
                last_nl = chunk.rfind(b"\n")
                if last_nl == -1:
                    return []
                complete = chunk[: last_nl + 1]
                self._file_offset += last_nl + 1
                for raw in complete.split(b"\n"):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        frames.append(json.loads(raw.decode("utf-8")))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
        except OSError:
            pass
        return frames

    # ------------------------------------------------------------------
    # Viewport calculation
    # ------------------------------------------------------------------

    def _compute_viewport(self, state: Dict[str, Any]) -> None:
        """
        @brief Compute origin and scale so the lot fits inside the window.

        Uses the lot corner polygon if available, falling back to the ego
        position. Sets self._origin and self._scale.
        @param state: Any frame from the episode.
        """
        corners_raw = state.get("corners", [])
        if corners_raw:
            pts = np.array([[c["x"], c["y"]] for c in corners_raw])
        else:
            ego = state.get("ego", {})
            cx = float(ego.get("x", 0.0))
            cy = float(ego.get("y", 0.0))
            pts = np.array([[cx - 30, cy - 30], [cx + 30, cy + 30]])

        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn
        span = np.where(span < 1.0, 1.0, span)

        usable_w = _MAP_W - 2 * _MARGIN_PX
        usable_h = _WINDOW_H - 2 * _MARGIN_PX
        self._scale = min(usable_w / span[0], usable_h / span[1])
        self._origin = mn

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_waiting(self) -> None:
        """@brief Render a 'waiting for data' splash screen."""
        self._screen.fill(_C_BG)
        _draw_legend(self._screen)
        font = pygame.font.SysFont("monospace", 18)
        txt = font.render(
            f"Waiting for first valid episode "
            f"(need >= {_MIN_EPISODE_FRAMES} frames)...",
            True, (80, 80, 80),
        )
        self._screen.blit(
            txt,
            ((_MAP_W - txt.get_width()) // 2,
             (_WINDOW_H - txt.get_height()) // 2),
        )
        pygame.display.flip()

    def _draw_paused(self) -> None:
        """@brief Render the last drawn frame with a 'paused' HUD overlay."""
        n_hist = len(self._episode_history)
        if self._static_surf is not None:
            self._screen.blit(self._static_surf, (0, 0))
        _draw_legend(self._screen)
        last_eid = self._episode_history[-1][0].get("episode_id", "?")
        cur_eid = (
            self._episode_history[self._play_index][0].get("episode_id", "?")
            if 0 <= self._play_index < n_hist else "?"
        )
        self._draw_hud(
            f"Paused ep {cur_eid}  |  "
            f"{n_hist} eps buffered (latest: ep {last_eid})  |  "
            f"<- -> navigate",
            y_offset=8,
        )
        pygame.display.flip()

    def _draw_frame(self, state: Dict[str, Any]) -> None:
        """
        @brief Render a single frame: rebuild static surface if episode changed,
               then blit static + dynamic layers.
        @param state: Parsed frame dict from vis_history.jsonl.
        """
        episode_id = state.get("episode_id")

        # Rebuild static surface once per episode
        if episode_id != self._static_episode_id:
            self._static_episode_id = episode_id
            self._compute_viewport(state)
            self._static_surf = _build_static_surface(
                state, self._origin, self._scale
            )

        assert self._static_surf is not None
        # Blit the pre-rendered static scene (map area only)
        self._screen.blit(self._static_surf, (0, 0))
        # Legend panel -- drawn every frame so it is never overwritten
        _draw_legend(self._screen)

        origin = self._origin
        scale = self._scale

        # -- Trail (alpha-blended, clipped to map area) ----------------------
        trail = state.get("trajectory", [])
        if len(trail) > 1:
            pts = trail[-_TRAIL_MAX_POINTS:]
            world = np.array(pts)
            spts = _world_to_screen(world, origin, scale)
            if len(spts) >= 2:
                trail_surf = pygame.Surface((_MAP_W, _WINDOW_H), pygame.SRCALPHA)
                pygame.draw.lines(trail_surf, _C_TRAIL, False, spts, 2)
                self._screen.blit(trail_surf, (0, 0))

        # -- Patrol NPC vehicles ---------------------------------------------
        for actor in state.get("actors", []):
            if actor.get("type", "static") == "npc":
                corners = _rot_corners(
                    actor["x"], actor["y"],
                    _NPC_HALF_L, _NPC_HALF_W,
                    actor.get("yaw", 0.0),
                )
                spts = _world_to_screen(corners, origin, scale)
                if len(spts) >= 3:
                    pygame.draw.polygon(self._screen, _C_PATROL_VEHICLE, spts)

        # -- Pedestrians -----------------------------------------------------
        for ped in state.get("pedestrians", []):
            sx, sy = _w2s_single(ped["x"], ped["y"], origin, scale)
            r = max(2, int(0.4 * scale))
            pygame.draw.circle(self._screen, _C_PEDESTRIAN, (sx, sy), r)

        # -- Ego vehicle -----------------------------------------------------
        ego = state.get("ego", {})
        if ego:
            ex, ey = float(ego["x"]), float(ego["y"])
            eyaw = float(ego["yaw"])
            corners = _rot_corners(ex, ey, _EGO_HALF_L, _EGO_HALF_W, eyaw)
            spts = _world_to_screen(corners, origin, scale)
            if len(spts) >= 3:
                pygame.draw.polygon(self._screen, _C_EGO, spts)

            # Heading arrow
            arrow_len = 3.5
            yaw_r = math.radians(eyaw)
            tip = (ex + arrow_len * math.cos(yaw_r),
                   ey + arrow_len * math.sin(yaw_r))
            s0 = _w2s_single(ex, ey, origin, scale)
            s1 = _w2s_single(tip[0], tip[1], origin, scale)
            pygame.draw.line(self._screen, _C_EGO, s0, s1, 2)
            dx, dy = s1[0] - s0[0], s1[1] - s0[1]
            length = math.hypot(dx, dy) or 1.0
            ux, uy = dx / length, dy / length
            left = (int(s1[0] - ux * 8 + uy * 5),
                    int(s1[1] - uy * 8 - ux * 5))
            right = (int(s1[0] - ux * 8 - uy * 5),
                     int(s1[1] - uy * 8 + ux * 5))
            pygame.draw.polygon(self._screen, _C_EGO, [s1, left, right])

        # -- HUD -------------------------------------------------------------
        n_hist = len(self._episode_history)
        first_eid = (
            self._episode_history[0][0].get("episode_id", "?")
            if n_hist > 0 else "?"
        )
        last_eid = (
            self._episode_history[-1][0].get("episode_id", "?")
            if n_hist > 0 else "?"
        )
        hud = (
            f"Floor: {state.get('floor_plan', '?')}  "
            f"Ep: {episode_id}  "
            f"Step: {state.get('episode_step', '?')}  "
            f"t={state.get('sim_time', 0.0):.2f}s  "
            f"[{self._play_index + 1}/{n_hist} ep{first_eid}-{last_eid}  <- ->]"
        )
        self._draw_hud(hud, y_offset=8)

        # Debug HUD line -- only rendered when the frame carries debug data
        dbg = state.get("debug")
        if dbg:
            debug_hud = (
                f"err={dbg.get('pos_err', 0.0):.2f}m "
                f"yaw={dbg.get('yaw_err_deg', 0.0):.1f}deg "
                f"spd={dbg.get('speed', 0.0):.2f}m/s "
                f"rwd={dbg.get('reward', 0.0):.3f} | "
                f"cov={dbg.get('cov_rms', 0.0):.3f} "
                f"drift={dbg.get('ekf_drift', 0.0):.2f}m "
                f"lidar={dbg.get('lidar_pts', 0)}pts | "
                f"obs={dbg.get('obs_dist', 0.0):.1f}m({dbg.get('obs_type', 'sta')}) "
                f"act=[{dbg.get('steer', 0.0):.2f} "
                f"{dbg.get('throttle', 0.0):.2f} "
                f"{dbg.get('brake', 0.0):.2f}]"
            )
            self._draw_hud(debug_hud, y_offset=34)

        pygame.display.flip()

    def _draw_hud(self, text: str, y_offset: int = 8) -> None:
        """
        @brief Render a semi-transparent HUD bar at the top of the screen.
        @param text: Text to display in the bar.
        @param y_offset: Vertical position in pixels from the top of the window.
        """
        txt_surf = self._hud_font.render(text, True, _C_HUD_TEXT)
        bar = pygame.Surface((txt_surf.get_width() + 16, txt_surf.get_height() + 8),
                              pygame.SRCALPHA)
        bar.fill(_C_HUD_BG)
        bar.blit(txt_surf, (8, 4))
        self._screen.blit(bar, (8, y_offset))
