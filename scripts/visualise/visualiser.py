"""
@file visualiser.py
@brief Live 2D bird's-eye visualiser for CARLA parking training and evaluation.

Tails outputs/vis_history.jsonl and renders each frame as it arrives using
Pygame. Creates outputs/.vis_active so the environment starts writing frames;
removing it (on window close) stops the env writing. The GNSS fix-state tier
drives a colour-coded panel and an uncertainty ring around the ego vehicle
scaled to that tier's 1-sigma position noise. Optional MP4 recording
(@see recorder.FrameRecorder) captures the window.
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
    HEX_EGO,
    HEX_LOT,
    HEX_PATROL_VEHICLE,
    HEX_PEDESTRIAN_ZONE,
    HEX_STATIC_VEHICLE,
    HEX_TARGET_BAY,
    hex_to_rgb,
)
from scripts.visualise.gnss_tiers import (
    GnssTier,
    display_label,
    load_gnss_tiers,
    resolve_tier,
    unknown_colour,
)
from scripts.visualise.recorder import FrameRecorder

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
_C_TRAIL = (*hex_to_rgb(HEX_EGO), 120)
_C_BG = (248, 248, 248)
_C_HUD_BG = (24, 24, 24)
_C_HUD_TEXT = (245, 245, 245)
_C_HUD_DIM = (170, 170, 170)
_C_LEGEND_BG = (235, 235, 235)
_C_LEGEND_TEXT = (50, 50, 50)

# Alpha applied to the ego uncertainty ring. Low enough that bays and the lot
# stay readable underneath a 5 m degraded-tier ring, which can span the width of
# several bays.
_RING_ALPHA = 55
_RING_EDGE_ALPHA = 170

# Every layout in this project is perpendicular-only, so the angled/parallel
# entries never fire in practice; they are kept so a future layout renders
# correctly rather than falling back to the perpendicular style.
_BAY_STYLE: Dict[str, Tuple[Any, int]] = {
    "angled": (_C_ANGLED_BAY, 2),
    "parallel": (_C_PARALLEL_BAY, 2),
    "perpendicular": (_C_PERP_BAY, 2),
}
_BAY_STYLE_DEFAULT: Tuple[Any, int] = (_C_PERP_BAY, 2)

# Font sizes and the legend width are multiplied by the UI scale at startup
# (see _Layout), so the whole interface can be enlarged for a projector or a
# recording with a single --ui-scale value.
_UI_SCALE_DEFAULT = 1.5

_MAP_W = 900
_BASE_LEGEND_W = 160
_BASE_HUD_FONT = 13
_BASE_LABEL_FONT = 12
_BASE_TIER_FONT = 21

# Map area height (HUD is drawn in a band BELOW the map, not overlaid on it).
# _MAP_H is only the INITIAL height (used for the waiting splash); once the
# first frame arrives, _compute_viewport() resizes it to the lot's aspect
# ratio. _MAP_H_MAX caps it so a tall lot cannot exceed the screen.
_MAP_H = 600
_MAP_H_MIN = 300
_MAP_H_MAX = 900

_FPS_CAP = 120
_MARGIN_PX = 40
_EGO_HALF_L = 2.25
_EGO_HALF_W = 1.0
_NPC_HALF_L = 2.35
_NPC_HALF_W = 1.05
_TRAIL_MAX_POINTS = 500
_POLL_SLEEP = 0.02

_DEFAULT_HISTORY_FILE = Path("outputs/vis_history.jsonl")
_SIGNAL_FILE = Path("outputs/.vis_active")
_DEFAULT_RECORD_DIR = Path("outputs/recordings")


class _Layout:
    """
    @class _Layout
    @brief Scale-dependent font sizes and panel geometry.

    Centralises every pixel size that depends on --ui-scale so the drawing code
    never multiplies by the scale itself.
    """

    def __init__(self, ui_scale: float) -> None:
        """
        @brief Derive sizes from the UI scale.
        @param ui_scale: Multiplier applied to fonts and the legend width.
        """
        self.scale = max(0.5, ui_scale)
        self.hud_font = self.scaled(_BASE_HUD_FONT)
        self.label_font = self.scaled(_BASE_LABEL_FONT)
        self.tier_font = self.scaled(_BASE_TIER_FONT)
        self.legend_w = self.scaled(_BASE_LEGEND_W)
        self.window_w = _MAP_W + self.legend_w

        # HUD band height: sized from font metrics and always reserving the
        # debug bar (shown only when debug=True in env_config.yaml), so
        # nothing is clipped at any scale or in either debug mode.
        self.bar_pitch = self.hud_font + 8
        self.tier_block_h = self.tier_font + self.hud_font + 12
        self.hud_band_h = self.tier_block_h + 3 * self.bar_pitch + self.scaled(12)
        # Stroke widths, so shapes survive being scaled down into a GIF.
        self.thick_line = max(2, int(round(3 * self.scale / 1.5)))

    def scaled(self, base: int) -> int:
        """
        @brief Scale a base pixel size.
        @param base: Unscaled size in pixels.
        @return Scaled size, at least 1.
        """
        return max(1, int(round(base * self.scale)))


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
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (rotation @ local.T).T + np.array([cx, cy])


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


def _draw_arrow_head(
    surface: pygame.Surface,
    colour: Any,
    start: Tuple[int, int],
    end: Tuple[int, int],
    size: int,
) -> None:
    """
    @brief Draw a filled triangular head at the end of a line.
    @param surface: Target surface.
    @param colour: Fill colour.
    @param start: Line start pixel (sets the direction).
    @param end: Line end pixel (where the head is drawn).
    @param size: Head length in pixels.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    half = size * 0.6
    left = (int(end[0] - ux * size + uy * half), int(end[1] - uy * size - ux * half))
    right = (int(end[0] - ux * size - uy * half), int(end[1] - uy * size + ux * half))
    pygame.draw.polygon(surface, colour, [end, left, right])


def _build_static_surface(
    state: Dict[str, Any],
    origin: np.ndarray,
    scale: float,
    map_h: int,
    layout: _Layout,
) -> pygame.Surface:
    """
    @brief Render the static scene elements into a surface (rebuilt once per episode).

    Includes lot boundary, bay outlines, target bay arrows, and static parked vehicles.

    @param state: Any frame from the episode (static data is identical across frames).
    @param origin: World origin for the viewport.
    @param scale: Pixels per metre.
    @param map_h: Current map-area height in pixels.
    @param layout: Scale-dependent sizes.
    @return Opaque Surface with static scene painted on it.
    """
    surf = pygame.Surface((_MAP_W, map_h))
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
            line_w = layout.thick_line + 1
        else:
            colour, line_w = _BAY_STYLE.get(
                bay.get("bay_type", "perpendicular"), _BAY_STYLE_DEFAULT
            )

        half_d = float(bay.get("depth", 5.0)) / 2.0
        half_w = float(bay.get("width", 2.5)) / 2.0
        yaw_deg = float(bay.get("yaw_deg", bay.get("yaw", 0.0)))
        bx, by = float(bay["x"]), float(bay["y"])
        spts = _world_to_screen(
            _rot_corners(bx, by, half_d, half_w, yaw_deg), origin, scale
        )
        pygame.draw.polygon(surf, colour, spts, line_w)

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
                pygame.draw.line(surf, _C_TARGET_BAY, s0, s1, layout.thick_line)
                _draw_arrow_head(surf, _C_TARGET_BAY, s0, s1, 4 + layout.thick_line * 2)

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


class LiveVisualiser:
    """
    @class LiveVisualiser
    @brief Live 2D bird's-eye Pygame visualiser for CARLA parking.

    Tails outputs/vis_history.jsonl and renders each frame as it arrives.
    The static scene (lot, bays, parked vehicles) is rebuilt once per episode.

    Controls:
        R        - start/stop MP4 recording
        F        - toggle fullscreen
        ESC / Q  - exit
    """

    def __init__(
        self,
        history_file: Optional[Path] = None,
        ui_scale: float = _UI_SCALE_DEFAULT,
        record: bool = False,
        record_dir: Optional[Path] = None,
        fps: int = 30,
        show_episode: bool = False,
    ) -> None:
        """
        @brief Initialise Pygame window and state.
        @param history_file: Path to vis_history.jsonl. Defaults to
               outputs/vis_history.jsonl.
        @param ui_scale: Font/legend scale multiplier.
        @param record: Start recording immediately on launch.
        @param record_dir: Directory for MP4 output.
        @param fps: Recording frame rate.
        @param show_episode: Include the episode number in the context line.
        """
        self._history_file = history_file or _DEFAULT_HISTORY_FILE
        self._signal_file = _SIGNAL_FILE
        self._layout = _Layout(ui_scale)
        self._show_episode = show_episode

        # Map and window heights start at the defaults and are resized to the
        # lot aspect ratio once the first frame arrives (see _compute_viewport).
        self._map_h: int = _MAP_H
        self._window_h: int = _MAP_H + self._layout.hud_band_h

        pygame.init()
        pygame.font.init()
        self._screen = pygame.display.set_mode((self._layout.window_w, self._window_h))
        pygame.display.set_caption(
            "CARLA Parking Visualiser  |  R record  |  F fullscreen  |  ESC quit"
        )
        self._clock = pygame.time.Clock()
        self._init_fonts()

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
        # Wall-clock time the visualiser started, used by the waiting splash
        # to show how long it has been waiting for the first frame.
        self._start_time: float = time.time()
        # Trail surface is reallocated whenever the map area is resized.
        self._trail_surf: pygame.Surface = pygame.Surface(
            (_MAP_W, self._map_h), pygame.SRCALPHA
        )

        # GNSS tier presentation, loaded once from the profiles YAML.
        self._gnss_tiers = load_gnss_tiers()

        self._fps = fps
        self._record_dir = record_dir or _DEFAULT_RECORD_DIR
        self._recorder: Optional[FrameRecorder] = None
        self._record_on_start = record

        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._signal_file.touch()

    def _init_fonts(self) -> None:
        """@brief Create the fonts for the current UI scale."""
        self._hud_font = pygame.font.SysFont("monospace", self._layout.hud_font)
        self._label_font = pygame.font.SysFont("monospace", self._layout.label_font)
        self._label_bold = pygame.font.SysFont(
            "monospace", self._layout.label_font, bold=True
        )
        self._tier_font = pygame.font.SysFont(
            "monospace", self._layout.tier_font, bold=True
        )

    def run(self) -> None:
        """@brief Open the Pygame window and enter the main loop."""
        self._exit_requested = False

        def _sigint(_sig: int, _frame: Any) -> None:
            self._exit_requested = True

        signal.signal(signal.SIGINT, _sigint)

        if self._record_on_start:
            self._toggle_recording()

        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._finish_recording()
            try:
                self._signal_file.unlink(missing_ok=True)
            except OSError:
                pass
            pygame.quit()

    def _toggle_recording(self) -> None:
        """
        @brief Start recording, or stop and finalise the current file.

        A missing ffmpeg is surfaced in the HUD rather than raised, so the
        viewer keeps running on a host without it.
        """
        if self._recorder is not None:
            self._finish_recording()
            return

        stamp = time.strftime("%d-%m-%Y-%H%M%S")
        output = self._record_dir / f"{stamp}.mp4"
        recorder = FrameRecorder(
            output, self._layout.window_w, self._window_h, fps=self._fps
        )
        try:
            recorder.start()
        except RuntimeError as exc:
            print(f"Recording unavailable: {exc}")
            return
        self._recorder = recorder
        print(f"Recording to {output}")

    def _finish_recording(self) -> None:
        """@brief Stop the recorder and report where the file landed."""
        if self._recorder is None:
            return
        written = self._recorder.stop()
        if written is not None:
            print(f"Recording saved: {written}")
        else:
            print("Recording stopped before any frame was written.")
        self._recorder = None

    def _run_loop(self) -> None:
        """
        @brief Read new JSONL frames and render them as fast as they arrive.

        Polls the JSONL file each iteration. When no new data is available
        the loop sleeps briefly to avoid busy-waiting. The signal file is
        re-asserted every iteration so a stale cleanup cannot stop the env
        from writing while this window is open.
        """
        while not self._exit_requested:
            self._handle_events()

            # Keep the signal file present so the env never stops writing.
            if not self._signal_file.exists():
                try:
                    self._signal_file.touch()
                except OSError:
                    pass

            frames = self._read_new_frames()
            if frames:
                self._ever_received_frame = True
                # Render only the latest frame to avoid falling behind
                self._draw_frame(frames[-1])
            elif not self._ever_received_frame:
                self._draw_waiting()
                time.sleep(_POLL_SLEEP)
            else:
                # Data stream dry - hold the last frame rather than flicker,
                # but still offer it to the recorder so a paused stream
                # records as a still rather than compressing video time.
                self._capture_frame()
                time.sleep(_POLL_SLEEP)

            self._clock.tick(_FPS_CAP)

    def _capture_frame(self) -> None:
        """@brief Offer the current window to the recorder, if active."""
        if self._recorder is not None:
            self._recorder.capture(self._screen)

    def _handle_events(self) -> None:
        """@brief Process Pygame event queue."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._exit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self._exit_requested = True
                elif event.key == pygame.K_r:
                    self._toggle_recording()
                elif event.key == pygame.K_f:
                    self._fullscreen = not self._fullscreen
                    flags = pygame.FULLSCREEN if self._fullscreen else 0
                    self._screen = pygame.display.set_mode(
                        (self._layout.window_w, self._window_h), flags
                    )
                    self._static_episode_id = None  # Force static surface rebuild
                    self._vis_trail_episode_id = None  # Force trail reset

    def _read_new_frames(self) -> List[Dict[str, Any]]:
        """
        @brief Read all complete JSONL lines written since the last call.

        The env truncates vis_history.jsonl every N episodes (see
        CARLAParkingEnv._vis_rotation_interval). When the file shrinks below
        _file_offset the offset is reset to zero so the visualiser reads from
        the start of the new data without stalling.

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

    def _compute_viewport(self, state: Dict[str, Any]) -> None:
        """
        @brief Compute origin and scale so the lot fits inside the map viewport.

        Also resizes the map area (and therefore the window) to the lot's
        aspect ratio: the lot is scaled to fill the full map width, and the
        map height is set to whatever that scale needs - clamped to
        [_MAP_H_MIN, _MAP_H_MAX]. This removes the empty vertical strip a
        wide-and-short lot would otherwise leave below the layout.

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
        # Scale so the lot fills the full map width, then size the map height
        # to match - the lot then touches both side margins with no slack.
        self._scale = usable_w / span[0]
        needed_h = int(span[1] * self._scale) + 2 * _MARGIN_PX
        new_map_h = max(_MAP_H_MIN, min(_MAP_H_MAX, needed_h))
        # When a tall lot hits the height clamp, fall back to the fit-both
        # scale so it is not cropped at the bottom.
        usable_h = new_map_h - 2 * _MARGIN_PX
        self._scale = min(self._scale, usable_h / span[1])
        self._origin = mn
        self._resize_map(new_map_h)

    def _resize_map(self, new_map_h: int) -> None:
        """
        @brief Resize the map area and recreate the window/trail surfaces.
        @param new_map_h: New map-area height in pixels.
        """
        if new_map_h == self._map_h:
            return
        self._map_h = new_map_h
        self._window_h = new_map_h + self._layout.hud_band_h
        flags = pygame.FULLSCREEN if self._fullscreen else 0
        self._screen = pygame.display.set_mode(
            (self._layout.window_w, self._window_h), flags
        )
        self._trail_surf = pygame.Surface((_MAP_W, self._map_h), pygame.SRCALPHA)
        # Force a static-surface rebuild: it is sized to the map height.
        self._static_episode_id = None

    def _draw_waiting(self) -> None:
        """
        @brief Render a diagnostic waiting splash when no data has arrived yet.

        Shows an animated spinner, elapsed wait time, and the state of the two
        files the pipeline depends on (the signal file and the history file),
        so a still-loading viewer is visibly distinct from a stalled one.
        """
        self._screen.fill(_C_BG)
        self._draw_legend()

        elapsed = time.time() - self._start_time
        spinner = "|/-\\"[int(elapsed * 4) % 4]

        history_exists = self._history_file.exists()
        if history_exists:
            history_line = f"History file: present ({self._history_file})"
        else:
            history_line = f"History file: not created yet ({self._history_file})"
        signal_line = (
            "Signal file: created (env will write frames once driving)"
            if self._signal_file.exists()
            else "Signal file: MISSING - env will not write frames"
        )

        # The demo driver takes time to load the checkpoint, connect to
        # CARLA, and reset the first episode before any frame is written.
        hint = (
            "Demo is loading the model and connecting to CARLA - "
            "this can take 30-60s."
            if elapsed < 90
            else "Still no frames after 90s - check the demo container logs "
            "(docker ps / docker logs)."
        )

        lines = [
            (self._tier_font, f"{spinner} Waiting for data...  ({elapsed:4.0f}s)"),
            (self._hud_font, ""),
            (self._hud_font, signal_line),
            (self._hud_font, history_line),
            (self._hud_font, ""),
            (self._hud_font, hint),
        ]

        total_h = sum(f.get_height() + 6 for f, _ in lines)
        y = (self._map_h - total_h) // 2
        for font, text in lines:
            if text:
                surf = font.render(text, True, (70, 70, 70))
                self._screen.blit(surf, ((_MAP_W - surf.get_width()) // 2, y))
            y += font.get_height() + 6

        pygame.display.flip()
        self._capture_frame()

    def _draw_frame(self, state: Dict[str, Any]) -> None:
        """
        @brief Render a single frame: rebuild static surface if episode changed,
               then blit static + dynamic layers.
        @param state: Parsed frame dict from vis_history.jsonl.
        """
        episode_id = state.get("episode_id")

        if episode_id != self._static_episode_id:
            # _compute_viewport may resize the map (and reset
            # _static_episode_id to None); set the id afterwards so the
            # rebuilt static surface is not discarded on the next frame.
            self._compute_viewport(state)
            self._static_surf = _build_static_surface(
                state, self._origin, self._scale, self._map_h, self._layout
            )
            self._static_episode_id = episode_id

        assert self._static_surf is not None
        self._screen.blit(self._static_surf, (0, 0))
        self._draw_legend()

        origin, scale = self._origin, self._scale
        ego = state.get("ego", {})

        # Track the live GNSS tier before drawing anything that depends on it.
        tier_name = str(ego.get("gnss_tier", ""))
        tier = resolve_tier(self._gnss_tiers, tier_name)

        # Accumulated ego trail: clear on episode reset, then append current position.
        if episode_id != self._vis_trail_episode_id:
            self._vis_trail = []
            self._vis_trail_episode_id = episode_id
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
            pygame.draw.lines(
                self._trail_surf, _C_TRAIL, False, spts, self._layout.thick_line
            )
            self._screen.blit(self._trail_surf, (0, 0))

        # Uncertainty ring, drawn beneath every actor so it never hides one.
        if ego and tier is not None:
            self._draw_uncertainty_ring(ego, tier, origin, scale)

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
            s0 = _w2s(ex, ey, origin, scale)
            s1 = _w2s(
                ex + 3.5 * math.cos(yaw_r), ey + 3.5 * math.sin(yaw_r), origin, scale
            )
            pygame.draw.line(self._screen, _C_EGO, s0, s1, self._layout.thick_line)
            _draw_arrow_head(
                self._screen, _C_EGO, s0, s1, 4 + self._layout.thick_line * 2
            )

        self._draw_hud_band(state, ego, tier, tier_name)

        pygame.display.flip()
        self._capture_frame()

    def _draw_uncertainty_ring(
        self,
        ego: Dict[str, Any],
        tier: GnssTier,
        origin: np.ndarray,
        scale: float,
    ) -> None:
        """
        @brief Draw the GNSS 1-sigma noise ring around the ego vehicle.

        The radius is the tier's CONFIGURED position noise (metric_stddev_m),
        not the EKF's live covariance estimate - it shows how much positional
        error the fix state admits, which is what differs between tiers.

        @param ego: Ego block from the frame.
        @param tier: Resolved tier for this frame.
        @param origin: World origin for the viewport.
        @param scale: Pixels per metre.
        """
        radius_px = int(tier.stddev_m * scale)
        if radius_px < 1:
            return
        centre = _w2s(float(ego["x"]), float(ego["y"]), origin, scale)
        diameter = radius_px * 2
        ring = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(
            ring, (*tier.colour, _RING_ALPHA), (radius_px, radius_px), radius_px
        )
        pygame.draw.circle(
            ring,
            (*tier.colour, _RING_EDGE_ALPHA),
            (radius_px, radius_px),
            radius_px,
            max(1, self._layout.thick_line - 1),
        )
        self._screen.blit(ring, (centre[0] - radius_px, centre[1] - radius_px))

    def _draw_hud_band(
        self,
        state: Dict[str, Any],
        ego: Dict[str, Any],
        tier: Optional[GnssTier],
        tier_name: str,
    ) -> None:
        """
        @brief Draw the HUD band below the map.

        Layout, top to bottom: the GNSS tier panel (card and accuracy), the
        run/step context line, the speed and action line, and the optional
        debug line.

        @param state: Current frame.
        @param ego: Ego block from the frame.
        @param tier: Resolved tier, or None when unknown.
        @param tier_name: Raw tier name from the frame.
        """
        band_top = self._map_h
        pygame.draw.rect(
            self._screen,
            _C_HUD_BG,
            (0, band_top, self._layout.window_w, self._layout.hud_band_h),
        )

        pad = self._layout.scaled(10)
        y = band_top + pad // 2

        y = self._draw_tier_panel(tier, tier_name, pad, y)

        # Context line: which episode/step this is, in dimmer text since it is
        # reference information rather than the headline. The episode number is
        # bookkeeping that only means something to whoever launched the run, so
        # it is opt-in rather than always shown.
        episode = f"Ep: {state.get('episode_id', '?')}   " if self._show_episode else ""
        context = (
            f"Floor: {state.get('floor_plan', '?')}   "
            f"{episode}"
            f"Step: {state.get('episode_step', '?')}   "
            f"t={state.get('sim_time', 0.0):.1f}s"
        )
        self._screen.blit(self._hud_font.render(context, True, _C_HUD_DIM), (pad, y))
        y += self._layout.bar_pitch

        # Speed and the action vector are fundamental state, not debug info -
        # the env writes ego.speed and action.* on every frame regardless of
        # the debug flag. Show them by default so the policy's behaviour
        # (is it braking? how fast is it over the bay?) is always visible.
        act = state.get("action", {})
        status = (
            f"spd {ego.get('speed', 0.0):5.2f} m/s   "
            f"steer {act.get('steer', 0.0):+.2f}   "
            f"throttle {act.get('throttle', 0.0):.2f}   "
            f"brake {act.get('brake', 0.0):.2f}"
        )
        self._screen.blit(self._hud_font.render(status, True, _C_HUD_TEXT), (pad, y))
        y += self._layout.bar_pitch

        # Diagnostic fields (pos error, reward, covariance) are only written
        # when debug=True in env_config.yaml.
        dbg = state.get("debug")
        if dbg:
            debug_line = (
                f"err {dbg.get('pos_err', 0.0):.2f} m   "
                f"yaw {dbg.get('yaw_err_deg', 0.0):.1f} deg   "
                f"reward {dbg.get('reward', 0.0):+.3f}   "
                f"EKF cov {dbg.get('cov_rms', 0.0):.3f}"
            )
            self._screen.blit(
                self._hud_font.render(debug_line, True, _C_HUD_DIM), (pad, y)
            )

    def _draw_tier_panel(
        self, tier: Optional[GnssTier], tier_name: str, pad: int, y: int
    ) -> int:
        """
        @brief Draw the GNSS tier card and its positional accuracy.
        @param tier: Resolved tier, or None when unknown.
        @param tier_name: Raw tier name from the frame.
        @param pad: Left padding in pixels.
        @param y: Top of the panel in pixels.
        @return The y coordinate below the panel.
        """
        colour = tier.colour if tier is not None else unknown_colour()
        label = display_label(tier, tier_name)

        # The card is always filled, so a tier change reads as a colour change
        # in a fixed shape rather than a box that appears and disappears.
        text_surf = self._tier_font.render(f"GNSS: {label}", True, (20, 20, 20))
        card = pygame.Surface((text_surf.get_width() + pad, text_surf.get_height() + 4))
        card.fill(colour)
        card.blit(text_surf, (pad // 2, 2))
        self._screen.blit(card, (pad, y))

        y += card.get_height() + self._layout.scaled(6)

        if tier is not None:
            # Stated as a tolerance rather than as "1-sigma X m": the viewer
            # needs to read the number as "how far off the car's idea of its
            # own position can be", which is what the ring shows.
            accuracy = f"Position known to +/- {tier.stddev_m:.2f} m (1-sigma)"
            self._screen.blit(
                self._hud_font.render(accuracy, True, _C_HUD_TEXT), (pad, y)
            )
        return y + self._layout.hud_font + 8

    def _draw_legend(self) -> None:
        """
        @brief Draw the colour legend in the right-hand panel.

        Swatches mirror how each element is actually drawn - filled squares for
        filled shapes, outlines for the outline-drawn bays, a line for the
        trail - so the legend decodes the map rather than merely listing names.
        """
        win_h = self._screen.get_height()
        legend_x = _MAP_W
        legend_w = self._layout.legend_w
        pygame.draw.rect(
            self._screen, _C_LEGEND_BG, pygame.Rect(legend_x, 0, legend_w, win_h)
        )
        pygame.draw.line(
            self._screen, (180, 180, 180), (legend_x, 0), (legend_x, win_h), 2
        )

        x0 = legend_x + self._layout.scaled(10)
        y0 = self._layout.scaled(14)
        swatch = self._layout.label_font
        row_h = swatch + self._layout.scaled(9)

        self._screen.blit(
            self._label_bold.render("Legend", True, (40, 40, 40)), (x0, y0)
        )
        y0 += row_h

        # Filled shapes.
        for colour, label in (
            (_C_EGO, "Ego"),
            (_C_STATIC_VEHICLE, "Parked"),
        ):
            pygame.draw.rect(self._screen, colour, (x0, y0, swatch, swatch))
            self._blit_legend_label(label, x0 + swatch + 8, y0)
            y0 += row_h

        # Outlined shapes, drawn as outlines on the map too.
        for colour, label in (
            (_C_TARGET_BAY, "Target bay"),
            (_C_PERP_BAY, "Parking bay"),
        ):
            pygame.draw.rect(self._screen, colour, (x0, y0, swatch, swatch), 2)
            self._blit_legend_label(label, x0 + swatch + 8, y0)
            y0 += row_h

        # Ego trail: a line, as drawn.
        mid = y0 + swatch // 2
        pygame.draw.line(
            self._screen,
            _C_EGO,
            (x0, mid),
            (x0 + swatch, mid),
            self._layout.thick_line,
        )
        self._blit_legend_label("Ego trail", x0 + swatch + 8, y0)
        y0 += row_h + self._layout.scaled(8)

        self._draw_tier_legend(x0, y0, swatch, row_h)

    def _draw_tier_legend(self, x0: int, y0: int, swatch: int, row_h: int) -> None:
        """
        @brief Draw the GNSS tier colour key and ring explanation.
        @param x0: Left edge in pixels.
        @param y0: Top edge in pixels.
        @param swatch: Swatch size in pixels.
        @param row_h: Row pitch in pixels.
        """
        if not self._gnss_tiers:
            return

        self._screen.blit(
            self._label_bold.render("Position accuracy", True, (40, 40, 40)), (x0, y0)
        )
        y0 += row_h

        for tier in sorted(self._gnss_tiers.values(), key=lambda t: t.severity):
            pygame.draw.rect(self._screen, tier.colour, (x0, y0, swatch, swatch))
            self._blit_legend_label(f"+/- {tier.stddev_m:.2f} m", x0 + swatch + 8, y0)
            y0 += row_h

    def _blit_legend_label(self, text: str, x: int, y: int) -> None:
        """
        @brief Draw a legend row label.
        @param text: Label text.
        @param x: Left edge in pixels.
        @param y: Top edge in pixels.
        """
        self._screen.blit(self._label_font.render(text, True, _C_LEGEND_TEXT), (x, y))


def main() -> None:
    """@brief Parse arguments and run the visualiser."""
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
    parser.add_argument(
        "--ui-scale",
        type=float,
        default=_UI_SCALE_DEFAULT,
        help="Font and legend scale multiplier. Raise for a projector or a "
        f"recording (default: {_UI_SCALE_DEFAULT}).",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording an MP4 immediately. Recording can also be "
        "toggled at any time with the R key.",
    )
    parser.add_argument(
        "--show-episode",
        action="store_true",
        help="Show the episode number in the HUD context line. Off by default, "
        "since the number is run bookkeeping rather than something an "
        "audience can interpret.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=_DEFAULT_RECORD_DIR,
        help=f"Directory for recorded MP4s (default: {_DEFAULT_RECORD_DIR}).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frame rate (default: 30).",
    )
    args = parser.parse_args()

    LiveVisualiser(
        history_file=args.history_file,
        ui_scale=args.ui_scale,
        record=args.record,
        record_dir=args.record_dir,
        fps=args.fps,
        show_episode=args.show_episode,
    ).run()


if __name__ == "__main__":
    main()
