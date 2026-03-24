"""
@file visualiser.py
@brief Live 2D bird's-eye visualiser for CARLA parking training and evaluation.

Tails outputs/vis_history.jsonl in real-time, rendering each frame as a
top-down diagram. Creates a signal file (outputs/.vis_active) so the
environment knows to write frames. Closing the window removes the signal
file and the env stops writing.

Layers drawn (back to front):
  1. Lot boundary polygon (light grey fill)
  2. Bay outlines by type: perpendicular=blue, angled=yellow, parallel=violet
  3. Target bay (bright green, thick outline + heading arrow)
  4. Static parked vehicles (dark grey rectangles)
  5. Patrol NPC vehicles (orange rectangles)
  6. Pedestrians (magenta circles)
  7. Ego vehicle (cyan rectangle + heading arrow + faded trail)

Usage:
    make visualise               # Live window during training
    make eval-visualise-2d       # Live window during eval
    make replay EPISODE=<id>     # Replay a specific past episode
    make replay --list-episodes  # Print all episode IDs in history
"""

import json
import math
import signal
import sys
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from scripts.colours import (
    BAY_HEX,
    HEX_CONE,
    HEX_EGO,
    HEX_LOT,
    HEX_PATROL_VEHICLE,
    HEX_PEDESTRIAN_ZONE,
    HEX_STATIC_VEHICLE,
    HEX_TARGET_BAY,
)

# Colours -- all sourced from scripts/colours.py
_COLOUR_LOT = HEX_LOT
_COLOUR_TARGET_BAY = HEX_TARGET_BAY
_COLOUR_PERP_BAY = BAY_HEX["perpendicular"]
_COLOUR_ANGLED_BAY = BAY_HEX["angled"]
_COLOUR_PARALLEL_BAY = BAY_HEX["parallel"]
_COLOUR_STATIC_VEHICLE = HEX_STATIC_VEHICLE
_COLOUR_PATROL_VEHICLE = HEX_PATROL_VEHICLE
_COLOUR_PEDESTRIAN = HEX_PEDESTRIAN_ZONE
_COLOUR_EGO = HEX_EGO
_COLOUR_TRAIL = HEX_EGO
_COLOUR_CONE = HEX_CONE

# Vehicle dimensions for rectangle drawing (metres).
# Ego: BMW Grand Tourer (4.5m x 2.0m). NPC/static: generic saloon (4.7m x 2.1m).
_EGO_HALF_LENGTH = 2.25
_EGO_HALF_WIDTH = 1.0
_NPC_HALF_LENGTH = 2.35
_NPC_HALF_WIDTH = 1.05

# Trail transparency
_TRAIL_ALPHA = 0.4

# Default paths
_DEFAULT_HISTORY_FILE = Path("outputs/vis_history.jsonl")
_SIGNAL_FILE = Path("outputs/.vis_active")

# How often to check for new lines when waiting (seconds)
_POLL_INTERVAL = 0.05


def _rot_rect(
    cx: float, cy: float, half_l: float, half_w: float, yaw_deg: float
) -> np.ndarray:
    """
    @brief Compute the four corners of a rotated rectangle.
    @param cx: Centre x (metres).
    @param cy: Centre y (metres).
    @param half_l: Half-length (forward direction).
    @param half_w: Half-width (lateral direction).
    @param yaw_deg: Yaw angle in degrees (CARLA convention).
    @return Array of shape (4, 2) with corner coordinates.
    """
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )

    R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    world = (R @ local.T).T + np.array([cx, cy])
    return world


def _draw_bay(
    ax: plt.Axes,
    bay: Dict[str, Any],
    is_target: bool,
) -> None:
    """
    @brief Draw a single bay outline as a rotated rectangle.
    @param ax: Matplotlib axes.
    @param bay: Bay dict with x, y, yaw, width, depth, bay_type keys.
    @param is_target: If True, draw with bright green thick outline.
    """
    bay_type = bay.get("bay_type", "perpendicular")
    if is_target:
        colour = _COLOUR_TARGET_BAY
        lw = 2.5
    elif bay_type == "angled":
        colour = _COLOUR_ANGLED_BAY
        lw = 1.0
    elif bay_type == "parallel":
        colour = _COLOUR_PARALLEL_BAY
        lw = 1.0
    else:
        colour = _COLOUR_PERP_BAY
        lw = 1.0

    half_d = float(bay.get("depth", 5.0)) / 2.0
    half_w = float(bay.get("width", 2.5)) / 2.0
    yaw_deg = float(bay.get("yaw_deg", bay.get("yaw", 0.0)))

    corners = _rot_rect(
        float(bay["x"]), float(bay["y"]), half_d, half_w, yaw_deg
    )
    poly = plt.Polygon(
        corners, closed=True, fill=False, edgecolor=colour, linewidth=lw
    )
    ax.add_patch(poly)

    # Both-direction arrows for target bay (nose-in and nose-out both valid).
    if is_target:
        arrow_len = half_d * 0.8
        bx, by = float(bay["x"]), float(bay["y"])
        for yaw_r in (math.radians(yaw_deg), math.radians(yaw_deg + 180.0)):
            ax.annotate(
                "",
                xy=(bx + arrow_len * math.cos(yaw_r), by + arrow_len * math.sin(yaw_r)),
                xytext=(bx, by),
                arrowprops=dict(arrowstyle="->", color=_COLOUR_TARGET_BAY, lw=2.0),
            )


def _draw_vehicle_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    yaw_deg: float,
    colour: str,
    alpha: float = 1.0,
) -> None:
    """
    @brief Draw a filled vehicle rectangle (2.0m x 4.5m).
    @param ax: Matplotlib axes.
    @param x: Centre x.
    @param y: Centre y.
    @param yaw_deg: Yaw in degrees.
    @param colour: Fill and edge colour.
    @param alpha: Transparency.
    """
    corners = _rot_rect(x, y, _EGO_HALF_LENGTH, _EGO_HALF_WIDTH, yaw_deg)
    poly = plt.Polygon(
        corners,
        closed=True,
        facecolor=colour,
        edgecolor=colour,
        alpha=alpha,
        linewidth=0.5,
    )
    ax.add_patch(poly)


class LiveVisualiser:
    """
    @class LiveVisualiser
    @brief Live 2D bird's-eye visualiser for CARLA parking.

    Tails vis_history.jsonl and plays frames at real-time speed
    (1 sim-second = 1 wall-second). Creates outputs/.vis_active on
    start so the env writes frames; removes it on close.

    Key bindings (while the window is focused):
      l          -- jump to the latest episode in the history
      0-9        -- type an episode number then Enter to jump to it
      Escape     -- cancel a partially-typed episode number
    """

    def __init__(
        self,
        history_file: Optional[Path] = None,
    ) -> None:
        """
        @brief Initialise the live visualiser.
        @param history_file: Path to vis_history.jsonl. Defaults to
               outputs/vis_history.jsonl.
        """
        self._history_file = history_file or _DEFAULT_HISTORY_FILE
        self._signal_file = _SIGNAL_FILE

        # Set up matplotlib figure. If interactive backend fails, fall back to Agg.
        try:
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
        except ImportError:
            matplotlib.use("Agg")
            self._fig, self._ax = plt.subplots(figsize=(10, 10))
        self._fig.tight_layout()
        self._ax.text(
            0.5,
            0.5,
            "Waiting for data...",
            transform=self._ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Small Tkinter help window listing key bindings (no matplotlib overhead).
        try:
            _root = self._fig.canvas.get_tk_widget().winfo_toplevel()
            self._help_win: Optional[tk.Toplevel] = tk.Toplevel(_root)
            self._help_win.title("Controls")
            self._help_win.resizable(False, False)
            self._help_win.configure(bg="#1E1E1E")
            _help_lines = (
                "  Visualiser controls\n"
                "  --------------------------------\n"
                "  l              latest episode\n"
                "  0-9 + Enter    jump to episode\n"
                "  Esc            cancel input\n"
                "  Ctrl+C         exit + delete history"
            )
            tk.Label(
                self._help_win,
                text=_help_lines,
                font=("Courier", 10),
                bg="#1E1E1E",
                fg="#D4D4D4",
                justify="left",
                padx=10,
                pady=10,
            ).pack()
        except Exception:
            # Non-Tk backend (e.g. Agg headless) -- skip help window silently.
            self._help_win = None

        # Jump-to-episode state: target episode ID to seek to, or None for live tail.
        # _digit_buf accumulates typed digits before Enter is pressed.
        self._jump_episode: Optional[int] = None
        self._digit_buf: str = ""
        # Set True by SIGINT handler to exit the run loop cleanly without
        # printing a Tkinter traceback.
        self._exit_requested: bool = False

        # Truncate any stale history so we start fresh. If the file is owned
        # by root (from Docker), skip truncation and just read existing data.
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        if self._history_file.exists():
            try:
                self._history_file.write_text("")
            except PermissionError:
                # File owned by root (Docker container); proceed with read-only
                pass

        # Create the signal file so the env starts writing
        self._signal_file.touch()

    # ------------------------------------------------------------------
    # Signal file management
    # ------------------------------------------------------------------

    def _remove_signal(self) -> None:
        """
        @brief Remove the signal file so the env stops writing.
        """
        try:
            self._signal_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _delete_history(self) -> None:
        """
        @brief Delete the history file to free disk space on exit.
        """
        try:
            self._history_file.unlink(missing_ok=True)
        except OSError:
            pass

    def _on_close(self, event: Any) -> None:
        """
        @brief Handle window close: remove signal file, delete history, and exit.
        @param event: Matplotlib close event (unused).
        """
        self._remove_signal()
        self._delete_history()
        sys.exit(0)

    def _on_key(self, event: Any) -> None:
        """
        @brief Handle key press for episode navigation.

        Bindings:
          l      -- jump to the latest (highest) episode ID in history
          0-9    -- accumulate digit into episode number buffer
          enter  -- confirm buffered episode number and jump to it
          escape -- cancel buffered episode number
        @param event: Matplotlib key event.
        """
        key = event.key if event.key is not None else ""

        if key == "l":
            latest = self._latest_episode_id()
            if latest is not None:
                self._jump_episode = latest
                self._digit_buf = ""
                self._update_status(f"Jumping to latest episode {latest}...")
            else:
                self._update_status("No episodes in history yet.")

        elif key in "0123456789":
            self._digit_buf += key
            self._update_status(
                f"Episode: {self._digit_buf}_ (Enter to jump, Esc to cancel)"
            )

        elif key == "enter" and self._digit_buf:
            self._jump_episode = int(self._digit_buf)
            self._digit_buf = ""
            self._update_status(f"Jumping to episode {self._jump_episode}...")

        elif key == "escape":
            self._digit_buf = ""
            self._update_status("Cancelled.")

    def _update_status(self, msg: str) -> None:
        """
        @brief Show a temporary status message in the figure title area.
        @param msg: Message to display.
        """
        self._ax.set_title(msg, fontsize=10)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        @brief Show the matplotlib window and start tailing the JSONL.
        """
        # Install a SIGINT handler that sets a flag rather than raising
        # KeyboardInterrupt inside Tkinter's C-level event loop, which
        # would otherwise print a noisy traceback before Python can catch it.
        self._exit_requested = False

        def _sigint_handler(signum: int, frame: Any) -> None:
            self._exit_requested = True

        signal.signal(signal.SIGINT, _sigint_handler)

        plt.ion()
        plt.show()
        try:
            self._run_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._remove_signal()
            self._delete_history()

    def _run_loop(self) -> None:
        """
        @brief Tail the JSONL file and play frames at real-time speed.

        Tracks the relationship between sim_time (from the env) and wall
        clock time. If training is faster than real-time, the visualiser
        sleeps to maintain 1:1 pacing. If training is slower, frames are
        displayed as soon as they arrive.

        When _jump_episode is set (via key press), the loop seeks to the
        first frame of that episode and resumes playback from there. After
        the episode ends it returns to live-tail mode.
        """
        wall_start: Optional[float] = None
        sim_start: Optional[float] = None
        file_pos: int = 0

        while True:
            if self._exit_requested:
                break
            try:
                # Check if a jump was requested via key press
                if self._jump_episode is not None:
                    target = self._jump_episode
                    self._jump_episode = None
                    seek_pos = self._find_episode_start(target)
                    if seek_pos is not None:
                        file_pos = seek_pos
                        wall_start = None
                        sim_start = None
                    else:
                        self._update_status(
                            f"Episode {target} not found in history."
                        )

                frame = self._read_next_frame(file_pos)
                if frame is None:
                    plt.pause(_POLL_INTERVAL)
                    continue

                file_pos = frame["_file_pos"]
                state = frame["state"]

                sim_time = state.get("sim_time", 0.0)
                episode_step = state.get("episode_step", 0)

                # Reset wall clock reference on episode boundary
                if episode_step <= 1:
                    wall_start = time.monotonic()
                    sim_start = sim_time

                # Pace at real-time: sleep if ahead of sim clock
                if wall_start is not None and sim_start is not None:
                    sim_elapsed = sim_time - sim_start
                    wall_elapsed = time.monotonic() - wall_start
                    sleep_time = sim_elapsed - wall_elapsed
                    if sleep_time > 0.001:
                        plt.pause(sleep_time)

                # Draw and flush
                self._draw_frame(state)
                self._fig.canvas.draw_idle()
                self._fig.canvas.flush_events()

            except Exception:
                plt.pause(_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # JSONL reader
    # ------------------------------------------------------------------

    def _read_next_frame(
        self, file_pos: int
    ) -> Optional[Dict[str, Any]]:
        """
        @brief Read the next JSON line from the history file.
        @param file_pos: Byte offset to read from.
        @return Dict with 'state' and '_file_pos', or None if no new data.
        """
        if not self._history_file.exists():
            return None

        try:
            with open(self._history_file, "r") as f:
                f.seek(file_pos)
                line = f.readline()
                if not line or not line.endswith("\n"):
                    return None
                new_pos = f.tell()

            state = json.loads(line)
            return {"state": state, "_file_pos": new_pos}
        except (json.JSONDecodeError, OSError):
            return None

    def _find_episode_start(self, episode_id: int) -> Optional[int]:
        """
        @brief Scan the JSONL from the beginning and return the byte offset
               of the first frame belonging to episode_id.
        @param episode_id: Target episode ID to seek to.
        @return Byte offset of the first matching frame, or None if not found.
        """
        if not self._history_file.exists():
            return None
        try:
            with open(self._history_file, "r") as f:
                while True:
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        state = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if state.get("episode_id") == episode_id:
                        return pos
        except OSError:
            pass
        return None

    def _latest_episode_id(self) -> Optional[int]:
        """
        @brief Scan the JSONL and return the highest episode_id present.
        @return Highest episode ID found, or None if the file is empty.
        """
        if not self._history_file.exists():
            return None
        latest: Optional[int] = None
        try:
            with open(self._history_file, "r") as f:
                for line in f:
                    try:
                        state = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = state.get("episode_id")
                    if isinstance(eid, int):
                        if latest is None or eid > latest:
                            latest = eid
        except OSError:
            pass
        return latest

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_frame(self, state: Dict[str, Any]) -> None:
        """
        @brief Redraw the entire scene from a state dict.
        @param state: Parsed frame dict from vis_history.jsonl.
        """
        ax = self._ax
        ax.cla()
        ax.set_aspect("equal")
        ax.set_facecolor("#F8F8F8")

        # Lot boundary polygon
        corners_raw = state.get("corners", [])
        if corners_raw:
            lot_pts = np.array([[c["x"], c["y"]] for c in corners_raw])
            lot_poly = plt.Polygon(
                lot_pts,
                closed=True,
                facecolor=_COLOUR_LOT,
                edgecolor="#999999",
                linewidth=1.0,
            )
            ax.add_patch(lot_poly)

        # Bay outlines
        target_bay = state.get("target_bay", {})
        target_id = target_bay.get("bay_id", "")
        for bay in state.get("bays", []):
            # Layout YAML uses "id"; target_bay dict uses "bay_id"
            bay_id = bay.get("id", bay.get("bay_id", ""))
            _draw_bay(ax, bay, is_target=(bay_id == target_id))

        # Static and patrol vehicles
        for actor in state.get("actors", []):
            actor_type = actor.get("type", "static")
            colour = (
                _COLOUR_PATROL_VEHICLE
                if actor_type == "npc"
                else _COLOUR_STATIC_VEHICLE
            )
            _draw_vehicle_rect(
                ax, actor["x"], actor["y"], actor.get("yaw", 0.0), colour
            )

        # Pedestrians
        for ped in state.get("pedestrians", []):
            ax.add_patch(
                plt.Circle(
                    (ped["x"], ped["y"]),
                    radius=0.4,
                    facecolor=_COLOUR_PEDESTRIAN,
                    edgecolor=_COLOUR_PEDESTRIAN,
                )
            )

        # Ego trajectory trail
        trail = state.get("trajectory", [])
        if len(trail) > 1:
            trail_arr = np.array(trail)
            ax.plot(
                trail_arr[:, 0],
                trail_arr[:, 1],
                color=_COLOUR_TRAIL,
                alpha=_TRAIL_ALPHA,
                linewidth=1.5,
            )

        # Ego vehicle
        ego = state.get("ego", {})
        if ego:
            ex = float(ego["x"])
            ey = float(ego["y"])
            eyaw_deg = float(ego["yaw"])
            _draw_vehicle_rect(ax, ex, ey, eyaw_deg, _COLOUR_EGO)

            # Heading arrow
            arrow_len = 3.0
            eyaw_r = math.radians(eyaw_deg)
            ax.annotate(
                "",
                xy=(
                    ex + arrow_len * math.cos(eyaw_r),
                    ey + arrow_len * math.sin(eyaw_r),
                ),
                xytext=(ex, ey),
                arrowprops=dict(
                    arrowstyle="->", color=_COLOUR_EGO, lw=2.0
                ),
            )

        # Episode info
        episode_id = state.get("episode_id", "?")
        step = state.get("episode_step", "?")
        sim_time = state.get("sim_time", 0.0)
        floor_plan = state.get("floor_plan", "?")
        ax.set_title(
            f"Floor plan: {floor_plan}  |  "
            f"Episode: {episode_id}  |  "
            f"Step: {step}  |  "
            f"t = {sim_time:.2f}s",
            fontsize=10,
        )

        # Legend
        legend_handles = [
            mpatches.Patch(
                color=_COLOUR_TARGET_BAY, label="Target bay"
            ),
            mpatches.Patch(
                color=_COLOUR_PERP_BAY, label="Perpendicular"
            ),
            mpatches.Patch(color=_COLOUR_ANGLED_BAY, label="Angled"),
            mpatches.Patch(
                color=_COLOUR_PARALLEL_BAY, label="Parallel"
            ),
            mpatches.Patch(color=_COLOUR_EGO, label="Ego"),
            mpatches.Patch(
                color=_COLOUR_PATROL_VEHICLE, label="Patrol NPC"
            ),
            mpatches.Patch(
                color=_COLOUR_STATIC_VEHICLE, label="Parked"
            ),
            mpatches.Patch(
                color=_COLOUR_PEDESTRIAN, label="Pedestrian"
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.7,
        )

        ax.autoscale_view()
