"""
@file visualise_training.py
@brief Detachable 2D bird's-eye visualiser for CARLA parking training.

Polls outputs/vis_state.json every 100 ms and redraws the parking lot scene
each time the file changes. Training always runs headless and is unaffected
by this process. Close the window at any time -- training continues.

Usage:
    make visualise               # live window
    make visualise-record        # live window + saves MP4 on close

Layers drawn (back to front):
  1. Lot boundary polygon (light grey fill)
  2. Perimeter cones (red circles)
  3. Bay outlines by type: perpendicular=blue, angled=yellow, parallel=violet
  4. Target bay (bright green, thick outline + heading arrow)
  5. Static parked vehicles (dark grey rectangles)
  6. Patrol NPC vehicles (orange rectangles)
  7. Pedestrians (magenta circles)
  8. Ego vehicle (cyan rectangle + heading arrow + faded trail)
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Default path to the shared state file written by VisStateWriter
_DEFAULT_STATE_FILE = Path("outputs/vis_state.json")

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

# Vehicle dimensions for rectangle drawing (metres)
_EGO_HALF_LENGTH = 2.25
_EGO_HALF_WIDTH = 1.0

# Polling interval (seconds)
_POLL_INTERVAL = 0.1

# Trail transparency
_TRAIL_ALPHA = 0.4

# Recording FPS
_RECORD_FPS = 10


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace with state_file and record attributes.
    """
    parser = argparse.ArgumentParser(
        description="Detachable 2D bird's-eye visualiser for parking training."
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=_DEFAULT_STATE_FILE,
        help="Path to vis_state.json written by training.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save frames and write MP4 on window close.",
    )
    return parser.parse_args()


def _load_state(state_file: Path) -> Optional[Dict[str, Any]]:
    """
    @brief Load and parse vis_state.json.
    @param state_file: Path to the state JSON file.
    @return Parsed state dict, or None if the file is missing or corrupt.
    """
    try:
        text = state_file.read_text()
        return json.loads(text)  # type: ignore[return-value]
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _rot_rect(
    cx: float, cy: float, half_l: float, half_w: float, yaw_deg: float
) -> np.ndarray:
    """
    @brief Compute the four corners of a rotated rectangle.
    @param cx: Centre x (metres).
    @param cy: Centre y (metres).
    @param half_l: Half-length (forward direction).
    @param half_w: Half-width (lateral direction).
    @param yaw_deg: Yaw angle in degrees (CARLA convention: 0 = east, CCW positive).
    @return Array of shape (4, 2) with corner coordinates.
    """
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    # Corners in local frame (forward=x, left=y)
    local = np.array(
        [
            [half_l, half_w],
            [-half_l, half_w],
            [-half_l, -half_w],
            [half_l, -half_w],
        ]
    )

    # Rotate into world frame
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
    yaw_deg = math.degrees(float(bay.get("yaw", 0.0)))

    corners = _rot_rect(float(bay["x"]), float(bay["y"]), half_d, half_w, yaw_deg)
    poly = plt.Polygon(corners, closed=True, fill=False, edgecolor=colour, linewidth=lw)
    ax.add_patch(poly)

    # Heading arrow for target bay
    if is_target:
        arrow_len = half_d * 0.8
        yaw_r = math.radians(yaw_deg)
        ax.annotate(
            "",
            xy=(
                float(bay["x"]) + arrow_len * math.cos(yaw_r),
                float(bay["y"]) + arrow_len * math.sin(yaw_r),
            ),
            xytext=(float(bay["x"]), float(bay["y"])),
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
    corners = _rot_rect(x, y, 2.25, 1.0, yaw_deg)
    poly = plt.Polygon(
        corners,
        closed=True,
        facecolor=colour,
        edgecolor=colour,
        alpha=alpha,
        linewidth=0.5,
    )
    ax.add_patch(poly)


def _draw_frame(
    ax: plt.Axes,
    state: Dict[str, Any],
) -> None:
    """
    @brief Redraw the entire scene from a state dict.
    @param ax: Matplotlib axes to draw into.
    @param state: Parsed vis_state.json dict.
    """
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
        _draw_bay(ax, bay, is_target=(bay.get("bay_id", "") == target_id))

    # Static and patrol vehicles
    for actor in state.get("actors", []):
        actor_type = actor.get("type", "static")
        colour = (
            _COLOUR_PATROL_VEHICLE if actor_type == "npc" else _COLOUR_STATIC_VEHICLE
        )
        _draw_vehicle_rect(ax, actor["x"], actor["y"], actor.get("yaw", 0.0), colour)

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
        ex, ey, eyaw_deg = float(ego["x"]), float(ego["y"]), float(ego["yaw"])
        _draw_vehicle_rect(ax, ex, ey, eyaw_deg, _COLOUR_EGO)

        # Heading arrow
        arrow_len = 3.0
        eyaw_r = math.radians(eyaw_deg)
        ax.annotate(
            "",
            xy=(ex + arrow_len * math.cos(eyaw_r), ey + arrow_len * math.sin(eyaw_r)),
            xytext=(ex, ey),
            arrowprops=dict(arrowstyle="->", color=_COLOUR_EGO, lw=2.0),
        )

    # Episode info
    ep = state.get("episode_info", {})
    step = ep.get("episode_step", state.get("episode_step", "?"))
    floor_plan = state.get("floor_plan", ep.get("floor_plan", "?"))
    ax.set_title(
        f"Floor plan: {floor_plan}  |  Step: {step}",
        fontsize=10,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color=_COLOUR_TARGET_BAY, label="Target bay"),
        mpatches.Patch(color=_COLOUR_PERP_BAY, label="Perpendicular"),
        mpatches.Patch(color=_COLOUR_ANGLED_BAY, label="Angled"),
        mpatches.Patch(color=_COLOUR_PARALLEL_BAY, label="Parallel"),
        mpatches.Patch(color=_COLOUR_EGO, label="Ego"),
        mpatches.Patch(color=_COLOUR_PATROL_VEHICLE, label="Patrol NPC"),
        mpatches.Patch(color=_COLOUR_STATIC_VEHICLE, label="Parked"),
        mpatches.Patch(color=_COLOUR_PEDESTRIAN, label="Pedestrian"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    ax.autoscale_view()


def main() -> None:
    """
    @brief Entry point for the 2D bird's-eye visualiser.

    Polls vis_state.json every 100 ms. If --record is set, accumulates frames
    and saves an MP4 on window close. Training is unaffected by this process.
    """
    args = _parse_args()

    frames: List[Any] = []  # Stores captured figures for MP4 output
    last_mtime: float = 0.0

    matplotlib.use("TkAgg") if os.environ.get("DISPLAY") else matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.tight_layout()

    # Show "Waiting..." if file not yet available
    ax.text(
        0.5,
        0.5,
        "Waiting for training...",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
    )

    def _on_close(event: Any) -> None:
        """Save MP4 when window is closed, if --record was set."""
        if args.record and frames:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out_dir = Path("outputs/recordings")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{ts}.mp4"
            try:
                from matplotlib.animation import FFMpegWriter

                writer = FFMpegWriter(fps=_RECORD_FPS)
                with writer.saving(fig, str(out_path), dpi=100):
                    for frame_state in frames:
                        _draw_frame(ax, frame_state)
                        writer.grab_frame()
                print(f"Recording saved: {out_path}")
            except Exception as exc:
                print(f"Could not save recording: {exc}")
        sys.exit(0)

    fig.canvas.mpl_connect("close_event", _on_close)
    plt.ion()
    plt.show()

    while True:
        try:
            state_file: Path = args.state_file

            if state_file.exists():
                mtime = state_file.stat().st_mtime
                if mtime > last_mtime:
                    last_mtime = mtime
                    state = _load_state(state_file)
                    if state is not None:
                        _draw_frame(ax, state)
                        if args.record:
                            # Store a shallow copy of the state for later MP4 write
                            frames.append(state)
                        fig.canvas.draw_idle()

            plt.pause(_POLL_INTERVAL)

        except Exception:
            # Non-fatal errors (e.g. window resize during redraw) -- continue
            plt.pause(_POLL_INTERVAL)


if __name__ == "__main__":
    main()
