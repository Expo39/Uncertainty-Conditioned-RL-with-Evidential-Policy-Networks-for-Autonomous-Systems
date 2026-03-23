"""
@file geometry.py
@brief Shared geometry utilities for parking lot layout processing.

Provides helpers for converting zone dictionaries (from layout YAML files)
into normalised bounding-box tuples, interpolating cone positions along a
polygon perimeter, and computing target bay pose in the ego vehicle body frame.
Used by both the training environment and the inspect_layout script so the
geometry logic lives in one place.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


def zone_bbox(zone_raw: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    @brief Convert a pedestrian zone dict to a bounding-box tuple.

    Handles two YAML formats produced by scripts/generate_layouts.py:

    Format A -- explicit extents::

        x_min: <float>
        x_max: <float>
        y_min: <float>
        y_max: <float>

    Format B -- centre + half-extents::

        centre_x: <float>
        centre_y: <float>
        half_width: <float>
        half_height: <float>

    @param zone_raw: Raw zone dict loaded from the layout YAML.
    @return Tuple (x_min, x_max, y_min, y_max) as world-frame floats.
    """
    if "x_min" in zone_raw:
        return (
            float(zone_raw["x_min"]),
            float(zone_raw["x_max"]),
            float(zone_raw["y_min"]),
            float(zone_raw["y_max"]),
        )

    # Format B: derive extents from centre + half-extents
    cx = float(zone_raw["centre_x"])
    cy = float(zone_raw["centre_y"])
    hw = float(zone_raw["half_width"])
    hh = float(zone_raw["half_height"])
    return cx - hw, cx + hw, cy - hh, cy + hh


def _interpolate_cone_positions(
    corners: List[Tuple[float, float]],
    spacing: float,
    entrance_point: Optional[Tuple[float, float]] = None,
    entrance_half_width: float = 4.0,
    extra_entrance_points: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float, float]]:
    """
    @brief Interpolate evenly spaced positions along a closed polygon perimeter.
    @param corners: List of (x, y) polygon vertices in order (last edge closes
                   back to first vertex automatically).
    @param spacing: Desired spacing between consecutive cones (metres).
    @param entrance_point: Optional (x, y) of the primary entrance centre. Cones
                           within entrance_half_width metres are omitted.
    @param entrance_half_width: Half-width of each entrance gap in metres (default 4.0).
    @param extra_entrance_points: Optional list of additional (x, y) entrance centres
                                  (e.g. extra spawn transforms). Each receives the same
                                  entrance_half_width gap as the primary entrance.
    @return List of (x, y, yaw_deg) tuples. yaw_deg is the edge direction in degrees
            so markers align with the perimeter wall.

    @note Uses adaptive spacing so the last cone on each edge aligns exactly
          with the corner rather than leaving a gap.
    """
    # Collect all entrance centres into one list for uniform gap logic.
    all_entrances: List[Tuple[float, float]] = []
    if entrance_point is not None:
        all_entrances.append(entrance_point)
    if extra_entrance_points:
        all_entrances.extend(extra_entrance_points)

    positions: List[Tuple[float, float, float]] = []
    n = len(corners)

    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]

        edge_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if edge_len < 1e-6:
            continue

        # Edge direction in degrees for marker alignment
        edge_yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))

        num_intervals = max(1, int(round(edge_len / spacing)))
        dx = (x1 - x0) / num_intervals
        dy = (y1 - y0) / num_intervals

        for k in range(num_intervals):
            cx = x0 + k * dx
            cy = y0 + k * dy
            in_gap = any(
                math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) < entrance_half_width
                for ex, ey in all_entrances
            )
            if not in_gap:
                positions.append((cx, cy, edge_yaw_deg))

    return positions


def _compute_relative_target_pose(
    x_ego: float,
    y_ego: float,
    yaw_ego: float,
    x_target: float,
    y_target: float,
    yaw_target: float,
) -> Tuple[float, float, float]:
    """
    @brief Compute target bay pose in the ego vehicle body frame.
    @param x_ego: Ego x position (metres).
    @param y_ego: Ego y position (metres).
    @param yaw_ego: Ego heading (radians).
    @param x_target: Target bay x position (metres).
    @param y_target: Target bay y position (metres).
    @param yaw_target: Target bay heading (radians).
    @return Tuple (dx, dy, dyaw) where dx/dy are in the ego body frame and
            dyaw is the heading error wrapped to (-pi, pi].

    @note Body frame: +x forward, +y left. dx > 0 means target is ahead.
    """
    dx_world = x_target - x_ego
    dy_world = y_target - y_ego

    cos_yaw = math.cos(yaw_ego)
    sin_yaw = math.sin(yaw_ego)

    dx = cos_yaw * dx_world + sin_yaw * dy_world
    dy = -sin_yaw * dx_world + cos_yaw * dy_world

    raw_dyaw = yaw_target - yaw_ego
    dyaw = math.atan2(math.sin(raw_dyaw), math.cos(raw_dyaw))

    return dx, dy, dyaw
