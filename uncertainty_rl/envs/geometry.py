"""
@file geometry.py
@brief Pure geometry helpers for parking lot layout computations.

All functions are free of CARLA and ROS 2 dependencies and are unit-testable
on CPU without any simulator running.
"""

import math
from typing import List, Optional, Tuple


def _interpolate_cone_positions(
    corners: List[Tuple[float, float]],
    spacing: float,
    entrance_point: Optional[Tuple[float, float]] = None,
    entrance_half_width: float = 4.0,
) -> List[Tuple[float, float]]:
    """
    @brief Interpolate evenly spaced positions along a closed polygon perimeter.
    @param corners: List of (x, y) polygon vertices in order (last edge closes
                   back to first vertex automatically).
    @param spacing: Desired spacing between consecutive cones (metres).
    @param entrance_point: Optional (x, y) of the entrance centre. Cones within
                           entrance_half_width metres of this point are omitted,
                           creating a driveable gap in the perimeter wall.
    @param entrance_half_width: Half-width of the entrance gap in metres (default 4.0).
    @return List of (x, y) positions for cone placement.

    @note Uses adaptive spacing so the last cone on each edge aligns exactly
          with the corner rather than leaving a gap.
    """
    positions: List[Tuple[float, float]] = []
    n = len(corners)

    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]

        edge_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if edge_len < 1e-6:
            continue

        num_intervals = max(1, int(round(edge_len / spacing)))
        dx = (x1 - x0) / num_intervals
        dy = (y1 - y0) / num_intervals

        for k in range(num_intervals):
            cx = x0 + k * dx
            cy = y0 + k * dy
            if entrance_point is not None:
                ex, ey = entrance_point
                dist = math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
                if dist < entrance_half_width:
                    continue
            positions.append((cx, cy))

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
