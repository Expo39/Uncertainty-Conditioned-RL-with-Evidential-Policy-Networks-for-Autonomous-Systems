"""
@file geometry.py
@brief Shared geometry utilities for parking lot layout processing.

Shared by the training environment and the inspect_layout script so the
geometry logic lives in one place.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def zone_bbox(zone_raw: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    @brief Convert a pedestrian zone dict to a bounding-box tuple.

    @note scripts/generate_layouts.py emits two zone formats: explicit extents
          (x_min/x_max/y_min/y_max) or centre + half-extents (centre_x/centre_y/
          half_width/half_height). Both are accepted here.

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
    all_entrances: List[Tuple[float, float]] = []
    if entrance_point is not None:
        all_entrances.append(entrance_point)
    if extra_entrance_points:
        all_entrances.extend(extra_entrance_points)

    # Compare squared distances to avoid sqrt per candidate.
    gap_sq = entrance_half_width * entrance_half_width

    positions: List[Tuple[float, float, float]] = []
    n = len(corners)

    for i in range(n):
        x0, y0 = corners[i]
        x1, y1 = corners[(i + 1) % n]

        edge_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if edge_len < 1e-6:
            continue

        edge_yaw_deg = math.degrees(math.atan2(y1 - y0, x1 - x0))

        num_intervals = max(1, int(round(edge_len / spacing)))
        dx = (x1 - x0) / num_intervals
        dy = (y1 - y0) / num_intervals

        for k in range(num_intervals):
            cx = x0 + k * dx
            cy = y0 + k * dy
            in_gap = any(
                (cx - ex) ** 2 + (cy - ey) ** 2 < gap_sq for ex, ey in all_entrances
            )
            if not in_gap:
                positions.append((cx, cy, edge_yaw_deg))

    return positions


def car_fully_inside_bay(
    car_x: float,
    car_y: float,
    car_yaw: float,
    car_half_length: float,
    car_half_width: float,
    bay_x: float,
    bay_y: float,
    bay_yaw: float,
    bay_width: float,
    bay_depth: float,
    margin: float = 0.0,
) -> bool:
    """
    @brief Test whether all four corners of the car lie inside the bay rectangle.

    @note The bay yaw points along the bay's depth (entry) axis, matching the
          layout YAML convention.

    @param car_x: Car centre x in world frame (metres).
    @param car_y: Car centre y in world frame (metres).
    @param car_yaw: Car heading in world frame (radians).
    @param car_half_length: Half the car's longitudinal extent (m).
    @param car_half_width:  Half the car's lateral extent (m).
    @param bay_x: Bay centre x in world frame (metres).
    @param bay_y: Bay centre y in world frame (metres).
    @param bay_yaw: Bay heading in world frame (radians).
    @param bay_width: Lateral extent of the bay (m, perpendicular to bay axis).
    @param bay_depth: Longitudinal extent of the bay (m, along bay axis).
    @param margin: Positive value shrinks the bay inward by `margin` metres
                   on every side (strict fit); zero accepts "inside or on
                   the line"; negative inflates the bay (slack).
    @return True when every car corner lies inside the (margin-adjusted) bay.
    """
    half_depth = bay_depth / 2.0 - margin
    half_width = bay_width / 2.0 - margin
    if half_depth <= 0.0 or half_width <= 0.0:
        return False

    cos_c, sin_c = math.cos(car_yaw), math.sin(car_yaw)
    cos_b, sin_b = math.cos(bay_yaw), math.sin(bay_yaw)

    car_corners_local = (
        (car_half_length, car_half_width),
        (car_half_length, -car_half_width),
        (-car_half_length, -car_half_width),
        (-car_half_length, car_half_width),
    )

    for lx, ly in car_corners_local:
        wx = car_x + cos_c * lx - sin_c * ly
        wy = car_y + sin_c * lx + cos_c * ly
        # Corner into the bay frame (inverse rotation).
        dx = wx - bay_x
        dy = wy - bay_y
        bx = cos_b * dx + sin_b * dy
        by = -sin_b * dx + cos_b * dy
        if abs(bx) > half_depth or abs(by) > half_width:
            return False
    return True


def bay_containment_fraction(
    car_x: float,
    car_y: float,
    car_yaw: float,
    car_half_length: float,
    car_half_width: float,
    bay_x: float,
    bay_y: float,
    bay_yaw: float,
    bay_width: float,
    bay_depth: float,
    reference: float,
    margin: float = 0.0,
) -> float:
    """
    @brief Smooth [0, 1] measure of how far the car is inside the bay.

    Returns 1.0 when every car corner lies inside the (margin-adjusted) bay
    rectangle, ramping linearly to 0.0 as the worst-overhanging corner moves
    `reference` metres outside the boundary. Used as a continuous in-bay gate
    on the endgame shaping terms so the reward gains a gradient that points
    INTO the bay, instead of a flat plateau that lets a centred-but-short stop
    earn the same shaping as a true park (a stop-short local optimum).

    @note Bay frame and margin convention match `car_fully_inside_bay`.

    @param reference: Overhang distance (m) at which the factor reaches 0.0.
                      Sized to the order of one car half-extent so the gradient
                      is alive across the last metre of the approach.
    @return Containment factor in [0, 1]; 1.0 iff all corners are inside.
    """
    half_depth = bay_depth / 2.0 - margin
    half_width = bay_width / 2.0 - margin
    if half_depth <= 0.0 or half_width <= 0.0 or reference <= 0.0:
        return 0.0

    cos_c, sin_c = math.cos(car_yaw), math.sin(car_yaw)
    cos_b, sin_b = math.cos(bay_yaw), math.sin(bay_yaw)

    car_corners_local = (
        (car_half_length, car_half_width),
        (car_half_length, -car_half_width),
        (-car_half_length, -car_half_width),
        (-car_half_length, car_half_width),
    )

    # Worst (largest) overhang of any corner beyond the nearest bay edge.
    max_overhang = 0.0
    for lx, ly in car_corners_local:
        wx = car_x + cos_c * lx - sin_c * ly
        wy = car_y + sin_c * lx + cos_c * ly
        dx = wx - bay_x
        dy = wy - bay_y
        bx = cos_b * dx + sin_b * dy
        by = -sin_b * dx + cos_b * dy
        overhang = max(abs(bx) - half_depth, abs(by) - half_width, 0.0)
        if overhang > max_overhang:
            max_overhang = overhang

    return max(0.0, 1.0 - max_overhang / reference)


def point_in_polygon(x: float, y: float, corners: List[Tuple[float, float]]) -> bool:
    """
    @brief Ray-casting point-in-polygon test.

    @note Used for OOB detection against the actual lot boundary rather than its
          axis-aligned bounding box, which over-extends at non-rectangular
          corners (trapezoid, irregular_a layouts).

    @param x: Query point x coordinate.
    @param y: Query point y coordinate.
    @param corners: Ordered polygon vertices as (x, y) pairs (closed automatically).
    @return True if the point is inside the polygon.
    """
    arr = np.asarray(corners, dtype=np.float64)
    xi = arr[:, 0]
    yi = arr[:, 1]
    xj = np.roll(xi, 1)
    yj = np.roll(yi, 1)
    cond1 = (yi > y) != (yj > y)
    cond2 = x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
    return bool(np.count_nonzero(cond1 & cond2) % 2)


def inflate_polygon(
    corners: List[Tuple[float, float]], margin: float
) -> List[Tuple[float, float]]:
    """
    @brief Offset a polygon outward by a uniform `margin` metres on every edge.

    Each edge is pushed out along its outward normal by exactly `margin`; each
    vertex moves to the intersection of its two offset edges, i.e. along the
    edge-normal bisector by `margin / sin(half-interior-angle)`.

    @note Winding is detected from the signed area so "outward" is correct for
          either orientation: layouts arrive here in CARLA's left-handed frame,
          so their winding is the mirror of the right-handed source.
    @note Reflex vertices on a concave polygon can overshoot, but the only
          concave layout (irregular_a) is held out for evaluation and the skirt
          is a soft out-of-bounds boundary, not a hard wall.

    @param corners: Ordered polygon vertices as (x, y) pairs.
    @param margin: Outward offset distance in metres.
    @return Offset polygon vertices in the same winding as the input. Inputs
            with fewer than three vertices are returned unchanged.
    """
    n = len(corners)
    if n < 3:
        return list(corners)

    # Signed area (shoelace): positive => CCW. The outward edge normal is the
    # edge direction rotated -90 deg for CCW winding, +90 deg for CW.
    signed_area = 0.5 * sum(
        corners[i][0] * corners[(i + 1) % n][1]
        - corners[(i + 1) % n][0] * corners[i][1]
        for i in range(n)
    )
    outward_sign = 1.0 if signed_area > 0.0 else -1.0

    def _edge_normal(
        p0: Tuple[float, float], p1: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Unit outward normal of the directed edge p0 -> p1."""
        ex, ey = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(ex, ey)
        if length < 1e-12:
            return (0.0, 0.0)
        # Rotate edge direction by -/+90 deg (winding-dependent) for outward.
        return (outward_sign * ey / length, -outward_sign * ex / length)

    offset: List[Tuple[float, float]] = []
    for i in range(n):
        prev_pt = corners[(i - 1) % n]
        curr_pt = corners[i]
        next_pt = corners[(i + 1) % n]
        n_in = _edge_normal(prev_pt, curr_pt)
        n_out = _edge_normal(curr_pt, next_pt)
        bx, by = n_in[0] + n_out[0], n_in[1] + n_out[1]
        bisector_len = math.hypot(bx, by)
        if bisector_len < 1e-9:
            # Degenerate (180 deg) vertex: push straight out along one normal.
            offset.append(
                (curr_pt[0] + margin * n_out[0], curr_pt[1] + margin * n_out[1])
            )
            continue
        # Distance along the unit bisector so both offset edges sit `margin`
        # out: margin / cos(angle between bisector and either edge normal).
        bux, buy = bx / bisector_len, by / bisector_len
        cos_half = bux * n_out[0] + buy * n_out[1]
        scale = margin / cos_half if abs(cos_half) > 1e-9 else margin
        offset.append((curr_pt[0] + scale * bux, curr_pt[1] + scale * buy))
    return offset


def yaw_from_quaternion(q_x: float, q_y: float, q_z: float, q_w: float) -> float:
    """
    @brief Extract yaw angle from a quaternion (2D mode), wrapped to [-pi, pi].

    @warning Standard ZYX Euler decomposition, valid only for 2D operation
             (z-axis rotation only - roll and pitch are assumed zero).

    @param q_x: Quaternion x component.
    @param q_y: Quaternion y component.
    @param q_z: Quaternion z component.
    @param q_w: Quaternion w component.
    @return Yaw angle in radians, wrapped to [-pi, pi].
    """
    siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle_symmetric(angle: float) -> float:
    """
    @brief Wrap an angle to (-pi, pi] with 180-degree parking symmetry.

    @param angle: Raw heading error in radians.
    @return Heading error in (-pi, pi] with 180-deg symmetry applied.
    """
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    if wrapped > math.pi / 2:
        return wrapped - math.pi
    if wrapped < -math.pi / 2:
        return wrapped + math.pi
    return wrapped


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
    @param x_ego: Ego x position (metres, CARLA world frame).
    @param y_ego: Ego y position (metres, CARLA world frame).
    @param yaw_ego: Ego heading (radians, CARLA convention - positive = CW
                    from above = right turn).
    @param x_target: Target bay x position (metres).
    @param y_target: Target bay y position (metres).
    @param yaw_target: Target bay heading (radians).
    @return Tuple (dx, dy, dyaw) in the ego body frame using a left-positive
            convention consistent with the LiDAR sectoring and the obs[1]
            vyaw sign convention:
            - dx > 0: target is ahead;  dx < 0: target is behind.
            - dy > 0: target is on the left;  dy < 0: target is on the right.
            - dyaw > 0: target requires a left rotation from ego;
              dyaw < 0: target requires a right rotation.
            All three shrink in magnitude to zero at the bay. dyaw is wrapped
            with 180-degree parking symmetry (target = ego or target = ego+pi
            both count as aligned) and is then in (-pi/2, pi/2].

    @note CARLA's world frame is left-handed (+x east, +y south). The standard
          CCW rotation matrix used here preserves chirality, so without the
          explicit `-` on dy and dyaw the body frame would inherit CARLA's
          left-handed convention (+y_body = right). The negation rewrites the
          body frame in REP-103 convention (+y_body = left), matching the
          LiDAR scan sectoring in extract_obstacle_features.
    """
    dx_world = x_target - x_ego
    dy_world = y_target - y_ego

    cos_yaw = math.cos(yaw_ego)
    sin_yaw = math.sin(yaw_ego)

    dx = cos_yaw * dx_world + sin_yaw * dy_world
    # Sign flipped vs the standard CCW rotation matrix to put body +y on the
    # left (REP-103) rather than on the right (CARLA-world chirality).
    dy = sin_yaw * dx_world - cos_yaw * dy_world

    # Sign flipped for the same reason: positive dyaw = left rotation needed.
    dyaw = -wrap_angle_symmetric(yaw_target - yaw_ego)
    return dx, dy, dyaw
