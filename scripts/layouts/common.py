"""
@file common.py
@brief Shared geometry helpers, world-frame transform, YAML output, and PNG plotting
       for parking lot layout generation.

All layout modules (rectangle.py, trapezoid.py, irregular_a.py) import from here.
Nothing in this module is layout-specific.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from uncertainty_rl.utils.geometry import point_in_polygon

# ---------------------------------------------------------------------------
# Bay dimension constants (German EAR 05)
# ---------------------------------------------------------------------------

BAY_DIMS: Dict[str, Dict[str, float]] = {
    "perpendicular": {"width": 2.5, "depth": 5.0, "aisle": 6.0},
    "angled": {"width": 2.5, "depth": 5.4, "aisle": 3.6},
    "parallel": {"width": 2.5, "depth": 8.0, "aisle": 4.0},
}

BAYS_PER_TYPE = 5  # Exactly 5 bays per type per floor plan

# Shared spacing constants used by all layout modules.
# _WALL_GAP: minimum clearance between a bay's back face and the perimeter wall/cones.
# PED_STRIP: width of a pedestrian zone strip alongside an aisle face (m).
# _PED_MARGIN: inset applied to both ends of a pedestrian zone so it does not
#              overlap the perimeter cone boundary.
_WALL_GAP: float = 0.5
PED_STRIP: float = 3.0
_PED_MARGIN: float = 0.5


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------


def _rotate(x: float, y: float, heading_rad: float) -> Tuple[float, float]:
    """
    @brief Rotate point (x, y) by heading_rad about the origin.
    @param x: Local x coordinate.
    @param y: Local y coordinate.
    @param heading_rad: Rotation angle in radians (CCW positive).
    @return Rotated (x, y) tuple.
    """
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    return (cos_h * x - sin_h * y, sin_h * x + cos_h * y)


def _translate(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    heading_rad: float,
) -> Tuple[float, float]:
    """
    @brief Rotate then translate a local point to world frame.
    @param x: Local x coordinate.
    @param y: Local y coordinate.
    @param origin_x: World-frame x of the lot origin.
    @param origin_y: World-frame y of the lot origin.
    @param heading_rad: Lot heading in radians.
    @return World-frame (x, y) tuple.
    """
    rx, ry = _rotate(x, y, heading_rad)
    return (rx + origin_x, ry + origin_y)


def _world_yaw(local_yaw_deg: float, heading_deg: float) -> float:
    """
    @brief Convert a local yaw angle to world-frame yaw.
    @param local_yaw_deg: Yaw in the lot's local frame (degrees).
    @param heading_deg: Lot heading offset (degrees).
    @return World-frame yaw in degrees.
    """
    return (local_yaw_deg + heading_deg) % 360.0


# ---------------------------------------------------------------------------
# Bay layout generators (local frame)
# ---------------------------------------------------------------------------


def _bay_corners(
    cx: float,
    cy: float,
    yaw_deg: float,
    width: float,
    depth: float,
) -> List[Tuple[float, float]]:
    """
    @brief Return the four corner points of a bay rectangle in local frame.

    The bay rectangle has depth along the vehicle's heading axis and width
    perpendicular to it. Corners are ordered CCW starting from (-d/2, -w/2).

    @param cx: Bay centre x.
    @param cy: Bay centre y.
    @param yaw_deg: Bay heading in degrees (vehicle nose direction).
    @param width: Bay width (perpendicular to heading), metres.
    @param depth: Bay depth (along heading), metres.
    @return List of four (x, y) corner tuples.
    """
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    hw = width / 2.0
    hd = depth / 2.0
    local = [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
    return [
        (cx + cos_y * lx - sin_y * ly, cy + sin_y * lx + cos_y * ly) for lx, ly in local
    ]


def validate_bays_in_polygon(
    bays: List[Dict[str, Any]],
    corners: List[Dict[str, float]],
    shape_name: str,
    margin: float = 0.05,
) -> None:
    """
    @brief Raise ValueError if any bay corner lies outside the lot polygon.

    Checks all four corners of every bay rectangle. A small margin is used
    to allow bays that are flush against a wall (floating-point tolerance).

    @param bays: List of bay dicts with local_x, local_y, local_yaw_deg, width, depth.
    @param corners: Lot perimeter as list of {x, y} dicts (local frame).
    @param shape_name: Floor plan name for error messages.
    @param margin: Outward expansion of polygon for boundary-touching bays.
    @raises ValueError: If any bay corner is outside the expanded polygon.
    """
    poly = [(c["x"], c["y"]) for c in corners]
    cx_avg = sum(p[0] for p in poly) / len(poly)
    cy_avg = sum(p[1] for p in poly) / len(poly)
    expanded = [
        (
            cx_avg + (1.0 + margin) * (px - cx_avg),
            cy_avg + (1.0 + margin) * (py - cy_avg),
        )
        for px, py in poly
    ]

    for bay in bays:
        bay_corners_pts = _bay_corners(
            bay["local_x"],
            bay["local_y"],
            bay["local_yaw_deg"],
            bay["width"],
            bay["depth"],
        )
        for corner in bay_corners_pts:
            if not point_in_polygon(corner[0], corner[1], expanded):
                raise ValueError(
                    f"[{shape_name}] Bay '{bay.get('bay_type', '?')}' at "
                    f"({bay['local_x']:.2f}, {bay['local_y']:.2f}) has a corner "
                    f"at ({corner[0]:.2f}, {corner[1]:.2f}) outside the lot boundary."
                )


def warn_narrow_corridors(
    bays: List[Dict[str, Any]],
    shape_name: str,
    min_width: float = 6.0,
) -> None:
    """
    @brief Warn if the minimum gap between any two facing bay clusters is narrower
           than min_width (EAR 05 minimum corridor width = 6.0 m).

    @param bays: All bays in the layout (local frame), each with local_x, local_y,
                 local_yaw_deg, width, depth.
    @param shape_name: Floor plan name for warning messages.
    @param min_width: Minimum acceptable corridor width in metres (default 6.0 m).
    """
    for i in range(len(bays)):
        for j in range(i + 1, len(bays)):
            a = bays[i]
            b = bays[j]
            same_type = a.get("bay_type") == b.get("bay_type")
            yaw_diff = abs(a["local_yaw_deg"] - b["local_yaw_deg"]) % 360.0
            same_yaw = yaw_diff < 1.0 or abs(yaw_diff - 360.0) < 1.0
            if same_type and same_yaw:
                continue
            a_half_x = a["depth"] / 2.0
            a_half_y = a["width"] / 2.0
            b_half_x = b["depth"] / 2.0
            b_half_y = b["width"] / 2.0
            gap_x = abs(a["local_x"] - b["local_x"]) - a_half_x - b_half_x
            gap_y = abs(a["local_y"] - b["local_y"]) - a_half_y - b_half_y
            gap = max(gap_x, gap_y, 0.0) if gap_x > 0 or gap_y > 0 else 0.0
            if gap < min_width:
                print(
                    f"  WARNING [{shape_name}]: corridor between "
                    f"'{a.get('bay_type', '?')}'"
                    f" ({a['local_x']:.1f}, {a['local_y']:.1f}) "
                    f"and '{b.get('bay_type', '?')}'"
                    f" ({b['local_x']:.1f}, {b['local_y']:.1f}) "
                    f"is {gap:.2f} m (min {min_width:.1f} m)."
                )


# ---------------------------------------------------------------------------
# Reusable bay row builders
# ---------------------------------------------------------------------------


def angled_bays_along_wall(
    n: int,
    wall_x0: float,
    wall_y0: float,
    wall_dx: float,
    wall_dy: float,
    wall_len: float,
    offset_from_wall: float,
    start_along_wall: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n angled (45-deg) bay centres along an arbitrary wall.
    @param n: Number of bays.
    @param wall_x0: Wall start point x.
    @param wall_y0: Wall start point y.
    @param wall_dx: Wall unit direction x (normalised).
    @param wall_dy: Wall unit direction y (normalised).
    @param wall_len: Total wall length (kept for documentation).
    @param offset_from_wall: Distance from wall to bay centre (inward).
    @param start_along_wall: Distance along wall to the first bay centre.
    @param facing_yaw_deg: Yaw of a parked vehicle (degrees, local frame).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["angled"]
    spacing = dims["width"] / math.sin(math.radians(45.0))
    # Inward normal: rotate wall direction 90 deg CCW
    nx = -wall_dy
    ny = wall_dx
    bays = []
    for i in range(n):
        along = start_along_wall + i * spacing
        bx = wall_x0 + wall_dx * along + nx * offset_from_wall
        by = wall_y0 + wall_dy * along + ny * offset_from_wall
        bays.append(
            {
                "bay_type": "angled",
                "local_x": bx,
                "local_y": by,
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
            }
        )
    return bays


# ---------------------------------------------------------------------------
# Bay offset helpers
# ---------------------------------------------------------------------------


def ang_offset_from_wall(bay_depth: float, bay_width: float) -> float:
    """
    @brief Inward offset from a flat wall to angled bay centre (45 deg parking).
    @param bay_depth: Bay depth (metres).
    @param bay_width: Bay width (metres).
    @return Perpendicular offset from wall to bay centre.
    """
    return (bay_depth / 2.0 + bay_width / 2.0) * math.sin(math.radians(45.0))


def ang_x_margin(bay_depth: float, bay_width: float) -> float:
    """
    @brief Minimum along-wall start offset so leftmost bay corner sits at wall edge.
    @param bay_depth: Bay depth (metres).
    @param bay_width: Bay width (metres).
    @return Start offset along the wall direction.
    """
    return (bay_depth / 2.0 + bay_width / 2.0) * math.cos(math.radians(45.0)) + 0.5


# ---------------------------------------------------------------------------
# Asymmetric landmark generation
# ---------------------------------------------------------------------------

# Minimum edge length to receive a landmark (metres).
_LANDMARK_MIN_EDGE: float = 10.0
# Extra clearance between a landmark centre and bay edges (metres).
# Set to 0.0: landmarks sit on the perimeter wall and _WALL_GAP (0.5 m)
# already keeps every bay back face 0.5 m inside the wall, so any point on the
# perimeter is geometrically outside all bays.  A non-zero value would
# over-filter landmarks near wall-adjacent bays.
_LANDMARK_BAY_MARGIN: float = 0.0

# Per-edge fractional positions for landmark placement. Each inner list gives
# the fraction(s) along that edge (0 = start corner, 1 = end corner) where a
# landmark is placed. The pattern is deliberately lopsided:
#   Edge 0: 3 landmarks clustered toward the start
#   Edge 1: 1 landmark near the far end
#   Edge 2: 2 landmarks in the second half
#   Edge 3+: no landmarks
# This guarantees that no rotation, reflection, or 180-degree flip of the lot
# produces the same LiDAR signature, giving Cartographer a unique scan match
# hypothesis at every position.
_LANDMARK_EDGE_FRACS: List[List[float]] = [
    [0.15, 0.40, 0.70],  # Edge 0 (bottom): 3 landmarks, spread across first 70%
    [0.25, 0.80],         # Edge 1 (right):  2 landmarks, near start and far end
    [0.45, 0.85],         # Edge 2 (top):    2 landmarks, second half
    [0.30, 0.65],         # Edge 3 (left):   2 landmarks -- every edge needs features
]


def _point_in_bay(
    px: float,
    py: float,
    bay: Dict[str, Any],
    margin: float,
) -> bool:
    """
    @brief Check if a point falls inside a bay rectangle (with margin).

    Transforms the point into bay-local coordinates and tests against the
    half-extents expanded by *margin* on each side.

    @param px: Point x in local frame.
    @param py: Point y in local frame.
    @param bay: Bay dict with local_x, local_y, local_yaw_deg, width, depth.
    @param margin: Extra clearance around the bay rectangle (metres).
    @return True if the point is inside the expanded bay rectangle.
    """
    dx = px - float(bay["local_x"])
    dy = py - float(bay["local_y"])
    yaw_rad = math.radians(float(bay["local_yaw_deg"]))
    cos_y = math.cos(-yaw_rad)
    sin_y = math.sin(-yaw_rad)
    lx = dx * cos_y - dy * sin_y
    ly = dx * sin_y + dy * cos_y
    half_d = float(bay["depth"]) / 2.0 + margin
    half_w = float(bay["width"]) / 2.0 + margin
    return abs(lx) <= half_d and abs(ly) <= half_w


def compute_landmarks(
    corners: List[Dict[str, float]],
    bays: Optional[List[Dict[str, Any]]] = None,
    edge_fracs: Optional[List[List[float]]] = None,
) -> List[Dict[str, float]]:
    """
    @brief Compute deterministic, asymmetric landmark positions from polygon edges.

    Landmarks are placed at pre-defined fractional offsets along each edge of
    the lot polygon, using the deliberately lopsided profile in
    ``_LANDMARK_EDGE_FRACS`` (or a caller-supplied override).  Edge 0 gets 3
    landmarks (spread), edge 1 gets 2 (near ends), edge 2 gets 2 (second
    half), edge 3 gets 2 (first two thirds).  Every long edge has at least 2
    landmarks so that movement along any edge produces a distinguishable scan
    change for Cartographer.

    Landmarks sit directly on the perimeter wall (no inset) and replace the
    perimeter cones at those positions.  Any candidate that falls inside a
    parking bay (expanded by ``_LANDMARK_BAY_MARGIN``) is discarded.

    @param corners: Polygon vertices as list of {x, y} dicts (local frame, CCW).
    @param bays: Optional list of bay dicts (local frame) for collision filtering.
    @param edge_fracs: Optional per-edge fractional positions override.  When
           None, ``_LANDMARK_EDGE_FRACS`` is used.  Callers with non-rectangular
           polygons (e.g. asymmetric notch) should supply a custom profile that
           matches the edge count.
    @return List of landmark dicts with x, y, yaw_deg keys (local frame).
    """
    fracs_table = edge_fracs if edge_fracs is not None else _LANDMARK_EDGE_FRACS
    pts = [(c["x"], c["y"]) for c in corners]
    n = len(pts)
    landmarks: List[Dict[str, float]] = []
    bays = bays or []

    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]

        edge_dx = x1 - x0
        edge_dy = y1 - y0
        edge_len = math.sqrt(edge_dx * edge_dx + edge_dy * edge_dy)

        if edge_len < _LANDMARK_MIN_EDGE:
            continue

        # Look up fractional positions for this edge index.
        # Edges beyond the profile length get no landmarks.
        if i >= len(fracs_table):
            continue
        fracs = fracs_table[i]
        if not fracs:
            continue

        # Unit vector along the edge.
        ux = edge_dx / edge_len
        uy = edge_dy / edge_len

        # Yaw: aligned with the edge direction (degrees).
        yaw_deg = math.degrees(math.atan2(uy, ux))

        for frac in fracs:
            # Clamp to (0.05, 0.95) to stay clear of corners.
            frac = max(0.05, min(0.95, frac))

            # Place directly on the perimeter edge (no inward offset).
            lx = x0 + ux * edge_len * frac
            ly = y0 + uy * edge_len * frac

            # Skip if inside or too close to any parking bay.
            if any(
                _point_in_bay(lx, ly, b, _LANDMARK_BAY_MARGIN)
                for b in bays
            ):
                continue

            landmarks.append({"x": lx, "y": ly, "yaw_deg": yaw_deg})

    return landmarks


# ---------------------------------------------------------------------------
# World-frame transformation
# ---------------------------------------------------------------------------


def to_world_frame(
    local_layout: Dict[str, Any],
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
) -> Dict[str, Any]:
    """
    @brief Transform a local-frame layout to CARLA world frame.

    All (x, y) coordinates are rotated by heading_deg and translated by
    (origin_x, origin_y). Yaw angles are offset by heading_deg.

    @param local_layout: Layout dict from one of the layout modules.
    @param origin_x: World-frame x of lot origin.
    @param origin_y: World-frame y of lot origin.
    @param origin_z: World-frame z of lot ground (CARLA z, typically 0.3).
    @param heading_deg: Heading of the lot in world frame (degrees, CCW positive).
    @return World-frame layout dict ready for YAML serialisation.
    """
    h_rad = math.radians(heading_deg)

    world_corners = []
    for c in local_layout["corners"]:
        wx, wy = _translate(c["x"], c["y"], origin_x, origin_y, h_rad)
        world_corners.append({"x": round(wx, 3), "y": round(wy, 3)})

    world_bays = []
    for i, b in enumerate(local_layout["bays"]):
        wx, wy = _translate(b["local_x"], b["local_y"], origin_x, origin_y, h_rad)
        world_yaw = _world_yaw(b["local_yaw_deg"], heading_deg)
        world_bay: Dict[str, Any] = {
            "id": f"{b['bay_type']}_{i}",
            "bay_type": b["bay_type"],
            "x": round(wx, 3),
            "y": round(wy, 3),
            "z": origin_z,
            "yaw_deg": round(world_yaw, 2),
            "width": b["width"],
            "depth": b["depth"],
        }
        if b.get("always_empty"):
            world_bay["always_empty"] = True
        if b.get("occupant"):
            world_bay["occupant"] = b["occupant"]
        world_bays.append(world_bay)

    sp = local_layout["spawn"]
    sx, sy = _translate(sp["x"], sp["y"], origin_x, origin_y, h_rad)
    world_spawn = {
        "x": round(sx, 3),
        "y": round(sy, 3),
        "z": origin_z,
        "yaw_deg": round(_world_yaw(sp["yaw_deg"], heading_deg), 2),
    }

    world_extra_spawns = []
    for esp in local_layout.get("extra_spawns", []):
        esx, esy = _translate(esp["x"], esp["y"], origin_x, origin_y, h_rad)
        world_extra_spawns.append(
            {
                "x": round(esx, 3),
                "y": round(esy, 3),
                "z": origin_z,
                "yaw_deg": round(_world_yaw(esp["yaw_deg"], heading_deg), 2),
            }
        )

    world_patrol = []
    for wp in local_layout["patrol_waypoints"]:
        wx, wy = _translate(wp["x"], wp["y"], origin_x, origin_y, h_rad)
        world_patrol.append({"x": round(wx, 3), "y": round(wy, 3)})

    world_ped_zones = []
    for zone in local_layout["pedestrian_zones"]:
        cx = (zone["x_min"] + zone["x_max"]) / 2.0
        cy = (zone["y_min"] + zone["y_max"]) / 2.0
        half_w = (zone["x_max"] - zone["x_min"]) / 2.0
        half_h = (zone["y_max"] - zone["y_min"]) / 2.0
        wcx, wcy = _translate(cx, cy, origin_x, origin_y, h_rad)
        world_ped_zones.append(
            {
                "centre_x": round(wcx, 3),
                "centre_y": round(wcy, 3),
                "half_width": round(half_w, 3),
                "half_height": round(half_h, 3),
            }
        )

    # Transform obstacle rectangles to world frame.
    # Each obstacle is stored as centre + half-extents so the CARLA spawner
    # can place cones around the perimeter without re-computing the bounds.
    world_obstacles = []
    for obs in local_layout.get("obstacles", []):
        cx_loc = (obs["x_min"] + obs["x_max"]) / 2.0
        cy_loc = (obs["y_min"] + obs["y_max"]) / 2.0
        half_w = (obs["x_max"] - obs["x_min"]) / 2.0
        half_h = (obs["y_max"] - obs["y_min"]) / 2.0
        wcx, wcy = _translate(cx_loc, cy_loc, origin_x, origin_y, h_rad)
        world_obstacles.append(
            {
                "centre_x": round(wcx, 3),
                "centre_y": round(wcy, 3),
                "half_width": round(half_w, 3),
                "half_height": round(half_h, 3),
            }
        )

    # Transform asymmetric landmarks to world frame.
    world_landmarks = []
    for lm in local_layout.get("landmarks", []):
        wx, wy = _translate(lm["x"], lm["y"], origin_x, origin_y, h_rad)
        world_landmarks.append(
            {
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z": origin_z,
                "yaw_deg": round(_world_yaw(lm["yaw_deg"], heading_deg), 2),
            }
        )

    # Transform optional asymmetric corner polygon and its landmarks.
    world_asym_corners = []
    for c in local_layout.get("asymmetric_corners", []):
        wx, wy = _translate(c["x"], c["y"], origin_x, origin_y, h_rad)
        world_asym_corners.append({"x": round(wx, 3), "y": round(wy, 3)})

    world_asym_landmarks = []
    for lm in local_layout.get("asymmetric_landmarks", []):
        wx, wy = _translate(lm["x"], lm["y"], origin_x, origin_y, h_rad)
        world_asym_landmarks.append(
            {
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z": origin_z,
                "yaw_deg": round(_world_yaw(lm["yaw_deg"], heading_deg), 2),
            }
        )

    result: Dict[str, Any] = {
        "corners": world_corners,
        "bays": world_bays,
        "spawn_transform": world_spawn,
        "extra_spawn_transforms": world_extra_spawns,
        "patrol_waypoints": world_patrol,
        "pedestrian_zones": world_ped_zones,
        "obstacles": world_obstacles,
        "landmarks": world_landmarks,
    }
    if world_asym_corners:
        result["asymmetric_corners"] = world_asym_corners
    if world_asym_landmarks:
        result["asymmetric_landmarks"] = world_asym_landmarks
    return result


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def write_layout_yaml(
    shape: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
    world_layout: Dict[str, Any],
    output_path: Path,
    ood: bool,
) -> None:
    """
    @brief Serialise a world-frame layout to YAML.

    @param shape: Floor plan shape name (rectangle, trapezoid, irregular_a).
    @param origin_x: World-frame x of lot origin.
    @param origin_y: World-frame y of lot origin.
    @param origin_z: World-frame z of lot origin.
    @param heading_deg: Lot heading in world frame (degrees).
    @param world_layout: World-frame layout dict.
    @param output_path: Path to write the YAML file.
    @param ood: Whether this floor plan is held out for OOD evaluation.
    """
    doc_clean = {
        "floor_plan": shape,
        "ood": ood,
        "origin": {
            "x": origin_x,
            "y": origin_y,
            "z": origin_z,
            "heading_deg": heading_deg,
        },
        "spawn_transform": world_layout["spawn_transform"],
        "extra_spawn_transforms": world_layout["extra_spawn_transforms"],
        "corners": world_layout["corners"],
        "bays": world_layout["bays"],
        "patrol_waypoints": world_layout["patrol_waypoints"],
        "pedestrian_zones": world_layout["pedestrian_zones"],
        "obstacles": world_layout.get("obstacles", []),
        "landmarks": world_layout.get("landmarks", []),
    }
    if world_layout.get("asymmetric_corners"):
        doc_clean["asymmetric_corners"] = world_layout["asymmetric_corners"]
    if world_layout.get("asymmetric_landmarks"):
        doc_clean["asymmetric_landmarks"] = world_layout["asymmetric_landmarks"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Parking lot layout - generated by scripts/generate_layouts.py\n")
        f.write("# Edit origin.x/y and re-run 'make generate-layouts' after\n")
        f.write("# measuring CARLA world-frame coordinates via:\n")
        f.write("#   make docker-inspect INSPECT_LAYOUT=<floor_plan>\n")
        f.write("#\n")
        yaml.dump(
            doc_clean, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Bird's-eye PNG plot
# ---------------------------------------------------------------------------


def plot_layout(
    shape: str,
    world_layout: Dict[str, Any],
    plot_path: Path,
) -> None:
    """
    @brief Render a bird's-eye PNG of the lot layout.

    Layers (back to front): grey lot polygon, bay rectangles (colour-coded by
    type), yaw arrows, spawn triangles, patrol path, pedestrian zone overlays.

    @param shape: Floor plan shape name for title.
    @param world_layout: World-frame layout dict.
    @param plot_path: Path to save the PNG.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Floor plan: {shape}", fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    corner_pts = [(c["x"], c["y"]) for c in world_layout["corners"]]
    from matplotlib.patches import Polygon as MPoly

    lot_patch = MPoly(
        corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="none", linewidth=0
    )
    ax.add_patch(lot_patch)

    # Draw full perimeter -- no gaps (spawns are inside the lot, not on the wall).
    n = len(corner_pts)
    for i in range(n):
        p0 = corner_pts[i]
        p1 = corner_pts[(i + 1) % n]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color="black",
            linewidth=2,
            solid_capstyle="butt",
        )

    # Draw asymmetric corner polygon as a dashed blue overlay when present.
    asym_corners = world_layout.get("asymmetric_corners", [])
    if asym_corners:
        asym_pts = [(c["x"], c["y"]) for c in asym_corners]
        asym_n = len(asym_pts)
        for i in range(asym_n):
            p0 = asym_pts[i]
            p1 = asym_pts[(i + 1) % asym_n]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="blue",
                linewidth=2,
                linestyle="--",
                solid_capstyle="butt",
            )

    from scripts.colours import (
        BAY_HEX,
        HEX_LOT,
        HEX_PATROL_PATH,
        HEX_PEDESTRIAN_ZONE,
        HEX_PEDESTRIAN_ZONE_EDGE,
    )

    for bay in world_layout["bays"]:
        bay_type = bay["bay_type"]
        is_motorcycle = bay_type == "motorcycle"
        colour = "#888888" if is_motorcycle else BAY_HEX.get(bay_type, "grey")
        bx, by = bay["x"], bay["y"]
        yaw_rad = math.radians(bay["yaw_deg"])
        w, d = bay["width"], bay["depth"]
        local_corners = [
            (-d / 2.0, -w / 2.0),
            (d / 2.0, -w / 2.0),
            (d / 2.0, w / 2.0),
            (-d / 2.0, w / 2.0),
        ]
        world_rect = [
            (
                bx + math.cos(yaw_rad) * lx - math.sin(yaw_rad) * ly,
                by + math.sin(yaw_rad) * lx + math.cos(yaw_rad) * ly,
            )
            for lx, ly in local_corners
        ]
        rect_patch = MPoly(
            world_rect,
            closed=True,
            facecolor=colour,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.6,
            zorder=3,
        )
        ax.add_patch(rect_patch)

    for zone in world_layout.get("pedestrian_zones", []):
        zw = zone["half_width"] * 2.0
        zh = zone["half_height"] * 2.0
        cloud = mpatches.FancyBboxPatch(
            (
                zone["centre_x"] - zone["half_width"],
                zone["centre_y"] - zone["half_height"],
            ),
            zw,
            zh,
            boxstyle="round,pad=0.4",
            facecolor=HEX_PEDESTRIAN_ZONE,
            edgecolor=HEX_PEDESTRIAN_ZONE_EDGE,
            alpha=0.40,
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        )
        ax.add_patch(cloud)

    for obs in world_layout.get("obstacles", []):
        obs_patch = mpatches.Rectangle(
            (
                obs["centre_x"] - obs["half_width"],
                obs["centre_y"] - obs["half_height"],
            ),
            obs["half_width"] * 2.0,
            obs["half_height"] * 2.0,
            linewidth=2.5,
            edgecolor="black",
            facecolor="white",
            zorder=5,
        )
        ax.add_patch(obs_patch)

    for idx, sp in enumerate(
        [world_layout["spawn_transform"]]
        + world_layout.get("extra_spawn_transforms", [])
    ):
        yaw_rad = math.radians(sp["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        tri_size = 1.2
        tri_local = [
            (tri_size, 0.0),
            (-tri_size * 0.6, tri_size * 0.6),
            (-tri_size * 0.6, -tri_size * 0.6),
        ]
        tri_world = [
            (
                sp["x"] + cos_y * lx - sin_y * ly,
                sp["y"] + sin_y * lx + cos_y * ly,
            )
            for lx, ly in tri_local
        ]
        tri_patch = MPoly(
            tri_world,
            closed=True,
            facecolor="cyan",
            edgecolor="white",
            linewidth=1,
            zorder=6,
        )
        ax.add_patch(tri_patch)
        label = f"SPAWN {idx + 1}"
        label_offset_perp = 1.5 if idx > 0 else 1.9
        label_y_nudge = 2.0 if idx > 0 else -0.5
        ax.text(
            sp["x"] + cos_y * 0.2 - sin_y * label_offset_perp,
            sp["y"] + sin_y * 0.2 + cos_y * label_offset_perp + label_y_nudge,
            label,
            fontsize=10,
            color="cyan",
            fontweight="bold",
            zorder=7,
            path_effects=[
                __import__(
                    "matplotlib.patheffects", fromlist=["withStroke"]
                ).withStroke(linewidth=2, foreground="black")
            ],
        )

    patrol = world_layout["patrol_waypoints"]
    if patrol:
        px = [wp["x"] for wp in patrol] + [patrol[0]["x"]]
        py = [wp["y"] for wp in patrol] + [patrol[0]["y"]]
        ax.plot(
            px,
            py,
            "--",
            color=HEX_PATROL_PATH,
            linewidth=1.5,
            alpha=0.9,
            label="Patrol path",
        )

    from matplotlib.lines import Line2D

    handles = [
        mpatches.Patch(color=BAY_HEX["perpendicular"], label="Perpendicular bays"),
        mpatches.Patch(color=BAY_HEX["angled"], label="Angled (45 deg) bays"),
        mpatches.Patch(color=BAY_HEX["parallel"], label="Parallel bays"),
        mpatches.Patch(color=HEX_LOT, edgecolor="black", label="Lot boundary"),
        Line2D(
            [0],
            [0],
            color=HEX_PATROL_PATH,
            linestyle="--",
            linewidth=1.5,
            label="Patrol path",
        ),
        mpatches.FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.2",
            facecolor=HEX_PEDESTRIAN_ZONE,
            edgecolor=HEX_PEDESTRIAN_ZONE_EDGE,
            alpha=0.5,
            linestyle="--",
            label="Pedestrian zones",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot:    {plot_path}")
