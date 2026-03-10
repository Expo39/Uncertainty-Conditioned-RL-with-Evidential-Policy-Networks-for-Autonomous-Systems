"""
@file generate_lot_layout.py
@brief Offline parking lot layout generator for the uncertainty-conditioned RL project.

Generates pre-computed world-frame lot geometry (corners, bay positions, spawn transform,
patrol waypoints, pedestrian zones) from shape dimensions and a world-frame origin.
Writes layout YAML files consumed directly by CARLAParkingEnv and optionally
produces bird's-eye PNG plots for visual inspection.

Three floor plan shapes are supported:

  rectangle   -- Standard axis-aligned rectangle, widened to hold 5-bay rows.
  trapezoid   -- Wider at the entrance end, narrower at the rear. One pair of
                 non-parallel sides gives a subtly different LiDAR wall profile.
  irregular_a -- Five-sided polygon (OOD floor plan). Held out from training.
                 One corner cut diagonally to break axis-aligned symmetry.

Bay dimensions follow German EAR 05 / EU harmonised practice (FGSV 2005):

  Perpendicular (90 deg): 2.5 m wide x 5.0 m deep, 6.0 m aisle
  Angled (45 deg):        2.5 m wide x 5.4 m deep, 3.6 m aisle
  Parallel (pull-in):     2.5 m wide x 8.0 m deep, 4.0 m aisle

Layout YAML format (consumed by CARLAParkingEnv):

  floor_plan: rectangle
  origin: {x: 0.0, y: 0.0, z: 0.3, heading_deg: 0.0}
  corners: [{x: ..., y: ...}, ...]
  spawn_transform: {x: ..., y: ..., z: 0.3, yaw_deg: ...}
  bays:
    - {id: perp_0, bay_type: perpendicular, x: ..., y: ..., yaw_deg: ...,
       width: 2.5, depth: 5.0, always_empty: false}
    ...
  patrol_waypoints: [{x: ..., y: ...}, ...]
  pedestrian_zones: [{x_min: ..., x_max: ..., y_min: ..., y_max: ...}]

Usage:
  # Generate all three layouts with placeholder origins (no CARLA needed):
  make generate-layouts

  # Generate one layout explicitly:
  python scripts/generate_lot_layout.py \\
      --shape rectangle --origin 0 0 0.3 --heading 0 \\
      --output configs/layouts/rectangle.yaml \\
      --plot outputs/layouts/rectangle.png

  # After recording CARLA world origins via --mark mode:
  python scripts/generate_lot_layout.py \\
      --shape rectangle --origin -200.0 0.0 0.3 --heading 0 \\
      --output configs/layouts/rectangle.yaml \\
      --plot outputs/layouts/rectangle.png
"""

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Bay dimension constants (German EAR 05)
# ---------------------------------------------------------------------------

BAY_DIMS: Dict[str, Dict[str, float]] = {
    "perpendicular": {"width": 2.5, "depth": 5.0, "aisle": 6.0},
    "angled": {"width": 2.5, "depth": 5.4, "aisle": 3.6},
    "parallel": {"width": 2.5, "depth": 8.0, "aisle": 4.0},
}

BAYS_PER_TYPE = 5  # Exactly 5 bays per type per floor plan
FLOOR_Z = 0.3  # CARLA ground z for all lots


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _rotate(
    x: float,
    y: float,
    heading_rad: float,
) -> Tuple[float, float]:
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
# Bay layout generators (local frame, origin at lot corner)
# ---------------------------------------------------------------------------


def _perpendicular_bays(
    n: int,
    row_x: float,
    row_y_start: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n perpendicular bay centres in a row.

    Bays are arranged side-by-side along the Y axis. The vehicle drives
    in along +X (facing_yaw_deg = 0 means drive along +X into the bay).

    @param n: Number of bays to generate.
    @param row_x: X position of bay centre row (local frame).
    @param row_y_start: Y position of the first bay centre.
    @param facing_yaw_deg: Yaw of a vehicle parked in this bay (degrees, local).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["perpendicular"]
    bays = []
    for i in range(n):
        bays.append(
            {
                "bay_type": "perpendicular",
                "local_x": row_x,
                "local_y": row_y_start + i * dims["width"],
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
                "always_empty": False,
            }
        )
    return bays


def _angled_bays(
    n: int,
    row_x: float,
    row_y_start: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n angled (45-deg) bay centres in a row.

    Bays are arranged side-by-side. Pitch of each bay is 45 deg from aisle axis.

    @param n: Number of bays to generate.
    @param row_x: X position of row front edge.
    @param row_y_start: Y position of the first bay centre.
    @param facing_yaw_deg: Yaw of a vehicle parked in this bay (degrees, local).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["angled"]
    bays = []
    # At 45 deg, lateral spacing between bay centres along the row axis
    # is width / sin(45) = width * sqrt(2)
    lateral_spacing = dims["width"] / math.sin(math.radians(45.0))
    for i in range(n):
        bays.append(
            {
                "bay_type": "angled",
                "local_x": row_x,
                "local_y": row_y_start + i * lateral_spacing,
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
                "always_empty": False,
            }
        )
    return bays


def _parallel_bays(
    n: int,
    row_y: float,
    row_x_start: float,
    facing_yaw_deg: float,
) -> List[Dict[str, Any]]:
    """
    @brief Generate n parallel (pull-in) bay centres along the X axis.

    Parallel bays are arranged front-to-back along the X axis. Front and rear
    neighbour slots get always_empty=True automatically.

    @param n: Number of bays to generate.
    @param row_y: Y position of bay centre row.
    @param row_x_start: X position of the first bay centre.
    @param facing_yaw_deg: Yaw of a vehicle parked in this bay (degrees, local).
    @return List of bay dicts.
    """
    dims = BAY_DIMS["parallel"]
    bays = []
    for i in range(n):
        bays.append(
            {
                "bay_type": "parallel",
                "local_x": row_x_start + i * dims["depth"],
                "local_y": row_y,
                "local_yaw_deg": facing_yaw_deg,
                "width": dims["width"],
                "depth": dims["depth"],
                # First and last slots are always empty to give clearance for pull-in
                "always_empty": (i == 0 or i == n - 1),
            }
        )
    return bays


# ---------------------------------------------------------------------------
# Floor plan geometry definitions
# ---------------------------------------------------------------------------


def _rectangle_layout(
    width: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Generate a rectangular floor plan layout in local frame.

    Layout:
      - Perpendicular bay row along rear wall (high x)
      - 45-deg bay row in the middle
      - Parallel bay row along one side
      - Spawn at front-centre (low x, mid y)

    @param width: Lot width (Y axis extent), metres.
    @param depth: Lot depth (X axis extent), metres.
    @return Dict with corners, bays, spawn, patrol, pedestrian zones (local frame).
    """
    # Perimeter corners (local frame, CCW from bottom-left)
    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width},
        {"x": 0.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular row: rear of lot, bays face -X (yaw=180 deg in local frame)
    perp_row_x = depth - dims_perp["depth"] - 1.0  # 1 m margin from rear wall
    perp_y_start = 1.5  # 1.5 m from side wall
    perp_bays = _perpendicular_bays(BAYS_PER_TYPE, perp_row_x, perp_y_start, 180.0)

    # 45-deg row: mid-lot, bays face 225 deg (back-left in local frame)
    ang_row_x = (
        perp_row_x
        - dims_perp["aisle"]
        - dims_ang["depth"] * math.cos(math.radians(45.0))
    )
    ang_y_start = 1.5
    ang_bays = _angled_bays(BAYS_PER_TYPE, ang_row_x, ang_y_start, 225.0)

    # Parallel row: along one side wall, bays face 180 deg (nose pointing +X)
    par_row_y = width - dims_par["width"] / 2.0 - 1.0  # 1 m from side wall
    par_x_start = 2.0  # 2 m from front wall
    par_bays = _parallel_bays(BAYS_PER_TYPE, par_row_y, par_x_start, 90.0)

    all_bays = perp_bays + ang_bays + par_bays

    # Spawn: front-centre, facing into lot (+X direction, yaw=0)
    spawn = {"x": 1.5, "y": width / 2.0, "yaw_deg": 0.0}

    # Patrol waypoints: rectangular loop around inner lot
    margin = 4.0
    patrol = [
        {"x": margin, "y": margin},
        {"x": depth - margin, "y": margin},
        {"x": depth - margin, "y": width - margin},
        {"x": margin, "y": width - margin},
    ]

    # Pedestrian zones: two strips along the aisles
    ped_zones = [
        {
            "x_min": ang_row_x - 1.0,
            "x_max": ang_row_x + dims_ang["aisle"],
            "y_min": 1.0,
            "y_max": width - 1.0,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }


def _trapezoid_layout(
    width_front: float,
    width_rear: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Generate a trapezoidal floor plan layout in local frame.

    Wider at the front (entrance), narrower at the rear. This produces a
    different LiDAR wall-distance profile compared to the rectangle.

    @param width_front: Width at the entrance side (y extent at x=0).
    @param width_rear: Width at the rear side (y extent at x=depth).
    @param depth: Lot depth (X axis extent), metres.
    @return Dict with corners, bays, spawn, patrol, pedestrian zones (local frame).
    """
    # Y offset of rear wall to create trapezoid shape
    # Lot is symmetric: both sides taper by (width_front - width_rear)/2
    y_offset = (width_front - width_rear) / 2.0

    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": y_offset},
        {"x": depth, "y": width_front - y_offset},
        {"x": 0.0, "y": width_front},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular row at rear (narrower end)
    perp_row_x = depth - dims_perp["depth"] - 1.0
    perp_y_start = y_offset + 1.5
    perp_bays = _perpendicular_bays(BAYS_PER_TYPE, perp_row_x, perp_y_start, 180.0)

    # 45-deg row in mid-lot
    ang_row_x = (
        perp_row_x
        - dims_perp["aisle"]
        - dims_ang["depth"] * math.cos(math.radians(45.0))
    )
    ang_y_start = 1.5
    ang_bays = _angled_bays(BAYS_PER_TYPE, ang_row_x, ang_y_start, 225.0)

    # Parallel row near front (wider end)
    par_row_y = width_front - dims_par["width"] / 2.0 - 1.0
    par_x_start = 2.0
    par_bays = _parallel_bays(BAYS_PER_TYPE, par_row_y, par_x_start, 90.0)

    all_bays = perp_bays + ang_bays + par_bays

    # Spawn at front-centre
    spawn = {"x": 1.5, "y": width_front / 2.0, "yaw_deg": 0.0}

    margin = 4.0
    patrol = [
        {"x": margin, "y": margin},
        {"x": depth - margin, "y": y_offset + margin},
        {"x": depth - margin, "y": width_front - y_offset - margin},
        {"x": margin, "y": width_front - margin},
    ]

    ped_zones = [
        {
            "x_min": ang_row_x - 1.0,
            "x_max": ang_row_x + dims_ang["aisle"],
            "y_min": 1.0,
            "y_max": width_front - 1.0,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }


def _irregular_a_layout(
    width: float,
    depth: float,
) -> Dict[str, Any]:
    """
    @brief Generate a five-sided irregular polygon floor plan (OOD only).

    One corner of the rectangle is cut diagonally, creating a five-sided
    polygon that breaks the axis-aligned symmetry seen in training shapes.

    @param width: Nominal lot width (Y axis extent), metres.
    @param depth: Nominal lot depth (X axis extent), metres.
    @return Dict with corners, bays, spawn, patrol, pedestrian zones (local frame).
    """
    # Cut the rear-right corner diagonally: remove 8m x 8m triangle
    cut = 8.0
    corners = [
        {"x": 0.0, "y": 0.0},
        {"x": depth, "y": 0.0},
        {"x": depth, "y": width - cut},  # cut starts here
        {"x": depth - cut, "y": width},  # cut ends here
        {"x": 0.0, "y": width},
    ]

    dims_perp = BAY_DIMS["perpendicular"]
    dims_ang = BAY_DIMS["angled"]
    dims_par = BAY_DIMS["parallel"]

    # Perpendicular row: slightly shortened to avoid the cut corner
    perp_row_x = depth - dims_perp["depth"] - 1.5
    perp_y_start = 1.5
    perp_bays = _perpendicular_bays(BAYS_PER_TYPE, perp_row_x, perp_y_start, 180.0)

    # 45-deg row
    ang_row_x = (
        perp_row_x
        - dims_perp["aisle"]
        - dims_ang["depth"] * math.cos(math.radians(45.0))
    )
    ang_y_start = 1.5
    ang_bays = _angled_bays(BAYS_PER_TYPE, ang_row_x, ang_y_start, 225.0)

    # Parallel row
    par_row_y = width - dims_par["width"] / 2.0 - 1.0
    par_x_start = 2.0
    par_bays = _parallel_bays(BAYS_PER_TYPE, par_row_y, par_x_start, 90.0)

    all_bays = perp_bays + ang_bays + par_bays

    spawn = {"x": 1.5, "y": width / 2.0, "yaw_deg": 0.0}

    margin = 4.0
    patrol = [
        {"x": margin, "y": margin},
        {"x": depth - margin, "y": margin},
        {"x": depth - margin, "y": width - cut - margin},
        {"x": depth - cut - margin, "y": width - margin},
        {"x": margin, "y": width - margin},
    ]

    ped_zones = [
        {
            "x_min": ang_row_x - 1.0,
            "x_max": ang_row_x + dims_ang["aisle"],
            "y_min": 1.0,
            "y_max": width - 1.0,
        },
    ]

    return {
        "corners": corners,
        "bays": all_bays,
        "spawn": spawn,
        "patrol_waypoints": patrol,
        "pedestrian_zones": ped_zones,
    }


# ---------------------------------------------------------------------------
# World-frame transformation
# ---------------------------------------------------------------------------


def _to_world_frame(
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

    @param local_layout: Layout dict from one of the _*_layout() functions.
    @param origin_x: World-frame x of lot origin.
    @param origin_y: World-frame y of lot origin.
    @param origin_z: World-frame z of lot ground (CARLA z, typically 0.3).
    @param heading_deg: Heading of the lot in world frame (degrees, CCW positive).
    @return World-frame layout dict ready for YAML serialisation.
    """
    h_rad = math.radians(heading_deg)

    # Transform corners
    world_corners = []
    for c in local_layout["corners"]:
        wx, wy = _translate(c["x"], c["y"], origin_x, origin_y, h_rad)
        world_corners.append({"x": round(wx, 3), "y": round(wy, 3)})

    # Transform bays
    world_bays = []
    for i, b in enumerate(local_layout["bays"]):
        wx, wy = _translate(b["local_x"], b["local_y"], origin_x, origin_y, h_rad)
        world_yaw = _world_yaw(b["local_yaw_deg"], heading_deg)
        world_bays.append(
            {
                "id": f"{b['bay_type']}_{i}",
                "bay_type": b["bay_type"],
                "x": round(wx, 3),
                "y": round(wy, 3),
                "z": origin_z,
                "yaw_deg": round(world_yaw, 2),
                "width": b["width"],
                "depth": b["depth"],
                "always_empty": b["always_empty"],
            }
        )

    # Transform spawn
    sp = local_layout["spawn"]
    sx, sy = _translate(sp["x"], sp["y"], origin_x, origin_y, h_rad)
    world_spawn = {
        "x": round(sx, 3),
        "y": round(sy, 3),
        "z": origin_z,
        "yaw_deg": round(_world_yaw(sp["yaw_deg"], heading_deg), 2),
    }

    # Transform patrol waypoints
    world_patrol = []
    for wp in local_layout["patrol_waypoints"]:
        wx, wy = _translate(wp["x"], wp["y"], origin_x, origin_y, h_rad)
        world_patrol.append({"x": round(wx, 3), "y": round(wy, 3)})

    # Transform pedestrian zones (axis-aligned bounding boxes -> transform centre)
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

    return {
        "corners": world_corners,
        "bays": world_bays,
        "spawn_transform": world_spawn,
        "patrol_waypoints": world_patrol,
        "pedestrian_zones": world_ped_zones,
    }


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------


def _write_layout_yaml(
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
    doc = {
        "# Parking lot layout - generated by scripts/generate_lot_layout.py": None,
        "# Edit origin and re-run make generate-layouts after measuring CARLA coordinates.": None,
        "floor_plan": shape,
        "ood": ood,
        "origin": {
            "x": origin_x,
            "y": origin_y,
            "z": origin_z,
            "heading_deg": heading_deg,
        },
        "spawn_transform": world_layout["spawn_transform"],
        "corners": world_layout["corners"],
        "bays": world_layout["bays"],
        "patrol_waypoints": world_layout["patrol_waypoints"],
        "pedestrian_zones": world_layout["pedestrian_zones"],
    }

    # Remove sentinel comment keys (just used for readability above)
    doc_clean = {k: v for k, v in doc.items() if not k.startswith("#")}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Write header comment manually since PyYAML strips comments
        f.write("# Parking lot layout - generated by scripts/generate_lot_layout.py\n")
        f.write("# Edit origin.x/y and re-run 'make generate-layouts' after\n")
        f.write("# measuring CARLA world-frame coordinates via:\n")
        f.write("#   python scripts/explore_map.py --mark --town Town05_Opt\n")
        f.write("#\n")
        yaml.dump(
            doc_clean, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    print(f"  Written: {output_path}")


# ---------------------------------------------------------------------------
# Bird's-eye PNG plot
# ---------------------------------------------------------------------------


def _plot_layout(
    shape: str,
    world_layout: Dict[str, Any],
    plot_path: Path,
) -> None:
    """
    @brief Render a bird's-eye PNG of the lot layout.

    Layers (back to front):
      - Grey lot polygon
      - Bay rectangles (colour-coded by type)
      - Yaw arrows on each bay
      - Spawn triangle
      - Patrol path

    @param shape: Floor plan shape name for title.
    @param world_layout: World-frame layout dict.
    @param plot_path: Path to save the PNG.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Polygon
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Floor plan: {shape}", fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)

    # Lot boundary polygon
    corner_pts = [(c["x"], c["y"]) for c in world_layout["corners"]]
    lot_patch = Polygon(
        corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="black", linewidth=2
    )
    ax.add_patch(lot_patch)

    # Bay rectangles
    bay_colours = {
        "perpendicular": "steelblue",
        "angled": "darkorange",
        "parallel": "forestgreen",
    }
    for bay in world_layout["bays"]:
        colour = bay_colours.get(bay["bay_type"], "grey")
        alpha = 0.4 if bay["always_empty"] else 0.7
        lw = 1 if bay["always_empty"] else 2
        bx, by = bay["x"], bay["y"]
        yaw_rad = math.radians(bay["yaw_deg"])
        w, d = bay["width"], bay["depth"]

        # Rectangle corners (local: -d/2..d/2 in X, -w/2..w/2 in Y)
        local_corners = [
            (-d / 2.0, -w / 2.0),
            (d / 2.0, -w / 2.0),
            (d / 2.0, w / 2.0),
            (-d / 2.0, w / 2.0),
        ]
        world_rect = []
        for lx, ly in local_corners:
            wx = bx + math.cos(yaw_rad) * lx - math.sin(yaw_rad) * ly
            wy = by + math.sin(yaw_rad) * lx + math.cos(yaw_rad) * ly
            world_rect.append((wx, wy))

        rect_patch = Polygon(
            world_rect,
            closed=True,
            facecolor=colour,
            edgecolor=colour,
            linewidth=lw,
            alpha=alpha,
        )
        ax.add_patch(rect_patch)

        # Yaw arrow
        arrow_len = 1.5
        ax.annotate(
            "",
            xy=(bx + math.cos(yaw_rad) * arrow_len, by + math.sin(yaw_rad) * arrow_len),
            xytext=(bx, by),
            arrowprops={"arrowstyle": "->", "color": colour, "lw": 1.5},
        )

    # Spawn triangle
    sp = world_layout["spawn_transform"]
    yaw_rad = math.radians(sp["yaw_deg"])
    ax.plot(sp["x"], sp["y"], marker="^", markersize=12, color="cyan", zorder=5)
    ax.annotate(
        "",
        xy=(sp["x"] + math.cos(yaw_rad) * 2.5, sp["y"] + math.sin(yaw_rad) * 2.5),
        xytext=(sp["x"], sp["y"]),
        arrowprops={"arrowstyle": "->", "color": "cyan", "lw": 2},
    )
    ax.text(sp["x"] + 0.5, sp["y"] + 0.5, "SPAWN", fontsize=8, color="cyan")

    # Patrol path
    patrol = world_layout["patrol_waypoints"]
    if patrol:
        px = [wp["x"] for wp in patrol] + [patrol[0]["x"]]
        py = [wp["y"] for wp in patrol] + [patrol[0]["y"]]
        ax.plot(px, py, "m--", linewidth=1.5, alpha=0.7, label="Patrol path")

    # Legend
    handles = [
        mpatches.Patch(color="steelblue", label="Perpendicular bays"),
        mpatches.Patch(color="darkorange", label="Angled (45 deg) bays"),
        mpatches.Patch(color="forestgreen", label="Parallel bays"),
        mpatches.Patch(color="#DDDDDD", edgecolor="black", label="Lot boundary"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot:    {plot_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate parking lot layout YAML + bird's-eye PNG.\n"
            "Run without --shape to generate all three layouts at once."
        )
    )
    parser.add_argument(
        "--shape",
        choices=["rectangle", "trapezoid", "irregular_a"],
        default=None,
        help="Floor plan shape. Omit to generate all three.",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=[0.0, 0.0, 0.3],
        help="World-frame origin of the lot (x y z). Default: 0 0 0.3.",
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=0.0,
        help="Lot heading in world frame (degrees). Default: 0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path (single shape mode only).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output PNG path (single shape mode only).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/layouts",
        help="Output directory for YAML files (multi-shape mode). Default: configs/layouts.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="outputs/layouts",
        help="Output directory for PNG files (multi-shape mode). Default: outputs/layouts.",
    )
    return parser.parse_args()


def _generate_one(
    shape: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
    output_path: Path,
    plot_path: Optional[Path],
) -> None:
    """
    @brief Generate one floor plan layout and write to disk.

    @param shape: One of 'rectangle', 'trapezoid', 'irregular_a'.
    @param origin_x: World x of lot origin.
    @param origin_y: World y of lot origin.
    @param origin_z: World z of lot origin.
    @param heading_deg: Lot heading in world frame (degrees).
    @param output_path: YAML output path.
    @param plot_path: PNG output path, or None to skip.
    """
    ood = shape == "irregular_a"

    if shape == "rectangle":
        local_layout = _rectangle_layout(width=35.0, depth=35.0)
    elif shape == "trapezoid":
        local_layout = _trapezoid_layout(width_front=35.0, width_rear=25.0, depth=35.0)
    elif shape == "irregular_a":
        local_layout = _irregular_a_layout(width=35.0, depth=35.0)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    world_layout = _to_world_frame(
        local_layout, origin_x, origin_y, origin_z, heading_deg
    )

    _write_layout_yaml(
        shape, origin_x, origin_y, origin_z, heading_deg, world_layout, output_path, ood
    )

    if plot_path is not None:
        _plot_layout(shape, world_layout, plot_path)


def main() -> None:
    """
    @brief Entry point: generate one or all floor plan layouts.
    """
    args = _parse_args()
    ox, oy, oz = args.origin

    if args.shape is not None:
        # Single shape mode
        out = (
            Path(args.output)
            if args.output
            else Path(args.output_dir) / f"{args.shape}.yaml"
        )
        plot = (
            Path(args.plot) if args.plot else Path(args.plot_dir) / f"{args.shape}.png"
        )
        print(
            f"Generating {args.shape} layout (origin={ox},{oy},{oz}, heading={args.heading} deg)"
        )
        _generate_one(args.shape, ox, oy, oz, args.heading, out, plot)
    else:
        # Multi-shape mode: generate all three with well-separated origins on Town05_Opt
        # Placeholder origins: use actual CARLA coordinates from --mark mode
        # These are approximate positions in flat areas of Town05_Opt (z=0.30)
        configs_map = {
            "rectangle": {"ox": -200.0, "oy": 0.0, "oz": 0.3, "hdg": 0.0, "ood": False},
            "trapezoid": {"ox": 0.0, "oy": -90.0, "oz": 0.3, "hdg": 0.0, "ood": False},
            "irregular_a": {"ox": 100.0, "oy": 0.0, "oz": 0.3, "hdg": 0.0, "ood": True},
        }

        out_dir = Path(args.output_dir)
        plot_dir = Path(args.plot_dir)
        print("Generating all floor plan layouts...")
        print(f"  YAMLs -> {out_dir}/")
        print(f"  PNGs  -> {plot_dir}/")
        print()
        print("NOTE: Origins are approximate placeholders.")
        print("      Run 'python scripts/explore_map.py --mark --town Town05_Opt'")
        print(
            "      to record precise CARLA world coordinates, then re-run this script."
        )
        print()

        for shape, cfg in configs_map.items():
            _generate_one(
                shape=shape,
                origin_x=cfg["ox"],
                origin_y=cfg["oy"],
                origin_z=cfg["oz"],
                heading_deg=cfg["hdg"],
                output_path=out_dir / f"{shape}.yaml",
                plot_path=plot_dir / f"{shape}.png",
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
