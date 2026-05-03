"""
@file common.py
@brief World-frame transform, YAML serialisation, and PNG plotting for parking
       lot layouts.

This module is the *engine* side of layout generation: it turns a layout dict
(returned by LotBuilder.build()) into a CARLA-frame YAML file plus a bird's-eye
PNG. It contains no layout-specific code and no DSL primitives.

All bay-related concerns (constants, geometry primitives, validators, the DSL
itself) live in builder.py. Floor plan modules import only from builder.py;
generate_layouts.py imports the three engine functions from here.
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


# ---------------------------------------------------------------------------
# Low-level rotation/translation primitives used by to_world_frame
# ---------------------------------------------------------------------------


def _rotate(x: float, y: float, heading_rad: float) -> Tuple[float, float]:
    """
    @brief Rotate point (x, y) by heading_rad about the origin.
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
    """
    rx, ry = _rotate(x, y, heading_rad)
    return (rx + origin_x, ry + origin_y)


def _world_yaw(local_yaw_deg: float, heading_deg: float) -> float:
    """
    @brief Convert a local yaw angle to world-frame yaw.
    """
    return (local_yaw_deg + heading_deg) % 360.0


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

    Layout modules define geometry in a right-handed math frame (Y-up, yaw
    CCW-positive). CARLA uses UE4's left-handed frame (Y increases rightward,
    yaw CW-positive). This function applies rotation + translation in the
    math frame, then mirrors into CARLA's frame by negating Y and yaw.

    The origin_x/origin_y constants in each layout module are specified in
    CARLA world coordinates (left-handed). The internal math is done in the
    right-handed frame and converted at the end.

    @param local_layout: Layout dict from one of the layout modules.
    @param origin_x: CARLA world-frame x of lot origin.
    @param origin_y: CARLA world-frame y of lot origin (left-handed).
    @param origin_z: CARLA world-frame z of lot ground (typically 0.3).
    @param heading_deg: Heading of the lot in world frame (degrees, CCW positive).
    @return CARLA world-frame layout dict ready for YAML serialisation.
    """
    h_rad = math.radians(heading_deg)
    # The origin is in CARLA's left-handed frame. Negate origin_y so the
    # intermediate math stays in the right-handed frame; then negate the
    # final Y output to return to CARLA frame.
    rh_origin_y = -origin_y

    def _to_carla(x_rh: float, y_rh: float) -> Tuple[float, float]:
        """Convert right-handed (x, y) to CARLA left-handed (x, -y)."""
        # + 0.0 avoids negative zero in YAML output.
        return (x_rh + 0.0, -y_rh + 0.0)

    def _carla_yaw(yaw_rh_deg: float) -> float:
        """Convert right-handed yaw (CCW+) to CARLA yaw (CW+)."""
        return (-yaw_rh_deg) % 360.0

    world_corners = []
    for c in local_layout["corners"]:
        wx, wy = _translate(c["x"], c["y"], origin_x, rh_origin_y, h_rad)
        cx, cy = _to_carla(wx, wy)
        world_corners.append({"x": round(cx, 3), "y": round(cy, 3)})

    world_bays = []
    for i, b in enumerate(local_layout["bays"]):
        wx, wy = _translate(b["local_x"], b["local_y"], origin_x, rh_origin_y, h_rad)
        cx, cy = _to_carla(wx, wy)
        world_yaw = _carla_yaw(_world_yaw(b["local_yaw_deg"], heading_deg))
        world_bay: Dict[str, Any] = {
            "id": f"{b['bay_type']}_{i}",
            "bay_type": b["bay_type"],
            "x": round(cx, 3),
            "y": round(cy, 3),
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
    sx, sy = _translate(sp["x"], sp["y"], origin_x, rh_origin_y, h_rad)
    csx, csy = _to_carla(sx, sy)
    world_spawn = {
        "x": round(csx, 3),
        "y": round(csy, 3),
        "z": origin_z,
        "yaw_deg": round(
            _carla_yaw(_world_yaw(sp["yaw_deg"], heading_deg)), 2
        ),
    }

    world_extra_spawns = []
    for esp in local_layout.get("extra_spawns", []):
        esx, esy = _translate(esp["x"], esp["y"], origin_x, rh_origin_y, h_rad)
        cesx, cesy = _to_carla(esx, esy)
        world_extra_spawns.append(
            {
                "x": round(cesx, 3),
                "y": round(cesy, 3),
                "z": origin_z,
                "yaw_deg": round(
                    _carla_yaw(_world_yaw(esp["yaw_deg"], heading_deg)), 2
                ),
            }
        )

    world_patrol = []
    for wp in local_layout["patrol_waypoints"]:
        wx, wy = _translate(wp["x"], wp["y"], origin_x, rh_origin_y, h_rad)
        cx, cy = _to_carla(wx, wy)
        world_patrol.append({"x": round(cx, 3), "y": round(cy, 3)})

    world_ped_zones = []
    for zone in local_layout["pedestrian_zones"]:
        zx = (zone["x_min"] + zone["x_max"]) / 2.0
        zy = (zone["y_min"] + zone["y_max"]) / 2.0
        half_w = (zone["x_max"] - zone["x_min"]) / 2.0
        half_h = (zone["y_max"] - zone["y_min"]) / 2.0
        wcx, wcy = _translate(zx, zy, origin_x, rh_origin_y, h_rad)
        ccx, ccy = _to_carla(wcx, wcy)
        world_ped_zones.append(
            {
                "centre_x": round(ccx, 3),
                "centre_y": round(ccy, 3),
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
        wcx, wcy = _translate(cx_loc, cy_loc, origin_x, rh_origin_y, h_rad)
        ccx, ccy = _to_carla(wcx, wcy)
        world_obstacles.append(
            {
                "centre_x": round(ccx, 3),
                "centre_y": round(ccy, 3),
                "half_width": round(half_w, 3),
                "half_height": round(half_h, 3),
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
    }
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
    }

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
    # Layout YAMLs are in CARLA's left-handed frame (Y increases rightward).
    # Invert Y so the PNG matches an intuitive bird's-eye view (north = up).
    ax.invert_yaxis()

    corner_pts = [(c["x"], c["y"]) for c in world_layout["corners"]]
    from matplotlib.patches import Polygon as MPoly

    lot_patch = MPoly(
        corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="none", linewidth=0
    )
    ax.add_patch(lot_patch)

    # Draw full perimeter - no gaps (spawns are inside the lot, not on the wall).
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
