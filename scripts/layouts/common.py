"""
@file common.py
@brief World-frame transform, YAML serialisation, and PNG plotting for parking
       lot layouts.

This module is the *engine* side of layout generation: it turns a layout dict
(returned by LotBuilder.build()) into a CARLA-frame YAML file plus a bird's-eye
PNG.
"""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


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

    @param local_layout: Layout dict from one of the layout modules.
    @param origin_x: CARLA world-frame x of lot origin.
    @param origin_y: CARLA world-frame y of lot origin (left-handed).
    @param origin_z: CARLA world-frame z of lot ground (typically 0.3).
    @param heading_deg: Heading of the lot in world frame (degrees, CCW positive).
    @return CARLA world-frame layout dict ready for YAML serialisation.
    """
    h_rad = math.radians(heading_deg)
    cos_h = math.cos(h_rad)
    sin_h = math.sin(h_rad)
    # The origin is in CARLA's left-handed frame. Negate origin_y so the
    # intermediate math stays in the right-handed frame; then negate the
    # final Y output to return to CARLA frame.
    rh_oy = -origin_y

    def _xform(x: float, y: float) -> Tuple[float, float]:
        """Rotate + translate right-handed (x,y), then mirror to CARLA frame."""
        rx = cos_h * x - sin_h * y
        ry = sin_h * x + cos_h * y
        # + 0.0 avoids negative zero in YAML output.
        return (rx + origin_x + 0.0, -(ry + rh_oy) + 0.0)

    def _yaw_to_carla(local_yaw_deg: float) -> float:
        """Convert local yaw (CCW+) to CARLA yaw (CW+)."""
        return (-(local_yaw_deg + heading_deg)) % 360.0

    def _xform_spawn(sp: Dict[str, Any]) -> Dict[str, Any]:
        cx, cy = _xform(sp["x"], sp["y"])
        return {
            "x": round(cx, 3),
            "y": round(cy, 3),
            "z": origin_z,
            "yaw_deg": round(_yaw_to_carla(sp["yaw_deg"]), 2),
        }

    def _xform_box(box: Dict[str, float]) -> Dict[str, float]:
        """Transform a min/max box to world-frame centre + half-extents."""
        lx = (box["x_min"] + box["x_max"]) / 2.0
        ly = (box["y_min"] + box["y_max"]) / 2.0
        cx, cy = _xform(lx, ly)
        return {
            "centre_x": round(cx, 3),
            "centre_y": round(cy, 3),
            "half_width": round((box["x_max"] - box["x_min"]) / 2.0, 3),
            "half_height": round((box["y_max"] - box["y_min"]) / 2.0, 3),
        }

    world_corners = [
        {"x": round(cx, 3), "y": round(cy, 3)}
        for c in local_layout["corners"]
        for cx, cy in (_xform(c["x"], c["y"]),)
    ]

    world_bays = []
    for i, b in enumerate(local_layout["bays"]):
        cx, cy = _xform(b["local_x"], b["local_y"])
        world_bay: Dict[str, Any] = {
            "id": f"{b['bay_type']}_{i}",
            "bay_type": b["bay_type"],
            "x": round(cx, 3),
            "y": round(cy, 3),
            "z": origin_z,
            "yaw_deg": round(_yaw_to_carla(b["local_yaw_deg"]), 2),
            "width": b["width"],
            "depth": b["depth"],
        }
        if b.get("always_empty"):
            world_bay["always_empty"] = True
        if b.get("occupant"):
            world_bay["occupant"] = b["occupant"]
        world_bays.append(world_bay)

    world_patrol = [
        {"x": round(cx, 3), "y": round(cy, 3)}
        for wp in local_layout["patrol_waypoints"]
        for cx, cy in (_xform(wp["x"], wp["y"]),)
    ]

    world_extra_spawns = [
        _xform_spawn(esp) for esp in local_layout.get("extra_spawns", [])
    ]

    world_ped_zones = [
        _xform_box(zone) for zone in local_layout["pedestrian_zones"]
    ]

    world_obstacles = [
        _xform_box(obs) for obs in local_layout.get("obstacles", [])
    ]

    return {
        "corners": world_corners,
        "bays": world_bays,
        "spawn_transform": _xform_spawn(local_layout["spawn"]),
        "extra_spawn_transforms": world_extra_spawns,
        "patrol_waypoints": world_patrol,
        "pedestrian_zones": world_ped_zones,
        "obstacles": world_obstacles,
    }


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
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon as MPoly
        from matplotlib.patheffects import withStroke
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    from scripts.colours import (
        BAY_HEX,
        HEX_LOT,
        HEX_PATROL_PATH,
        HEX_PEDESTRIAN_ZONE,
        HEX_PEDESTRIAN_ZONE_EDGE,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Floor plan: {shape}", fontsize=14)
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    # Layout YAMLs are in CARLA's left-handed frame (Y increases rightward).
    # Invert Y so the PNG matches an intuitive bird's-eye view (north = up).
    ax.invert_yaxis()

    corner_pts = [(c["x"], c["y"]) for c in world_layout["corners"]]
    lot_patch = MPoly(
        corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="none", linewidth=0
    )
    ax.add_patch(lot_patch)

    # Draw full perimeter as a single closed polygon outline.
    perim_patch = MPoly(
        corner_pts,
        closed=True,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
        capstyle="butt",
    )
    ax.add_patch(perim_patch)

    for bay in world_layout["bays"]:
        bay_type = bay["bay_type"]
        colour = "#888888" if bay_type == "motorcycle" else BAY_HEX.get(bay_type, "grey")
        bx, by = bay["x"], bay["y"]
        yaw_rad = math.radians(bay["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        hw, hd = bay["width"] / 2.0, bay["depth"] / 2.0
        local_corners = [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
        world_rect = [
            (bx + cos_y * lx - sin_y * ly, by + sin_y * lx + cos_y * ly)
            for lx, ly in local_corners
        ]
        ax.add_patch(MPoly(
            world_rect,
            closed=True,
            facecolor=colour,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.6,
            zorder=3,
        ))

    for zone in world_layout.get("pedestrian_zones", []):
        ax.add_patch(mpatches.FancyBboxPatch(
            (zone["centre_x"] - zone["half_width"], zone["centre_y"] - zone["half_height"]),
            zone["half_width"] * 2.0,
            zone["half_height"] * 2.0,
            boxstyle="round,pad=0.4",
            facecolor=HEX_PEDESTRIAN_ZONE,
            edgecolor=HEX_PEDESTRIAN_ZONE_EDGE,
            alpha=0.40,
            linewidth=1.5,
            linestyle="--",
            zorder=4,
        ))

    for obs in world_layout.get("obstacles", []):
        ax.add_patch(mpatches.Rectangle(
            (obs["centre_x"] - obs["half_width"], obs["centre_y"] - obs["half_height"]),
            obs["half_width"] * 2.0,
            obs["half_height"] * 2.0,
            linewidth=2.5,
            edgecolor="black",
            facecolor="white",
            zorder=5,
        ))

    stroke_effect = [withStroke(linewidth=2, foreground="black")]
    tri_local = [(1.2, 0.0), (-0.72, 0.72), (-0.72, -0.72)]
    for idx, sp in enumerate(
        [world_layout["spawn_transform"]] + world_layout.get("extra_spawn_transforms", [])
    ):
        yaw_rad = math.radians(sp["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        tri_world = [
            (sp["x"] + cos_y * lx - sin_y * ly, sp["y"] + sin_y * lx + cos_y * ly)
            for lx, ly in tri_local
        ]
        ax.add_patch(MPoly(
            tri_world,
            closed=True,
            facecolor="cyan",
            edgecolor="white",
            linewidth=1,
            zorder=6,
        ))
        label_offset_perp = 1.5 if idx > 0 else 1.9
        label_y_nudge = 2.0 if idx > 0 else -0.5
        ax.text(
            sp["x"] + cos_y * 0.2 - sin_y * label_offset_perp,
            sp["y"] + sin_y * 0.2 + cos_y * label_offset_perp + label_y_nudge,
            f"SPAWN {idx + 1}",
            fontsize=10,
            color="cyan",
            fontweight="bold",
            zorder=7,
            path_effects=stroke_effect,
        )

    patrol = world_layout["patrol_waypoints"]
    if patrol:
        px = [wp["x"] for wp in patrol] + [patrol[0]["x"]]
        py = [wp["y"] for wp in patrol] + [patrol[0]["y"]]
        ax.plot(px, py, "--", color=HEX_PATROL_PATH, linewidth=1.5, alpha=0.9, label="Patrol path")

    handles = [
        mpatches.Patch(color=BAY_HEX["perpendicular"], label="Perpendicular bays"),
        mpatches.Patch(color=BAY_HEX["angled"], label="Angled (45 deg) bays"),
        mpatches.Patch(color=BAY_HEX["parallel"], label="Parallel bays"),
        mpatches.Patch(color=HEX_LOT, edgecolor="black", label="Lot boundary"),
        Line2D([0], [0], color=HEX_PATROL_PATH, linestyle="--", linewidth=1.5, label="Patrol path"),
        mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
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
