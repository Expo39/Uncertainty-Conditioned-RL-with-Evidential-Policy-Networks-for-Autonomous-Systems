"""
@file common.py
@brief World-frame transform, YAML serialisation, and PNG plotting for
       parking lot layouts.

The engine side of layout generation: turns a layout dict (from
LotBuilder.build()) into a CARLA-frame YAML file plus a bird's-eye PNG.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Motorcycle bays render in a neutral grey rather than their palette hue so the
# always-empty bays read as distinct from drivable perpendicular/angled bays.
MOTORCYCLE_DRAW_HEX = "#888888"

# Bay-type legend entries in stable display order. Only the types actually
# present in a layout are emitted (see plot_layout), so a lot with no angled or
# motorcycle bays never lists them.
_BAY_LEGEND_ORDER: List[Tuple[str, str]] = [
    ("perpendicular", "Perpendicular bays"),
    ("angled", "Angled (45 deg) bays"),
    ("parallel", "Parallel bays"),
    ("motorcycle", "Motorcycle bays"),
]


def _bay_legend_colour(bay_type: str) -> str:
    """
    @brief Resolve the plot colour for a bay type (drawing and legend share this).
    @param bay_type: Bay type key (e.g. "perpendicular", "motorcycle").
    @return Hex colour string; motorcycle bays use the neutral draw grey.
    """
    from scripts.colours import BAY_HEX

    if bay_type == "motorcycle":
        return MOTORCYCLE_DRAW_HEX
    return BAY_HEX.get(bay_type, "grey")


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

    world_ped_zones = [_xform_box(zone) for zone in local_layout["pedestrian_zones"]]

    world_obstacles = [_xform_box(obs) for obs in local_layout.get("obstacles", [])]

    return {
        "corners": world_corners,
        "bays": world_bays,
        "spawn_transform": _xform_spawn(local_layout["spawn"]),
        "extra_spawn_transforms": world_extra_spawns,
        "patrol_waypoints": world_patrol,
        "pedestrian_zones": world_ped_zones,
        "obstacles": world_obstacles,
    }


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


def plot_layout(
    shape: str,
    world_layout: Dict[str, Any],
    plot_path: Path,
    legend_loc: str = "upper right",
    show_patrol: bool = True,
    show_pedestrians: bool = True,
    show_extra_spawns: bool = True,
    oob_inflation_margin: Optional[float] = None,
) -> None:
    """
    @brief Render a bird's-eye PNG of the lot layout.

    Layers (back to front): grey lot polygon, bay rectangles (colour-coded by
    type), yaw arrows, spawn triangles, patrol path, pedestrian zone overlays,
    and an optional soft out-of-bounds boundary. The patrol path, pedestrian
    zones, and extra spawns are drawn only when the corresponding flags are set,
    so the PNG reflects the environment the agent actually trains in. The primary
    spawn is always drawn.

    @param shape: Floor plan shape name for title.
    @param world_layout: World-frame layout dict.
    @param plot_path: Path to save the PNG.
    @param legend_loc: Legend location (default: 'upper right').
    @param show_patrol: Draw the patrol path and its legend entry.
    @param show_pedestrians: Draw the pedestrian zones and their legend entry.
    @param show_extra_spawns: Draw the extra (non-primary) spawn triangles. The
           primary spawn is always drawn regardless of this flag.
    @param oob_inflation_margin: When not None, draw the lot polygon inflated by
           this many metres as a dashed soft out-of-bounds boundary.
    """
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Polygon as MPoly
        from matplotlib.patheffects import withStroke

        from scripts import figure_style as fs
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot.")
        return

    from scripts.colours import (
        HEX_LOT,
        HEX_OOB_BOUNDARY,
        HEX_PATROL_PATH,
        HEX_PEDESTRIAN_ZONE,
        HEX_PEDESTRIAN_ZONE_EDGE,
    )
    from uncertainty_rl.utils.geometry import inflate_polygon

    fs.apply()
    fig, ax = plt.subplots(figsize=fs.WIDE_TALL)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
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

    # Soft out-of-bounds boundary: the lot polygon inflated outward by the margin.
    if oob_inflation_margin is not None:
        oob_pts = inflate_polygon(corner_pts, oob_inflation_margin)
        ax.add_patch(
            MPoly(
                oob_pts,
                closed=True,
                facecolor="none",
                edgecolor=HEX_OOB_BOUNDARY,
                linewidth=1.5,
                linestyle="--",
            )
        )

    for bay in world_layout["bays"]:
        bay_type = bay["bay_type"]
        colour = _bay_legend_colour(bay_type)
        bx, by = bay["x"], bay["y"]
        yaw_rad = math.radians(bay["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        hw, hd = bay["width"] / 2.0, bay["depth"] / 2.0
        local_corners = [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
        world_rect = [
            (bx + cos_y * lx - sin_y * ly, by + sin_y * lx + cos_y * ly)
            for lx, ly in local_corners
        ]
        ax.add_patch(
            MPoly(
                world_rect,
                closed=True,
                facecolor=colour,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.6,
                zorder=3,
            )
        )
        # Add bay ID label at the centre of the bay
        bay_id = bay.get("id", "").split("_")[-1]  # Extract just the number
        ax.text(
            bx,
            by,
            bay_id,
            ha="center",
            va="center",
            fontsize=fs.FS_NOTE,
            color="black",
            weight="bold",
            path_effects=[withStroke(linewidth=1, foreground="white")],
            zorder=4,
        )

    if show_pedestrians:
        for zone in world_layout.get("pedestrian_zones", []):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (
                        zone["centre_x"] - zone["half_width"],
                        zone["centre_y"] - zone["half_height"],
                    ),
                    zone["half_width"] * 2.0,
                    zone["half_height"] * 2.0,
                    boxstyle="round,pad=0.4",
                    facecolor=HEX_PEDESTRIAN_ZONE,
                    edgecolor=HEX_PEDESTRIAN_ZONE_EDGE,
                    alpha=0.40,
                    linewidth=1.5,
                    linestyle="--",
                    zorder=4,
                )
            )

    for obs in world_layout.get("obstacles", []):
        ax.add_patch(
            mpatches.Rectangle(
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
        )

    stroke_effect = [withStroke(linewidth=2, foreground="black")]
    tri_local = [(1.2, 0.0), (-0.72, 0.72), (-0.72, -0.72)]
    # The primary spawn is always drawn; extras only when show_extra_spawns is set.
    spawns = [world_layout["spawn_transform"]]
    if show_extra_spawns:
        spawns += world_layout.get("extra_spawn_transforms", [])
    for idx, sp in enumerate(spawns):
        yaw_rad = math.radians(sp["yaw_deg"])
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        tri_world = [
            (sp["x"] + cos_y * lx - sin_y * ly, sp["y"] + sin_y * lx + cos_y * ly)
            for lx, ly in tri_local
        ]
        ax.add_patch(
            MPoly(
                tri_world,
                closed=True,
                facecolor="cyan",
                edgecolor="white",
                linewidth=1,
                zorder=6,
            )
        )
        label_offset_perp = 1.5 if idx > 0 else 1.9
        label_y_nudge = 2.0 if idx > 0 else -0.5
        ax.text(
            sp["x"] + cos_y * 0.2 - sin_y * label_offset_perp,
            sp["y"] + sin_y * 0.2 + cos_y * label_offset_perp + label_y_nudge,
            f"SPAWN {idx + 1}",
            fontsize=fs.FS_NOTE,
            color="cyan",
            fontweight="bold",
            zorder=7,
            path_effects=stroke_effect,
        )

    patrol = world_layout["patrol_waypoints"]
    if show_patrol and patrol:
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

    # Build the legend from only the elements that were actually drawn, so it
    # never lists a bay type, patrol path, or pedestrian zone absent from this
    # layout. Bay-type entries follow _BAY_LEGEND_ORDER for a stable order.
    present_types = {bay["bay_type"] for bay in world_layout["bays"]}
    handles: List[Any] = [
        mpatches.Patch(color=_bay_legend_colour(bay_type), label=label)
        for bay_type, label in _BAY_LEGEND_ORDER
        if bay_type in present_types
    ]
    handles.append(
        mpatches.Patch(color=HEX_LOT, edgecolor="black", label="Lot boundary")
    )
    if oob_inflation_margin is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=HEX_OOB_BOUNDARY,
                linestyle="--",
                linewidth=1.5,
                label=f"OOB boundary (lot + {oob_inflation_margin:g} m)",
            )
        )
    if show_patrol:
        handles.append(
            Line2D(
                [0],
                [0],
                color=HEX_PATROL_PATH,
                linestyle="--",
                linewidth=1.5,
                label="Patrol path",
            )
        )
    if show_pedestrians:
        handles.append(
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
            )
        )
    fs.legend_strip(
        fig, (handles, [h.get_label() for h in handles]), side="below", ncol=2
    )
    fs.grid(ax)

    # Pin the data limits to the lot extent: legend handles such as the
    # pedestrian-zone FancyBboxPatch carry a data-space footprint at (0, 0),
    # and without explicit limits bbox_inches="tight" expands the figure to
    # enclose it, producing a runaway multi-gigapixel PNG.
    pad = 5.0
    # Clears the soft OOB skirt (extends oob_inflation_margin beyond the lot)
    # with headroom, so the dashed boundary is never drawn at the axis edge.
    if oob_inflation_margin is not None:
        pad = oob_inflation_margin + 5.0
    xs = [p[0] for p in corner_pts]
    ys = [p[1] for p in corner_pts]
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)

    fs.save(fig, plot_path)
    print(f"  Plot:    {plot_path}")
