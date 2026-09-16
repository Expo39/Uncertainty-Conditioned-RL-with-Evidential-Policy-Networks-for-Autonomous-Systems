"""
@file lot_layouts.py
@brief Lot layout figure: both lots, one shared legend.

Reads the generated layout YAMLs (already world-frame) and draws them as two
panels of a single figure. Presentation only: no geometry is recomputed, so
bays, spawns and lot corners are exactly what generate_layouts.py wrote.

Both lots are drawn as one \\textwidth figure rather than two subfigures at
0.48\\linewidth each: a narrower subfigure would scale fs.FS_LEGEND (8.5 pt)
down to an illegible size on the page and would need a legend per panel,
against the figure_style rule that a figure carries at most one.

Both panels share one data extent (the larger lot's padded span, centred on
each lot), so the two outlines print at the same metres-per-inch and occupy
the same area. sharey keeps one y axis for the pair.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon as MPoly  # noqa: E402

from scripts import figure_style as fs  # noqa: E402
from scripts.colours import HEX_LOT, HEX_OOB_BOUNDARY  # noqa: E402
from scripts.layouts.common import _BAY_LEGEND_ORDER, _bay_legend_colour  # noqa: E402
from uncertainty_rl.utils.constants import OOB_INFLATION_MARGIN  # noqa: E402
from uncertainty_rl.utils.geometry import inflate_polygon  # noqa: E402

CONFIGS = Path("configs/layouts")
OUT = Path("outputs/raw_derived/figures")

LAYOUTS = ("rectangle", "irregular_a")
PANEL_TITLES = {"rectangle": "(a) rectangle", "irregular_a": "(b) irregular_a"}

# Per-side padding beyond the OOB skirt. plot_layout uses 5.0 m of headroom for
# on-screen inspection; 1.0 m keeps the dashed boundary off the axis spine in
# print and reclaims 8 m of dead margin per side.
PAD_HEADROOM = 1.0

# Full text width: the figure is included at \linewidth, so house font sizes
# land at their intended print size.
FIG_W = 7.2


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def _shared_span(layouts: dict, oob_margin: float) -> tuple[float, float]:
    """The larger padded span in each axis, so both panels share one scale."""
    pad = 2.0 * (oob_margin + PAD_HEADROOM)
    widths, heights = [], []
    for layout in layouts.values():
        xs = [c["x"] for c in layout["corners"]]
        ys = [c["y"] for c in layout["corners"]]
        widths.append(max(xs) - min(xs) + pad)
        heights.append(max(ys) - min(ys) + pad)
    return max(widths), max(heights)


def _draw_panel(ax, layout: dict, oob_margin: float, span: tuple[float, float]) -> None:
    corner_pts = [(c["x"], c["y"]) for c in layout["corners"]]
    xs = [p[0] for p in corner_pts]
    ys = [p[1] for p in corner_pts]

    ax.add_patch(
        MPoly(
            corner_pts, closed=True, facecolor="#DDDDDD", edgecolor="none", linewidth=0
        )
    )
    ax.add_patch(
        MPoly(
            corner_pts,
            closed=True,
            facecolor="none",
            edgecolor="black",
            linewidth=2,
            capstyle="butt",
        )
    )
    ax.add_patch(
        MPoly(
            inflate_polygon(corner_pts, oob_margin),
            closed=True,
            facecolor="none",
            edgecolor=HEX_OOB_BOUNDARY,
            linewidth=1.5,
            linestyle="--",
        )
    )

    for bay in layout["bays"]:
        bx, by = bay["x"], bay["y"]
        yaw = math.radians(bay["yaw_deg"])
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        hw, hd = bay["width"] / 2.0, bay["depth"] / 2.0
        rect = [
            (bx + cos_y * lx - sin_y * ly, by + sin_y * lx + cos_y * ly)
            for lx, ly in [(-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw)]
        ]
        # Bay indices are deliberately not drawn: 47 two-digit labels across
        # rows roughly 2.5 m wide cannot print legibly at this scale, and no
        # reported passage refers to a bay by number. The bay-type
        # colours in the legend carry everything a reader needs here.
        ax.add_patch(
            MPoly(
                rect,
                closed=True,
                facecolor=_bay_legend_colour(bay["bay_type"]),
                edgecolor="white",
                linewidth=1.5,
                alpha=0.6,
                zorder=3,
            )
        )

    # Primary spawn only: use_extra_spawns is False in the sim env config.
    sp = layout["spawn_transform"]
    yaw = math.radians(sp["yaw_deg"])
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    tri = [
        (sp["x"] + cos_y * lx - sin_y * ly, sp["y"] + sin_y * lx + cos_y * ly)
        for lx, ly in [(1.2, 0.0), (-0.72, 0.72), (-0.72, -0.72)]
    ]
    # The marker is named in the shared legend rather than labelled in place:
    # one spawn per lot means an in-plot label repeats the same word twice and
    # competes with the geometry for space.
    ax.add_patch(
        MPoly(
            tri,
            closed=True,
            facecolor="cyan",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
    )

    # One extent for both panels, centred on this lot: equal scale, equal area.
    span_w, span_h = span
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    ax.set_xlim(cx - span_w / 2.0, cx + span_w / 2.0)
    # Inverted y: CARLA's frame is left-handed, so north reads as up.
    ax.set_ylim(cy + span_h / 2.0, cy - span_h / 2.0)
    ax.set_aspect("equal", adjustable="box")
    fs.grid(ax)


def render(out) -> Path:
    """Draw both lots as one figure. `out` is a path without extension."""
    layouts = {name: _load(name) for name in LAYOUTS}
    span = _shared_span(layouts, OOB_INFLATION_MARGIN)
    print(f"shared extent: {span[0]:.1f} x {span[1]:.1f} m")

    fs.apply()
    fig_h = FIG_W / 2.0 * (span[1] / span[0]) + 0.9
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, fig_h), sharey=True)

    for ax, name in zip(axes, LAYOUTS):
        _draw_panel(ax, layouts[name], OOB_INFLATION_MARGIN, span)
        ax.set_xlabel("x (m)")
        ax.set_title(PANEL_TITLES[name], fontsize=fs.FS_LABEL)
    axes[0].set_ylabel("y (m)")

    # One legend for the pair: union of the bay types present in either lot,
    # so a type unique to one panel is still named.
    present = {b["bay_type"] for lay in layouts.values() for b in lay["bays"]}
    handles = [
        mpatches.Patch(facecolor=_bay_legend_colour(t), label=lab)
        for t, lab in _BAY_LEGEND_ORDER
        if t in present
    ]
    handles.append(
        mpatches.Patch(facecolor=HEX_LOT, edgecolor="black", label="Lot boundary")
    )
    # Legend key for the spawn triangle: the arrow points along the spawn
    # heading, so the entry names both the position and orientation it encodes.
    handles.append(
        Line2D(
            [0],
            [0],
            marker=">",
            color="none",
            markerfacecolor="cyan",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7,
            label="Spawn pose (arrow: heading)",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=HEX_OOB_BOUNDARY,
            linestyle="--",
            linewidth=1.5,
            label=f"OOB boundary (lot + {OOB_INFLATION_MARGIN:g} m)",
        )
    )
    fs.legend_strip(fig, (handles, [h.get_label() for h in handles]), side="below")

    return fs.save(fig, out)


if __name__ == "__main__":
    render(OUT / "lot_layouts")
