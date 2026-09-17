"""
@file figure_style.py
@brief House style for every rendered figure: rcParams, palette and legends.

Single source of truth for figure appearance across scripts/. Appearance
only - never re-bins, re-orders, filters or converts units, so a restyled
figure plots exactly the numbers it plotted before.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# PNG everywhere, well above screen resolution so the raster survives printing.
EXT = ".png"
DPI = 400

# (key, label, colour, marker, linestyle). The key is the raw string in the
# CSVs; the label is what a reader sees. No raw key may reach a rendered
# legend, axis or tick. Marker and linestyle vary with colour so the figures
# stay readable in greyscale.
ARMS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("vanilla_ppo", "vanilla", "#4c72b0", "o", "-"),
    ("input_uncertainty", "input", "#dd8452", "s", "--"),
    ("output_uncertainty", "output", "#55a868", "^", "-."),
    ("full_method", "full", "#c44e52", "D", "-"),
)

ARM_ORDER: List[str] = [a[0] for a in ARMS]
ARM_LABEL: Dict[str, str] = {a[0]: a[1] for a in ARMS}
ARM_COLOUR: Dict[str, str] = {a[0]: a[2] for a in ARMS}
ARM_MARKER: Dict[str, str] = {a[0]: a[3] for a in ARMS}
ARM_LINESTYLE: Dict[str, str] = {a[0]: a[4] for a in ARMS}

# Two-line forms keep tick labels horizontal at the four-to-five categories the
# pooled figures carry. The held-tier pair appears only in per-run sweeps.
CONDITION_LABEL: Dict[str, str] = {
    "anchor_deployment": "anchor\ndeployment",
    "anchor_empty": "anchor\nempty",
    "gnss_degrade_one_way": "GNSS degraded\n(one way)",
    "lidar_degraded": "lidar\ndegraded",
    "ood_irregular_rtk_fixed": "OOD irregular\n(RTK fixed)",
    "gnss_fixed": "GNSS held\n(RTK fixed)",
    "gnss_degraded": "GNSS held\n(degraded)",
}

TIER_LABEL: Dict[str, str] = {
    "rtk_fixed": "RTK fixed",
    "rtk_float": "RTK float",
    "standalone": "standalone",
    "degraded": "degraded",
}

# Green -> amber -> orange -> red severity ramp. Adjacent tiers must stay
# distinguishable at TIER_ALPHA, so rtk_float is amber rather than a second
# green, which washes out against rtk_fixed at band weight.
TIER_BAND: Dict[str, str] = {
    "rtk_fixed": "#2e7d32",
    "rtk_float": "#f9a825",
    "standalone": "#ef6c00",
    "degraded": "#c62828",
}
TIER_ALPHA = 0.18

SIGNAL_LABEL: Dict[str, str] = {
    "ekf_std_pos_max_m": "max EKF position std",
    "max_epistemic": "max epistemic",
    "max_total": "max total predictive",
}

AXIS_LABEL: Dict[str, str] = {
    "std_pos": "predicted position std (m)",
    "std_yaw": "predicted heading std (rad)",
    "abs_err_pos": "absolute position error (m)",
    "abs_err_yaw": "absolute heading error (rad)",
}

ACCENT = "#c44e52"  # single highlighted series (binned mean, trend line)
ACCENT_ALT = "#4c72b0"  # second series on a twin axis
SCATTER = "#4c72b0"  # dense scatter clouds
TRACE = "#222222"  # a logged trace drawn over shaded bands
MUTED = "#666666"  # annotations, chance diagonals
RULE = "#cccccc"  # legend frames, reference rules

# Inches. Sized so text renders at body size for the width the figure is placed
# at; pick the one matching the intended placement.
SINGLE = (5.4, 3.2)  # one panel at 0.85 text width
WIDE = (7.2, 3.6)  # one panel at full text width
WIDE_TALL = (7.2, 4.4)  # one panel needing vertical room
GRID_2X2 = (7.2, 5.0)  # 2x2 panel grid
STACK_2 = (7.2, 6.0)  # 2 stacked panels
SIDE_2 = (7.2, 3.2)  # 2 side-by-side panels

LW = 1.3  # data line width
MS = 4.2  # marker size
GRID_ALPHA = 0.25
GRID_LW = 0.5

FS_LABEL = 8.5
FS_TICK = 7.5
FS_LEGEND = 8.5
FS_NOTE = 7.0


def apply() -> None:
    """
    @brief Install the shared rcParams. Call once, before creating any figure.

    The only place a font is chosen; no figure module may override it.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "axes.titlesize": FS_LABEL,
            "axes.labelsize": FS_LABEL,
            "xtick.labelsize": FS_TICK,
            "ytick.labelsize": FS_TICK,
            "legend.fontsize": FS_LEGEND,
            "text.usetex": False,
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _label(table: Dict[str, str], key: str) -> str:
    """
    @brief Look up a display label, falling back to the key itself.
    @param table: One of the *_LABEL maps.
    @param key: Raw key as it appears in the CSVs.
    @return The mapped label, or str(key) when unmapped.
    """
    return table.get(str(key), str(key))


def arm_label(arm: str) -> str:
    """@brief Display label for an arm key."""
    return _label(ARM_LABEL, arm)


def condition_label(condition: str) -> str:
    """@brief Display label for a condition key."""
    return _label(CONDITION_LABEL, condition)


def tier_label(tier: str) -> str:
    """@brief Display label for a GNSS fix-state tier key."""
    return _label(TIER_LABEL, tier)


def signal_label(signal: str) -> str:
    """@brief Display label for a safety-gate signal key."""
    return _label(SIGNAL_LABEL, signal)


def axis_label(column: str) -> str:
    """@brief Axis label for a raw dataframe column."""
    return _label(AXIS_LABEL, column)


def arm_kw(arm: str, *, marker: bool = True) -> Dict[str, Any]:
    """
    @brief Plot kwargs for an arm: colour, linestyle, marker and weights.
    @param arm: Raw arm key.
    @param marker: False for dense traces, where markers smear into a band.
    @return Keyword dict to splat into a Matplotlib plotting call.
    """
    key = str(arm)
    kw: Dict[str, Any] = {
        "color": ARM_COLOUR.get(key, MUTED),
        "linestyle": ARM_LINESTYLE.get(key, "-"),
        "lw": LW,
    }
    if marker:
        kw["marker"] = ARM_MARKER.get(key, "o")
        kw["ms"] = MS
    return kw


def arm_palette(arms: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """
    @brief Colour mapping for a seaborn `palette=` argument.
    @param arms: Arm keys to map; None uses ARM_ORDER.
    @return Mapping of arm key to hex colour.
    """
    keys = list(arms) if arms is not None else ARM_ORDER
    return {k: ARM_COLOUR.get(str(k), MUTED) for k in keys}


def grid(ax: Any, axis: str = "both") -> None:
    """
    @brief Apply the house grid, drawn behind the data.
    @param ax: Axes to grid.
    @param axis: "both", "x" or "y".
    """
    ax.grid(True, axis=axis, alpha=GRID_ALPHA, lw=GRID_LW, which="major")
    ax.set_axisbelow(True)


def condition_ticks(ax: Any, conditions: Sequence[str]) -> None:
    """
    @brief Relabel x ticks from condition keys to display labels, unrotated.
    @param ax: Axes to relabel.
    @param conditions: Condition keys; plotted order is preserved, never sorted.
    """
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [condition_label(c) for c in conditions], rotation=0, ha="center"
    )


def relabel_arm_legend(ax: Any) -> None:
    """
    @brief Strip a seaborn legend title and map raw arm keys to display labels.
    @param ax: Axes carrying the seaborn-generated legend.

    Seaborn writes the column name ("arm") as a heading and the raw keys as
    entries; neither is reader-facing.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title(None)
    for text in legend.get_texts():
        text.set_text(arm_label(text.get_text()))


def note(ax: Any, text: str, corner: str = "right", y: float = 0.94) -> Any:
    """
    @brief Small grey in-axes annotation, e.g. a panel's expected direction.
    @param ax: Axes to annotate.
    @param text: Annotation text.
    @param corner: "left" or "right".
    @param y: Vertical position in axes fraction.
    @return The created annotation.
    """
    x = 0.03 if corner == "left" else 0.97
    return ax.annotate(
        text,
        xy=(x, y),
        xycoords="axes fraction",
        ha="left" if corner == "left" else "right",
        va="top",
        fontsize=FS_NOTE,
        color=MUTED,
    )


# Exactly two legend formats, both OUTSIDE the plotting area, at most one per
# figure. Both read frame, font, padding and handle length from _LEGEND_KW so
# they cannot drift apart - do not pass styling overrides at the call site.
_LEGEND_KW: Dict[str, Any] = dict(
    fontsize=FS_LEGEND,
    frameon=True,
    framealpha=0.9,
    edgecolor=RULE,
    borderpad=0.5,
    handlelength=2.2,
    handletextpad=0.6,
    borderaxespad=0.0,
)

# corner -> (axes-fraction anchor, legend alignment).
_BLOCK_ANCHORS: Dict[str, Tuple[Tuple[float, float], str]] = {
    "upper right": ((1.0, 1.0), "upper left"),
    "lower right": ((1.0, 0.0), "lower left"),
    "upper left": ((0.0, 1.0), "upper right"),
    "lower left": ((0.0, 0.0), "lower right"),
}


def _handles(source: Any) -> Tuple[List[Any], List[str]]:
    """
    @brief Normalise a legend source to explicit handles and labels.
    @param source: An Axes to harvest from, or an explicit (handles, labels).
    @return Tuple of (handles, labels).
    """
    if isinstance(source, tuple):
        return source
    return source.get_legend_handles_labels()


def _drop_axes_legends(fig: Any) -> None:
    """
    @brief Remove any per-axes legend, so a figure never carries more than one.
    @param fig: Figure to strip.
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def legend_strip(
    fig: Any,
    source: Any,
    side: str = "below",
    ncol: Optional[int] = None,
    offset: float = 0.02,
) -> Any:
    """
    @brief One horizontal strip spanning the figure, outside the plot.
    @param fig: Figure to attach the legend to.
    @param source: An Axes to harvest handles from, or (handles, labels).
    @param side: "above" or "below" the plotting area.
    @param ncol: Column count; None puts every entry on one row, which is what
           makes it read as a strip. Set it only when one row would overrun.
    @param offset: Gap from the axes, in figure fraction.
    @return The created legend.
    @throws ValueError If side is not "above" or "below".
    """
    handles, labels = _handles(source)
    _drop_axes_legends(fig)
    if ncol is None:
        ncol = max(1, len(handles))
    if side == "above":
        loc, anchor = "lower center", (0.5, 1.0 + offset)
    elif side == "below":
        loc, anchor = "upper center", (0.5, -offset)
    else:
        raise ValueError(f"side must be 'above' or 'below', got {side!r}")
    return fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=anchor,
        columnspacing=1.6,
        **_LEGEND_KW,
    )


def legend_block(
    fig: Any, source: Any, corner: str = "upper right", offset: float = 0.015
) -> Any:
    """
    @brief One vertical block beside the plot, outside the axes.
    @param fig: Figure to attach the legend to.
    @param source: An Axes to harvest handles from, or (handles, labels).
    @param corner: A key of _BLOCK_ANCHORS.
    @param offset: Gap from the axes, in figure fraction.
    @return The created legend.
    @throws ValueError If corner is not one of the four supported names.
    """
    handles, labels = _handles(source)
    _drop_axes_legends(fig)
    if corner not in _BLOCK_ANCHORS:
        raise ValueError(
            f"corner must be one of {sorted(_BLOCK_ANCHORS)}, got {corner!r}"
        )
    (ax_x, ax_y), loc = _BLOCK_ANCHORS[corner]
    # Push outward along x from whichever side the block sits on.
    anchor = (ax_x + offset if ax_x >= 1.0 else ax_x - offset, ax_y)
    return fig.legend(
        handles, labels, loc=loc, ncol=1, bbox_to_anchor=anchor, **_LEGEND_KW
    )


def save(
    fig: Any,
    out: Any,
    *,
    pad: float = 0.4,
    rect: Optional[Tuple[float, float, float, float]] = None,
    close: bool = True,
) -> Path:
    """
    @brief Tight-layout and write the figure as PNG.
    @param fig: Figure to write.
    @param out: Destination path; any extension (or none) is normalised to PNG,
           so no caller can emit a stray PDF or a stale second format.
    @param pad: Tight-layout padding.
    @param rect: Optional tight-layout rect, reserving room for an outside legend.
    @param close: Close the figure after writing, freeing its memory.
    @return The path written.
    """
    path = Path(out).with_suffix(EXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout(pad=pad)
    else:
        fig.tight_layout(rect=rect, pad=pad)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"wrote {path}")
    return path
