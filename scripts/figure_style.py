"""
@file figure_style.py
@brief Single source of truth for how every dissertation data plot looks.

Every figure that ends up in the write-up imports this module. Nothing else
sets rcParams, picks a colour, or builds a legend by hand. That is the whole
point: the figures previously carried four different palettes and two
different fonts between them, which is what made the chapter look inconsistent.

This module governs APPEARANCE ONLY. It never touches the data: no
re-binning, no re-ordering, no filtering, no unit conversion. A figure
restyled through here plots exactly the numbers it plotted before, so the
prose and tables that quote those numbers stay correct.

Usage:

    from figure_style import style as fs

    fs.apply()
    fig, ax = plt.subplots(figsize=fs.WIDE)
    for arm in fs.ARM_ORDER:
        ax.plot(x, y[arm], label=fs.arm_label(arm), **fs.arm_kw(arm))
    fs.grid(ax)
    fs.legend_strip(fig, ax, side="below")
    fs.save(fig, out_dir / "f13_ablation_by_condition")

Legend rule. There are exactly TWO legend formats and both sit OUTSIDE the
plotting area. No figure puts a legend inside its axes.

    STRIP (legend_strip) -- one horizontal row spanning the figure width,
                            placed above or below the plot.
    BLOCK (legend_block) -- one vertical column beside the plot, placed at
                            any of the four corners.

Placement may vary to avoid colliding with the traces; nothing else about the
two may vary. Both draw their frame, font, padding and handle length from the
one _LEGEND_KW dict below, so they cannot drift apart. Do not pass styling
overrides at the call site.

Output is PNG for every figure, at print-grade dpi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Output ----------------------------------------------------------------
# Every data figure is a PNG. DPI is well above screen resolution so the raster
# still holds up when an examiner zooms into the printed page.
EXT = ".png"
DPI = 400

# --- Arm identity ----------------------------------------------------------
# (key, display label, colour, marker, linestyle). The key is the raw string
# used in the CSVs and dataframes; the display label is what a reader sees. No
# raw key may ever reach a rendered legend, axis or tick.
#
# Colours are the seaborn-deep set. Marker and linestyle vary alongside colour
# so the figures stay readable in greyscale.
ARMS: tuple[tuple[str, str, str, str, str], ...] = (
    ("vanilla_ppo", "vanilla", "#4c72b0", "o", "-"),
    ("input_uncertainty", "input", "#dd8452", "s", "--"),
    ("output_uncertainty", "output", "#55a868", "^", "-."),
    ("full_method", "full", "#c44e52", "D", "-"),
)

ARM_ORDER: list[str] = [a[0] for a in ARMS]
ARM_LABEL: dict[str, str] = {a[0]: a[1] for a in ARMS}
ARM_COLOUR: dict[str, str] = {a[0]: a[2] for a in ARMS}
ARM_MARKER: dict[str, str] = {a[0]: a[3] for a in ARMS}
ARM_LINESTYLE: dict[str, str] = {a[0]: a[4] for a in ARMS}

# --- Condition and tier identity -------------------------------------------
# Raw scenario keys mapped to the prose wording used in the chapter. The
# two-line forms keep x tick labels horizontal at full text width instead of
# rotating them 45 degrees, which was making several figures hard to read.
CONDITION_LABEL: dict[str, str] = {
    "anchor_deployment": "anchor\ndeployment",
    "anchor_empty": "anchor\nempty",
    "gnss_degrade_one_way": "GNSS degraded\n(one way)",
    "lidar_degraded": "lidar\ndegraded",
    "ood_irregular_rtk_fixed": "OOD irregular\n(RTK fixed)",
    # The held-tier pair. Pooled figures drop them (each pins one fix state all
    # episode, so nothing degrades within an episode), but a per-run sweep
    # covers every eval_config condition and still needs to name them.
    "gnss_fixed": "GNSS held\n(RTK fixed)",
    "gnss_degraded": "GNSS held\n(degraded)",
}

# GNSS fix-state tiers, for the sawtooth trace and the covariance probe.
TIER_LABEL: dict[str, str] = {
    "rtk_fixed": "RTK fixed",
    "rtk_float": "RTK float",
    "standalone": "standalone",
    "degraded": "degraded",
}

# Background shading for fix-state tiers, ordered clean to degraded. These form
# a green -> amber -> orange -> red severity ramp: adjacent tiers must stay
# distinguishable at TIER_ALPHA, so rtk_float is amber rather than a second
# green, which washed out against rtk_fixed at band weight.
TIER_BAND: dict[str, str] = {
    "rtk_fixed": "#2e7d32",
    "rtk_float": "#f9a825",
    "standalone": "#ef6c00",
    "degraded": "#c62828",
}
TIER_ALPHA = 0.18

# Signals compared in the gate ROC, and the axis quantities elsewhere.
SIGNAL_LABEL: dict[str, str] = {
    "ekf_std_pos_max_m": "max EKF position std",
    "max_epistemic": "max epistemic",
    "max_total": "max total predictive",
}

AXIS_LABEL: dict[str, str] = {
    "std_pos": "predicted position std (m)",
    "std_yaw": "predicted heading std (rad)",
    "abs_err_pos": "absolute position error (m)",
    "abs_err_yaw": "absolute heading error (rad)",
}

# --- Non-arm accents -------------------------------------------------------
ACCENT = "#c44e52"  # single highlighted series (binned mean, trend line)
ACCENT_ALT = "#4c72b0"  # second series on a twin axis
SCATTER = "#4c72b0"  # dense scatter clouds
TRACE = "#222222"  # a logged trace drawn over shaded bands
MUTED = "#666666"  # annotations, chance diagonals
RULE = "#cccccc"  # legend frames, reference rules

# --- Geometry --------------------------------------------------------------
# Widths in inches, chosen so text renders at the document's body size for the
# \includegraphics width each figure is used at. Pick the one that matches.
SINGLE = (5.4, 3.2)  # one panel at 0.85\textwidth
WIDE = (7.2, 3.6)  # one panel at \textwidth
WIDE_TALL = (7.2, 4.4)  # one panel at \textwidth needing vertical room
GRID_2X2 = (7.2, 5.0)  # 2x2 panel grid at \textwidth
STACK_2 = (7.2, 6.0)  # 2 stacked panels at \textwidth
SIDE_2 = (7.2, 3.2)  # 2 side-by-side panels at \textwidth

LW = 1.3  # data line width
MS = 4.2  # marker size
GRID_ALPHA = 0.25
GRID_LW = 0.5

FS_LABEL = 8.5
FS_TICK = 7.5
FS_LEGEND = 8.5
FS_NOTE = 7.0


def apply() -> None:
    """Install the shared rcParams. Call once, before creating any figure.

    Serif throughout, matching the document body font. This is the only place
    a font is chosen; no script may override it.
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


# --- Label helpers ---------------------------------------------------------
def arm_label(arm: str) -> str:
    """Display label for an arm key. Falls back to the key if unmapped."""
    return ARM_LABEL.get(str(arm), str(arm))


def condition_label(condition: str) -> str:
    """Prose label for a condition key."""
    return CONDITION_LABEL.get(str(condition), str(condition))


def tier_label(tier: str) -> str:
    """Prose label for a GNSS fix-state tier key."""
    return TIER_LABEL.get(str(tier), str(tier))


def signal_label(signal: str) -> str:
    """Prose label for a gate signal key."""
    return SIGNAL_LABEL.get(str(signal), str(signal))


def axis_label(column: str) -> str:
    """Prose axis label for a raw dataframe column."""
    return AXIS_LABEL.get(str(column), str(column))


def arm_kw(arm: str, *, marker: bool = True) -> dict:
    """Plot kwargs for an arm: colour, marker, linestyle and weights.

    Pass marker=False for dense traces where markers would smear into a band.
    """
    key = str(arm)
    kw = {
        "color": ARM_COLOUR.get(key, MUTED),
        "linestyle": ARM_LINESTYLE.get(key, "-"),
        "lw": LW,
    }
    if marker:
        kw["marker"] = ARM_MARKER.get(key, "o")
        kw["ms"] = MS
    return kw


def arm_palette(arms: Iterable[str] | None = None) -> dict[str, str]:
    """Colour mapping for seaborn's `palette=` argument."""
    keys = list(arms) if arms is not None else ARM_ORDER
    return {k: ARM_COLOUR.get(str(k), MUTED) for k in keys}


# --- Axes furniture --------------------------------------------------------
def grid(ax, axis: str = "both") -> None:
    """Apply the house grid, drawn behind the data."""
    ax.grid(True, axis=axis, alpha=GRID_ALPHA, lw=GRID_LW, which="major")
    ax.set_axisbelow(True)


def condition_ticks(ax, conditions: Sequence[str]) -> None:
    """Relabel x ticks from raw condition keys to prose, unrotated.

    Order is preserved exactly as given: this is presentation only and never
    re-sorts the underlying categories.
    """
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [condition_label(c) for c in conditions], rotation=0, ha="center"
    )


def relabel_arm_legend(ax) -> None:
    """Strip a seaborn legend title and map raw arm keys to display labels.

    Seaborn writes the dataframe column name ("arm") as a legend heading and
    the raw keys as entries; both are wrong for the write-up.
    """
    legend = ax.get_legend()
    if legend is None:
        return
    legend.set_title(None)
    for text in legend.get_texts():
        text.set_text(arm_label(text.get_text()))


def note(ax, text: str, corner: str = "right", y: float = 0.94):
    """Small grey in-axes annotation, e.g. a panel's expected direction."""
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


# --- Legends: exactly two formats, both outside the plot area --------------
_LEGEND_KW = dict(
    fontsize=FS_LEGEND,
    frameon=True,
    framealpha=0.9,
    edgecolor=RULE,
    borderpad=0.5,
    handlelength=2.2,
    handletextpad=0.6,
    borderaxespad=0.0,
)


def _handles(source):
    """Accept an axes to harvest handles from, or an explicit (handles, labels)."""
    if isinstance(source, tuple):
        return source
    return source.get_legend_handles_labels()


def _drop_axes_legends(fig) -> None:
    """Remove any per-axes legend, so a figure never carries more than one."""
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def legend_strip(
    fig, source, side: str = "below", ncol: int | None = None, offset: float = 0.02
):
    """Format 1: one horizontal strip spanning the figure, outside the plot.

    `side` is "above" or "below". `ncol` defaults to a single row holding
    every entry, which is what makes it read as a strip; pass a value only
    when one row would overrun the figure width.
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


def legend_block(fig, source, corner: str = "upper right", offset: float = 0.015):
    """Format 2: one vertical block beside the plot, outside the axes.

    `corner` is "upper right", "lower right", "upper left" or "lower left".
    """
    handles, labels = _handles(source)
    _drop_axes_legends(fig)
    anchors = {
        "upper right": ((1.0 + offset, 1.0), "upper left"),
        "lower right": ((1.0 + offset, 0.0), "lower left"),
        "upper left": ((0.0 - offset, 1.0), "upper right"),
        "lower left": ((0.0 - offset, 0.0), "lower right"),
    }
    if corner not in anchors:
        raise ValueError(f"corner must be one of {sorted(anchors)}, got {corner!r}")
    anchor, loc = anchors[corner]
    return fig.legend(
        handles, labels, loc=loc, ncol=1, bbox_to_anchor=anchor, **_LEGEND_KW
    )


# --- Output ----------------------------------------------------------------
def save(fig, out, *, pad: float = 0.4, rect=None, close: bool = True) -> Path:
    """Tight-layout and write the figure as PNG.

    `out` may carry any extension or none; it is normalised to PNG so no
    caller can emit a stray PDF or a stale second format.
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
