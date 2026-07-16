"""
@file plot_ekf_sawtooth.py
@brief Generate methodology figure F-5: the reported EKF sigma_x sawtooth over one
       episode, with GNSS fix-state tiers shown as shaded background bands.

Reads one per-episode demo-trace CSV (written by demo_drive.py) and plots the
reported EKF position standard deviation sigma_x = `ekf_std_x` against time. sigma_x
is exactly the covariance feature the policy observes; its sawtooth (growth during
EKF prediction, collapse at each accepted GNSS fix) is the physical origin of
observation indices 2-4.

Tier bands: the GNSS fix-state tier is taken from the logged `gnss_tier` column,
which is the true Markov-chain state driving the injected noise (added to the trace
writer for exactly this figure). That is the faithful signal and is drawn as-is (no
smoothing): every state, including a genuine single-tick one, is real.

Fallback: for OLD traces written before `gnss_tier` existed, the tier is
reconstructed from the true localisation error ||ekf_xy - gt_xy|| banded at the tier
edges, and despeckled for legibility. This is NOISY near tier boundaries (true error
jitters across a threshold between corrections) and prints a warning; re-run
demo_drive.py to get a faithful figure. See trace_tier_breakdown.py for why the
reported std itself cannot label the tier (it saturates ~1.1 m).

The saturation annotation compares the peak reported sigma_x to the DESIGN noise
floor (Table T-7 / gnss_noise_profiles.yaml) of the worst tier the episode reached,
a stable per-tier constant, not a per-tick error sample.

Tier band palette is an ordered severity ramp (fixed=green ... degraded=red),
validated colourblind-safe on adjacent pairs (OKLab dE >= 8 for normal/deutan/protan);
tier is additionally encoded by ladder position and a legend, so colour is never the
sole channel.

@note Read-only w.r.t. the trace. Writes a single PDF. Pure CPU, no torch/CARLA.
Default output path targets the dissertation figures directory.

Usage:
    python scripts/miscellaneous/plot_ekf_sawtooth.py \
        --trace outputs/demo_traces/<baseline>/<leaf>/<stamp>/episode_<N>.csv \
        --out   ../Dissertation_WriteUp/content/chapters/3_methodology/figures/f5_ekf_sawtooth.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to write a PDF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GNSS callback / EKF rate. Step index -> seconds.
HZ = 20.0

# Tier order, best -> worst. Fixes legend order and the severity ramp.
_TIER_ORDER = ["rtk_fixed", "rtk_float", "standalone", "degraded"]

# Design noise floor (1-sigma position, m) of each tier, from
# configs/deployment/sim/gnss_noise_profiles.yaml (metric_stddev_m) and Table T-7.
# Used only for the saturation annotation (a stable per-tier constant), never to
# label the tier.
_TIER_FLOOR_M = {
    "rtk_fixed": 0.020,
    "rtk_float": 0.360,
    "standalone": 1.802,
    "degraded": 5.0,
}

# FALLBACK ONLY. Upper edges (inclusive) for reconstructing the tier from TRUE
# localisation error ||ekf_xy - gt_xy|| when the trace lacks a gnss_tier column.
# This reconstruction is noisy near the edges (true error jitters across a
# threshold between corrections) and is a last resort; the faithful signal is the
# logged gnss_tier column. Boundaries sit at the tier metric_stddev_m.
_TIER_EDGES: List[Tuple[str, float]] = [
    ("rtk_fixed", 0.36),
    ("rtk_float", 1.80),
    ("standalone", 5.0),
    ("degraded", float("inf")),
]

# Ordered severity ramp (fixed -> degraded). Doubles as a status palette; validated
# colourblind-safe on adjacent pairs. Kept light so the foreground line stays dominant.
_TIER_COLOUR = {
    "rtk_fixed": "#2e7d32",   # green   (best)
    "rtk_float": "#f9a825",   # amber
    "standalone": "#ef6c00",  # orange
    "degraded": "#c62828",    # red     (worst)
}
_TIER_LABEL = {
    "rtk_fixed": "rtk_fixed",
    "rtk_float": "rtk_float",
    "standalone": "standalone",
    "degraded": "degraded",
}

# Bands shorter than this are merged into the preceding tier (legibility of the
# background only; the sigma_x curve is never smoothed).
MIN_RUN_TICKS = 2


def _tier_of(true_err_m: float) -> str:
    """@brief Map a true localisation error (m) to its GNSS tier name."""
    for name, upper in _TIER_EDGES:
        if true_err_m <= upper:
            return name
    return _TIER_EDGES[-1][0]


def _tier_runs(tiers: np.ndarray, despeckle: bool) -> List[Tuple[str, int, int]]:
    """
    @brief Run-length encode a per-tick tier array.
    @param tiers: Per-tick tier-name array.
    @param despeckle: If True, merge runs shorter than MIN_RUN_TICKS into the
           preceding run (legibility only). Use for the noisy true-error FALLBACK.
           The logged gnss_tier is the real Markov state and must NOT be despeckled,
           so pass False for it: a genuine single-tick state is real and kept.
    @return List of (tier_name, start_idx, end_idx_exclusive) covering all ticks.
    """
    # Initial RLE.
    runs: List[Tuple[str, int, int]] = []
    start = 0
    for i in range(1, len(tiers) + 1):
        if i == len(tiers) or tiers[i] != tiers[start]:
            runs.append((str(tiers[start]), start, i))
            start = i

    if not despeckle:
        return runs

    # Merge any run shorter than MIN_RUN_TICKS into the preceding run, then
    # coalesce equal neighbours.
    merged: List[List] = []
    for name, a, b in runs:
        if (b - a) < MIN_RUN_TICKS and merged:
            merged[-1][2] = b  # extend previous run over the sliver
        elif merged and merged[-1][0] == name:
            merged[-1][2] = b
        else:
            merged.append([name, a, b])
    coalesced: List[Tuple[str, int, int]] = []
    for name, a, b in merged:
        if coalesced and coalesced[-1][0] == name:
            coalesced[-1] = (coalesced[-1][0], coalesced[-1][1], b)
        else:
            coalesced.append((name, a, b))
    return coalesced


def _resolve_tiers(df: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    @brief Get the per-tick tier, preferring the logged gnss_tier column.
    @return (tier_name_array, is_reconstructed). is_reconstructed is True only
            when gnss_tier is absent and the noisy true-error fallback was used.
    """
    if "gnss_tier" in df.columns and df["gnss_tier"].notna().any() and (
        df["gnss_tier"].astype(str).str.len() > 0
    ).any():
        tiers = df["gnss_tier"].astype(str).to_numpy()
        # Guard against an unexpected tier name so colour/label lookups never KeyError.
        unknown = sorted(set(tiers) - set(_TIER_ORDER))
        if unknown:
            raise ValueError(f"trace has unknown gnss_tier value(s): {unknown}")
        return tiers, False

    # Fallback: reconstruct from true localisation error (noisy near edges).
    true_err = np.sqrt(
        (df["ekf_x"] - df["gt_x"]) ** 2 + (df["ekf_y"] - df["gt_y"]) ** 2
    ).to_numpy(dtype=float)
    tiers = np.array([_tier_of(v) for v in true_err])
    print(
        "WARNING: trace has no gnss_tier column; tier bands RECONSTRUCTED from "
        "true error and are noisy near tier boundaries. Re-run demo_drive.py "
        "(now logs gnss_tier) for a faithful figure."
    )
    return tiers, True


def _style() -> None:
    """@brief Restrained, print-friendly rcParams; serif to sit beside LaTeX body text."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
        }
    )


def plot(trace_csv: Path, out_pdf: Path) -> None:
    """@brief Render F-5 from one episode trace to a vector PDF."""
    df = pd.read_csv(trace_csv)
    required = {"ekf_std_x", "ekf_x", "ekf_y", "gt_x", "gt_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{trace_csv} missing columns: {sorted(missing)}")

    n = len(df)
    if n == 0:
        raise ValueError(f"{trace_csv} has no rows.")

    t = np.arange(n) / HZ  # seconds
    sigma_x = df["ekf_std_x"].to_numpy(dtype=float)

    tiers, reconstructed = _resolve_tiers(df)
    # The logged tier is the real Markov state: keep every run, even 1-tick ones.
    # Despeckle only the noisy true-error fallback, for band legibility.
    runs = _tier_runs(tiers, despeckle=reconstructed)
    worst = min(  # worst (highest-index) tier the episode actually visited
        (name for name, _, _ in runs),
        key=lambda nm: -_TIER_ORDER.index(nm),
        default="rtk_fixed",
    )

    _style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))

    # Shaded tier bands behind the curve. Half-tick padding closes seams.
    half = 0.5 / HZ
    seen = set()
    for name, a, b in runs:
        x0 = t[a] - half
        x1 = t[b - 1] + half
        ax.axvspan(
            x0,
            x1,
            color=_TIER_COLOUR[name],
            alpha=0.16,
            linewidth=0,
            zorder=0,
            label=_TIER_LABEL[name] if name not in seen else None,
        )
        seen.add(name)

    # Reported sigma_x sawtooth (raw, untouched).
    ax.plot(
        t,
        sigma_x,
        color="#1a1a1a",
        linewidth=1.3,
        zorder=3,
        solid_capstyle="round",
        label="reported $\\sigma_x$",
    )

    ax.set_xlabel("time (s)")
    ax.set_ylabel("reported EKF $\\sigma_x$ (m)")
    ax.set_xlim(t[0] - half, t[-1] + half)
    ax.set_ymargin(0.0)
    ax.set_ylim(0.0, float(np.nanmax(sigma_x)) * 1.20)

    # Callout. The y-axis already shows the plateau height, so the label only
    # states the defensible FACT: reported sigma_x stays near 1 m in the worst
    # tier, below the true localisation error it experiences there. Pooled over
    # the whole run's degraded ticks, reported sigma_x is median ~0.9 m (p90
    # ~1.1, max ~1.2) while the true error is median ~1.4 m (p90 ~5 m), so it is
    # consistently the smaller of the two. Deliberately NOT phrased as "below the
    # raw 5 m injection floor" (that invites "the filter is fusing correctly, as
    # intended") and NOT quoting a per-episode peak (sigma_x reaches ~1.24 m
    # elsewhere, so a single peak would misrepresent the ceiling).
    peak_i = int(np.nanargmax(sigma_x))
    peak_sigma = float(sigma_x[peak_i])
    # Anchor the leader on the plateau, and drop the text into the empty band
    # below it. Bias both toward mid-plateau (0.45 of the span) so the box clears
    # the sharp recovery drop near the end of the worst-tier dwell.
    mid_t = t[0] + 0.45 * (t[-1] - t[0])
    mid_i = int(np.argmin(np.abs(t - mid_t)))
    anchor_t = t[mid_i]
    anchor_sigma = float(sigma_x[mid_i])  # land the leader on the plateau curve
    text_t = anchor_t
    text_y = peak_sigma * 0.55
    ax.annotate(
        f"reported $\\sigma_x$ stays near 1 m in {_TIER_LABEL[worst]},\n"
        "below the true error it experiences",
        xy=(anchor_t, anchor_sigma),
        xytext=(text_t, text_y),
        fontsize=8,
        color="#333333",
        va="center",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc", lw=0.6,
                  alpha=0.9),
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
        zorder=5,
    )

    # Order the legend as line first, then the tier ladder. Only the tiers that
    # actually appear are in it (axvspan labelled them on use). Placed BELOW the
    # axes as a single horizontal row so it never covers the trace.
    handles, labels = ax.get_legend_handles_labels()
    order_key = {"reported $\\sigma_x$": -1}
    for i, name in enumerate(_TIER_ORDER):
        order_key[_TIER_LABEL[name]] = i
    pairs = sorted(zip(handles, labels), key=lambda hl: order_key.get(hl[1], 99))
    handles, labels = zip(*pairs)
    leg = ax.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    src = "reconstructed from true error" if reconstructed else "logged gnss_tier"
    print(f"wrote {out_pdf}  ({n} ticks, {n / HZ:.2f} s, "
          f"sigma_x max {np.nanmax(sigma_x):.3f} m, tier source: {src}, "
          f"worst tier: {worst})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[2]
    default_trace = (
        repo
        / "outputs/demo_traces/full_method/6_42_22062026-1502"
        / "15-07-2026-143405/episode_107.csv"
    )
    default_out = (
        repo.parent
        / "Dissertation_WriteUp/content/chapters/3_methodology/figures/f5_ekf_sawtooth.pdf"
    )
    p.add_argument("--trace", type=Path, default=default_trace,
                   help="Per-episode demo-trace CSV.")
    p.add_argument("--out", type=Path, default=default_out,
                   help="Output PDF path.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plot(args.trace, args.out)


if __name__ == "__main__":
    main()
