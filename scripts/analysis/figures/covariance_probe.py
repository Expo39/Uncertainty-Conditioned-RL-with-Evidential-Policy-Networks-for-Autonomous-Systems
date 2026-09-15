"""
@file covariance_probe.py
@brief Causal covariance probe: action response to a swept covariance.

Plots the deterministic-action delta and the head's epistemic estimate as the
covariance block alone is swept from the RTK-fixed level to the degraded level,
every other observation feature held fixed, for all three training seeds.

Data source: `scripts/analysis/covariance_probe.py` (synthetic single-state
mode) run per seed on the stage-6 full_method checkpoints, 31-07-2026:
  seed 42  6_42_22062026-1502
  seed 123 6_123_01072026-0404
  seed 7   6_7_05072026-1952
Only full_method can be probed: the covariance-blind arms (vanilla_ppo,
output_uncertainty) carry no covariance block to sweep (obs_dim 10 vs 13), and
input_uncertainty has the block but a standard head, which reports no epistemic
estimate. The sweep is therefore undefined for all three rather than flat.

The two quantities sit on separate panels rather than a twin axis: they are
unrelated scales, and the action delta's magnitude varies several-fold across
seeds while its shape does not, which a shared axis would obscure.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts import figure_style as fs  # noqa: E402

# Probe output, full_method stage-6 checkpoints, one row per tier.
STD_POS = [0.02, 0.36, 0.47, 1.00]
TIERS = ["rtk_fixed", "rtk_float", "standalone", "degraded"]

# seed -> (|d action| per tier, epistemic per tier)
SEEDS = {
    "42": ([0.0000, 0.2034, 0.2025, 0.2291], [0.1564, 0.4896, 0.6331, 1.0084]),
    "123": ([0.0000, 0.3884, 0.3757, 0.3191], [0.0659, 0.6724, 0.8462, 1.0908]),
    "7": ([0.0000, 1.6675, 1.6931, 1.7052], [0.0317, 0.2160, 0.3397, 0.5643]),
}

# One colour/marker/linestyle per seed, reusing the shared accents.
SEED_STYLE = {
    "42": (fs.ACCENT_ALT, "o", "-"),
    "123": (fs.ACCENT, "s", "--"),
    "7": (fs.TRACE, "^", "-."),
}


def render(out) -> None:
    """Draw the covariance probe to `out` (extension is normalised)."""
    fs.apply()

    fig, (ax_a, ax_e) = plt.subplots(1, 2, figsize=fs.SIDE_2, sharex=True)

    for seed, (delta, epistemic) in SEEDS.items():
        colour, marker, ls = SEED_STYLE[seed]
        for ax, series in ((ax_a, delta), (ax_e, epistemic)):
            ax.plot(
                STD_POS,
                series,
                marker=marker,
                linestyle=ls,
                color=colour,
                lw=fs.LW,
                ms=fs.MS,
                label=f"seed {seed}",
            )

    ax_a.set_ylabel(r"$|\Delta$ action$|$ from clean baseline")
    ax_e.set_ylabel("Epistemic estimate")
    for ax in (ax_a, ax_e):
        ax.set_xlabel("Swept EKF position standard deviation (m)")
        ax.set_ylim(bottom=0)
        fs.grid(ax)
        # Tier names on a secondary top axis rather than inside the plot area,
        # so they cannot clip the y limits. rtk_float and standalone sit 0.11 m
        # apart on the x axis, too close for horizontal labels, so the strip is
        # angled; that keeps all four on one baseline and preserves reading order.
        ax_top = ax.secondary_xaxis("top")
        ax_top.set_xticks(STD_POS)
        ax_top.set_xticklabels(
            [fs.tier_label(t) for t in TIERS],
            rotation=30,
            ha="left",
            rotation_mode="anchor",
        )
        ax_top.tick_params(
            axis="x", length=0, pad=2, labelsize=fs.FS_NOTE, colors=fs.MUTED
        )
        for spine in ax_top.spines.values():
            spine.set_visible(False)
        for x in STD_POS:
            ax.axvline(x, color=fs.RULE, lw=0.4, zorder=0)

    fs.legend_strip(fig, ax_a, side="below")

    fs.save(fig, Path(out))


if __name__ == "__main__":
    render(
        Path(sys.argv[1]) / "covariance_probe"
        if len(sys.argv) > 1
        else "covariance_probe"
    )
