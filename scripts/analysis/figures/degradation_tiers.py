"""
@file degradation_tiers.py
@brief Per-arm success across the TRUE-localisation GNSS tiers
       under the live anchor Markov chain, pooled over seeds.

The held gnss_fixed/gnss_degraded contrast is a weak stressor: the EKF suppresses
a static raw fault, so the two held tiers do not separate at the policy's input
(reported sigma p50 ~0.016 m in both). The real degradation axis lives inside the
anchor_deployment condition, where the fix state transitions mid-episode and the
TRUE localisation error sweeps the full tier range. This figure bands each anchor
episode by its WORST true localisation error ||ekf_xy - gt_xy|| (the faithful tier
proxy: the reported EKF std saturates ~1.3 m and understates the tier by up to 8x,
per trace_tiers.py), then plots per-arm success rate across the bands,
pooled over seeds 42/123/7.

Read-only over the frozen eval CSVs. Pure CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Make the repo root importable so the shared figure style resolves when this
# file is run directly (python scripts/analysis/figures/degradation_tiers.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts import figure_style as fs  # noqa: E402

# Episodes are banded into equal-population terciles of their MEAN true
# localisation error ||ekf_xy - gt_xy|| under the anchor chain. Mean (not the
# per-episode maximum) is the faithful notion of how hard localisation typically
# was: banding on the max rewards a single transient covariance spike and yields a
# non-monotone axis. The fixed-integer tier is not a band here because the live
# chain almost never holds cm-level accuracy for a whole episode (see caption); the
# terciles instead partition the realised difficulty into low / medium / high.
_BAND_LABELS = ["low", "medium", "high"]

_ARMS = ["vanilla_ppo", "input_uncertainty", "output_uncertainty", "full_method"]
_SEEDS = ["seed_42", "seed_123", "seed_7"]

# Four distinct, colour-blind-safe, print-friendly hues (Okabe-Ito derived).
# full_method carries the eye (heavy blue line); the three baselines each get a
# clearly separated hue + marker + linestyle so they never blur together.
# Draw order only: colour, marker and linestyle come from figure_style so this
# figure shares one palette with every other plot. full_method sits on top so
# the headline trace is never hidden behind a baseline.
_ARM_ZORDER: Dict[str, int] = {
    "full_method": 6,
    "output_uncertainty": 4,
    "input_uncertainty": 3,
    "vanilla_ppo": 2,
}


def _stage6_dir(base: Path, seed: str, arm: str) -> Path:
    d = base / seed / arm
    st6 = sorted(p for p in d.iterdir() if p.name.startswith("6_"))
    if not st6:
        raise FileNotFoundError(f"no stage-6 checkpoint under {d}")
    return st6[0] / "without_wrapper"


def _collect(base: Path) -> pd.DataFrame:
    """One row per anchor episode across all arms/seeds: arm, mean_err, success."""
    rows: List[Dict[str, object]] = []
    for arm in _ARMS:
        for seed in _SEEDS:
            wd = _stage6_dir(base, seed, arm)
            ps = pd.read_csv(wd / "per_step_records.csv")
            ps = ps[ps["condition"] == "anchor_deployment"]
            # Mean true error per episode is the faithful notion of how hard
            # localisation typically was; the per-episode maximum rewards a single
            # transient covariance spike and yields a non-monotone difficulty axis.
            mean_err = ps.groupby("episode")["abs_err_pos_m"].mean()
            ep = pd.read_csv(wd / "episode_records.csv")
            ep = ep[ep["condition"] == "anchor_deployment"]
            for _, r in ep.iterrows():
                e = r["episode"]
                if e not in mean_err.index:
                    continue
                rows.append(
                    dict(
                        arm=arm,
                        mean_err=float(mean_err.loc[e]),
                        success=int(float(r["success"])),
                    )
                )
    return pd.DataFrame(rows)


def _tercile_edges(mean_err: pd.Series) -> Tuple[float, float]:
    """Pooled 33rd/66th percentile of mean true error (equal-population bands)."""
    return float(mean_err.quantile(1 / 3)), float(mean_err.quantile(2 / 3))


def _band_of(v: float, e1: float, e2: float) -> str:
    if v <= e1:
        return "low"
    if v <= e2:
        return "medium"
    return "high"


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion (fractions)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def plot(base: Path, out_pdf: Path) -> None:
    fs.apply()
    df = _collect(base)
    e1, e2 = _tercile_edges(df["mean_err"])
    df["band"] = df["mean_err"].apply(lambda v: _band_of(v, e1, e2))

    x = np.arange(len(_BAND_LABELS))
    fig, ax = plt.subplots(figsize=fs.WIDE)
    for arm in _ARMS:
        sub = df[df["arm"] == arm]
        ys, lo, hi = [], [], []
        for lab in _BAND_LABELS:
            cell = sub[sub["band"] == lab]
            n = len(cell)
            s = int(cell["success"].sum())
            p = s / n if n else np.nan
            ys.append(p * 100 if n else np.nan)
            l, h = _wilson(s, n)
            lo.append((p - l) * 100 if n else 0.0)
            hi.append((h - p) * 100 if n else 0.0)
        ax.errorbar(
            x,
            ys,
            yerr=[lo, hi],
            capsize=2.5,
            elinewidth=0.9,
            zorder=_ARM_ZORDER.get(arm, 3),
            label=fs.arm_label(arm),
            **fs.arm_kw(arm),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"low\n(≤{e1:.2f} m)",
            f"medium\n({e1:.2f}–{e2:.2f} m)",
            f"high\n(>{e2:.2f} m)",
        ]
    )
    ax.set_xlabel("mean true localisation error over the episode (anchor chain)")
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 65)
    ax.set_xlim(-0.35, len(_BAND_LABELS) - 0.65)
    # Legend order follows the arm order shared across the document.
    handles, labels = ax.get_legend_handles_labels()
    order = [fs.arm_label(a) for a in fs.ARM_ORDER]
    pairs = sorted(zip(handles, labels), key=lambda hl: order.index(hl[1]))
    handles, labels = zip(*pairs)
    fs.legend_strip(fig, (list(handles), list(labels)), side="above")

    fs.save(fig, out_pdf)
    print(f"tercile edges of mean true error: e1={e1:.3f} m, e2={e2:.3f} m")

    print("\nPooled anchor success by mean-error tercile (success/n = %):")
    for arm in _ARMS:
        sub = df[df["arm"] == arm]
        cells = []
        for lab in _BAND_LABELS:
            cell = sub[sub["band"] == lab]
            n = len(cell)
            s = int(cell["success"].sum())
            cells.append(f"{lab}: {s}/{n}={s/n*100:4.1f}%" if n else f"{lab}: -")
        print(f"  {arm:18s} " + "  ".join(cells))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("outputs/raw/evaluation_results"))
    ap.add_argument("--out", type=Path, default=Path("degradation_tiers"))
    args = ap.parse_args()
    plot(args.base, args.out)


if __name__ == "__main__":
    main()
