"""
@file trace_tier_breakdown.py
@brief Offline diagnostic: resolve demo-trace outcomes by GNSS localisation tier.

Reads a directory of per-episode demo-trace CSVs (written by CARLAParkingEnv during
training) and answers one question: does the policy fail BECAUSE localisation is hard
(failures concentrated in high-EKF-covariance episodes), or does it fail even when
localisation is clean (a sign of exploration collapse rather than task difficulty)?

For each episode it reduces the per-step trace to:
 - success: 1 if the per-step success flag is ever set (it flips on the terminal
   success step and stays 0 otherwise).
 - final pos error (m): the last step's pos_error_m (terminal parking precision).
 - hardest localisation moment: the per-episode max TRUE localisation error
   ||ekf_xy - gt_xy|| (from the trace's ekf_x/y and gt_x/y), used as the proxy for the
   worst GNSS tier the episode visited. NOT the reported EKF std: the EKF posterior std
   is heavily damped by IMU + process-model fusion and saturates around ~1.3 m even when
   the estimate is 10 m off truth, so it understates the tier by up to ~8x. The reported
   std is kept only as a secondary view (it is the policy's covariance OBS input).

It then tabulates success rate and mean final pos error bucketed by that true error into
GNSS-tier bands (thresholds from gnss_noise_profiles.yaml metric_stddev_m), plus the same
table keyed on the reported std (to expose the saturation), and the correlation between
per-step aleatoric uncertainty and per-step EKF std (does the head's predicted uncertainty
even track localisation uncertainty, or has it gone flat?).

@note Read-only: never mutates traces, configs, or outputs. Pure CPU, no torch/CARLA.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Upper edges (inclusive) of the GNSS-tier bands, in metres of TRUE localisation
# error ||ekf_xy - gt_xy||. Boundaries follow the tiers' metric_stddev_m
# (rtk_fixed 0.020, rtk_float 0.360, standalone 1.802, degraded 5.0).
#
# @warning Band on TRUE error, NOT on the reported EKF std: the EKF posterior std
# (obs features 2-4) is heavily damped by IMU + process-model fusion and saturates
# around ~1.3 m even when the estimate is 10 m off truth, so it badly understates the
# tier and mislabels genuinely-degraded episodes. The true error (from the trace's
# gt_x/y and ekf_x/y) is the faithful tier proxy.
_BAND_EDGES_M: List[Tuple[str, float]] = [
    ("fixed (<=0.36)", 0.36),
    ("float (<=1.80)", 1.80),
    ("standalone (<=5.0)", 5.0),
    ("degraded (>5.0)", float("inf")),
]


def _band_for(value: float) -> str:
    """@brief Map a per-episode max localisation error (m) to its GNSS-tier band."""
    if np.isnan(value):
        return "no-ekf (NaN)"
    for label, upper in _BAND_EDGES_M:
        if value <= upper:
            return label
    return _BAND_EDGES_M[-1][0]


def _reduce_episode(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    @brief Reduce one trace DataFrame to per-episode scalars.
    @param df: Per-step trace for a single episode.
    @return Dict of per-episode metrics, or None if required columns are missing.
    """
    required = {
        "success",
        "pos_error_m",
        "ekf_std_x",
        "ekf_std_y",
        "ekf_x",
        "ekf_y",
        "gt_x",
        "gt_y",
    }
    if not required.issubset(df.columns) or len(df) == 0:
        return None

    success = float(np.nanmax(df["success"].to_numpy()))
    final_pos_err = float(df["pos_error_m"].to_numpy()[-1])
    # TRUE localisation error per step: ||ekf_xy - gt_xy||. The faithful tier proxy
    # (the reported ekf_std saturates ~1.3 m regardless of how far the estimate drifts).
    true_err = np.sqrt(
        (df["ekf_x"] - df["gt_x"]) ** 2 + (df["ekf_y"] - df["gt_y"]) ** 2
    ).to_numpy()
    max_true_err = float(np.nanmax(true_err))
    # Secondary view: worst reported posterior std (the policy's covariance input).
    max_std = float(np.nanmax(df[["ekf_std_x", "ekf_std_y"]].to_numpy()))
    return {
        "success": success,
        "final_pos_err": final_pos_err,
        "max_true_err": max_true_err,
        "max_std": max_std,
    }


def _aleatoric_ekf_correlation(
    frames: Sequence[pd.DataFrame],
) -> Tuple[Optional[float], int]:
    """
    @brief Pearson correlation of per-step aleatoric vs per-step EKF std_x.
    @param frames: Per-episode trace DataFrames.
    @return Tuple (correlation, n_pairs); correlation is None if too few valid pairs
            or either series has zero variance.
    """
    cols = ("aleatoric", "ekf_std_x")
    parts = [f.loc[:, cols] for f in frames if set(cols).issubset(f.columns)]
    if not parts:
        return None, 0
    stacked = pd.concat(parts, ignore_index=True).apply(pd.to_numeric, errors="coerce")
    stacked = stacked.dropna()
    n_pairs = len(stacked)
    if n_pairs < 2:
        return None, n_pairs
    ale = stacked["aleatoric"].to_numpy()
    std = stacked["ekf_std_x"].to_numpy()
    if ale.std() == 0.0 or std.std() == 0.0:
        return None, n_pairs
    return float(np.corrcoef(ale, std)[0, 1]), n_pairs


def _print_band_table(episodes: pd.DataFrame, std_col: str, title: str) -> None:
    """@brief Print success rate and mean final pos error bucketed by tier band."""
    labels = [label for label, _ in _BAND_EDGES_M] + ["no-ekf (NaN)"]
    bands = episodes[std_col].apply(_band_for)

    print(f"\n{title}")
    print(f"{'band':<22}{'n':>6}{'success_rate':>14}" f"{'mean_final_pos_err_m':>22}")
    for label in labels:
        sub = episodes[bands == label]
        if len(sub) == 0:
            continue
        sr = float(sub["success"].mean())
        mpe = float(sub["final_pos_err"].mean())
        print(f"{label:<22}{len(sub):>6}{sr:>14.3f}{mpe:>22.3f}")


def _run(trace_dir: Path) -> None:
    """@brief Load all traces under trace_dir and print the tier-resolved summary."""
    csv_paths = sorted(trace_dir.glob("episode_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No episode_*.csv files found in {trace_dir}")

    frames: List[pd.DataFrame] = []
    rows: List[Dict[str, float]] = []
    skipped = 0
    for path in csv_paths:
        df = pd.read_csv(path)
        reduced = _reduce_episode(df)
        if reduced is None:
            skipped += 1
            continue
        frames.append(df)
        rows.append(reduced)

    if not rows:
        raise ValueError(f"No usable traces in {trace_dir} (missing columns?).")

    episodes = pd.DataFrame(rows)
    n = len(episodes)
    overall_sr = float(episodes["success"].mean())
    overall_mpe = float(episodes["final_pos_err"].mean())

    print(f"Trace dir: {trace_dir}")
    print(
        f"Episodes: {n} usable"
        + (f" ({skipped} skipped)" if skipped else "")
        + f" | overall success {overall_sr:.3f} | "
        f"mean final pos err {overall_mpe:.3f} m"
    )

    # Primary: band by TRUE localisation error (the faithful tier proxy).
    _print_band_table(
        episodes,
        "max_true_err",
        "By worst TRUE localisation error ||ekf_xy - gt_xy|| (the real tier):",
    )
    # Secondary: band by the reported posterior std (the policy's covariance input) -
    # shown to expose how badly it understates the true tier (it saturates ~1.3 m).
    _print_band_table(
        episodes,
        "max_std",
        "By worst REPORTED EKF std (obs input; saturates, understates the tier):",
    )

    corr, n_pairs = _aleatoric_ekf_correlation(frames)
    print("\nPer-step aleatoric vs reported EKF std_x:")
    if corr is None:
        print(f"  correlation undefined (n_pairs={n_pairs}; zero variance or too few)")
    else:
        print(f"  Pearson r = {corr:+.3f}  (n_pairs={n_pairs})")
        print(
            "  near 0 => the head's uncertainty does NOT track localisation "
            "uncertainty (flat/collapsed)."
        )

    # Decision hint (case A vs B), keyed on TRUE error.
    clean = episodes[episodes["max_true_err"] <= _BAND_EDGES_M[0][1]]
    if len(clean) > 0:
        clean_sr = float(clean["success"].mean())
        print(
            f"\nClean-localisation episodes (max true err <= {_BAND_EDGES_M[0][1]} m): "
            f"{len(clean)}, success {clean_sr:.3f}."
        )
        print(
            "  Low clean-band success => failures occur even with good localisation "
            "(case A, exploration collapse).\n"
            "  High clean-band success but low degraded-band success => task is hard "
            "under degradation (case B)."
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace-dir",
        type=Path,
        required=True,
        help="Directory of episode_*.csv demo traces (one timestamped dump folder).",
    )
    return p.parse_args()


def main() -> None:
    """@brief CLI entry point."""
    args = _parse_args()
    _run(args.trace_dir)


if __name__ == "__main__":
    main()
