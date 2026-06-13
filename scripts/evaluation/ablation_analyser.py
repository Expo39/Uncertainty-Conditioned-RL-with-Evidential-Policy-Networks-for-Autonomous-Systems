"""
@file ablation_analyser.py
@brief Cross-arm analysis of the input-covariance ablation from eval CSVs.

Host-side, read-only diagnostic that loads every ablation arm's per-episode
records (outputs/evaluation_results/<baseline>/<leaf>/episode_records.csv),
attaches the arm name from the directory tree, joins on condition, and computes
the contrasts the dissertation actually claims:

  - Per-condition success rate and final position error for each arm.
  - The covariance contrast deltas with bootstrap confidence intervals:
    input_uncertainty - vanilla_ppo (standard heads, covariance on/off) and
    full_method - output_uncertainty (evidential heads, covariance on/off).
  - A single "degradation slope" per arm: how far success falls and position
    error grows from the cleanest GNSS tier (rtk_fixed) to the worst (degraded).
    The thesis is that the covariance arms degrade more gracefully (shallower
    slope), so this is the headline number.

A per-condition bar grid (success + position error, arms side by side) and a
degradation-slope figure are written to the output directory. Pure pandas /
numpy / matplotlib on the host .venv - no CARLA, ROS 2, or torch. Run via
`make analyse-ablation` (never python directly).

@author Antonio Galdes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# The four ablation arms in plot order: covariance-off then covariance-on,
# standard heads then evidential heads. The contrast pairs are adjacent.
_ARM_ORDER: List[str] = [
    "vanilla_ppo",
    "input_uncertainty",
    "output_uncertainty",
    "full_method",
]

# The single-variable covariance contrasts the study claims (covariance arm
# minus its matched no-covariance arm with the same policy head).
_CONTRAST_PAIRS: List[Tuple[str, str, str]] = [
    ("standard_head", "input_uncertainty", "vanilla_ppo"),
    ("evidential_head", "full_method", "output_uncertainty"),
]

# GNSS tier endpoints for the degradation slope (cleanest -> worst held tier).
_SLOPE_CLEAN_CONDITION = "gnss_fixed"
_SLOPE_DEGRADED_CONDITION = "gnss_degraded"

# Number of bootstrap resamples for delta confidence intervals.
_N_BOOTSTRAP = 10000


def _discover_arm_csvs(results_root: Path) -> Dict[str, Path]:
    """
    @brief Find each ablation arm's episode_records.csv under the nested tree.
    @param results_root: outputs/evaluation_results (the <baseline>/<leaf> root).
    @return Mapping arm name -> path to its episode_records.csv. If an arm has
            several leaves (re-runs), the most recently modified one is used.

    The arm name is the first path component under the root (the baseline
    directory). Only the four known ablation arms are kept.
    """
    found: Dict[str, Path] = {}
    for csv_path in sorted(results_root.glob("*/*/episode_records.csv")):
        arm = csv_path.parent.parent.name
        if arm not in _ARM_ORDER:
            continue
        if arm not in found or csv_path.stat().st_mtime > found[arm].stat().st_mtime:
            found[arm] = csv_path
    return found


def _load_records(arm_csvs: Dict[str, Path]) -> pd.DataFrame:
    """
    @brief Concatenate every arm's per-episode records into one tidy frame.
    @param arm_csvs: Mapping arm name -> episode_records.csv path.
    @return Long-form DataFrame with an added "arm" column (categorical, in
            _ARM_ORDER) and a numeric "success" column.
    """
    frames: List[pd.DataFrame] = []
    for arm, path in arm_csvs.items():
        df = pd.read_csv(path)
        df["arm"] = arm
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            "No episode_records.csv found for any known ablation arm."
        )
    records = pd.concat(frames, ignore_index=True)
    records["arm"] = pd.Categorical(records["arm"], categories=_ARM_ORDER, ordered=True)
    # success is written as int (0/1); coerce defensively in case of NaN rows.
    records["success"] = pd.to_numeric(records["success"], errors="coerce")
    return records


def _condition_summary(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-arm, per-condition aggregate of the headline metrics.
    @param records: Tidy per-episode frame from _load_records.
    @return DataFrame indexed by (condition, arm) with success_rate (percent),
            mean_pos_error_m, n_episodes, and mean EKF position std.
    """
    grouped = records.groupby(["condition", "arm"], observed=True)
    summary = grouped.agg(
        success_rate=("success", lambda s: 100.0 * s.mean()),
        mean_pos_error_m=("final_pos_error_m", "mean"),
        n_episodes=("success", "size"),
        ekf_std_pos_mean_m=("ekf_std_pos_mean_m", "mean"),
    ).reset_index()
    return summary


def _bootstrap_delta_ci(
    treat: np.ndarray,
    control: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    @brief Bootstrap the (treatment - control) difference of a per-episode metric.
    @param treat: Per-episode values for the covariance-on arm.
    @param control: Per-episode values for the matched covariance-off arm.
    @param statistic: "mean" (e.g. success fraction or position error mean).
    @param rng: Seeded NumPy generator for reproducibility.
    @return Tuple (point_estimate, ci_low, ci_high) at the 95% percentile level.

    Treatment and control are resampled independently (the arms are evaluated on
    separate episode draws, so this is a two-sample difference, not paired).
    NaNs (e.g. position error on a handed-off episode) are dropped per arm first.
    """
    treat = treat[~np.isnan(treat)]
    control = control[~np.isnan(control)]
    if treat.size == 0 or control.size == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(treat.mean() - control.mean())
    t_idx = rng.integers(0, treat.size, size=(_N_BOOTSTRAP, treat.size))
    c_idx = rng.integers(0, control.size, size=(_N_BOOTSTRAP, control.size))
    deltas = treat[t_idx].mean(axis=1) - control[c_idx].mean(axis=1)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)


def _contrast_table(records: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    @brief Covariance contrast deltas with bootstrap CIs, per condition.
    @param records: Tidy per-episode frame.
    @param seed: Bootstrap RNG seed.
    @return DataFrame with one row per (pair, condition): the success-rate delta
            (percentage points) and position-error delta (m), each with a 95% CI
            and a "significant" flag (CI excludes zero).

    A positive success delta or a negative position-error delta favours the
    covariance arm. Significance is the CI not straddling zero - the formal
    statement that covariance helped at that condition.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    conditions = list(records["condition"].unique())

    for pair_name, treat_arm, control_arm in _CONTRAST_PAIRS:
        for condition in conditions:
            cond = records[records["condition"] == condition]
            t = cond[cond["arm"] == treat_arm]
            c = cond[cond["arm"] == control_arm]
            if t.empty or c.empty:
                continue

            # Success delta in percentage points (100 * fraction).
            s_point, s_lo, s_hi = _bootstrap_delta_ci(
                100.0 * t["success"].to_numpy(dtype=float),
                100.0 * c["success"].to_numpy(dtype=float),
                "mean",
                rng,
            )
            # Position-error delta in metres (negative = covariance arm tighter).
            p_point, p_lo, p_hi = _bootstrap_delta_ci(
                t["final_pos_error_m"].to_numpy(dtype=float),
                c["final_pos_error_m"].to_numpy(dtype=float),
                "mean",
                rng,
            )

            rows.append(
                {
                    "pair": pair_name,
                    "treatment": treat_arm,
                    "control": control_arm,
                    "condition": condition,
                    "success_delta_pp": s_point,
                    "success_ci_low": s_lo,
                    "success_ci_high": s_hi,
                    "success_significant": not (s_lo <= 0.0 <= s_hi),
                    "pos_error_delta_m": p_point,
                    "pos_error_ci_low": p_lo,
                    "pos_error_ci_high": p_hi,
                    "pos_error_significant": not (p_lo <= 0.0 <= p_hi),
                }
            )
    return pd.DataFrame(rows)


def _degradation_slope(summary: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-arm fall in success / rise in position error from clean to degraded GNSS.
    @param summary: Per-condition aggregate from _condition_summary.
    @return DataFrame, one row per arm: success at the clean and degraded held
            tiers, the drop (percentage points), and the position-error growth (m).

    Both endpoint conditions must be present (the held-tier GNSS axis). A smaller
    success drop and smaller position-error growth mean a more graceful
    degradation - the property the covariance arms are claimed to have.
    """
    pivot = summary.pivot_table(
        index="arm", columns="condition", values="success_rate", observed=True
    )
    pos_pivot = summary.pivot_table(
        index="arm", columns="condition", values="mean_pos_error_m", observed=True
    )
    rows: List[Dict[str, object]] = []
    for arm in pivot.index:
        if (
            _SLOPE_CLEAN_CONDITION not in pivot.columns
            or _SLOPE_DEGRADED_CONDITION not in pivot.columns
        ):
            break
        clean = pivot.loc[arm, _SLOPE_CLEAN_CONDITION]
        degraded = pivot.loc[arm, _SLOPE_DEGRADED_CONDITION]
        pos_clean = pos_pivot.loc[arm, _SLOPE_CLEAN_CONDITION]
        pos_degraded = pos_pivot.loc[arm, _SLOPE_DEGRADED_CONDITION]
        rows.append(
            {
                "arm": arm,
                "success_clean_pct": clean,
                "success_degraded_pct": degraded,
                "success_drop_pp": clean - degraded,
                "pos_error_clean_m": pos_clean,
                "pos_error_degraded_m": pos_degraded,
                "pos_error_growth_m": pos_degraded - pos_clean,
            }
        )
    return pd.DataFrame(rows)


def _plot_condition_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    """
    @brief Side-by-side success and position-error bars, arms grouped per condition.
    @param summary: Per-condition aggregate.
    @param out_dir: Directory for the saved figure.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, 1.1 * summary["condition"].nunique()), 9)
    )
    for ax, metric, label in (
        (axes[0], "success_rate", "Success rate (%)"),
        (axes[1], "mean_pos_error_m", "Mean final position error (m)"),
    ):
        sns.barplot(
            data=summary,
            x="condition",
            y=metric,
            hue="arm",
            hue_order=_ARM_ORDER,
            ax=ax,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="arm", fontsize=8)
    for tick in axes[1].get_xticklabels():
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_by_condition.png", dpi=150)
    plt.close(fig)


def _plot_degradation_slope(slope: pd.DataFrame, out_dir: Path) -> None:
    """
    @brief Clean-vs-degraded success per arm - the graceful-degradation headline.
    @param slope: Per-arm slope table from _degradation_slope.
    @param out_dir: Directory for the saved figure.
    """
    if slope.empty:
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in slope.iterrows():
        ax.plot(
            [0, 1],
            [row["success_clean_pct"], row["success_degraded_pct"]],
            marker="o",
            label=str(row["arm"]),
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["RTK fixed (clean)", "Degraded (~5 m)"])
    ax.set_ylabel("Success rate (%)")
    ax.set_title("GNSS degradation slope by ablation arm")
    ax.legend(title="arm")
    fig.tight_layout()
    fig.savefig(out_dir / "degradation_slope.png", dpi=150)
    plt.close(fig)


def analyse(results_root: Path, out_dir: Path, seed: int) -> None:
    """
    @brief Run the full cross-arm analysis and write tables + figures.
    @param results_root: outputs/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory to write the CSV tables and PNG figures into.
    @param seed: Bootstrap RNG seed (reproducibility).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_csvs = _discover_arm_csvs(results_root)
    if not arm_csvs:
        raise FileNotFoundError(
            f"No arm episode_records.csv under {results_root}. "
            "Run make docker-eval for each BASELINE first."
        )
    print("Loaded arms:")
    for arm in _ARM_ORDER:
        if arm in arm_csvs:
            print(f"  {arm:20s} <- {arm_csvs[arm]}")
        else:
            print(f"  {arm:20s} (MISSING - contrast involving it is skipped)")

    records = _load_records(arm_csvs)
    summary = _condition_summary(records)
    contrasts = _contrast_table(records, seed)
    slope = _degradation_slope(summary)

    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    contrasts.to_csv(out_dir / "covariance_contrasts.csv", index=False)
    slope.to_csv(out_dir / "degradation_slope.csv", index=False)

    _plot_condition_bars(summary, out_dir)
    _plot_degradation_slope(slope, out_dir)

    _print_headline(contrasts, slope)
    print(f"\nTables and figures written to {out_dir}")


def _print_headline(contrasts: pd.DataFrame, slope: pd.DataFrame) -> None:
    """
    @brief Console summary of the two claims: covariance contrast + slope.
    @param contrasts: Contrast table from _contrast_table.
    @param slope: Slope table from _degradation_slope.
    """
    print("\n=== Covariance contrast (treatment - control), 95% bootstrap CI ===")
    if contrasts.empty:
        print("  (no overlapping conditions across arms)")
    else:
        for _, r in contrasts.iterrows():
            star = "*" if r["success_significant"] else " "
            print(
                f"  [{r['pair']:15s}] {r['condition']:24s} "
                f"success {r['success_delta_pp']:+6.1f}pp "
                f"[{r['success_ci_low']:+6.1f}, {r['success_ci_high']:+6.1f}] {star}  "
                f"pos {r['pos_error_delta_m']:+5.2f}m"
            )
        print("  (* = success CI excludes zero: covariance significantly helped)")

    print("\n=== GNSS degradation slope (clean -> degraded) ===")
    if slope.empty:
        print(
            f"  (need both {_SLOPE_CLEAN_CONDITION} and "
            f"{_SLOPE_DEGRADED_CONDITION} conditions)"
        )
    else:
        for _, r in slope.iterrows():
            print(
                f"  {str(r['arm']):20s} success {r['success_clean_pct']:5.1f}% -> "
                f"{r['success_degraded_pct']:5.1f}%  (drop {r['success_drop_pp']:5.1f}pp)  "
                f"pos err +{r['pos_error_growth_m']:.2f}m"
            )
        print(
            "  Shallower drop + smaller pos-error growth = more graceful (the claim)."
        )


def main() -> None:
    """
    @brief CLI entry point for the cross-arm ablation analysis.
    """
    parser = argparse.ArgumentParser(
        description="Cross-arm analysis of the input-covariance ablation."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="outputs/evaluation_results",
        help="Root holding <baseline>/<leaf>/episode_records.csv for each arm.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation_analysis",
        help="Directory for the analysis tables and figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed for reproducible confidence intervals.",
    )
    args = parser.parse_args()
    analyse(Path(args.results_root), Path(args.output_dir), args.seed)


if __name__ == "__main__":
    main()
