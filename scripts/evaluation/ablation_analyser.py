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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/evaluation/ablation_analyser.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from scripts.evaluation._discovery import discover_arm_csvs  # noqa: E402

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

# Quantile bins for the behaviour-vs-EKF-std cross-section.
_N_STD_BINS = 5


def _discover_arm_csvs(
    results_root: Path, leaf: Optional[str] = None
) -> Dict[str, Path]:
    """
    @brief Find each ablation arm's episode_records.csv under the nested tree.
    @param results_root: outputs/evaluation_results (the <baseline>/<leaf> root).
    @param leaf: Optional checkpoint leaf to pin to (single-arm runs); None lets
           each arm's newest run win.
    @return Mapping arm name -> path to its episode_records.csv. Per arm the
            without_wrapper variant (free-running policy - the caution/precision
            reads) is preferred, then the most recent leaf.

    The arm name is the first path component under the root (the baseline
    directory); discover_arm_csvs handles both the two-level (legacy) and
    three-level (wrapper-variant) layouts. Only the known ablation arms are kept.
    """
    return {
        arm: path
        for arm, path in discover_arm_csvs(
            results_root,
            "episode_records.csv",
            prefer_variant="without_wrapper",
            leaf=leaf,
        ).items()
        if arm in _ARM_ORDER
    }


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


def _degradation_slope(
    summary: pd.DataFrame,
    clean_condition: str = _SLOPE_CLEAN_CONDITION,
    degraded_condition: str = _SLOPE_DEGRADED_CONDITION,
) -> pd.DataFrame:
    """
    @brief Per-arm fall in success / rise in position error from clean to degraded GNSS.
    @param summary: Per-condition aggregate from _condition_summary.
    @param clean_condition: Slope-start condition name (cleanest held GNSS tier).
    @param degraded_condition: Slope-end condition name (worst held GNSS tier).
    @return DataFrame, one row per arm: success at the clean and degraded held
            tiers, the drop (percentage points), and the position-error growth (m).

    Both endpoint conditions must be present (the held-tier GNSS axis). A smaller
    success drop and smaller position-error growth mean a more graceful
    degradation - the property the covariance arms are claimed to have. The
    endpoints default to the occupancy-0.5 cell but can be pointed at the
    empty-lot cell (gnss_empty_fixed -> gnss_empty_degraded), where the
    intermediate EKF error is not swallowed by neighbours and the slope is clean.
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
            clean_condition not in pivot.columns
            or degraded_condition not in pivot.columns
        ):
            break
        clean = pivot.loc[arm, clean_condition]
        degraded = pivot.loc[arm, degraded_condition]
        pos_clean = pos_pivot.loc[arm, clean_condition]
        pos_degraded = pos_pivot.loc[arm, degraded_condition]
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


def _behaviour_by_std(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Final pos-error and approach speed per EKF-std bin, per arm.
    @param records: Tidy per-episode frame from _load_records.
    @return DataFrame: per (arm, std_bin) the mean final position error and mean
            final speed, with the std-bin range.

    The mechanism behind the outcome gap: binning episodes by their EKF position
    std (ekf_std_pos_mean_m, logged for every arm) and reading off how each arm
    behaves. The claim is that as std rises, the covariance arm holds its final
    position error roughly flat (it compensates), while the blind arm's error
    grows; the covariance arm may also slow down (emergent caution). Speed and
    error are per-episode finals, so this is a behavioural cross-section, not a
    causal per-step trace - that is what the calibration analysis covers.
    """
    std_col = "ekf_std_pos_mean_m"
    if std_col not in records.columns:
        return pd.DataFrame()
    # Caution metrics added later (mean_speed_moving_ms etc.); older CSVs lack
    # them, so aggregate only those present. final_pos_error_m is the precision
    # outcome; final_speed_ms is kept for backward compatibility.
    _CAUTION_COLS = [
        "final_pos_error_m",
        "final_speed_ms",
        "mean_speed_moving_ms",
        "mean_abs_vyaw_rads",
        "mean_brake_cmd",
        "mean_action_jerk",
    ]
    present = [c for c in _CAUTION_COLS if c in records.columns]
    df = records[[std_col, "arm", *present]].dropna(subset=[std_col])
    if len(df) < _N_STD_BINS:
        return pd.DataFrame()
    try:
        df = df.assign(std_bin=pd.qcut(df[std_col], _N_STD_BINS, duplicates="drop"))
    except ValueError:
        return pd.DataFrame()
    grouped = df.groupby(["arm", "std_bin"], observed=True)
    agg_spec: Dict[str, Tuple[str, object]] = {
        "std_low": (std_col, lambda s: float(s.min())),
        "std_high": (std_col, lambda s: float(s.max())),
        "n": (std_col, "size"),
    }
    # Each present caution metric becomes a mean_<col> column.
    for col in present:
        agg_spec[f"mean_{col}"] = (col, "mean")
    out = grouped.agg(**agg_spec).reset_index()
    out["std_mid"] = (out["std_low"] + out["std_high"]) / 2.0
    return out


def _plot_behaviour(behaviour: pd.DataFrame, out_dir: Path) -> None:
    """
    @brief Final pos-error and approach speed vs EKF-std bin, arms overlaid.
    @param behaviour: Per (arm, std_bin) table from _behaviour_by_std.
    @param out_dir: Directory for the saved figure.
    """
    if behaviour.empty:
        return
    sns.set_style("whitegrid")
    # Plot every caution metric present (precision outcome first, then the
    # drive-carefully signals). Expected directions as std rises: pos-error flat
    # for the covariance arm; speed/jerk/vyaw DOWN; brake UP.
    panels = [
        ("mean_final_pos_error_m", "Mean final position error (m)"),
        ("mean_mean_speed_moving_ms", "Mean approach speed (m/s)"),
        ("mean_mean_brake_cmd", "Mean brake command"),
        ("mean_mean_abs_vyaw_rads", "Mean |yaw rate| (rad/s)"),
        ("mean_mean_action_jerk", "Mean action jerk"),
    ]
    panels = [(c, lbl) for c, lbl in panels if c in behaviour.columns]
    if not panels:
        return
    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (metric, label) in zip(axes, panels):
        for arm in _ARM_ORDER:
            a = behaviour[behaviour["arm"] == arm]
            if a.empty:
                continue
            ax.plot(a["std_mid"], a[metric], "o-", label=arm)
        ax.set_xlabel("EKF position std bin midpoint (m)")
        ax.set_ylabel(label)
        ax.legend(title="arm", fontsize=8)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle("Behaviour vs localisation uncertainty (caution mechanism)")
    fig.tight_layout()
    fig.savefig(out_dir / "behaviour_by_std.png", dpi=150)
    plt.close(fig)


# Caution metrics and the direction that means "more conservative as std rises".
# Sign is the EXPECTED sign of the Spearman(metric, EKF std) for a cautious
# policy: brake should rise (+1), speed / yaw-rate / jerk should fall (-1). The
# cross-arm DIFFERENCE in these slopes (covariance arm minus blind arm) is the
# causal test: caution attributable to SEEING the covariance, not to the episode
# merely being harder (which the blind arm also experiences).
_CAUTION_DIRECTION: List[Tuple[str, int]] = [
    ("mean_brake_cmd", +1),
    ("mean_speed_moving_ms", -1),
    ("mean_abs_vyaw_rads", -1),
    ("mean_action_jerk", -1),
]


def _caution_slopes(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-arm Spearman of each caution metric against EKF position std.
    @param records: Tidy per-episode frame from _load_records.
    @return DataFrame: one row per (arm, metric) with the rank correlation, its
            expected cautious sign, whether the sign matches, and n. Empty if the
            caution columns are absent (older CSVs).

    A negative speed/yaw/jerk slope and a positive brake slope mean the arm drives
    more conservatively as localisation uncertainty rises. This is the WITHIN-arm
    read; the cross-arm difference (_caution_contrast) is the causal claim.
    """
    std_col = "ekf_std_pos_mean_m"
    metrics = [m for m, _ in _CAUTION_DIRECTION if m in records.columns]
    if std_col not in records.columns or not metrics:
        return pd.DataFrame()
    direction = dict(_CAUTION_DIRECTION)
    rows: List[Dict[str, object]] = []
    for arm in [a for a in _ARM_ORDER if a in set(records["arm"])]:
        sub = records[records["arm"] == arm]
        for metric in metrics:
            pair = sub[[std_col, metric]].dropna()
            if len(pair) < 3 or pair[std_col].nunique() < 2:
                continue
            spearman = float(pair[std_col].rank().corr(pair[metric].rank()))
            exp = direction[metric]
            rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "spearman_vs_std": spearman,
                    "expected_sign": exp,
                    "is_cautious": bool(spearman * exp > 0),
                    "n": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def _caution_levels(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-arm ABSOLUTE caution level and the precision/success payoff.
    @param records: Tidy per-episode frame from _load_records.
    @return DataFrame: one row per arm with mean approach speed, mean brake,
            success rate, median final pos error, and collision rate. Empty if
            the caution columns are absent.

    The slope (_caution_slopes) measures how behaviour CHANGES with std; this
    measures the baseline LEVEL. A covariance arm that drives slower overall is
    only "cautious" rather than "undertrained" if the slowness buys accuracy -
    higher success and lower final pos error - so those are reported alongside.
    """
    if "mean_speed_moving_ms" not in records.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for arm in [a for a in _ARM_ORDER if a in set(records["arm"])]:
        sub = records[records["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "mean_speed_moving_ms": float(sub["mean_speed_moving_ms"].mean()),
                "mean_brake_cmd": float(sub.get("mean_brake_cmd", pd.Series()).mean()),
                "success_rate": float(100.0 * sub["success"].mean()),
                "median_pos_error_m": float(sub["final_pos_error_m"].median()),
                "collision_rate": float(100.0 * (sub["outcome"] == "collision").mean()),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _caution_contrast(slopes: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Cross-arm caution-slope difference for each covariance contrast pair.
    @param slopes: Per-(arm, metric) slope table from _caution_slopes.
    @return DataFrame: per (pair, metric) the treatment slope, control slope, and
            their difference oriented so POSITIVE = the covariance arm is more
            cautious than its blind control. Empty if either arm is missing.

    This is the causal claim for "input uncertainty makes the car conservative":
    the covariance arm (treatment) should show a stronger cautious slope than the
    matched no-covariance arm (control). Difference is signed by the metric's
    expected direction so a positive value always means "covariance => more
    caution", whichever metric.
    """
    if slopes.empty:
        return pd.DataFrame()
    direction = dict(_CAUTION_DIRECTION)
    by_arm_metric = {
        (r["arm"], r["metric"]): float(r["spearman_vs_std"])
        for _, r in slopes.iterrows()
    }
    rows: List[Dict[str, object]] = []
    for pair_name, treat_arm, control_arm in _CONTRAST_PAIRS:
        for metric, exp in _CAUTION_DIRECTION:
            t = by_arm_metric.get((treat_arm, metric))
            c = by_arm_metric.get((control_arm, metric))
            if t is None or c is None:
                continue
            # Orient by expected direction: positive difference = treatment more
            # cautious. For a "down" metric (exp=-1) a more-negative treatment
            # slope is more cautious, so multiply the raw (t - c) by exp.
            diff = (t - c) * direction[metric]
            rows.append(
                {
                    "pair": pair_name,
                    "treatment": treat_arm,
                    "control": control_arm,
                    "metric": metric,
                    "treat_slope": t,
                    "control_slope": c,
                    "caution_diff": diff,
                    "covariance_more_cautious": bool(diff > 0),
                }
            )
    return pd.DataFrame(rows)


def analyse(
    results_root: Path,
    out_dir: Path,
    seed: int,
    slope_clean: str = _SLOPE_CLEAN_CONDITION,
    slope_degraded: str = _SLOPE_DEGRADED_CONDITION,
    leaf: Optional[str] = None,
) -> None:
    """
    @brief Run the full cross-arm analysis and write tables + figures.
    @param results_root: outputs/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory to write the CSV tables and PNG figures into.
    @param seed: Bootstrap RNG seed (reproducibility).
    @param slope_clean: Degradation-slope start condition (cleanest GNSS tier).
    @param slope_degraded: Degradation-slope end condition (worst GNSS tier).
    @param leaf: Optional checkpoint leaf to pin to (single-arm runs); None lets
           each arm's newest run win.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_csvs = _discover_arm_csvs(results_root, leaf=leaf)
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
    slope = _degradation_slope(summary, slope_clean, slope_degraded)
    behaviour = _behaviour_by_std(records)
    caution_slopes = _caution_slopes(records)
    caution_contrast = _caution_contrast(caution_slopes)
    caution_levels = _caution_levels(records)

    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    contrasts.to_csv(out_dir / "covariance_contrasts.csv", index=False)
    slope.to_csv(out_dir / "degradation_slope.csv", index=False)
    if not behaviour.empty:
        behaviour.to_csv(out_dir / "behaviour_by_std.csv", index=False)
    if not caution_slopes.empty:
        caution_slopes.to_csv(out_dir / "caution_slopes.csv", index=False)
    if not caution_contrast.empty:
        caution_contrast.to_csv(out_dir / "caution_contrast.csv", index=False)
    if not caution_levels.empty:
        caution_levels.to_csv(out_dir / "caution_levels.csv", index=False)

    _plot_condition_bars(summary, out_dir)
    _plot_degradation_slope(slope, out_dir)
    _plot_behaviour(behaviour, out_dir)

    _print_headline(contrasts, slope, slope_clean, slope_degraded)
    _print_caution(caution_slopes, caution_contrast)
    _print_caution_levels(caution_levels)
    print(f"\nTables and figures written to {out_dir}")


def _print_headline(
    contrasts: pd.DataFrame,
    slope: pd.DataFrame,
    slope_clean: str = _SLOPE_CLEAN_CONDITION,
    slope_degraded: str = _SLOPE_DEGRADED_CONDITION,
) -> None:
    """
    @brief Console summary of the two claims: covariance contrast + slope.
    @param contrasts: Contrast table from _contrast_table.
    @param slope: Slope table from _degradation_slope.
    @param slope_clean: Slope-start condition name (for the missing-data hint).
    @param slope_degraded: Slope-end condition name (for the missing-data hint).
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
        print(f"  (need both {slope_clean} and {slope_degraded} conditions)")
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


def _print_caution(slopes: pd.DataFrame, contrast: pd.DataFrame) -> None:
    """
    @brief Console summary of the caution-vs-uncertainty result.
    @param slopes: Per-(arm, metric) slope table from _caution_slopes.
    @param contrast: Cross-arm difference table from _caution_contrast.

    Two reads: WITHIN-arm (does each arm drive more cautiously as std rises?) and
    CROSS-arm (is the caution stronger for the covariance arm - the causal claim).
    The cross-arm block is empty until both arms of a pair are evaluated.
    """
    print("\n=== Caution vs EKF std: within-arm slopes (Spearman) ===")
    if slopes.empty:
        print("  (no caution columns - re-run eval with the updated evaluate.py)")
    else:
        for arm in [a for a in _ARM_ORDER if a in set(slopes["arm"])]:
            a = slopes[slopes["arm"] == arm]
            parts = "  ".join(
                f"{r['metric'].replace('mean_', '').replace('_', ''):14s}"
                f"{r['spearman_vs_std']:+.3f}{'ok' if r['is_cautious'] else 'XX'}"
                for _, r in a.iterrows()
            )
            print(f"  {arm:20s} {parts}")
        print("  brake should be +, speed/vyaw/jerk should be - (ok = cautious sign).")

    print("\n=== Caution CROSS-arm (covariance - blind): the causal claim ===")
    if contrast.empty:
        print(
            "  (need BOTH arms of a contrast pair: input_uncertainty vs vanilla_ppo,\n"
            "   full_method vs output_uncertainty - only then is caution attributable\n"
            "   to SEEING the covariance, not to the episode merely being harder)"
        )
    else:
        for pair in contrast["pair"].unique():
            p = contrast[contrast["pair"] == pair]
            wins = int(p["covariance_more_cautious"].sum())
            print(f"  [{pair}] covariance more cautious on {wins}/{len(p)} metrics:")
            for _, r in p.iterrows():
                mark = (
                    "->covariance" if r["covariance_more_cautious"] else "->blind/equal"
                )
                print(
                    f"      {r['metric']:22s} treat {r['treat_slope']:+.3f} "
                    f"vs control {r['control_slope']:+.3f}  diff {r['caution_diff']:+.3f} {mark}"
                )


def _print_caution_levels(levels: pd.DataFrame) -> None:
    """
    @brief Console summary of absolute caution LEVEL and its accuracy payoff.
    @param levels: Per-arm level table from _caution_levels.

    Speed-vs-std SLOPE can be flat even when an arm drives cautiously at a low
    baseline; this shows the absolute speed/brake level alongside success and
    pos-error so a slower arm can be read as cautious (slower AND more accurate)
    rather than merely undertrained (slower AND worse).
    """
    print("\n=== Caution LEVEL (absolute) + accuracy payoff ===")
    if levels.empty:
        print("  (no caution columns - re-run eval with the updated evaluate.py)")
        return
    print(
        f"  {'arm':20s} {'speed':>6} {'brake':>6} {'succ%':>6} "
        f"{'pos_err':>8} {'collide%':>9}"
    )
    for _, r in levels.iterrows():
        print(
            f"  {str(r['arm']):20s} {r['mean_speed_moving_ms']:6.2f} "
            f"{r['mean_brake_cmd']:6.3f} {r['success_rate']:6.1f} "
            f"{r['median_pos_error_m']:8.2f} {r['collision_rate']:9.1f}"
        )
    print(
        "  A covariance arm that is SLOWER and MORE accurate (higher succ%, lower\n"
        "  pos_err) is cautious, not undertrained - the slowness buys precision."
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
    parser.add_argument(
        "--slope-clean",
        type=str,
        default=_SLOPE_CLEAN_CONDITION,
        help="Degradation-slope start condition (cleanest held GNSS tier).",
    )
    parser.add_argument(
        "--slope-degraded",
        type=str,
        default=_SLOPE_DEGRADED_CONDITION,
        help="Degradation-slope end condition (worst held GNSS tier).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pin to one checkpoint leaf, e.g. 1_42_19062026-0120 "
        "(single-arm runs; default: each arm's newest run).",
    )
    args = parser.parse_args()
    analyse(
        Path(args.results_root),
        Path(args.output_dir),
        args.seed,
        args.slope_clean,
        args.slope_degraded,
        leaf=args.checkpoint,
    )


if __name__ == "__main__":
    main()
