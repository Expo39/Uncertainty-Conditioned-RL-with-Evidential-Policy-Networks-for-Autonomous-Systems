"""
@file ablation.py
@brief Cross-arm analysis of the input-covariance ablation from eval CSVs.

Computes per-condition success/position error per arm, the covariance
contrast deltas with bootstrap CIs, and the GNSS degradation slope per arm
- a shallower slope is the hypothesised covariance advantage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/analysis/ablation.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis._discovery import discover_arm_csvs  # noqa: E402

# The six ablation arms in plot order: standard, heteroscedastic then evidential
# heads, covariance-off before covariance-on. The contrast pairs are adjacent.
_ARM_ORDER: List[str] = [
    "vanilla_ppo",
    "input_uncertainty",
    "heteroscedastic",
    "heteroscedastic_input",
    "output_uncertainty",
    "full_method",
]

# The single-variable covariance contrasts the study claims (covariance arm
# minus its matched no-covariance arm with the same policy head). The
# heteroscedastic pair comes last so the bootstrap draws of the original two
# pairs, and so their intervals under a fixed seed, are unchanged.
_CONTRAST_PAIRS: List[Tuple[str, str, str]] = [
    ("standard_head", "input_uncertainty", "vanilla_ppo"),
    ("evidential_head", "full_method", "output_uncertainty"),
    ("heteroscedastic_head", "heteroscedastic_input", "heteroscedastic"),
]

# GNSS tier endpoints for the degradation slope (cleanest -> worst held tier).
_SLOPE_CLEAN_CONDITION = "gnss_fixed"
_SLOPE_DEGRADED_CONDITION = "gnss_degraded"

# Held-tier conditions dropped from every summary and figure by default: each
# pins one GNSS fix state for the whole episode, so the EKF suppresses the
# static raw fault and the "slope" between them is flat by construction.
_HELD_TIER_CONDITIONS: List[str] = ["gnss_fixed", "gnss_degraded"]

# lidar_degraded corrupts only the obstacle channel: the EKF never consumes
# LiDAR, so localisation std stays pinned at its RTK-fixed floor and the
# condition carries no localisation-uncertainty signal for the headline
# figures (dropped by drop_unreported() at the figure boundary only).
_UNREPORTED_CONDITIONS: List[str] = ["lidar_degraded"]

# Conditions whose GNSS tier varies WITHIN an episode. The behaviour and
# calibration readings need uncertainty that moves while the vehicle drives,
# which excludes the held tiers (pinned all episode) and the OOD layout (held
# at RTK fixed, and zero successes on every arm).
_VARYING_CONDITIONS: List[str] = [
    "anchor_deployment",
    "anchor_empty",
    "gnss_degrade_one_way",
]

# Number of bootstrap resamples for delta confidence intervals.
_N_BOOTSTRAP = 10000

# Quantile bins for the behaviour-vs-EKF-std cross-section.
_N_STD_BINS = 5


def _discover_arm_csvs(
    results_root: Path, leaf: Optional[str] = None, stage: Optional[str] = None
) -> Dict[str, Path]:
    """
    @brief Find each known ablation arm's episode_records.csv, preferring the
           without_wrapper variant, restricted to arms in _ARM_ORDER.
    """
    return {
        arm: path
        for arm, path in discover_arm_csvs(
            results_root,
            "episode_records.csv",
            prefer_variant="without_wrapper",
            leaf=leaf,
            stage=stage,
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


def drop_held_tiers(
    records: pd.DataFrame, conditions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    @brief Drop the held-tier conditions from any frame carrying a "condition" column.
    @param records: Any per-episode / per-step frame; passed through untouched if
           it has no "condition" column or is empty.
    @param conditions: Condition names to drop; None uses _HELD_TIER_CONDITIONS.
    @return A copy with those conditions removed.
    @note Applied at every load boundary so the held tiers cannot reach a summary
          or figure by any path; kept public so cross_seed.py filters identically.
    """
    if conditions is None:
        conditions = _HELD_TIER_CONDITIONS
    if not conditions or records.empty or "condition" not in records.columns:
        return records
    return records[~records["condition"].isin(conditions)].copy()


def drop_unreported(
    records: pd.DataFrame, conditions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    @brief Drop conditions outside the headline set from any "condition" frame.
    @param records: Any frame; passed through untouched if it has no "condition"
           column or is empty.
    @param conditions: Condition names to drop; None uses _UNREPORTED_CONDITIONS.
    @return A copy with those conditions removed.
    @note Applied at the FIGURE boundary, not the analysis boundary: the pooled
          CSVs keep every condition inspectable; figures show only the headline set.
    """
    if conditions is None:
        conditions = _UNREPORTED_CONDITIONS
    if not conditions or records.empty or "condition" not in records.columns:
        return records
    return records[~records["condition"].isin(conditions)].copy()


def keep_varying(
    records: pd.DataFrame, conditions: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    @brief Restrict a frame to the conditions whose GNSS tier varies in-episode.
    @param records: Any frame; passed through untouched if it has no "condition"
           column or is empty.
    @param conditions: Conditions to keep; None uses _VARYING_CONDITIONS.
    @return A copy holding only those conditions.
    @note A pinned tier gives no within-episode variation for the behaviour-vs-std
          and calibration readings to correlate against.
    """
    if conditions is None:
        conditions = _VARYING_CONDITIONS
    if not conditions or records.empty or "condition" not in records.columns:
        return records
    return records[records["condition"].isin(conditions)].copy()


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
    @return Tuple (point_estimate, ci_low, ci_high) at the 95% percentile level.
    @note Treatment and control are resampled independently - a two-sample
          difference, not paired, since the arms ran on separate episode draws.
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
    @return DataFrame with one row per (pair, condition): success-rate delta
            (pp) and position-error delta (m), each with a 95% CI and a
            "significant" flag (CI excludes zero).
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
    @note A smaller success drop and smaller position-error growth mean a more
          graceful degradation - the property the covariance arms are claimed
          to have. Both endpoint conditions must be present.
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


def _behaviour_by_std(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Final pos-error and approach speed per EKF-std bin, per arm.
    @note Per-episode finals, not a per-step trace - calibration.py covers that.
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


# Expected sign of Spearman(metric, EKF std) for a cautious policy: brake
# should rise (+1), speed / yaw-rate / jerk should fall (-1). The cross-arm
# DIFFERENCE in these slopes is the causal test for caution attributable to
# SEEING the covariance, not to the episode merely being harder.
_CAUTION_DIRECTION: List[Tuple[str, int]] = [
    ("mean_brake_cmd", +1),
    ("mean_speed_moving_ms", -1),
    ("mean_abs_vyaw_rads", -1),
    ("mean_action_jerk", -1),
]


def _caution_slopes(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-arm Spearman of each caution metric against EKF position std -
           the WITHIN-arm read; _caution_contrast is the cross-arm causal claim.
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
    @brief Per-arm absolute caution level (speed, brake) and its precision/
           success payoff, distinguishing cautious from merely undertrained.
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
    @brief Cross-arm caution-slope difference per contrast pair: the causal
           claim that seeing the covariance, not a harder episode, drives
           caution. Positive means the covariance arm is more cautious.
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
            # Multiply by exp so positive always means "treatment more cautious"
            # (for a "down" metric, exp=-1 flips a more-negative slope positive).
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
    stage: Optional[str] = None,
    keep_held_tiers: bool = False,
) -> None:
    """
    @brief Run the full cross-arm analysis and write tables + figures.
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory to write the CSV tables and PNG figures into.
    @param seed: Bootstrap RNG seed (reproducibility).
    @param slope_clean: Degradation-slope start condition (cleanest GNSS tier).
    @param slope_degraded: Degradation-slope end condition (worst GNSS tier).
    @param leaf: Optional checkpoint leaf to pin to (single-arm runs); None lets
           each arm's newest run win.
    @param stage: Optional curriculum stage (e.g. "1") to compare all arms at the
           same stage; None uses each arm's newest leaf (may mix stages).
    @param keep_held_tiers: Retain the held-tier conditions and the degradation
           slope they define; default False drops them (the five-condition set).
    """
    # Nest by stage (or pinned leaf) so STAGE=1 and STAGE=2 runs never overwrite
    # each other; an unpinned mixed-stage run lands in "latest".
    if stage is not None:
        out_dir = out_dir / f"stage{stage}"
    elif leaf is not None:
        out_dir = out_dir / leaf
    else:
        out_dir = out_dir / "latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_csvs = _discover_arm_csvs(results_root, leaf=leaf, stage=stage)
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
    # Drop the held tiers before ANY statistic is computed, so no table or figure
    # downstream can reintroduce them.
    if not keep_held_tiers:
        n_before = len(records)
        records = drop_held_tiers(records)
        print(
            f"\nDropped held-tier conditions "
            f"({', '.join(_HELD_TIER_CONDITIONS)}): "
            f"{n_before - len(records)} of {n_before} episodes removed; "
            f"{records['condition'].nunique()} conditions retained."
        )
    summary = _condition_summary(records)
    contrasts = _contrast_table(records, seed)
    # The slope is defined BY the two held tiers, so it exists only when they are
    # kept; an empty frame here is the correct result, not missing data.
    slope = (
        _degradation_slope(summary, slope_clean, slope_degraded)
        if keep_held_tiers
        else pd.DataFrame()
    )
    behaviour = _behaviour_by_std(records)
    caution_slopes = _caution_slopes(records)
    caution_contrast = _caution_contrast(caution_slopes)
    caution_levels = _caution_levels(records)

    summary.to_csv(out_dir / "condition_summary.csv", index=False)
    contrasts.to_csv(out_dir / "covariance_contrasts.csv", index=False)
    if not slope.empty:
        slope.to_csv(out_dir / "degradation_slope.csv", index=False)
    if not behaviour.empty:
        behaviour.to_csv(out_dir / "behaviour_by_std.csv", index=False)
    if not caution_slopes.empty:
        caution_slopes.to_csv(out_dir / "caution_slopes.csv", index=False)
    if not caution_contrast.empty:
        caution_contrast.to_csv(out_dir / "caution_contrast.csv", index=False)
    if not caution_levels.empty:
        caution_levels.to_csv(out_dir / "caution_levels.csv", index=False)

    _print_headline(contrasts, slope, slope_clean, slope_degraded, keep_held_tiers)
    _print_caution(caution_slopes, caution_contrast)
    _print_caution_levels(caution_levels)
    print(f"\nCSVs written to {out_dir}")


def _print_headline(
    contrasts: pd.DataFrame,
    slope: pd.DataFrame,
    slope_clean: str = _SLOPE_CLEAN_CONDITION,
    slope_degraded: str = _SLOPE_DEGRADED_CONDITION,
    held_tiers_kept: bool = True,
) -> None:
    """
    @brief Console summary of the two claims: covariance contrast + slope.
    @param held_tiers_kept: False reports an empty slope as deliberately
           skipped rather than as missing data.
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

    if slope.empty and not held_tiers_kept:
        # Deliberately dropped, not missing: say so rather than hinting at absent data.
        print(
            "\n=== GNSS degradation slope: SKIPPED (held tiers dropped) ===\n"
            "  The slope is defined by the held tiers; pass --keep-held-tiers to "
            "compute it."
        )
        return

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
    @brief Console summary of the caution-vs-uncertainty result: within-arm
           slopes, then the cross-arm causal contrast.
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
        pairs = ", ".join(f"{cov} vs {blind}" for _, cov, blind in _CONTRAST_PAIRS)
        print(
            f"  (need BOTH arms of a contrast pair: {pairs} - only then is caution\n"
            "   attributable to SEEING the covariance, not to the episode merely\n"
            "   being harder)"
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
    @brief Console summary of absolute caution LEVEL and its accuracy payoff -
           distinguishes a slower-but-more-accurate arm from an undertrained one.
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
        default="outputs/raw/evaluation_results",
        help="Root holding <baseline>/<leaf>/episode_records.csv for each arm.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/raw_derived/ablation_analysis",
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
        "--keep-held-tiers",
        action="store_true",
        help=(
            "Keep the held-tier conditions ("
            + ", ".join(_HELD_TIER_CONDITIONS)
            + ") and the degradation slope they define. Default drops them, "
            "matching the five headline conditions."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pin to one checkpoint leaf, e.g. 1_42_19062026-0120 "
        "(single-arm runs; default: each arm's newest run).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Compare all arms at this curriculum stage (e.g. 1), instead of "
        "each arm's newest leaf which may sit at different stages.",
    )
    args = parser.parse_args()
    analyse(
        Path(args.results_root),
        Path(args.output_dir),
        args.seed,
        args.slope_clean,
        args.slope_degraded,
        leaf=args.checkpoint,
        stage=args.stage,
        keep_held_tiers=args.keep_held_tiers,
    )


if __name__ == "__main__":
    main()
