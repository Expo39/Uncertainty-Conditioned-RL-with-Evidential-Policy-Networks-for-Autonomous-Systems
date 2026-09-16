"""
@file cross_seed.py
@brief Pool every seed's eval into one headline + a per-seed robustness table.

A cross-arm difference on one seed is indistinguishable from seed luck, so
this pools every seed into one sample alongside the per-arm mean/[min, max]
across seeds. Run via `make analyse-cross-seed`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Make the repo root importable so the shared helpers resolve when this file is
# run directly (python scripts/analysis/cross_seed.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import pandas as pd  # noqa: E402

from scripts.analysis._discovery import discover_records, seed_roots  # noqa: E402
from scripts.analysis.ablation import (  # noqa: E402
    _ARM_ORDER,
    _CONTRAST_PAIRS,
    _HELD_TIER_CONDITIONS,
    _behaviour_by_std,
    _caution_contrast,
    _caution_levels,
    _caution_slopes,
    _condition_summary,
    _contrast_table,
    _degradation_slope,
    _discover_arm_csvs,
    _load_records,
    _print_caution,
    _print_caution_levels,
    _print_headline,
    drop_held_tiers,
)
from scripts.analysis.calibration import (  # noqa: E402
    _add_combined_columns,
    _binned_table,
    _correlations,
)
from scripts.analysis.gate_roc import _evaluate_signals  # noqa: E402
from scripts.analysis.gate_roc import _load as _gate_load  # noqa: E402
from scripts.analysis.handover_timing import _REGIME, _latency  # noqa: E402

# Arms pinned for the single-CSV pools. The EKF is identical across arms, so
# any one characterises calibration; the handover wrapper only fires on an
# evidential head. Pinning keeps both reads reproducible.
CALIBRATION_ARM = "vanilla_ppo"
HANDOVER_ARM = "full_method"


def _seed_label(seed_root: Path) -> int:
    """
    @brief The integer seed of a seed_<N> sub-root (42 for ".../seed_42").
    @return The parsed seed, or -1 for an unlabelled root (pool-of-one fallback,
            kept visibly distinct in per-seed tables).
    """
    name = seed_root.name
    if name.startswith("seed_") and name[len("seed_") :].isdigit():
        return int(name[len("seed_") :])
    return -1


def _pool_episodes(results_root: Path, stage: str) -> pd.DataFrame:
    """
    @brief Concatenate every seed's per-arm episode_records into one tidy frame.
    @return Long-form frame with the ordered "arm" Categorical, a numeric
            "success" column, and an added "seed" column.
    @note The Categorical is re-asserted on the pooled frame so a seed missing
          an arm cannot silently change the dtype.
    """
    frames: List[pd.DataFrame] = []
    for seed_root in seed_roots(results_root):
        arm_csvs = _discover_arm_csvs(seed_root, stage=stage)
        if not arm_csvs:
            continue
        df = _load_records(arm_csvs)
        df["seed"] = _seed_label(seed_root)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No arm episode_records.csv under {results_root}/seed_*/ at stage "
            f"{stage}. Run make docker-eval for each BASELINE and seed first "
            "(expected seed_<N>/<baseline>/<leaf>/<variant>/episode_records.csv)."
        )
    pooled = pd.concat(frames, ignore_index=True)
    pooled["arm"] = pd.Categorical(pooled["arm"], categories=_ARM_ORDER, ordered=True)
    return pooled


def _pool_gate(results_root: Path, stage: str) -> pd.DataFrame:
    """
    @brief Pool the gate frame (handoff dropped, is_failure added) across seeds,
           via gate_roc._load per seed, ready for _evaluate_signals.
    """
    frames: List[pd.DataFrame] = []
    for seed_root in seed_roots(results_root):
        arm_csvs = _discover_arm_csvs(seed_root, stage=stage)
        if not arm_csvs:
            continue
        gframe = _gate_load(arm_csvs)
        gframe["seed"] = _seed_label(seed_root)
        frames.append(gframe)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _pool_single_csv(
    results_root: Path,
    name: str,
    prefer_variant: str,
    arm: Optional[str] = None,
    stage: Optional[str] = None,
) -> pd.DataFrame:
    """
    @brief Pool a per-run single CSV (not arm-mapped) across seeds, tagged by seed.
    @param arm: Baseline to pin to.
    @param stage: Curriculum stage to pin to.
    @return Concatenated frame with an added "seed" column; empty if none found.
    @warning arm and stage should both be pinned for any quoted quantity:
             discover_records otherwise falls back to whichever file has the
             newest mtime, so an unpinned read can silently change which run a
             reported statistic describes.
    """
    frames: List[pd.DataFrame] = []
    for seed_root in seed_roots(results_root):
        candidates = discover_records(
            seed_root, name, arm=arm, prefer_variant=prefer_variant, stage=stage
        )
        if not candidates:
            continue
        df = pd.read_csv(candidates[0])
        df["seed"] = _seed_label(seed_root)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _per_seed_summary(pooled: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per (arm, condition, seed) success rate, mean pos error and episode count.
    """
    grouped = pooled.groupby(["arm", "condition", "seed"], observed=True)
    return grouped.agg(
        success_rate=("success", lambda s: 100.0 * s.mean()),
        mean_pos_error_m=("final_pos_error_m", "mean"),
        n_episodes=("success", "size"),
    ).reset_index()


def _seed_robustness(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per (arm, condition) mean and [min, max] across seeds - the range
           check: an effect smaller than the range is not robust, whatever the
           pooled CI says.
    """
    grouped = per_seed.groupby(["arm", "condition"], observed=True)
    out = grouped.agg(
        n_seeds=("seed", "nunique"),
        success_mean_pct=("success_rate", "mean"),
        success_min_pct=("success_rate", "min"),
        success_max_pct=("success_rate", "max"),
        pos_error_mean_m=("mean_pos_error_m", "mean"),
        pos_error_min_m=("mean_pos_error_m", "min"),
        pos_error_max_m=("mean_pos_error_m", "max"),
    ).reset_index()
    out["success_range_pp"] = out["success_max_pct"] - out["success_min_pct"]
    out["pos_error_range_m"] = out["pos_error_max_m"] - out["pos_error_min_m"]
    return out


def _per_seed_contrasts(pooled: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Each covariance contrast's success/pos-error delta computed PER seed.
    @return Frame, one row per (pair, condition, seed): the treatment minus
            control success delta (pp) and position-error delta (m).
    @note No bootstrap here - a single seed's episode count does not warrant a
          within-seed CI. Feeds _seed_robustness_contrasts for the cross-seed
          mean/range, which can differ from the pooled delta since pooling
          weights by episode while this weights every seed equally.
    """
    rows: List[Dict[str, object]] = []
    seeds = sorted(pooled["seed"].unique())
    for seed in seeds:
        s_df = pooled[pooled["seed"] == seed]
        for pair_name, treat_arm, control_arm in _CONTRAST_PAIRS:
            for condition in s_df["condition"].unique():
                cond = s_df[s_df["condition"] == condition]
                t = cond[cond["arm"] == treat_arm]
                c = cond[cond["arm"] == control_arm]
                if t.empty or c.empty:
                    continue
                rows.append(
                    {
                        "pair": pair_name,
                        "treatment": treat_arm,
                        "control": control_arm,
                        "condition": condition,
                        "seed": int(seed),
                        "success_delta_pp": 100.0
                        * (t["success"].mean() - c["success"].mean()),
                        "pos_error_delta_m": (
                            t["final_pos_error_m"].mean()
                            - c["final_pos_error_m"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _seed_robustness_contrasts(per_seed_contrasts: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per (pair, condition) mean and [min, max] of the per-seed contrast
           delta, reported beside the pooled bootstrap delta.
    """
    if per_seed_contrasts.empty:
        return pd.DataFrame()
    grouped = per_seed_contrasts.groupby(["pair", "condition"], observed=True)
    out = grouped.agg(
        n_seeds=("seed", "nunique"),
        success_delta_mean_pp=("success_delta_pp", "mean"),
        success_delta_min_pp=("success_delta_pp", "min"),
        success_delta_max_pp=("success_delta_pp", "max"),
        pos_error_delta_mean_m=("pos_error_delta_m", "mean"),
        pos_error_delta_min_m=("pos_error_delta_m", "min"),
        pos_error_delta_max_m=("pos_error_delta_m", "max"),
    ).reset_index()
    return out


def _handover_table(pooled_wrapper: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Per-condition handover-timing table over the pooled with_wrapper frame.
    @return Per-condition regime / handoff fraction / latency table, or empty if
            the frame lacks the handoff_step column (no wrapper-on eval pooled).
    @note Mirrors handover_timing.analyse's inline aggregation (not exposed as a
          function there) on the pool, reusing the imported _latency and _REGIME.
    """
    if pooled_wrapper.empty or "handoff_step" not in pooled_wrapper.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for cond in sorted(pooled_wrapper["condition"].unique()):
        sub = pooled_wrapper[pooled_wrapper["condition"] == cond]
        regime = _REGIME.get(str(cond), "spawn")
        n = len(sub)
        n_handoff = int(sub["handoff_step"].notna().sum())
        latencies = sub.apply(lambda r: _latency(r, regime), axis=1).dropna()
        rows.append(
            {
                "condition": cond,
                "regime": regime,
                "n_episodes": n,
                "handoff_frac": round(n_handoff / n, 3) if n else float("nan"),
                "n_latency": int(len(latencies)),
                "median_latency_steps": (
                    float(latencies.median()) if len(latencies) else float("nan")
                ),
                "p90_latency_steps": (
                    float(latencies.quantile(0.9)) if len(latencies) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _print_robustness(robustness: pd.DataFrame, n_seeds: int) -> None:
    """
    @brief Console summary of the cross-seed success range and the seed-luck caveat.
    @param robustness: The per (arm, condition) robustness table.
    @param n_seeds: How many seeds were pooled (1 = pool of one; range is 0).
    """
    print("\n=== Cross-seed robustness: success mean and [min, max] over seeds ===")
    if robustness.empty:
        print("  (no pooled conditions)")
        return
    if n_seeds <= 1:
        print(f"  Only {n_seeds} seed present - range is 0; this is a pool of one.")
    for arm in [a for a in _ARM_ORDER if a in set(robustness["arm"].astype(str))]:
        a = robustness[robustness["arm"].astype(str) == arm]
        parts = "  ".join(
            f"{str(r['condition'])[:14]:14s}"
            f"{r['success_mean_pct']:5.1f}%[{r['success_min_pct']:4.0f},"
            f"{r['success_max_pct']:4.0f}]"
            for _, r in a.iterrows()
        )
        print(f"  {arm:20s} {parts}")
    print(
        "\n  Pooling per-episode records and bootstrapping the pool treats seed as a\n"
        "  fixed nuisance: the pooled CI reflects episode-level variance but UNDER-\n"
        "  represents between-seed variance. The [min, max] range above is the honest\n"
        "  cross-seed-stability check - an effect smaller than a rival's range is not\n"
        "  robust, whatever the pooled CI says."
    )


def analyse(
    results_root: Path,
    out_dir: Path,
    seed: int,
    stage: str,
    slope_clean: str,
    slope_degraded: str,
    keep_held_tiers: bool = False,
) -> None:
    """
    @brief Pool every seed, run the existing stats on the pool, write tables + figures.
    @param results_root: Root holding seed_<N>/<baseline>/<leaf>/... (the PARENT of
           the per-seed trees, e.g. outputs/raw/evaluation_results - NOT a seed_<N> dir).
    @param out_dir: Output root; results nest under all_seeds/stage<S>/.
    @param seed: Bootstrap RNG seed for the pooled contrast CIs (reproducibility,
           NOT an experiment seed - the experiment seeds are the pooled trees).
    @param stage: Curriculum stage every arm is pinned to (e.g. "6").
    @param slope_clean: Degradation-slope start condition (cleanest GNSS tier).
    @param slope_degraded: Degradation-slope end condition (worst GNSS tier).
    @param keep_held_tiers: Retain the held-tier conditions and the degradation
           slope they define; default False drops them from every pool.
    """
    out_dir = out_dir / "all_seeds" / f"stage{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pool the per-episode records ONCE; every ablation table + figure reads it.
    pooled = _pool_episodes(results_root, stage)
    # Drop the held tiers from the episode pool BEFORE any statistic is computed;
    # the gate and calibration pools are filtered at their own load points below,
    # so no figure can reintroduce them by a different path.
    if not keep_held_tiers:
        n_before = len(pooled)
        pooled = drop_held_tiers(pooled)
        print(
            f"Dropped held-tier conditions ({', '.join(_HELD_TIER_CONDITIONS)}): "
            f"{n_before - len(pooled)} of {n_before} episodes removed; "
            f"{pooled['condition'].nunique()} conditions retained."
        )
    n_seeds = int(pooled["seed"].nunique())
    seeds_present = sorted(int(s) for s in pooled["seed"].unique())
    print(f"Pooled {len(pooled)} episodes across {n_seeds} seed(s): {seeds_present}")
    for arm in _ARM_ORDER:
        if arm in set(pooled["arm"].astype(str)):
            n = int((pooled["arm"].astype(str) == arm).sum())
            print(f"  {arm:20s} {n} episodes pooled")
        else:
            print(f"  {arm:20s} (MISSING - contrast involving it is skipped)")

    # --- Pooled headline statistics (the EXISTING functions, on the larger pool).
    summary = _condition_summary(pooled)
    contrasts = _contrast_table(pooled, seed)
    # Defined BY the held tiers, so it exists only when they are kept.
    slope = (
        _degradation_slope(summary, slope_clean, slope_degraded)
        if keep_held_tiers
        else pd.DataFrame()
    )
    behaviour = _behaviour_by_std(pooled)
    caution_slopes = _caution_slopes(pooled)
    caution_contrast = _caution_contrast(caution_slopes)
    caution_levels = _caution_levels(pooled)

    # --- Per-seed robustness (mean / range), derived from the same pooled frame.
    per_seed = _per_seed_summary(pooled)
    robustness = _seed_robustness(per_seed)
    per_seed_contrasts = _per_seed_contrasts(pooled)
    robustness_contrasts = _seed_robustness_contrasts(per_seed_contrasts)

    # --- Pooled gate ROC AUC over every seed's episodes.
    gate_frame = _pool_gate(results_root, stage)
    if not keep_held_tiers:
        gate_frame = drop_held_tiers(gate_frame)
    gate_auc = _evaluate_signals(gate_frame) if not gate_frame.empty else pd.DataFrame()

    # --- Pooled calibration. The EKF is identical across arms, so one arm
    # characterises the filter; it is PINNED so the statistic cannot follow
    # whichever arm's file happens to be newest.
    calib = _pool_single_csv(
        results_root,
        "calibration_records.csv",
        "without_wrapper",
        arm=CALIBRATION_ARM,
        stage=stage,
    )
    if not keep_held_tiers:
        calib = drop_held_tiers(calib)
    calib_corr = pd.DataFrame()
    calib_binned = pd.DataFrame()
    if not calib.empty:
        calib = _add_combined_columns(calib)
        calib_corr = _correlations(calib)
        calib_binned = _binned_table(calib)

    # --- Pooled handover timing over the with_wrapper frame (EDL arms).
    handover_pool = _pool_single_csv(
        results_root,
        "episode_records.csv",
        "with_wrapper",
        arm=HANDOVER_ARM,
        stage=stage,
    )
    if not keep_held_tiers:
        handover_pool = drop_held_tiers(handover_pool)
    handover = _handover_table(handover_pool)

    # --- Write the pooled CSVs (cross_seed names them; prefixed "pooled_").
    summary.to_csv(out_dir / "pooled_condition_summary.csv", index=False)
    contrasts.to_csv(out_dir / "pooled_covariance_contrasts.csv", index=False)
    if not slope.empty:
        slope.to_csv(out_dir / "pooled_degradation_slope.csv", index=False)
    if not behaviour.empty:
        behaviour.to_csv(out_dir / "pooled_behaviour_by_std.csv", index=False)
    if not caution_slopes.empty:
        caution_slopes.to_csv(out_dir / "pooled_caution_slopes.csv", index=False)
    if not caution_contrast.empty:
        caution_contrast.to_csv(out_dir / "pooled_caution_contrast.csv", index=False)
    if not caution_levels.empty:
        caution_levels.to_csv(out_dir / "pooled_caution_levels.csv", index=False)
    if not gate_auc.empty:
        gate_auc.to_csv(out_dir / "pooled_gate_auc.csv", index=False)
    if not calib_corr.empty:
        calib_corr.to_csv(out_dir / "pooled_calibration_correlations.csv", index=False)
    if not calib_binned.empty:
        calib_binned.to_csv(out_dir / "pooled_calibration_binned.csv", index=False)
    if not handover.empty:
        handover.to_csv(out_dir / "pooled_handover_timing.csv", index=False)

    # --- Write the per-seed robustness CSVs (the mean/range the plan mandates).
    per_seed.to_csv(out_dir / "per_seed_summary.csv", index=False)
    robustness.to_csv(out_dir / "seed_robustness.csv", index=False)
    if not robustness_contrasts.empty:
        robustness_contrasts.to_csv(
            out_dir / "seed_robustness_contrasts.csv", index=False
        )

    # --- Console: the pooled headline, then the seed-robustness range + caveat.
    _print_headline(contrasts, slope, slope_clean, slope_degraded, keep_held_tiers)
    _print_caution(caution_slopes, caution_contrast)
    _print_caution_levels(caution_levels)
    _print_robustness(robustness, n_seeds)
    print(f"\nPooled CSVs written to {out_dir}")


def main() -> None:
    """
    @brief CLI entry point for the cross-seed pooled aggregator.
    """
    parser = argparse.ArgumentParser(
        description="Pool every seed's eval into pooled headline + per-seed robustness."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="outputs/raw/evaluation_results",
        help="PARENT root holding seed_<N>/<baseline>/<leaf>/... (not a seed_<N> dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/raw_derived/cross_seed_analysis",
        help="Output root; results nest under all_seeds/stage<S>/.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="6",
        help="Curriculum stage every arm is pinned to (the leg evals stage 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Bootstrap RNG seed for the pooled contrast CIs (reproducibility "
        "only - NOT an experiment seed; the experiment seeds are the pooled trees).",
    )
    parser.add_argument(
        "--slope-clean",
        type=str,
        default="gnss_fixed",
        help="Degradation-slope start condition (cleanest held GNSS tier).",
    )
    parser.add_argument(
        "--slope-degraded",
        type=str,
        default="gnss_degraded",
        help="Degradation-slope end condition (worst held GNSS tier).",
    )
    parser.add_argument(
        "--keep-held-tiers",
        action="store_true",
        help=(
            "Keep the held-tier conditions ("
            + ", ".join(_HELD_TIER_CONDITIONS)
            + ") and the degradation slope they define. Default drops them, "
            "matching the five reported conditions."
        ),
    )
    args = parser.parse_args()
    analyse(
        Path(args.results_root),
        Path(args.output_dir),
        args.seed,
        args.stage,
        args.slope_clean,
        args.slope_degraded,
        keep_held_tiers=args.keep_held_tiers,
    )


if __name__ == "__main__":
    main()
