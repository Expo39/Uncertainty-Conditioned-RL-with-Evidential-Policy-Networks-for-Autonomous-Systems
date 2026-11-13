"""
@file seed_level.py
@brief Seed-level inference for the 3x2 ablation: two-level bootstrap intervals,
       exact seed permutation tests, brake correlations and parking precision.

The episode-level bootstrap in ablation.py treats every episode as independent,
which understates between-seed variance. Here each arm draws its seeds with
replacement, then the episodes of each drawn seed, so an interval spans both
sources. Arms are resampled independently. Run via `make analyse-seed-level`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis._discovery import pooled_frame  # noqa: E402
from scripts.analysis.ablation import (  # noqa: E402
    _ARM_ORDER,
    _CONTRAST_PAIRS,
    _VARYING_CONDITIONS,
)

# Seed of the pre-registered analysis, so every interval is reproducible.
BOOTSTRAP_SEED = 20260927
N_BOOTSTRAP = 10000

# An estimate is a signed combination of per-arm success rates.
Estimate = Dict[str, float]


def _covariance_estimates() -> Dict[str, Estimate]:
    """
    @brief Covariance effect per head, their interactions, and the head effects.
    @return Mapping estimate name -> {arm: weight}.
    @note Interactions are differences of covariance contrasts; the head effects
          compare arms that differ only in the actor head.
    """
    contrasts = {
        f"covariance_{label}": {cov: 1.0, blind: -1.0}
        for label, cov, blind in _CONTRAST_PAIRS
    }
    std = contrasts["covariance_standard_head"]
    het = contrasts["covariance_heteroscedastic_head"]
    evid = contrasts["covariance_evidential_head"]
    return {
        **contrasts,
        "interaction_heteroscedastic_minus_standard": _combine(het, std),
        "interaction_evidential_minus_heteroscedastic": _combine(evid, het),
        "interaction_evidential_minus_standard": _combine(evid, std),
        "full_method_minus_heteroscedastic_input": {
            "full_method": 1.0,
            "heteroscedastic_input": -1.0,
        },
        "heteroscedastic_minus_vanilla_ppo": {
            "heteroscedastic": 1.0,
            "vanilla_ppo": -1.0,
        },
    }


def _combine(plus: Estimate, minus: Estimate) -> Estimate:
    """
    @brief Difference of two estimates, plus - minus.
    @param plus: Estimate taken with a positive sign.
    @param minus: Estimate taken with a negative sign.
    @return Combined arm weights.
    """
    out = dict(plus)
    for arm, weight in minus.items():
        out[arm] = out.get(arm, 0.0) - weight
    return out


def _two_level_means(
    groups: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_resamples: int,
    pool_episodes: bool = False,
) -> np.ndarray:
    """
    @brief Two-level bootstrap of a mean: seeds with replacement, then episodes.
    @param groups: One value array per seed.
    @param rng: Random generator, shared so the whole analysis is one stream.
    @param n_resamples: Number of bootstrap resamples.
    @param pool_episodes: If True, weight each drawn seed by its episode count
           (the mean over pooled episodes); otherwise average the seed means.
    @return Resampled means, shape (n_resamples,).
    @note Every drawn seed slot gets its own episode resample, so a seed drawn
          twice contributes two independent resamples.
    """
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.full(n_resamples, np.nan)
    k = len(groups)
    sizes = np.array([len(g) for g in groups], dtype=float)
    drawn = rng.integers(0, k, size=(n_resamples, k))
    rows = np.arange(n_resamples)
    total = np.zeros(n_resamples)
    weight = np.zeros(n_resamples)
    for j in range(k):
        means = np.column_stack(
            [
                g[rng.integers(0, len(g), size=(n_resamples, len(g)))].mean(1)
                for g in groups
            ]
        )
        w = sizes[drawn[:, j]] if pool_episodes else np.ones(n_resamples)
        total += w * means[rows, drawn[:, j]]
        weight += w
    return total / weight


def _seed_groups(
    records: pd.DataFrame, arm: str, seeds: Sequence[str], column: str
) -> List[np.ndarray]:
    """
    @brief Per-seed value arrays of one column for one arm.
    @param records: Episode records with arm and seed columns.
    @param arm: Arm name.
    @param seeds: Seed names, in a fixed order.
    @param column: Column to extract.
    @return One float array per seed (empty when that seed has no rows).
    """
    block = records[records["arm"] == arm]
    return [
        block.loc[block["seed"] == seed, column].to_numpy(dtype=float) for seed in seeds
    ]


def _estimate_intervals(
    records: pd.DataFrame,
    seeds: Sequence[str],
    estimates: Mapping[str, Estimate],
    rng: np.random.Generator,
    n_resamples: int,
) -> pd.DataFrame:
    """
    @brief Point estimate and 95% two-level interval of each success contrast.
    @param records: Episode records restricted to the varying conditions.
    @param seeds: Seed names.
    @param estimates: Mapping estimate name -> arm weights.
    @param rng: Random generator.
    @param n_resamples: Number of bootstrap resamples.
    @return One row per condition x estimate, in percentage points. Estimates
            needing an arm with no records are skipped.
    """
    present = set(records["arm"])
    rows: List[Dict[str, object]] = []
    for condition, block in records.groupby("condition", sort=False):
        point: Dict[str, float] = {}
        draws: Dict[str, np.ndarray] = {}
        for arm in (a for a in _ARM_ORDER if a in present):
            groups = _seed_groups(block, arm, seeds, "success")
            point[arm] = float(np.mean([g.mean() for g in groups if len(g)]))
            draws[arm] = _two_level_means(groups, rng, n_resamples)
        for name, weights in estimates.items():
            if not set(weights) <= set(draws):
                continue
            est = sum(w * point[a] for a, w in weights.items())
            dist = sum(w * draws[a] for a, w in weights.items())
            lo, hi = np.percentile(dist, [2.5, 97.5])
            rows.append(
                {
                    "condition": condition,
                    "estimate": name,
                    "point_pp": 100.0 * est,
                    "ci_low_pp": 100.0 * lo,
                    "ci_high_pp": 100.0 * hi,
                    "excludes_zero": bool(lo > 0.0 or hi < 0.0),
                }
            )
    return pd.DataFrame(rows)


def _permutation_tests(per_seed: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Exact one-sided seed-level permutation test for each covariance pair.
    @param per_seed: Success rate indexed by (condition, arm), one column per seed.
    @return One row per condition x pair: observed difference, separation flag
            and p over all C(2n, n) seed arrangements.
    @note With three seeds per arm the smallest attainable p is 1/20 = 0.05.
    """
    rows: List[Dict[str, object]] = []
    for condition in per_seed.index.get_level_values("condition").unique():
        for label, cov, blind in _CONTRAST_PAIRS:
            if (condition, cov) not in per_seed.index:
                continue
            if (condition, blind) not in per_seed.index:
                continue
            a = per_seed.loc[(condition, cov)].dropna().to_numpy()
            b = per_seed.loc[(condition, blind)].dropna().to_numpy()
            observed = a.mean() - b.mean()
            pool = np.concatenate([a, b])
            arrangements = list(itertools.combinations(range(len(pool)), len(a)))
            hits = sum(
                pool[list(c)].mean() - np.delete(pool, list(c)).mean()
                >= observed - 1e-12
                for c in arrangements
            )
            rows.append(
                {
                    "condition": condition,
                    "pair": label,
                    "observed_pp": 100.0 * observed,
                    "min_cov_above_max_blind": bool(a.min() > b.max()),
                    "hits": int(hits),
                    "arrangements": len(arrangements),
                    "p_one_sided": hits / len(arrangements),
                }
            )
    return pd.DataFrame(rows)


def _brake_correlations(records: pd.DataFrame, seeds: Sequence[str]) -> pd.DataFrame:
    """
    @brief Spearman of brake command against EKF position std, per seed and pooled.
    @param records: Episode records restricted to the varying conditions.
    @param seeds: Seed names.
    @return One row per arm: the within-condition correlation averaged over the
            varying conditions, per seed and over the pooled seeds.
    """

    def _mean_within(block: pd.DataFrame) -> float:
        # Spearman via ranks, so no scipy dependency (as in ablation.py).
        values = [
            float(part["ekf_std_pos_mean_m"].rank().corr(part["mean_brake_cmd"].rank()))
            for _, part in block.groupby("condition")
        ]
        values = [v for v in values if v == v]
        return float(np.mean(values)) if values else float("nan")

    rows: List[Dict[str, object]] = []
    for arm in (a for a in _ARM_ORDER if a in set(records["arm"])):
        block = records[records["arm"] == arm]
        row: Dict[str, object] = {"arm": arm}
        for seed in seeds:
            row[seed] = _mean_within(block[block["seed"] == seed])
        row["pooled"] = _mean_within(block)
        rows.append(row)
    return pd.DataFrame(rows)


def _parking_precision(
    records: pd.DataFrame,
    seeds: Sequence[str],
    rng: np.random.Generator,
    n_resamples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    @brief Final position error and decisions to park over successful episodes.
    @param records: Episode records restricted to the varying conditions.
    @param seeds: Seed names.
    @param rng: Random generator.
    @param n_resamples: Number of bootstrap resamples.
    @return Tuple of (per-arm summary, covariance-pair differences with
            two-level 95% intervals over the pooled successful episodes).
    """
    parked = records[records["success"] == 1]
    summary = (
        parked.groupby("arm", observed=True)
        .agg(
            n=("success", "size"),
            median_err_m=("final_pos_error_m", "median"),
            mean_err_m=("final_pos_error_m", "mean"),
            decisions=("steps", "mean"),
        )
        .reset_index()
    )
    rows: List[Dict[str, object]] = []
    present = set(parked["arm"])
    for label, cov, blind in _CONTRAST_PAIRS:
        if cov not in present or blind not in present:
            continue
        for column in ("final_pos_error_m", "steps"):
            diff = _two_level_means(
                _seed_groups(parked, cov, seeds, column),
                rng,
                n_resamples,
                pool_episodes=True,
            ) - _two_level_means(
                _seed_groups(parked, blind, seeds, column),
                rng,
                n_resamples,
                pool_episodes=True,
            )
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append(
                {
                    "pair": label,
                    "metric": column,
                    "mean_diff": float(diff.mean()),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
    return summary, pd.DataFrame(rows)


def analyse(
    results_root: Path,
    output_dir: Path,
    seeds: Sequence[str],
    stage: str,
    bootstrap_seed: int,
    n_resamples: int,
) -> None:
    """
    @brief Run the seed-level analysis and write its CSVs.
    @param results_root: outputs/raw/evaluation_results (the seed-nested parent).
    @param output_dir: Directory for the output CSVs.
    @param seeds: Seed sub-root names, in a fixed order for reproducibility.
    @param stage: Curriculum stage to read.
    @param bootstrap_seed: Seed of the single bootstrap random stream.
    @param n_resamples: Number of bootstrap resamples.
    """
    records = pooled_frame(
        results_root, seeds, _ARM_ORDER, "episode_records.csv", stage=stage
    )
    records = records[records["condition"].isin(_VARYING_CONDITIONS)]
    missing = [a for a in _ARM_ORDER if a not in set(records["arm"])]
    if missing:
        print(f"WARNING: no records for {missing}; estimates needing them skipped.")

    rng = np.random.default_rng(bootstrap_seed)
    per_seed = records.groupby(["condition", "arm", "seed"]).success.mean().unstack()
    pooled = records.groupby(["condition", "arm"]).success.mean().unstack() * 100.0
    pooled = pooled.reindex(columns=[a for a in _ARM_ORDER if a in pooled.columns])
    intervals = _estimate_intervals(
        records, seeds, _covariance_estimates(), rng, n_resamples
    )
    permutation = _permutation_tests(per_seed)
    brake = _brake_correlations(records, seeds)
    precision, precision_diff = _parking_precision(records, seeds, rng, n_resamples)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pooled_success.csv": pooled.reset_index(),
        "per_seed_success.csv": (per_seed * 100.0).reset_index(),
        "estimates.csv": intervals,
        "permutation.csv": permutation,
        "brake_spearman.csv": brake,
        "parking_precision.csv": precision,
        "parking_precision_contrasts.csv": precision_diff,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    print("Pooled success (%):\n", pooled.round(1).to_string())
    primary = intervals[intervals["estimate"].str.startswith("covariance_")]
    print("\nCovariance contrasts, two-level 95% intervals (pp):")
    for _, r in primary.iterrows():
        print(
            f"  {r['condition']:22s} {r['estimate']:34s} {r['point_pp']:+6.1f} "
            f"[{r['ci_low_pp']:+6.1f}, {r['ci_high_pp']:+6.1f}]"
        )
    print(f"\nWrote {len(outputs)} tables to {output_dir}")


def main() -> None:
    """
    @brief CLI entry point.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--results-root", type=Path, default=Path("outputs/raw/evaluation_results")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/raw_derived/seed_level")
    )
    parser.add_argument("--stage", type=str, default="6")
    parser.add_argument("--seeds", nargs="+", default=["seed_42", "seed_123", "seed_7"])
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    analyse(
        args.results_root,
        args.output_dir,
        args.seeds,
        args.stage,
        args.bootstrap_seed,
        args.n_bootstrap,
    )


if __name__ == "__main__":
    main()
