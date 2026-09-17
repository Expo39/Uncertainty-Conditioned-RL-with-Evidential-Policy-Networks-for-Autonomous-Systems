"""
@file calibration.py
@brief Is the EKF covariance fed to the policy an HONEST uncertainty signal?

Does the EKF's PREDICTED std track its ACTUAL error (ground truth minus EKF
estimate)? Reports the std-vs-error correlation overall and per condition.
Run via `make analyse-calibration`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/analysis/calibration.py), which puts the
# script's own directory on sys.path rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import pandas as pd  # noqa: E402

from scripts.analysis._discovery import arm_leaf_subpath, discover_records  # noqa: E402
from scripts.analysis.ablation import drop_held_tiers  # noqa: E402

# Std/error axis pairs to analyse: the predicted-std column against its matched
# actual-error column. Position uses the combined magnitudes.
_AXES: List[Tuple[str, str, str]] = [
    ("position", "std_pos", "abs_err_pos"),
    ("heading", "std_yaw", "abs_err_yaw"),
]

# Number of quantile bins for the error-vs-std monotonicity table.
_N_BINS = 5

# Conditions where the GNSS tier MOVES within the episode (Markov drift or the
# one-way good->bad drift): only a varying std gives a real calibration test,
# so held-tier conditions (std pinned all episode) are reported but excluded
# from the verdict. Names match configs/eval_config.yaml.
_VARYING_CONDITIONS = {
    "anchor_deployment",
    "anchor_empty",
    "gnss_degrade_one_way",
}

# Minimum std spread (p90 - p10, metres) for a condition's correlation to count
# as a real test. Below this the std is effectively constant (a held tier), so
# the correlation is flagged weak and kept out of the headline verdict.
_MIN_STD_SPREAD_M = 0.01


def _find_calibration_csv(
    results_root: Path, arm: Optional[str], leaf: Optional[str]
) -> Path:
    """
    @brief Locate a calibration_records.csv under the nested results tree.
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param arm: Optional baseline name to restrict to; None = any arm.
    @param leaf: Optional checkpoint leaf to pin to; None = newest run wins.
    @return Path to the most recently modified matching calibration_records.csv.

    evaluate.py writes <baseline>/<leaf>/<wrapper_variant>/calibration_records.csv
    (three levels). Older runs wrote <baseline>/<leaf>/ (two levels). Match both,
    then prefer the without_wrapper variant - the wrapper caps throttle and so
    corrupts the very uncertainty/error signal this analysis reads.
    """
    candidates = discover_records(
        results_root, "calibration_records.csv", arm, leaf=leaf
    )
    if not candidates:
        scope = "/".join(p for p in (arm, leaf) if p)
        raise FileNotFoundError(
            f"No calibration_records.csv under {results_root}/{scope}. "
            f"Re-run make docker-eval to generate it."
        )
    return candidates[0]


def _add_combined_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Add the combined position std column the analysis keys on.
    @param df: Raw calibration records (std_x, std_y, std_yaw, abs_err_*).
    @return The frame with std_pos (mean of std_x, std_y) added.

    The policy observes std_x and std_y separately; the actual error is a planar
    magnitude (abs_err_pos), so the matched predicted scalar is the mean of the
    two axis stds - the EKF's predicted planar position spread.
    """
    df = df.copy()
    df["std_pos"] = df[["std_x", "std_y"]].mean(axis=1)
    return df


def _correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Pearson and Spearman correlation of predicted std vs actual error.
    @param df: Calibration records with the combined columns added.
    @return DataFrame: one row per (scope, axis) with pearson, spearman, n,
            group (varying / held), and std_spread (p90 - p10 of the std).

    Spearman (rank) is the headline - monotone "more std => more error" without
    assuming linearity. Computed per condition plus a pooled "varying" scope over
    ONLY the conditions whose tier moves within the episode (_VARYING_CONDITIONS):
    a within-condition correlation needs the std to actually vary, which a held
    tier (pinned all episode) does not provide. The std_spread column exposes
    that directly - a near-zero spread means the row is a held tier and its
    correlation is not a real test.
    """
    rows = []
    varying_df = df[df["condition"].isin(_VARYING_CONDITIONS)]
    # Pooled scope is the VARYING conditions only - the genuine calibration test.
    scopes = [("varying_pooled", varying_df)] + [
        (str(c), df[df["condition"] == c]) for c in sorted(df["condition"].unique())
    ]
    for scope, sub in scopes:
        is_varying = scope == "varying_pooled" or scope in _VARYING_CONDITIONS
        for axis, std_col, err_col in _AXES:
            pair = sub[[std_col, err_col]].dropna()
            if len(pair) < 3 or pair[std_col].nunique() < 2:
                continue
            pearson = float(pair[std_col].corr(pair[err_col], method="pearson"))
            # Spearman = Pearson on ranks; computed this way so the script does
            # not pull in scipy (pandas method="spearman" requires it, and scipy
            # is absent from both the host .venv and the training container).
            spearman = float(
                pair[std_col].rank().corr(pair[err_col].rank(), method="pearson")
            )
            std_spread = float(
                pair[std_col].quantile(0.9) - pair[std_col].quantile(0.1)
            )
            rows.append(
                {
                    "scope": scope,
                    "axis": axis,
                    "group": "varying" if is_varying else "held",
                    "pearson": pearson,
                    "spearman": spearman,
                    "std_spread": std_spread,
                    "n": int(len(pair)),
                }
            )
    return pd.DataFrame(rows)


def _binned_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Mean actual error per quantile bin of predicted std (monotonicity).
    @param df: Calibration records with the combined columns added.
    @return DataFrame: per (axis, bin) the std range and mean actual error.

    A monotone rise in mean error across std bins is the plain-language statement
    of calibration: when the EKF says it is more uncertain, it is in fact more
    wrong, on average.
    """
    rows = []
    for axis, std_col, err_col in _AXES:
        pair = df[[std_col, err_col]].dropna()
        if len(pair) < _N_BINS:
            continue
        try:
            bins = pd.qcut(pair[std_col], _N_BINS, duplicates="drop")
        except ValueError:
            continue
        grouped = pair.groupby(bins, observed=True)
        for i, (interval, g) in enumerate(grouped):
            rows.append(
                {
                    "axis": axis,
                    "bin": i,
                    "std_low": float(interval.left),
                    "std_high": float(interval.right),
                    "mean_abs_error": float(g[err_col].mean()),
                    "n": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def analyse(
    results_root: Path,
    out_dir: Path,
    arm: Optional[str],
    leaf: Optional[str],
    keep_held_tiers: bool = False,
) -> None:
    """
    @brief Run the EKF calibration analysis and write the tables + figure.
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory for the CSV tables and PNG figure.
    @param arm: Optional baseline name to restrict the source CSV to.
    @param leaf: Optional checkpoint leaf to pin to; None = newest run wins.
    @param keep_held_tiers: Retain the held-tier conditions. Default False drops
           them, so the per-condition rows and the scatter cover only the five
           reported conditions. The pooled "varying" verdict is
           unaffected either way: _VARYING_CONDITIONS already excludes the held
           tiers, which is why the headline correlation does not move.
    """
    csv_path = _find_calibration_csv(results_root, arm, leaf)
    # Nest the output under <baseline>/<leaf> (mirroring the eval) so a different
    # arm or checkpoint never overwrites a previous calibration result.
    out_dir = out_dir / arm_leaf_subpath(csv_path, results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration source: {csv_path}")
    df = _add_combined_columns(pd.read_csv(csv_path))
    if not keep_held_tiers:
        df = drop_held_tiers(df)

    corr = _correlations(df)
    binned = _binned_table(df)
    corr.to_csv(out_dir / "calibration_correlations.csv", index=False)
    binned.to_csv(out_dir / "calibration_binned.csv", index=False)

    print("\n=== Predicted std vs actual error correlation (Spearman is headline) ===")
    if corr.empty:
        print("  (insufficient variation to correlate)")
    else:

        def _print_row(r: "pd.Series") -> None:
            flag = (
                "" if r["std_spread"] >= _MIN_STD_SPREAD_M else "  (std pinned - weak)"
            )
            print(
                f"  {str(r['scope']):20s} {str(r['axis']):9s} "
                f"spearman {r['spearman']:+.3f}  pearson {r['pearson']:+.3f}  "
                f"spread {r['std_spread']:.3f}  (n={int(r['n'])}){flag}"
            )

        print("\n  -- VARYING conditions (std moves within episode - the real test) --")
        for _, r in corr[corr["group"] == "varying"].iterrows():
            _print_row(r)
        print(
            "\n  -- HELD conditions (std pinned all episode - weak, not in verdict) --"
        )
        for _, r in corr[corr["group"] == "held"].iterrows():
            _print_row(r)

        # Verdict from the pooled VARYING position correlation only - the held
        # tiers cannot test calibration (no std spread to correlate against).
        verdict = corr[
            (corr["scope"] == "varying_pooled") & (corr["axis"] == "position")
        ]
        print("\n  -- Verdict (VARYING position only) --")
        if verdict.empty or verdict.iloc[0]["std_spread"] < _MIN_STD_SPREAD_M:
            print(
                "  No varying-condition position signal with real std spread - "
                "re-run\n  the FULL eval (anchors + one_way), not just the held "
                "endpoints."
            )
        else:
            s = float(verdict.iloc[0]["spearman"])
            tag = (
                "HONEST (conditioning justified)"
                if s >= 0.2
                else "WEAK/ABSENT - investigate before claiming"
            )
            print(f"  pooled VARYING position spearman = {s:+.3f} -> {tag}")

    print("\n=== Mean actual error per predicted-std bin (monotone = calibrated) ===")
    if binned.empty:
        print("  (insufficient data to bin)")
    else:
        for axis in binned["axis"].unique():
            b = binned[binned["axis"] == axis]
            trail = "  ".join(
                f"[{r.std_low:.3f}-{r.std_high:.3f}]:{r.mean_abs_error:.3f}"
                for r in b.itertuples()
            )
            print(f"  {axis:9s} {trail}")
    print(f"\nTables written to {out_dir}")


def main() -> None:
    """
    @brief CLI entry point for the EKF calibration analysis.
    """
    parser = argparse.ArgumentParser(
        description="Check whether the EKF covariance is an honest uncertainty signal."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="outputs/raw/evaluation_results",
        help="Root holding <baseline>/<leaf>/calibration_records.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/raw_derived/calibration_analysis",
        help="Directory for the calibration tables and figure.",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help="Restrict to one baseline's CSV (default: any, most recent).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pin to one checkpoint leaf, e.g. 1_42_19062026-0120 "
        "(default: newest run for the arm).",
    )
    parser.add_argument(
        "--keep-held-tiers",
        action="store_true",
        help="Keep the held-tier conditions (gnss_fixed, gnss_degraded). Default "
        "drops them, matching the five reported conditions.",
    )
    args = parser.parse_args()
    analyse(
        Path(args.results_root),
        Path(args.output_dir),
        args.arm,
        args.checkpoint,
        keep_held_tiers=args.keep_held_tiers,
    )


if __name__ == "__main__":
    main()
