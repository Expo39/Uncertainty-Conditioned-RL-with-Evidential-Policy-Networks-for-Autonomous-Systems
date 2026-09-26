"""
@file gate_roc.py
@brief Compare an EKF-std safety gate against an evidential-epistemic gate.

Which separates failures from successes better as a handoff threshold: EKF
position std (every arm) or evidential epistemic (evidential arms only)?
Sweeps the threshold, reports failure-catch vs false-abort AUC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/analysis/gate_roc.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis._discovery import discover_arm_csvs  # noqa: E402
from scripts.analysis.ablation import _N_BOOTSTRAP, drop_held_tiers  # noqa: E402

# Resamples per bootstrap chunk. Bounds the (chunk x n_failures) weight matrix
# so the interval costs a few tens of MB at the pooled episode counts.
_BOOTSTRAP_CHUNK = 500

# Outcomes the gate SHOULD pre-empt (a handoff before these is the desired
# behaviour). success is the only non-failure; "handoff" episodes already
# aborted and are excluded so we score the signal, not the policy's own gate.
_FAILURE_OUTCOMES = {"collision", "out_of_bounds", "near_miss", "stuck"}

# Candidate gate signals: column -> the arms it is defined for. EKF std exists
# for every arm; epistemic only for evidential heads.
_SIGNALS: Dict[str, Optional[List[str]]] = {
    "ekf_std_pos_max_m": None,  # None = all arms
    "max_epistemic": ["output_uncertainty", "full_method"],
}


def _discover_arm_csvs(
    results_root: Path, stage: Optional[str] = None
) -> Dict[str, Path]:
    """
    @brief Find each arm's episode_records.csv (without_wrapper preferred).
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param stage: Optional curriculum stage (e.g. "1") so the gate comparison uses
           arms at the SAME stage rather than each arm's newest leaf (which may sit
           at different stages); None = any stage.
    @return Mapping arm name -> episode_records.csv path.

    Handles both the two-level (legacy) and three-level (wrapper-variant)
    layouts; per arm the without_wrapper variant wins, then the newest leaf.
    """
    return discover_arm_csvs(
        results_root,
        "episode_records.csv",
        prefer_variant="without_wrapper",
        stage=stage,
    )


def _load(arm_csvs: Dict[str, Path]) -> pd.DataFrame:
    """
    @brief Concatenate arms into one frame with an "arm" column and "is_failure".
    @param arm_csvs: Mapping arm name -> episode_records.csv path.
    @return Tidy per-episode frame. Episodes whose own outcome is "handoff" are
            dropped (the policy already aborted; the signal cannot be scored
            against a ground-truth success/fail).
    """
    frames: List[pd.DataFrame] = []
    for arm, path in arm_csvs.items():
        df = pd.read_csv(path)
        df["arm"] = arm
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No episode_records.csv found under the root.")
    records = pd.concat(frames, ignore_index=True)
    records = records[records["outcome"] != "handoff"].copy()
    records["is_failure"] = records["outcome"].isin(_FAILURE_OUTCOMES).astype(int)
    return records


def _roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    @brief ROC of an abort-threshold sweep: catch failures vs false aborts.
    @param scores: Per-episode signal value (higher = more uncertain).
    @param labels: 1 for a failure episode, 0 for a success.
    @return Tuple (false_abort_rate, failure_catch_rate, auc). false_abort_rate
            is the fraction of true successes aborted (aborting them is the cost);
            failure_catch_rate is the fraction of failures the gate fires before
            (the benefit). AUC is the trapezoidal area, ranked-pairs equivalent.

    A failure is "caught" when its signal exceeds the threshold (we would have
    aborted). A success is a "false abort" when its signal also exceeds it.
    Sweeping every observed score as a threshold traces the full curve.
    """
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-scores)  # high score (abort) first
    sorted_labels = labels[order]
    tpr = np.concatenate(([0.0], np.cumsum(sorted_labels) / n_pos))
    fpr = np.concatenate(([0.0], np.cumsum(1 - sorted_labels) / n_neg))
    # Trapezoidal area under the ROC, computed directly so the script needs
    # neither np.trapz (removed in NumPy 2.x) nor sklearn.
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def _pair_wins(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    @brief Failure-vs-success pairwise win matrix of a gate signal.
    @param scores: Per-episode signal value (higher = more uncertain).
    @param labels: 1 for a failure episode, 0 for a success.
    @return Matrix of shape (n_failures, n_successes): 1 where the failure
            scores higher, 0.5 on a tie, 0 otherwise. Empty when either class
            is absent. NaN scores are dropped first.

    Its mean is the Mann-Whitney AUC with ties averaged, which a pinned signal
    (the EKF std sitting at its floor) needs: the trapezoidal sweep in
    _roc_curve depends on how argsort happens to order tied scores.
    """
    mask = ~np.isnan(scores)
    scores, labels = scores[mask], labels[mask]
    failures = scores[labels == 1]
    successes = scores[labels == 0]
    diff = failures[:, None] - successes[None, :]
    return (diff > 0).astype(float) + 0.5 * (diff == 0)


def _bootstrap_counts(n: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """
    @brief How often each of n items is drawn in each bootstrap resample.
    @param n: Items in the sample.
    @param n_resamples: Number of resamples.
    @param rng: Random generator.
    @return Integer matrix of shape (n_resamples, n) whose rows each sum to n.
    """
    draws = rng.integers(0, n, size=(n_resamples, n))
    offsets = n * np.arange(n_resamples)[:, None]
    flat = np.bincount((draws + offsets).ravel(), minlength=n_resamples * n)
    return flat.reshape(n_resamples, n)


def _bootstrap_auc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    """
    @brief Tie-aware rank AUC with a stratified bootstrap interval.
    @param scores: Per-episode signal value (higher = more uncertain).
    @param labels: 1 for a failure episode, 0 for a success.
    @param rng: Random generator; seed it for a reproducible interval.
    @return Tuple (auc, ci_low, ci_high) at the 95% percentile level; all NaN
            when either class is absent.
    @note Failures and successes are resampled separately, so every resample
          keeps the observed class counts and the AUC stays defined. An
          unstratified draw could empty the rarer class on a small condition.

    A resampled AUC is the pairwise win matrix weighted by how often each
    failure and each success was drawn, so the pairs are compared once and each
    resample reduces to a matrix product rather than a fresh sort.
    """
    wins = _pair_wins(scores, labels)
    n_failures, n_successes = wins.shape
    if n_failures == 0 or n_successes == 0:
        return float("nan"), float("nan"), float("nan")

    point = float(wins.mean())
    aucs = np.empty(_N_BOOTSTRAP)
    for start in range(0, _N_BOOTSTRAP, _BOOTSTRAP_CHUNK):
        size = min(_BOOTSTRAP_CHUNK, _N_BOOTSTRAP - start)
        fail_w = _bootstrap_counts(n_failures, size, rng)
        succ_w = _bootstrap_counts(n_successes, size, rng)
        weighted = np.sum((fail_w @ wins) * succ_w, axis=1)
        aucs[start : start + size] = weighted / (n_failures * n_successes)
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)


def _evaluate_signals(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Compute the gate ROC AUC for every (arm, signal) pair.
    @param records: Tidy per-episode frame from _load.
    @return DataFrame with arm, signal, auc, n_failures, n_success.

    A signal restricted to evidential arms is skipped for the others. EKF std is
    scored per arm so the covariance-blind arms (vanilla, input) get an EKF-std
    gate row too - the apples-to-apples baseline for the epistemic gate.
    """
    rows: List[Dict[str, object]] = []
    for arm in sorted(records["arm"].unique()):
        arm_df = records[records["arm"] == arm]
        labels = arm_df["is_failure"].to_numpy()
        for signal, allowed in _SIGNALS.items():
            if allowed is not None and arm not in allowed:
                continue
            if signal not in arm_df.columns:
                continue
            scores = arm_df[signal].to_numpy(dtype=float)
            if np.isnan(scores).all():
                continue
            _, _, auc = _roc_curve(scores, labels)
            rows.append(
                {
                    "arm": arm,
                    "signal": signal,
                    "auc": auc,
                    "n_failures": int(labels.sum()),
                    "n_success": int((1 - labels).sum()),
                }
            )
    return pd.DataFrame(rows)


def analyse(
    results_root: Path,
    out_dir: Path,
    stage: Optional[str] = None,
    keep_held_tiers: bool = False,
) -> None:
    """
    @brief Run the gate comparison and write the AUC table + ROC figure.
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory for the CSV table and PNG figure.
    @param stage: Optional curriculum stage (e.g. "1") to compare all arms at the
           same stage; None uses each arm's newest leaf (may mix stages).
    @param keep_held_tiers: Retain the held-tier conditions. Default False drops
           them so the ROC is scored over the five reported conditions.
    """
    # Nest by stage so STAGE=1 and STAGE=2 runs never overwrite; unpinned in "latest".
    out_dir = out_dir / (f"stage{stage}" if stage is not None else "latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_csvs = _discover_arm_csvs(results_root, stage=stage)
    records = _load(arm_csvs)
    if not keep_held_tiers:
        records = drop_held_tiers(records)
    auc_table = _evaluate_signals(records)
    auc_table.to_csv(out_dir / "gate_auc.csv", index=False)

    print("=== Safety-gate AUC (higher = better failure/success separation) ===")
    if auc_table.empty:
        print("  (no scorable signals - need failures AND successes per arm)")
    else:
        for _, r in auc_table.iterrows():
            print(
                f"  {str(r['arm']):20s} {str(r['signal']):20s} "
                f"AUC {r['auc']:.3f}  "
                f"(fail {r['n_failures']}, ok {r['n_success']})"
            )
        print(
            "\n  Compare ekf_std_pos_max_m vs max_epistemic on the SAME evidential "
            "arm:\n  a higher epistemic AUC means the policy's own confidence is the "
            "better\n  abort signal than the EKF covariance gate a blind system could "
            "build."
        )
    print(f"\nTable written to {out_dir}")


def main() -> None:
    """
    @brief CLI entry point for the safety-gate ROC comparison.
    """
    parser = argparse.ArgumentParser(
        description="Compare EKF-std vs evidential-epistemic safety gates."
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
        default="outputs/raw_derived/gate_analysis",
        help="Directory for the AUC table and ROC figure.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Compare all arms at this curriculum stage (e.g. 1), instead of "
        "each arm's newest leaf which may sit at different stages.",
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
        stage=args.stage,
        keep_held_tiers=args.keep_held_tiers,
    )


if __name__ == "__main__":
    main()
