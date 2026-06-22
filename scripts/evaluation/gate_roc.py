"""
@file gate_roc.py
@brief Compare an EKF-std safety gate against an evidential-epistemic gate.

Host-side, read-only diagnostic that asks the safety question behind the
ablation: if the vehicle aborts (hands off) when a scalar uncertainty signal
exceeds a threshold, which signal separates the failures from the successes
better? Two candidate signals, both already logged per episode in
episode_records.csv:

  - EKF position std (ekf_std_pos_max_m): available to EVERY arm, since the EKF
    always runs. This is the gate a covariance-blind system could still build.
  - Evidential epistemic (max_epistemic): available only to evidential arms
    (output_uncertainty, full_method). This is the policy's own confidence.

For each signal we sweep the abort threshold and trace the trade-off between
correctly aborting before a failure (collision / out_of_bounds / near_miss /
stuck) and needlessly aborting an episode that would have succeeded. The area
under that curve (AUC) is the single comparison number: a higher-AUC signal is
the better safety gate. Pure pandas / numpy / matplotlib on the host .venv.
Run via `make analyse-gate` (never python directly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/evaluation/gate_roc.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.evaluation._discovery import discover_arm_csvs  # noqa: E402

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
    @param results_root: outputs/evaluation_results (nested <baseline>/<leaf>).
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


def _plot_roc(records: pd.DataFrame, out_dir: Path) -> None:
    """
    @brief Overlay the gate ROC curves for each (arm, signal) on one axis.
    @param records: Tidy per-episode frame.
    @param out_dir: Directory for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
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
            fpr, tpr, auc = _roc_curve(scores, labels)
            if np.isnan(auc):
                continue
            ax.plot(fpr, tpr, label=f"{arm} / {signal} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False abort rate (successes needlessly aborted)")
    ax.set_ylabel("Failure catch rate (failures pre-empted)")
    ax.set_title("Safety-gate ROC: EKF std vs evidential epistemic")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "gate_roc.png", dpi=150)
    plt.close(fig)


def analyse(results_root: Path, out_dir: Path, stage: Optional[str] = None) -> None:
    """
    @brief Run the gate comparison and write the AUC table + ROC figure.
    @param results_root: outputs/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory for the CSV table and PNG figure.
    @param stage: Optional curriculum stage (e.g. "1") to compare all arms at the
           same stage; None uses each arm's newest leaf (may mix stages).
    """
    # Nest by stage so STAGE=1 and STAGE=2 runs never overwrite; unpinned in "latest".
    out_dir = out_dir / (f"stage{stage}" if stage is not None else "latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_csvs = _discover_arm_csvs(results_root, stage=stage)
    records = _load(arm_csvs)
    auc_table = _evaluate_signals(records)
    auc_table.to_csv(out_dir / "gate_auc.csv", index=False)
    _plot_roc(records, out_dir)

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
    print(f"\nTable and figure written to {out_dir}")


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
        default="outputs/evaluation_results",
        help="Root holding <baseline>/<leaf>/episode_records.csv for each arm.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gate_analysis",
        help="Directory for the AUC table and ROC figure.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Compare all arms at this curriculum stage (e.g. 1), instead of "
        "each arm's newest leaf which may sit at different stages.",
    )
    args = parser.parse_args()
    analyse(Path(args.results_root), Path(args.output_dir), stage=args.stage)


if __name__ == "__main__":
    main()
