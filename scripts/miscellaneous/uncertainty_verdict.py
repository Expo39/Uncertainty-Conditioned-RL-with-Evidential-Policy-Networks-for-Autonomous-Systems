"""
@file uncertainty_verdict.py
@brief One-shot verdict on whether the epistemic/aleatoric ratio varies with state.

Reads an eval run's per-step uncertainty trace (per_step_records.csv, written when
EVAL_PER_STEP_CAP > 0 for an evidential head) and reports, per condition, the epi/ale
ratio and implied nu. A FLAT ratio across clean vs novel/degraded conditions confirms the
single-head NIG conflation (epistemic = aleatoric/nu with nu state-independent) - the two
channels are one signal, as expected for this architecture. See the detailed note below.
For the safety controller, threshold the TOTAL uncertainty, not the (non-separating) split.

Read-only diagnostic. Run via `make uncertainty-verdict EVAL_DIR=<run output dir>`.
@see uncertainty_rl/evaluation/evaluate.py (per_step_records.csv), scripts/miscellaneous/CLAUDE.md.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Conditions grouped by whether they leave the training data manifold. CLEAN are
# in-distribution (clean GNSS / deployment anchor); HARD are the novelty / degradation
# conditions where epistemic SHOULD be relatively higher if the signal is state-dependent.
# Names match configs/eval_config.yaml; unlisted conditions are reported but not grouped.
_CLEAN = {"gnss_fixed", "anchor_empty", "anchor_deployment"}
_HARD = {
    "gnss_degraded",
    "lidar_degraded",
    "ood_irregular_rtk_fixed",
    "gnss_degrade_one_way",
}


def _load(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    @brief Load per-step (epistemic, aleatoric) pairs grouped by condition.
    @param path: Path to per_step_records.csv.
    @return Mapping condition -> list of (epistemic, aleatoric) tuples.
    """
    by_cond: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                epi = float(row["epistemic"])
                ale = float(row["aleatoric"])
            except (KeyError, ValueError):
                continue
            by_cond[row["condition"]].append((epi, ale))
    return by_cond


def _stats(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    @brief Summary statistics for one condition's (epistemic, aleatoric) pairs.
    @param pairs: List of (epistemic, aleatoric) per-step tuples.
    @return Dict with mean epistemic / aleatoric, epi/ale ratio, frac(epi>ale), implied nu.
    """
    n = len(pairs)
    if n == 0:
        return {
            "n": 0,
            "epi": 0.0,
            "ale": 0.0,
            "ratio": 0.0,
            "frac_epi_gt": 0.0,
            "nu": 0.0,
        }
    epi = sum(p[0] for p in pairs) / n
    ale = sum(p[1] for p in pairs) / n
    ratio = epi / ale if ale > 0 else 0.0
    frac = sum(1 for e, a in pairs if e > a) / n
    # Implied nu = aleatoric / epistemic (epistemic = aleatoric / nu); <1 = epistemic-dominant.
    nu = ale / epi if epi > 0 else 0.0
    return {
        "n": n,
        "epi": epi,
        "ale": ale,
        "ratio": ratio,
        "frac_epi_gt": frac,
        "nu": nu,
    }


def main() -> int:
    """
    @brief CLI entry point: print per-condition uncertainty stats and a separation verdict.
    @return Process exit code (0 always; this is a report, not a gate).
    """
    parser = argparse.ArgumentParser(
        description="Verdict on epistemic-vs-aleatoric separation from an eval run."
    )
    parser.add_argument(
        "eval_dir",
        type=str,
        help="Eval run output dir containing per_step_records.csv "
        "(e.g. outputs/evaluation_results/<baseline>/<leaf>/without_wrapper).",
    )
    args = parser.parse_args()

    csv_path = Path(args.eval_dir) / "per_step_records.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        print(
            "Run eval with EVAL_PER_STEP_CAP > 0 on an evidential head to produce it.",
            file=sys.stderr,
        )
        return 1

    by_cond = _load(csv_path)
    if not by_cond:
        print(f"ERROR: no usable rows in {csv_path}.", file=sys.stderr)
        return 1

    print(f"Per-step uncertainty by condition ({csv_path}):\n")
    header = f"{'condition':28s} {'n':>6} {'epi':>8} {'ale':>8} {'epi/ale':>8} {'nu':>7} {'epi>ale%':>9}"
    print(header)
    print("-" * len(header))
    clean_ratios: List[float] = []
    hard_ratios: List[float] = []
    for cond in sorted(by_cond):
        s = _stats(by_cond[cond])
        tag = "clean" if cond in _CLEAN else ("HARD" if cond in _HARD else "")
        print(
            f"{cond:28s} {s['n']:6d} {s['epi']:8.4f} {s['ale']:8.4f} "
            f"{s['ratio']:8.3f} {s['nu']:7.2f} {100 * s['frac_epi_gt']:8.1f}%  {tag}"
        )
        if cond in _CLEAN:
            clean_ratios.append(s["ratio"])
        elif cond in _HARD:
            hard_ratios.append(s["ratio"])

    print("\n--- Verdict ---")
    if not clean_ratios or not hard_ratios:
        print(
            "Not enough grouped conditions to judge separation "
            "(need at least one CLEAN and one HARD condition)."
        )
        return 0

    clean_mean = sum(clean_ratios) / len(clean_ratios)
    hard_mean = sum(hard_ratios) / len(hard_ratios)
    lift = hard_mean / clean_mean if clean_mean > 0 else 0.0
    print(f"  mean epi/ale on CLEAN conditions = {clean_mean:.3f}")
    print(f"  mean epi/ale on HARD  conditions = {hard_mean:.3f}")
    print(f"  HARD/CLEAN lift = {lift:.2f}x")
    # A useful handoff signal needs epistemic to rise RELATIVELY on hard conditions.
    # The thresholds are deliberately loose - this is a directional read, not a gate.
    if lift >= 1.5:
        print("  -> GOOD: epistemic separates - it rises on novel/degraded states.")
    elif lift >= 1.1:
        print("  -> WEAK: some separation, but the margin is small.")
    else:
        print("  -> FLAT: epistemic does NOT separate (one signal). Expected on the")
        print("     single-head NIG actor: epistemic = aleatoric/nu and RL leaves nu")
        print(
            "     unsupervised, so the two channels stay a fixed ratio. Not tunable -"
        )
        print(
            "     see documentation/detailed_notes/epistemic_aleatoric_disentanglement.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
