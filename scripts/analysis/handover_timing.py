"""
@file handover_timing.py
@brief When does the safety wrapper hand over, relative to the degradation onset?

Host-side, read-only diagnostic that turns the per-episode handover-timing columns
(handoff_step, degraded_onset_step, written by uncertainty_rl/evaluation/evaluate.py)
into a per-condition latency table. The claim is about TIMING, not rate: a useful
uncertainty-conditioned controller hands over soon after conditions degrade. The
reference point for "soon after" differs by condition, so each is tagged with an
onset regime and latency is only ever compared within a regime:

  - spawn regime: the condition is degraded/novel from episode start (gnss_degraded,
    lidar_degraded, OOD layout). Latency = handoff_step (steps from spawn). Reads as
    "how fast does the controller react to this standing condition?".
  - switch regime: the condition starts clean and the GNSS Markov chain drifts into
    the degraded tier mid-episode (gnss_degrade_one_way). Latency =
    handoff_step - degraded_onset_step (steps AFTER the drift crossing). This is the
    causal money shot: does the handover track the onset, not just the map?
  - none regime: clean/in-distribution conditions where no handover is expected; a
    LOW handover fraction here is the desired (low false-positive) result.

A spawn latency and a switch latency are different quantities and are never averaged
together. Run via `make handover-timing` (never python directly).
@see uncertainty_rl/evaluation/evaluate.py (episode_records.csv handoff_step columns).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Make the repo root importable so the shared discovery helper resolves when this
# file is run directly (python scripts/analysis/handover_timing.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from scripts.analysis._discovery import arm_leaf_subpath, discover_records  # noqa: E402

# Per-condition onset regime. switch = degradation arrives mid-episode (latency is
# measured after the drift crossing); spawn = degraded/novel from step 0 (latency
# from spawn); none = clean, no handover expected. Conditions absent from this map
# default to "spawn" - the safe reading for any new standing condition. Names match
# configs/eval_config.yaml.
_REGIME: Dict[str, str] = {
    "gnss_degrade_one_way": "switch",
    "gnss_degraded": "spawn",
    "lidar_degraded": "spawn",
    "ood_irregular_rtk_fixed": "spawn",
    "gnss_fixed": "none",
    "anchor_empty": "none",
    "anchor_deployment": "none",
}


def _latency(row: pd.Series, regime: str) -> float:
    """
    @brief Handover latency for one episode against its onset regime.
    @param row: One episode_records row (handoff_step, degraded_onset_step).
    @param regime: The condition's onset regime (switch / spawn / none).
    @return Steps from the regime's onset to the handover, or NaN if no handover
            fired (or, in the switch regime, the drift never reached degraded, or
            the handover preceded the crossing - not an onset-driven handover).
    """
    handoff = row.get("handoff_step", float("nan"))
    if pd.isna(handoff):
        return float("nan")
    if regime == "switch":
        onset = row.get("degraded_onset_step", float("nan"))
        if pd.isna(onset):
            return float("nan")
        latency = float(handoff) - float(onset)
        # A handover before the crossing is not onset-driven; exclude it from the
        # switch-latency summary (it is still counted in the handover fraction).
        return latency if latency >= 0.0 else float("nan")
    # spawn / none: steps from episode start.
    return float(handoff)


def analyse(
    results_root: Path, out_dir: Path, arm: Optional[str], leaf: Optional[str]
) -> None:
    """
    @brief Build the per-condition handover-timing table and write it.
    @param results_root: outputs/raw/evaluation_results (nested <baseline>/<leaf>).
    @param out_dir: Directory for the CSV table.
    @param arm: Optional baseline name to restrict to; None = most recent any-arm.
    @param leaf: Optional checkpoint leaf to pin to; None = newest run wins.
    """
    # Handovers only fire with the wrapper ON, so prefer the with_wrapper run.
    candidates = discover_records(
        results_root,
        "episode_records.csv",
        arm,
        prefer_variant="with_wrapper",
        leaf=leaf,
    )
    if not candidates:
        scope = "/".join(p for p in (arm, leaf) if p)
        raise FileNotFoundError(
            f"No episode_records.csv under {results_root}/{scope}. "
            f"Run make docker-eval (wrapper ON) first."
        )
    csv_path = candidates[0]
    # Nest under <baseline>/<leaf> (mirroring the eval) so a different arm or
    # checkpoint never overwrites a previous handover-timing result.
    out_dir = out_dir / arm_leaf_subpath(csv_path, results_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Handover-timing source: {csv_path}")
    df = pd.read_csv(csv_path)

    if "handoff_step" not in df.columns:
        raise KeyError(
            "episode_records.csv has no handoff_step column - re-run eval with "
            "the updated evaluate.py (and the SafetyWrapper ON, NO_SAFETY unset)."
        )

    rows: List[Dict[str, object]] = []
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
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
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "handover_timing.csv", index=False)

    print("\n=== Handover timing by condition ===")
    print(
        f"  {'condition':24s} {'regime':7s} {'handoff%':>9s} "
        f"{'med lat':>9s} {'p90 lat':>9s} {'n_lat':>6s}"
    )
    print("  " + "-" * 68)
    for _, r in table.iterrows():
        med = r["median_latency_steps"]
        p90 = r["p90_latency_steps"]
        med_s = f"{med:9.1f}" if pd.notna(med) else f"{'-':>9s}"
        p90_s = f"{p90:9.1f}" if pd.notna(p90) else f"{'-':>9s}"
        print(
            f"  {str(r['condition']):24s} {str(r['regime']):7s} "
            f"{100 * r['handoff_frac']:8.1f}% {med_s} {p90_s} {int(r['n_latency']):6d}"
        )
    print(
        "\n  switch regime: latency is steps AFTER the GNSS drift reaches degraded.\n"
        "  spawn regime: latency is steps from episode start (standing condition).\n"
        "  none regime: low handoff% is the desired low-false-positive result.\n"
        "  Latencies are comparable WITHIN a regime only."
    )
    print(f"\nTable written to {out_dir}")


def main() -> None:
    """
    @brief CLI entry point for the handover-timing analysis.
    """
    parser = argparse.ArgumentParser(
        description="Per-condition handover timing relative to degradation onset."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="outputs/raw/evaluation_results",
        help="Root holding <baseline>/<leaf>/episode_records.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/raw_derived/handover_timing",
        help="Directory for the handover-timing table.",
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
    args = parser.parse_args()
    analyse(Path(args.results_root), Path(args.output_dir), args.arm, args.checkpoint)


if __name__ == "__main__":
    main()
