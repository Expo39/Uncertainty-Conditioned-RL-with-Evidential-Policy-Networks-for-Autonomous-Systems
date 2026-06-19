"""
@file _discovery.py
@brief Shared locator for per-run eval CSVs under the nested results tree.

evaluate.py writes each run to
<results_root>/<baseline>/<leaf>/<wrapper_variant>/<name>.csv, where
wrapper_variant is "without_wrapper" or "with_wrapper" (see
uncertainty_rl/evaluation/evaluate.py). Older runs wrote one level shallower
(<baseline>/<leaf>/<name>.csv). The analysis scripts (calibration, gate_roc,
ablation_analyser) all need to find these CSVs, map each back to its arm
(baseline) name, and - for any uncertainty/behaviour reading - prefer the
without_wrapper variant, since the wrapper caps throttle and forces stops,
corrupting the free-running signal. This module is the single source of truth
for that matching so a future layout change is a one-file fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

# Wrapper-variant directory names written by evaluate.py. Order is preference:
# without_wrapper first so uncertainty/behaviour analyses read the free-running
# policy, not the intervention-shaped one.
_VARIANT_PREFERENCE: List[str] = ["without_wrapper", "with_wrapper"]


def _arm_of(csv_path: Path, results_root: Path) -> Optional[str]:
    """
    @brief Recover the arm (baseline) name for a discovered CSV.
    @param csv_path: Path to a matched per-run CSV.
    @param results_root: The evaluation_results root the glob ran from.
    @return The baseline directory name (immediate child of results_root), or
            None if the path does not sit under results_root.

    The arm is always the FIRST path component under results_root regardless of
    how many levels deep the CSV is, so this is robust to both the two-level
    (legacy) and three-level (wrapper-variant) layouts.
    """
    try:
        rel = csv_path.relative_to(results_root)
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def _variant_rank(csv_path: Path) -> int:
    """
    @brief Sort key putting without_wrapper ahead of with_wrapper, others last.
    @param csv_path: Path to a matched per-run CSV.
    @return Index into the variant preference list; len() for anything else.

    The variant is the CSV's parent directory name when the three-level layout
    is in use; for the legacy two-level layout it is the leaf (never a known
    variant), so it ranks last - which is correct, there is nothing to prefer.
    """
    parent = csv_path.parent.name
    return (
        _VARIANT_PREFERENCE.index(parent)
        if parent in _VARIANT_PREFERENCE
        else len(_VARIANT_PREFERENCE)
    )


def discover_records(
    results_root: Path, name: str, arm: Optional[str] = None
) -> List[Path]:
    """
    @brief Find per-run CSVs of a given name, newest and without_wrapper first.
    @param results_root: outputs/evaluation_results.
    @param name: CSV file name to match (e.g. "calibration_records.csv").
    @param arm: Optional baseline name to restrict to; None = any arm.
    @return Matching paths sorted by (variant preference, mtime descending).
            Empty if none match.

    Globs the two-level and three-level layouts and concatenates them. The
    variant preference dominates the sort so a without_wrapper CSV is returned
    ahead of a with_wrapper one even if the latter is marginally newer.
    """
    base = arm if arm else "*"
    patterns = [f"{base}/*/{name}", f"{base}/*/*/{name}"]
    seen: Dict[Path, None] = {}
    for pattern in patterns:
        for path in results_root.glob(pattern):
            seen[path] = None
    return sorted(
        seen, key=lambda p: (_variant_rank(p), -p.stat().st_mtime)
    )


def discover_arm_csvs(results_root: Path, name: str) -> Dict[str, Path]:
    """
    @brief Map each arm to its best per-run CSV of the given name.
    @param results_root: outputs/evaluation_results.
    @param name: CSV file name to match (e.g. "episode_records.csv").
    @return Mapping arm name -> chosen CSV path (without_wrapper preferred, then
            most recent). One entry per arm.

    For each arm the first hit in discover_records order wins, so the
    without_wrapper variant is chosen when present.
    """
    chosen: Dict[str, Path] = {}
    for path in discover_records(results_root, name):
        arm = _arm_of(path, results_root)
        if arm is None or arm in chosen:
            continue
        chosen[arm] = path
    return chosen
