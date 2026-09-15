"""
@file _discovery.py
@brief Shared locator for per-run eval CSVs under the nested results tree.

evaluate.py writes each run to
<results_root>/<baseline>/<leaf>/<wrapper_variant>/<name>.csv, where
wrapper_variant is "without_wrapper" or "with_wrapper" (see
uncertainty_rl/evaluation/evaluate.py). Older runs wrote one level shallower
(<baseline>/<leaf>/<name>.csv). The analysis scripts (calibration, gate_roc,
ablation) all need to find these CSVs, map each back to its arm
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


def arm_leaf_subpath(csv_path: Path, results_root: Path) -> Path:
    """
    @brief The <baseline>/<leaf> sub-path of a discovered per-run CSV.
    @param csv_path: Path to a matched per-run CSV.
    @param results_root: The evaluation_results root the glob ran from.
    @return Path("<baseline>/<leaf>") so a per-arm analysis can mirror the eval's
            nesting in its own output dir. Falls back to Path() if the CSV does
            not sit under results_root.

    Per-arm analyses (calibration, handover timing) write under
    <output_dir>/<baseline>/<leaf>/ so re-running on a different arm or checkpoint
    never overwrites a previous result - mirroring how the evals themselves nest.
    The leaf is the SECOND component (the wrapper variant, if present, is dropped
    so without_/with_wrapper analyses of the same run share a leaf dir).
    """
    try:
        rel = csv_path.relative_to(results_root)
    except ValueError:
        return Path()
    parts = rel.parts
    if len(parts) >= 2:
        return Path(parts[0]) / parts[1]
    return Path(parts[0]) if parts else Path()


def seed_roots(results_root: Path) -> List[Path]:
    """
    @brief The per-seed sub-roots under an output root, for cross-seed pooling.
    @param results_root: A root that may CONTAIN seed_<N>/ children, e.g.
           outputs/evaluation_results (whose per-seed trees are
           outputs/evaluation_results/seed_42/<baseline>/<leaf>/...).
    @return Sorted list of the seed_<N> child directories. If the root has no
            seed_*/ children (a single tree with no seed nesting, or a root
            already pinned to one seed), the singleton [results_root] is returned
            so a caller pools "one seed" transparently (pool of one).

    The cross-seed aggregator loops this and calls the existing per-seed discovery
    (discover_records / discover_arm_csvs) on each returned root, reading the seed
    LABEL from the directory name rather than re-parsing a CSV path - so the seed
    is always known from the loop, never recovered from a flat path list.
    """
    seeds = sorted(p for p in results_root.glob("seed_*") if p.is_dir())
    return seeds if seeds else [results_root]


def _variant_rank(csv_path: Path, preferred: Optional[str] = None) -> int:
    """
    @brief Sort key ranking the preferred wrapper variant first, others last.
    @param csv_path: Path to a matched per-run CSV.
    @param preferred: Variant to rank first ("without_wrapper" by default, or
           "with_wrapper" for analyses like handover timing that need the wrapper
           to have fired). None uses the module default order.
    @return Index into the (reordered) variant preference list; len() otherwise.

    The variant is the CSV's parent directory name when the three-level layout
    is in use; for the legacy two-level layout it is the leaf (never a known
    variant), so it ranks last - which is correct, there is nothing to prefer.
    """
    order = _VARIANT_PREFERENCE
    if preferred in _VARIANT_PREFERENCE:
        order = [preferred] + [v for v in _VARIANT_PREFERENCE if v != preferred]
    parent = csv_path.parent.name
    return (
        order.index(parent)
        if parent in _VARIANT_PREFERENCE
        else len(_VARIANT_PREFERENCE)
    )


def _leaf_stage(leaf_name: str) -> Optional[str]:
    """
    @brief Curriculum stage of a run leaf, from its <stage>_<seed>_<timestamp> name.
    @param leaf_name: The run directory name, e.g. "2_42_20062026-0451".
    @return The leading stage component ("2"), or None if the name has no "_".
    """
    head = leaf_name.split("_", 1)[0]
    return head if head else None


def discover_records(
    results_root: Path,
    name: str,
    arm: Optional[str] = None,
    prefer_variant: Optional[str] = None,
    leaf: Optional[str] = None,
    stage: Optional[str] = None,
) -> List[Path]:
    """
    @brief Find per-run CSVs of a given name, preferred variant then newest first.
    @param results_root: outputs/evaluation_results.
    @param name: CSV file name to match (e.g. "calibration_records.csv").
    @param arm: Optional baseline name to restrict to; None = any arm.
    @param prefer_variant: Wrapper variant to return first. Default
           "without_wrapper" (free-running policy - uncertainty/behaviour reads);
           pass "with_wrapper" for handover-timing, which needs the wrapper fired.
    @param leaf: Optional checkpoint leaf (run dir) to PIN to, e.g.
           "1_42_19062026-0120". When given, only that run's CSVs match - so
           analysis never silently reads a different checkpoint when several
           exist. When None, every leaf matches and the newest mtime wins.
    @param stage: Optional curriculum stage to restrict to (e.g. "1"), matched
           against the leaf's leading <stage>_ component. Keeps a cross-arm read
           apples-to-apples when arms sit at different stages; None = any stage.
    @return Matching paths sorted by (variant preference, mtime descending).
            Empty if none match.

    Globs the two-level and three-level layouts and concatenates them. The
    variant preference dominates the sort so the preferred variant is returned
    ahead of the other even if the other is marginally newer.
    """
    base = arm if arm else "*"
    run = leaf if leaf else "*"
    patterns = [f"{base}/{run}/{name}", f"{base}/{run}/*/{name}"]
    seen: Dict[Path, None] = {}
    for pattern in patterns:
        for path in results_root.glob(pattern):
            seen[path] = None
    matches = list(seen)
    if stage is not None:
        # Leaf is the component directly under the arm dir (parts[1] below root).
        matches = [
            p
            for p in matches
            if _leaf_stage(p.relative_to(results_root).parts[1]) == str(stage)
        ]
    return sorted(
        matches, key=lambda p: (_variant_rank(p, prefer_variant), -p.stat().st_mtime)
    )


def discover_arm_csvs(
    results_root: Path,
    name: str,
    prefer_variant: Optional[str] = None,
    leaf: Optional[str] = None,
    stage: Optional[str] = None,
) -> Dict[str, Path]:
    """
    @brief Map each arm to its best per-run CSV of the given name.
    @param results_root: outputs/evaluation_results.
    @param name: CSV file name to match (e.g. "episode_records.csv").
    @param prefer_variant: Wrapper variant to prefer per arm (see discover_records).
    @param leaf: Optional checkpoint leaf to pin to. Cross-arm callers (ablation)
           usually leave this None, since each arm has its own leaf; pin only when
           analysing a single arm's specific run.
    @param stage: Optional curriculum stage to restrict to (e.g. "1"), so a
           cross-arm read compares arms at the SAME stage instead of each arm's
           newest leaf (which may sit at different stages); None = any stage.
    @return Mapping arm name -> chosen CSV path (preferred variant, then most
            recent). One entry per arm.

    For each arm the first hit in discover_records order wins, so the preferred
    variant is chosen when present.
    """
    chosen: Dict[str, Path] = {}
    for path in discover_records(
        results_root, name, prefer_variant=prefer_variant, leaf=leaf, stage=stage
    ):
        arm = _arm_of(path, results_root)
        if arm is None or arm in chosen:
            continue
        chosen[arm] = path
    return chosen
