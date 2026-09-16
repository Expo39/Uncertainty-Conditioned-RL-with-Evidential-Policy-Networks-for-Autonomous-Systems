"""
@file _discovery.py
@brief Shared locator for per-run eval CSVs under the nested results tree.

evaluate.py writes each run to
<results_root>/<baseline>/<leaf>/<wrapper_variant>/<name>.csv (older runs wrote
one level shallower, <baseline>/<leaf>/<name>.csv). For uncertainty/behaviour
reads, without_wrapper is preferred over with_wrapper since the wrapper caps
throttle and forces stops, corrupting the free-running signal. Single source
of truth for this matching so a future layout change is a one-file fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    import pandas as pd

# Wrapper-variant directory names written by evaluate.py. Order is preference:
# without_wrapper first so uncertainty/behaviour analyses read the free-running
# policy, not the intervention-shaped one.
_VARIANT_PREFERENCE: List[str] = ["without_wrapper", "with_wrapper"]


def _arm_of(csv_path: Path, results_root: Path) -> Optional[str]:
    """
    @brief Arm (baseline) name for a discovered CSV: the first path component
           under results_root, valid for both the two- and three-level layouts.
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
    @return Path("<baseline>/<leaf>"), so per-arm output mirrors the eval's own
            nesting; Path() if the CSV does not sit under results_root.
    @note The wrapper-variant component, if present, is dropped so without_/
          with_wrapper analyses of the same run share one leaf dir.
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
           outputs/raw/evaluation_results/seed_42/<baseline>/<leaf>/...
    @return Sorted seed_<N> child directories, or [results_root] if none exist
            (so a caller pools "one seed" transparently).
    """
    seeds = sorted(p for p in results_root.glob("seed_*") if p.is_dir())
    return seeds if seeds else [results_root]


def _variant_rank(csv_path: Path, preferred: Optional[str] = None) -> int:
    """
    @brief Sort key ranking the preferred wrapper variant first, others last.
    @note Legacy two-level paths have no variant parent dir, so they rank last
          by falling through to len(_VARIANT_PREFERENCE) - correctly, since
          there is nothing to prefer.
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
    @param results_root: outputs/raw/evaluation_results.
    @param name: CSV file name to match (e.g. "calibration_records.csv").
    @param arm: Optional baseline name to restrict to; None = any arm.
    @param prefer_variant: Variant to rank first; default "without_wrapper".
    @param leaf: Optional checkpoint leaf to pin to; None matches any leaf.
    @param stage: Optional curriculum stage to restrict to; None = any stage.
    @return Matching paths sorted by (variant preference, mtime descending).
    @note Globs both the two-level and three-level layouts; variant preference
          dominates the sort, so it wins even over a marginally newer file.
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
    @param results_root: outputs/raw/evaluation_results.
    @param name: CSV file name to match (e.g. "episode_records.csv").
    @param prefer_variant: Variant to prefer per arm (see discover_records).
    @param leaf: Optional checkpoint leaf to pin to; None leaves each arm free
           to use its own leaf.
    @param stage: Optional curriculum stage, so a cross-arm read compares arms
           at the same stage rather than each arm's own newest leaf.
    @return Mapping arm name -> chosen CSV path. One entry per arm, taken as the
            first hit in discover_records order.
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


def stage_leaf(
    results_root: Path,
    seed: str,
    arm: str,
    stage: str = "6",
    variant: str = "without_wrapper",
) -> Path:
    """
    @brief The run directory for one seed/arm at a given curriculum stage.
    @param results_root: outputs/raw/evaluation_results (the seed-nested parent).
    @param seed: Seed sub-root name, e.g. "seed_42".
    @param arm: Baseline name, e.g. "full_method".
    @param stage: Curriculum stage the leaf name starts with.
    @param variant: Wrapper variant directory beneath the leaf.
    @return Path to <results_root>/<seed>/<arm>/<stage>_*/<variant>.
    @throws FileNotFoundError If no leaf for that stage exists.
    """
    arm_dir = results_root / seed / arm
    if not arm_dir.is_dir():
        raise FileNotFoundError(f"no arm directory {arm_dir}")
    leaves = sorted(p for p in arm_dir.iterdir() if p.name.startswith(f"{stage}_"))
    if not leaves:
        raise FileNotFoundError(f"no stage-{stage} checkpoint under {arm_dir}")
    return leaves[0] / variant


def pooled_frame(
    results_root: Path,
    seeds: Sequence[str],
    arms: Sequence[str],
    name: str,
    stage: str = "6",
    variant: str = "without_wrapper",
    usecols: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """
    @brief Concatenate one CSV across every seed and arm into a single frame.
    @param results_root: outputs/raw/evaluation_results.
    @param seeds: Seed sub-root names to pool.
    @param arms: Baseline names to pool.
    @param name: CSV file name, e.g. "episode_records.csv".
    @param stage: Curriculum stage to read.
    @param variant: Wrapper variant to read.
    @param usecols: Optional column subset, for the wide per-step files.
    @return One frame with "arm" and "seed" columns added. Missing seed/arm
            combinations are skipped rather than raising.
    @throws FileNotFoundError If no combination yielded a file.
    """
    import pandas as pd

    frames: List["pd.DataFrame"] = []
    for arm in arms:
        for seed in seeds:
            try:
                path = stage_leaf(results_root, seed, arm, stage, variant) / name
            except FileNotFoundError:
                continue
            if not path.exists():
                continue
            frame = pd.read_csv(path, usecols=usecols)
            frame["arm"] = arm
            frame["seed"] = seed
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no {name} under {results_root} at stage {stage}")
    return pd.concat(frames, ignore_index=True)
