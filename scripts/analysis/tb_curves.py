"""
@file tb_curves.py
@brief Export the seed-averaged training curves from the TensorBoard logs to CSV.

The training-curve figure is the one reading that does not come from the
evaluation CSVs: its source is the per-stage TensorBoard scalars written during
training. This module turns those logs into one tidy CSV so the figure has the
same provenance as every other - generated data, not a literal.

Reads `env/success_rate` and `env/collision_rate` from every
`logs/<arm>/<stage>_<seed>_<stamp>/` run, concatenates the six curriculum
stages end to end into a single cumulative decision axis, averages across
seeds at matched steps, and applies one exponential smoothing pass.

Writes `training_curves.csv` (arm, metric, decisions_m, value) plus
`training_stage_bounds.csv` (stage, decisions_m) for the stage boundaries.

@note Needs `tensorboard`, which lives in the training container, so this runs
via `make docker-training-curves` rather than on the host.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from scripts.diagnostics.tb_read import load_scalars  # noqa: E402

# Scalars behind the two panels, mapped to the metric name used in the CSV.
METRIC_TAGS: Dict[str, str] = {
    "env/success_rate": "success_rate",
    "env/collision_rate": "collision_rate",
}

# Run directory names are <stage>_<seed>_<timestamp>.
_LEAF = re.compile(r"^(?P<stage>\d+)_(?P<seed>\d+)_")

# Exponential smoothing factor. The raw per-rollout rates are noisy enough to
# obscure the trend across a stage; 0.9 is the TensorBoard UI default.
SMOOTHING = 0.9

# A decision is one policy step. The x axis is analysed in millions.
DECISIONS_PER_M = 1_000_000.0

# Points kept per curve after smoothing. The raw logs carry a few thousand
# rollouts per arm, far more than a printed line can resolve; thinning to an
# even stride keeps the shape and the endpoints while leaving the trace legible.
RESAMPLE_POINTS = 330


def _smooth(values: List[float], weight: float) -> List[float]:
    """
    @brief Exponential moving average, matching the TensorBoard UI smoother.
    @param values: Series in logged order.
    @param weight: Smoothing factor in [0, 1); 0 disables.
    @return Smoothed series of the same length.
    """
    out: List[float] = []
    last = values[0] if values else 0.0
    for v in values:
        last = last * weight + (1.0 - weight) * v
        out.append(last)
    return out


def _resample(frame: pd.DataFrame, points: int) -> pd.DataFrame:
    """
    @brief Thin a curve to an even stride, keeping the first and last points.
    @param frame: Single curve, sorted by its x column.
    @param points: Target point count; frames already shorter pass through.
    @return The thinned frame.
    """
    if points <= 0 or len(frame) <= points:
        return frame
    stride = len(frame) / float(points)
    keep = sorted({int(i * stride) for i in range(points)} | {len(frame) - 1})
    return frame.iloc[keep]


def _run_dirs(logs_root: Path, arm: str) -> List[Tuple[int, int, Path]]:
    """
    @brief Every (stage, seed, directory) for an arm, ordered by stage.
    @param logs_root: The logs/ root.
    @param arm: Baseline name.
    @return Sorted list of (stage, seed, path); unparseable names are skipped.
    """
    found: List[Tuple[int, int, Path]] = []
    arm_dir = logs_root / arm
    if not arm_dir.is_dir():
        return found
    for path in arm_dir.iterdir():
        match = _LEAF.match(path.name)
        if match is None or not path.is_dir():
            continue
        found.append((int(match["stage"]), int(match["seed"]), path))
    return sorted(found)


def _stage_series(run_dir: Path, tag: str) -> Optional[pd.DataFrame]:
    """
    @brief One run's scalar series for a tag.
    @param run_dir: TensorBoard run directory.
    @param tag: Scalar tag to read.
    @return Frame of step / value, or None when the tag is absent.
    """
    scalars = load_scalars(run_dir)
    if tag not in scalars:
        return None
    points = scalars[tag]
    if not points:
        return None
    return pd.DataFrame(points, columns=["step", "value"])


def build(logs_root: Path, arms: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    @brief Build the seed-averaged curves and the stage boundaries.
    @param logs_root: The logs/ root.
    @param arms: Baseline names to export.
    @return (curves, bounds). curves is arm / metric / decisions_m / value;
            bounds is stage / decisions_m.
    @throws FileNotFoundError If no run yielded either scalar.

    Stages are laid end to end on one cumulative axis, so each stage's steps
    are offset by the total length of the stages before it. That offset is
    taken from the longest seed in each stage, so every seed shares one axis
    and the boundaries land in the same place for all arms.
    """
    rows: List[pd.DataFrame] = []
    stage_lengths: Dict[int, int] = {}

    # First pass: the length of each stage, so the offsets are shared.
    for arm in arms:
        for stage, _seed, run_dir in _run_dirs(logs_root, arm):
            series = _stage_series(run_dir, next(iter(METRIC_TAGS)))
            if series is None:
                continue
            stage_lengths[stage] = max(
                stage_lengths.get(stage, 0), int(series["step"].max())
            )
    if not stage_lengths:
        raise FileNotFoundError(f"no readable scalars under {logs_root}")

    offsets: Dict[int, int] = {}
    running = 0
    for stage in sorted(stage_lengths):
        offsets[stage] = running
        running += stage_lengths[stage]

    # Second pass: collect every run onto the cumulative axis.
    for arm in arms:
        for stage, seed, run_dir in _run_dirs(logs_root, arm):
            for tag, metric in METRIC_TAGS.items():
                series = _stage_series(run_dir, tag)
                if series is None:
                    continue
                series["decisions"] = series["step"] + offsets[stage]
                series["arm"] = arm
                series["metric"] = metric
                series["seed"] = seed
                rows.append(series)

    frame = pd.concat(rows, ignore_index=True)

    # Average across seeds at matched steps, then smooth once per curve.
    averaged = (
        frame.groupby(["arm", "metric", "decisions"])["value"].mean().reset_index()
    )
    smoothed: List[pd.DataFrame] = []
    for (arm, metric), group in averaged.groupby(["arm", "metric"]):
        group = group.sort_values("decisions").copy()
        group["value"] = _smooth(group["value"].tolist(), SMOOTHING)
        smoothed.append(_resample(group, RESAMPLE_POINTS))
    curves = pd.concat(smoothed, ignore_index=True)
    curves["decisions_m"] = curves["decisions"] / DECISIONS_PER_M
    curves = curves[["arm", "metric", "decisions_m", "value"]]

    bounds = pd.DataFrame(
        {
            "stage": sorted(stage_lengths)[1:],
            "decisions_m": [
                offsets[s] / DECISIONS_PER_M for s in sorted(stage_lengths)[1:]
            ],
        }
    )
    return curves, bounds


def main() -> None:
    """@brief CLI: export the training curves and stage bounds as CSV."""
    parser = argparse.ArgumentParser(
        description="Export seed-averaged training curves from TensorBoard logs."
    )
    parser.add_argument("--logs-root", type=Path, default=Path("logs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/raw_derived/training")
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "vanilla_ppo",
            "input_uncertainty",
            "output_uncertainty",
            "full_method",
        ],
    )
    args = parser.parse_args()

    curves, bounds = build(args.logs_root, args.arms)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = args.output_dir / "training_curves.csv"
    bounds_path = args.output_dir / "training_stage_bounds.csv"
    curves.to_csv(curves_path, index=False)
    bounds.to_csv(bounds_path, index=False)
    print(f"wrote {curves_path} ({len(curves)} rows)")
    print(f"wrote {bounds_path} ({len(bounds)} boundaries)")


if __name__ == "__main__":
    main()
