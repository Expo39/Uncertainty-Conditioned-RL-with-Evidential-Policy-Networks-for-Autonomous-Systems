"""
@file tb_read.py
@brief Inspect scalar trajectories in TensorBoard event files from the CLI.

Ad-hoc diagnostic for reading a training run's logged scalars without the
TensorBoard UI. Prints, per tag, the full-series statistics (count, mean,
nonzero count, first/last and min/max with their steps) plus an evenly
sampled trajectory, so spiky sparse-positive metrics stay visible rather
than being hidden behind quantiles. Supports tag selection, smoothing, tail
inspection, multi-run comparison, and tidy CSV export.

Usage (via make):
  make tb-scalars LOG=logs/<run_dir>
  make tb-scalars LOG=logs/<run_dir> ARGS="--match success collision"
  make tb-scalars LOG=logs/<run_dir> ARGS="--tags env/success_rate --full"
  make tb-scalars LOG=logs/<run_a> ARGS="logs/<run_b> --match success"
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# (step, value) pairs in logged order.
Series = List[Tuple[int, float]]


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse CLI arguments for the scalar inspector.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Print scalar trajectories from TensorBoard event dirs."
    )
    parser.add_argument(
        "log_dirs",
        nargs="+",
        type=Path,
        help="One or more run log directories containing event files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available scalar tags and exit.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Exact tag names to print (e.g. env/success_rate).",
    )
    parser.add_argument(
        "--match",
        nargs="+",
        default=None,
        help="Case-insensitive substrings; any tag containing one is printed.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=12,
        help="Number of evenly spaced trajectory samples per tag (default 12).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every logged point instead of a sampled trajectory.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        metavar="N",
        help="Also print the last N raw points (plateau inspection).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        metavar="K",
        help="Trailing moving-average window applied before printing.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Export the selected raw series as tidy CSV (run,tag,step,value).",
    )
    return parser.parse_args()


def load_scalars(log_dir: Path) -> Dict[str, Series]:
    """
    @brief Load every scalar series from a TensorBoard event directory.
    @param log_dir: Run directory containing one or more event files.
    @return Mapping of tag -> list of (step, value) in logged order.
    """
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    series: Dict[str, Series] = {}
    for tag in accumulator.Tags()["scalars"]:
        series[tag] = [(e.step, float(e.value)) for e in accumulator.Scalars(tag)]
    return series


def select_tags(
    available: Sequence[str],
    exact: Sequence[str],
    match: Sequence[str],
) -> List[str]:
    """
    @brief Resolve the tag selection from exact names and substring filters.
    @param available: All tags present in the run.
    @param exact: Exact tag names requested (empty for none).
    @param match: Case-insensitive substrings requested (empty for none).
    @return Selected tags in sorted order; all tags when nothing is requested.
    """
    if not exact and not match:
        return sorted(available)
    chosen = set()
    for tag in available:
        if tag in exact:
            chosen.add(tag)
        if any(m.lower() in tag.lower() for m in match):
            chosen.add(tag)
    for tag in exact:
        if tag not in available:
            print(f"  (tag not in run: {tag})")
    return sorted(chosen)


def smooth_series(series: Series, window: int) -> Series:
    """
    @brief Apply a trailing moving average to a series.
    @param series: (step, value) pairs in logged order.
    @param window: Window size in points; <=1 returns the series unchanged.
    @return Smoothed series with the same steps.
    """
    if window <= 1:
        return list(series)
    values = [v for _, v in series]
    smoothed: Series = []
    for i, (step, _) in enumerate(series):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        smoothed.append((step, sum(chunk) / len(chunk)))
    return smoothed


def _fmt_step(step: int) -> str:
    """
    @brief Format a global step compactly (thousands as 'k').
    @param step: Global timestep.
    @return Human-readable step label.
    """
    return f"{step // 1000}k" if step >= 1000 else str(step)


def _fmt_point(step: int, value: float) -> str:
    """
    @brief Format one (step, value) sample as 'step:value'.
    @param step: Global timestep.
    @param value: Scalar value.
    @return Compact sample label.
    """
    return f"{_fmt_step(step)}:{value:.4g}"


def _sample(series: Series, points: int) -> Series:
    """
    @brief Pick evenly spaced samples across a series (first and last kept).
    @param series: (step, value) pairs in logged order.
    @param points: Number of samples requested; <=0 returns everything.
    @return Sampled subseries (deduplicated, order preserved).
    """
    n = len(series)
    if points <= 0 or n <= points:
        return list(series)
    indices = sorted({round(i * (n - 1) / (points - 1)) for i in range(points)})
    return [series[i] for i in indices]


def print_tag(tag: str, series: Series, args: argparse.Namespace) -> None:
    """
    @brief Print statistics and a trajectory for one tag.
    @param tag: Scalar tag name.
    @param series: Raw (step, value) pairs for the tag.
    @param args: Parsed CLI options (points/full/last/smooth).
    """
    if not series:
        print(f"{tag}  (empty)")
        return
    values = [v for _, v in series]
    n = len(values)
    mean = sum(values) / n
    nonzero = sum(1 for v in values if v != 0.0)
    min_i = min(range(n), key=lambda i: values[i])
    max_i = max(range(n), key=lambda i: values[i])
    print(f"{tag}  n={n}  mean={mean:.4g}  nonzero={nonzero}")
    print(
        f"  first={_fmt_point(*series[0])}  last={_fmt_point(*series[-1])}"
        f"  min={_fmt_point(*series[min_i])}  max={_fmt_point(*series[max_i])}"
    )
    shown = smooth_series(series, args.smooth)
    label = f"traj(smooth={args.smooth})" if args.smooth > 1 else "traj"
    sampled = shown if args.full else _sample(shown, args.points)
    print(f"  {label}: " + "  ".join(_fmt_point(s, v) for s, v in sampled))
    if args.last > 0:
        tail = series[-args.last :]
        print(f"  last{args.last}: " + "  ".join(_fmt_point(s, v) for s, v in tail))


def export_csv(
    path: Path, runs: List[Tuple[str, Dict[str, Series]]], tags: List[str]
) -> None:
    """
    @brief Write the selected raw series to a tidy CSV.
    @param path: Output CSV path.
    @param runs: (run_name, tag->series) pairs.
    @param tags: Tags to export.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "tag", "step", "value"])
        for run_name, series_map in runs:
            for tag in tags:
                for step, value in series_map.get(tag, []):
                    writer.writerow([run_name, tag, step, value])
    print(f"CSV written: {path}")


def main() -> None:
    """
    @brief Load each run, resolve the tag selection, and print trajectories.
    """
    args = _parse_args()
    runs: List[Tuple[str, Dict[str, Series]]] = []
    for log_dir in args.log_dirs:
        runs.append((log_dir.name, load_scalars(log_dir)))

    all_selected: List[str] = []
    for run_name, series_map in runs:
        print(f"=== {run_name} ===")
        if args.list:
            for tag in sorted(series_map):
                print(f"  {tag}  (n={len(series_map[tag])})")
            continue
        tags = select_tags(sorted(series_map), args.tags or [], args.match or [])
        all_selected.extend(t for t in tags if t not in all_selected)
        for tag in tags:
            print_tag(tag, series_map[tag], args)
        print()

    if args.csv is not None and not args.list:
        export_csv(args.csv, runs, all_selected)


if __name__ == "__main__":
    main()
