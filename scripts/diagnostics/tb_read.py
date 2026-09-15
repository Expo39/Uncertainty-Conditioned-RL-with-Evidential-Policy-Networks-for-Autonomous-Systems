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
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _varint(buf: bytes, pos: int) -> Tuple[int, int]:
    """
    @brief Read one protobuf base-128 varint.
    @param buf: Buffer to read from.
    @param pos: Offset to start at.
    @return (value, offset just past the varint).
    """
    result = shift = 0
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            break
        shift += 7
    return result, pos


def _scalars_from_event(payload: bytes) -> List[Tuple[int, str, float]]:
    """
    @brief Extract (step, tag, value) triples from one serialised Event.
    @param payload: The Event protobuf bytes of a single TFRecord.
    @return Every simple-value scalar the event carries; empty for other events.

    Hand-decoded rather than via the tensorboard package, which would pull a
    whole dashboard (gRPC, Werkzeug, a web server) onto the host just to parse
    a file. Only the fields needed are read - Event.step (field 2, varint),
    Event.summary (field 5), then Summary.Value.tag (field 1, bytes) and
    simple_value (field 2, 32-bit float). Any field this does not recognise is
    skipped by its wire type, so unknown fields cannot desynchronise the parse.
    """
    out: List[Tuple[int, str, float]] = []
    step = 0
    pos = 0
    while pos < len(payload):
        key, pos = _varint(payload, pos)
        field, wire = key >> 3, key & 0x07
        if wire == 0:
            value, pos = _varint(payload, pos)
            if field == 2:  # Event.step
                step = value
        elif wire == 1:  # 64-bit (Event.wall_time)
            pos += 8
        elif wire == 2:  # length-delimited
            length, pos = _varint(payload, pos)
            block = payload[pos : pos + length]
            pos += length
            if field == 5:  # Event.summary
                out.extend((step, tag, val) for tag, val in _summary_values(block))
        elif wire == 5:  # 32-bit
            pos += 4
        else:  # unknown wire type: cannot continue safely
            break
    return out


def _summary_values(block: bytes) -> List[Tuple[str, float]]:
    """
    @brief Extract (tag, simple_value) pairs from a serialised Summary.
    @param block: The Summary protobuf bytes.
    @return One pair per Value that carries a simple_value; others are skipped.
    """
    pairs: List[Tuple[str, float]] = []
    pos = 0
    while pos < len(block):
        key, pos = _varint(block, pos)
        field, wire = key >> 3, key & 0x07
        if wire == 2:
            length, pos = _varint(block, pos)
            value_block = block[pos : pos + length]
            pos += length
            if field == 1:  # Summary.value (repeated)
                tag: Optional[str] = None
                simple: Optional[float] = None
                vpos = 0
                while vpos < len(value_block):
                    vkey, vpos = _varint(value_block, vpos)
                    vfield, vwire = vkey >> 3, vkey & 0x07
                    if vwire == 2:
                        vlen, vpos = _varint(value_block, vpos)
                        if vfield == 1:  # Value.tag
                            tag = value_block[vpos : vpos + vlen].decode(
                                "utf-8", "replace"
                            )
                        vpos += vlen
                    elif vwire == 5:
                        if vfield == 2:  # Value.simple_value
                            (simple,) = struct.unpack_from("<f", value_block, vpos)
                        vpos += 4
                    elif vwire == 0:
                        _, vpos = _varint(value_block, vpos)
                    elif vwire == 1:
                        vpos += 8
                    else:
                        break
                if tag is not None and simple is not None:
                    pairs.append((tag, float(simple)))
        elif wire == 0:
            _, pos = _varint(block, pos)
        elif wire == 1:
            pos += 8
        elif wire == 5:
            pos += 4
        else:
            break
    return pairs


def load_scalars(log_dir: Path) -> Dict[str, Series]:
    """
    @brief Load every scalar series from a TensorBoard event directory.
    @param log_dir: Run directory containing one or more event files.
    @return Mapping of tag -> list of (step, value) in logged order.

    Reads the TFRecord framing directly: each record is an 8-byte little-endian
    length, a 4-byte CRC of that length, the payload, then a 4-byte CRC of the
    payload. The CRCs are not checked - a truncated final record (a run killed
    mid-write) simply ends the read, which is the behaviour wanted here.
    """
    series: Dict[str, Series] = {}
    for path in sorted(log_dir.glob("events.out.tfevents.*")):
        data = path.read_bytes()
        pos = 0
        while pos + 12 <= len(data):
            (length,) = struct.unpack_from("<Q", data, pos)
            start = pos + 12
            end = start + length
            if end > len(data):
                break  # truncated tail from an interrupted run
            for step, tag, value in _scalars_from_event(data[start:end]):
                series.setdefault(tag, []).append((step, value))
            pos = end + 4
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
