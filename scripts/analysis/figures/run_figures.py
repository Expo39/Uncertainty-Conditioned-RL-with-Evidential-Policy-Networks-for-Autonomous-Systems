"""
@file run_figures.py
@brief Per-run summary panels rendered from a single run's evaluation_results.csv.

Draws evaluation_plots.png (2x2 panel) and failure_modes.png (outcome
shares). Distinct from the pooled figures in build.py: these summarise one
run in isolation and are never pooled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Repo root on the path (NOT scripts/, whose `inspect` package would shadow the
# stdlib module that matplotlib imports).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts import figure_style as fs  # noqa: E402

# The per-condition aggregate CSV evaluate.py writes for every run.
RESULTS_CSV = "evaluation_results.csv"

# A run sweeps every eval_config condition, so these panels carry more
# categories than the pooled figures and need extra height for the angled tick
# labels. Wider and taller than fs.GRID_2X2 for that reason alone; these are
# screen-read diagnostics rather than figures placed at a fixed text width.
SUMMARY_SIZE = (9.0, 6.6)
OUTCOME_SIZE = (9.0, 4.2)

# Single-series bar panels: (column, y label, panel title).
_BAR_PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("success_rate", "success rate (%)", "Success rate"),
    ("average_reward", "average reward", "Average reward"),
    ("average_steps", "average steps", "Steps to termination"),
)

# Outcome taxonomy for the stacked breakdown, ordered success-first so the
# stack reads from the desired outcome upward through the failure modes.
_OUTCOME_PANELS: Tuple[Tuple[str, str, str], ...] = (
    ("success_rate", "success", fs.ARM_COLOUR["vanilla_ppo"]),
    ("near_miss_rate", "near miss", "#f9a825"),
    ("stuck_rate", "stuck", "#ef6c00"),
    ("handoff_rate", "handoff", fs.MUTED),
    ("out_of_bounds_rate", "out of bounds", "#8d6e63"),
    ("collision_rate", "collision", fs.ARM_COLOUR["full_method"]),
)

# The two evidential channels, drawn as grouped bars on the fourth panel.
_UNCERTAINTY_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("mean_epistemic_uncertainty", "epistemic", fs.ACCENT_ALT),
    ("mean_aleatoric_uncertainty", "aleatoric", fs.ACCENT),
)


def _has_uncertainty(df: pd.DataFrame) -> bool:
    """
    @brief Whether the run carries a live evidential uncertainty signal.
    @param df: Per-condition aggregate frame.
    @return True when at least one uncertainty column is non-zero.

    Standard Gaussian arms leave both columns at zero because the extraction
    path is gated on policy_type=evidential. Distinguishing that from an
    evidential head that genuinely output zero keeps a blank panel from being
    read as a measurement.
    """
    columns = [c for c, _, _ in _UNCERTAINTY_COLUMNS if c in df.columns]
    if not columns:
        return False
    return bool(df[columns].abs().sum(axis=1).gt(0.0).any())


def _condition_ticks(ax, conditions: Sequence[str]) -> None:
    """
    @brief Label the x axis with condition names, angled to stay legible.
    @param ax: Axes to label.
    @param conditions: Condition keys in plotted order.

    A run sweeps every condition in eval_config, including the held-tier pair
    that the pooled figures drop, so up to seven labels share one axis. The
    house two-line prose labels collide at that density, so these panels take
    the raw keys on a single line and angle them instead.
    """
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(
        [fs.condition_label(c).replace("\n", " ") for c in conditions],
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )


def _draw_summary(df: pd.DataFrame, conditions: List[str], out: Path) -> Path:
    """
    @brief Draw the 2x2 per-condition summary panel.
    @param df: Per-condition aggregate frame.
    @param conditions: Condition keys in plotted order.
    @param out: Output path; the suffix is normalised by fs.save.
    @return The written path.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(conditions), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=SUMMARY_SIZE)
    flat = axes.ravel()

    for ax, (column, ylabel, title) in zip(flat, _BAR_PANELS):
        ax.bar(x, df[column], width=0.6, color=fs.ARM_COLOUR["vanilla_ppo"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=fs.FS_LABEL)
        _condition_ticks(ax, conditions)
        fs.grid(ax, axis="y")

    ax_unc = flat[3]
    ax_unc.set_title("Policy uncertainty estimates", fontsize=fs.FS_LABEL)
    if _has_uncertainty(df):
        width = 0.35
        for offset, (column, label, colour) in zip(
            (-width / 2.0, width / 2.0), _UNCERTAINTY_COLUMNS
        ):
            ax_unc.bar(x + offset, df[column], width, label=label, color=colour)
        ax_unc.set_ylabel("uncertainty")
        _condition_ticks(ax_unc, conditions)
        fs.grid(ax_unc, axis="y")
        fs.legend_strip(fig, ax_unc, side="below")
    else:
        # Standard Gaussian head: no evidential signal exists to plot.
        ax_unc.text(
            0.5,
            0.5,
            "not applicable - standard head\n(no evidential uncertainty output)",
            ha="center",
            va="center",
            fontsize=fs.FS_NOTE,
            color=fs.MUTED,
            transform=ax_unc.transAxes,
        )
        ax_unc.set_xticks([])
        ax_unc.set_yticks([])

    return fs.save(fig, out)


def _draw_failure_modes(
    df: pd.DataFrame, conditions: List[str], out: Path
) -> Optional[Path]:
    """
    @brief Draw the stacked outcome-share breakdown.
    @param df: Per-condition aggregate frame.
    @param conditions: Condition keys in plotted order.
    @param out: Output path; the suffix is normalised by fs.save.
    @return The written path, or None when no outcome column is present.
    """
    import matplotlib.pyplot as plt

    present = [spec for spec in _OUTCOME_PANELS if spec[0] in df.columns]
    if not present:
        return None

    x = np.arange(len(conditions), dtype=float)
    fig, ax = plt.subplots(figsize=OUTCOME_SIZE)
    bottom = np.zeros(len(conditions))
    for column, label, colour in present:
        values = df[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=label, color=colour, width=0.6)
        bottom += values

    ax.set_ylabel("share of episodes (%)")
    _condition_ticks(ax, conditions)
    fs.grid(ax, axis="y")
    fs.legend_strip(fig, ax, side="below")
    return fs.save(fig, out)


def render(run_dir: Path, out_root: Path, raw_root: Path) -> List[Path]:
    """
    @brief Render both per-run panels for one evaluation run directory.
    @param run_dir: Directory holding the run's evaluation_results.csv.
    @param out_root: Figure tree root; the run's path under raw_root is
           mirrored beneath it.
    @param raw_root: Root the run path is relative to, for that mirroring.
    @return Paths of the figures written.
    @throws FileNotFoundError If the run directory has no aggregate CSV.

    Figures never land beside the CSVs: the evaluation tree holds raw data
    only, so the panels mirror the run's path under the figure root instead.
    """
    csv_path = run_dir / RESULTS_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    conditions = df["condition"].astype(str).tolist()

    try:
        stem = run_dir.resolve().relative_to(raw_root.resolve())
    except ValueError:
        stem = Path(run_dir.name)
    target = out_root / stem

    fs.apply()
    written = [_draw_summary(df, conditions, target / "evaluation_plots")]
    failure_modes = _draw_failure_modes(df, conditions, target / "failure_modes")
    if failure_modes is not None:
        written.append(failure_modes)
    return written


def find_run_dirs(root: Path) -> List[Path]:
    """
    @brief Every evaluation run directory beneath a root.
    @param root: Tree to search, e.g. outputs.
    @return Sorted parents of each evaluation_results.csv found.
    """
    return sorted(p.parent for p in root.rglob(RESULTS_CSV))


def main() -> None:
    """@brief CLI: render the per-run panels for one run or a whole tree."""
    parser = argparse.ArgumentParser(
        description="Render the per-run evaluation panels from a run's CSV."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="A single run directory. Omit to walk --root for every run.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/raw/evaluation_results"),
        help="Evaluation tree to search when --run-dir is not given.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/raw_derived/figures/per_run"),
        help="Figure root; each run's path under --root is mirrored beneath it.",
    )
    args = parser.parse_args()

    run_dirs = [args.run_dir] if args.run_dir else find_run_dirs(args.root)
    if not run_dirs:
        parser.error(f"no {RESULTS_CSV} found under {args.root}")

    failed: List[Path] = []
    for run_dir in run_dirs:
        try:
            render(run_dir, args.output_dir, args.root)
        except (OSError, ValueError, KeyError) as exc:
            print(f"  !! {run_dir} skipped: {type(exc).__name__}: {exc}")
            failed.append(run_dir)

    drawn = len(run_dirs) - len(failed)
    print(f"\nrendered {drawn}/{len(run_dirs)} run(s)")


if __name__ == "__main__":
    main()
