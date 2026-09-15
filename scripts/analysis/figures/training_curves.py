"""
@file training_curves.py
@brief Training success and collision rate across the curriculum.

Reads the CSVs that scripts/analysis/tb_curves.py exports from the
TensorBoard scalars (seed-averaged, smoothed, stages laid end to end on one
cumulative decision axis) and draws them as two stacked panels. Presentation
only: nothing is re-averaged or re-smoothed here.

x-axis is cumulative policy decisions across the whole curriculum; dotted
verticals mark the stage boundaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts import figure_style as fs  # noqa: E402

# Where scripts/analysis/tb_curves.py writes its export.
DATA_DIR = Path("outputs/training")
CURVES_CSV = "training_curves.csv"
BOUNDS_CSV = "training_stage_bounds.csv"

# (metric key, y label, y limits, y ticks), top panel first.
_PANELS: Tuple[Tuple[str, str, Tuple[float, float], List[float]], ...] = (
    ("success_rate", "success rate", (0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]),
    ("collision_rate", "collision rate", (0.0, 0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
)


def render(data_dir: Optional[Path], out: Path) -> None:
    """
    @brief Draw the training curves from the exported CSVs.
    @param data_dir: Directory holding the exports; None uses DATA_DIR.
    @param out: Output path; the suffix is normalised by fs.save.
    @throws FileNotFoundError If the curves CSV is missing.
    """
    root = Path(data_dir or DATA_DIR)
    curves_path = root / CURVES_CSV
    if not curves_path.exists():
        raise FileNotFoundError(
            f"{curves_path} not found - run `make docker-training-curves` first"
        )
    curves = pd.read_csv(curves_path)

    bounds_path = root / BOUNDS_CSV
    bounds = (
        pd.read_csv(bounds_path)["decisions_m"].tolist() if bounds_path.exists() else []
    )

    fs.apply()
    fig, axes = plt.subplots(2, 1, figsize=fs.STACK_2, sharex=True)
    handles: List = []
    handle_labels: List[str] = []

    for panel_idx, (ax, (metric, ylabel, ylim, yticks)) in enumerate(
        zip(axes, _PANELS)
    ):
        for arm in fs.ARM_ORDER:
            series = curves[(curves["arm"] == arm) & (curves["metric"] == metric)]
            if series.empty:
                continue
            series = series.sort_values("decisions_m")
            # Dense traces sampled every few thousand steps, so no markers.
            ax.plot(
                series["decisions_m"],
                series["value"],
                label=fs.arm_label(arm),
                **fs.arm_kw(arm, marker=False),
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        fs.grid(ax)
        # Stage boundaries, drawn behind the traces.
        for boundary in bounds:
            ax.axvline(boundary, color=fs.MUTED, lw=0.5, ls=":", alpha=0.7, zorder=0)
        if panel_idx == 0:
            handles, handle_labels = ax.get_legend_handles_labels()
            # Name each stage between its boundaries.
            edges = [0.0] + list(bounds) + [curves["decisions_m"].max()]
            for i in range(len(edges) - 1):
                ax.annotate(
                    f"S{i + 1}",
                    xy=((edges[i] + edges[i + 1]) / 2.0, ylim[1]),
                    xytext=(0, -3),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs.FS_NOTE,
                    color=fs.MUTED,
                )

    axes[-1].set_xlabel(r"cumulative policy decisions ($\times 10^{6}$)")
    axes[0].set_xlim(0, curves["decisions_m"].max())
    fs.legend_strip(fig, (handles, handle_labels), side="above")
    fs.save(fig, out)


def main() -> None:
    """@brief CLI: draw the training curves from the exported CSVs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.data_dir, args.out)


if __name__ == "__main__":
    main()
