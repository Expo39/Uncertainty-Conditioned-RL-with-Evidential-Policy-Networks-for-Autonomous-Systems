"""
@file training_curves.py
@brief Training success and collision rate across the curriculum.

The series come from the TensorBoard env/success_rate and env/collision_rate
scalars (four arms, six stages, seeds 7/42/123, seed-averaged and lightly
smoothed), carried in a ported JSON payload. Nothing is recomputed or
re-smoothed here, so the plotted values never drift from the reported ones.

x-axis is cumulative policy decisions across the whole curriculum; dashed
verticals mark the stage boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts import figure_style as fs  # noqa: E402

# The pgfplots series carried arm identity in their colour; map back to the arm
# keys so the shared palette drives the styling.
_STYLE_TO_ARM = {
    "okblue": "full_method",
    "okorange": "input_uncertainty",
    "okgreen": "output_uncertainty",
    "okgrey": "vanilla_ppo",
}

_PANELS = [
    ("success rate", (0.0, 1.0), [0, 0.25, 0.5, 0.75, 1.0]),
    ("collision rate", (0.0, 0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
]


def render(data: Path, out: Path) -> None:
    """Draw the training curves from the ported series JSON."""
    payload = json.loads(Path(data).read_text())
    series = payload["series"]
    bounds = payload["bounds"]
    labels = payload["labels"]

    fs.apply()
    fig, axes = plt.subplots(2, 1, figsize=fs.STACK_2, sharex=True)

    # Series arrive as four arms for the success panel, then the same four for
    # the collision panel, in the pgfplots draw order.
    per_panel = len(series) // 2
    handles: list = []
    handle_labels: list = []

    for panel_idx, (ax, (ylabel, ylim, yticks)) in enumerate(zip(axes, _PANELS)):
        for s in series[panel_idx * per_panel : (panel_idx + 1) * per_panel]:
            key = s["style"].split(",")[0].strip()
            arm = _STYLE_TO_ARM.get(key)
            if arm is None:
                continue
            xs = [p[0] for p in s["pts"]]
            ys = [p[1] for p in s["pts"]]
            # Dense traces sampled every few thousand steps, so no markers.
            ax.plot(xs, ys, label=fs.arm_label(arm), **fs.arm_kw(arm, marker=False))
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        fs.grid(ax)
        # Stage boundaries, drawn behind the traces.
        for b in bounds:
            ax.axvline(b, color=fs.MUTED, lw=0.5, ls=":", alpha=0.7, zorder=0)
        if panel_idx == 0:
            for x, _y, name in labels:
                ax.annotate(
                    name,
                    xy=(x, ylim[1]),
                    xytext=(0, -3),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=fs.FS_NOTE,
                    color=fs.MUTED,
                )
            handles, handle_labels = ax.get_legend_handles_labels()

    axes[-1].set_xlabel(r"cumulative policy decisions ($\times 10^{6}$)")
    axes[0].set_xlim(0, 10.007)

    # Arm order shared with every other figure, rather than pgfplots draw order.
    order = [fs.arm_label(a) for a in fs.ARM_ORDER]
    pairs = sorted(zip(handles, handle_labels), key=lambda hl: order.index(hl[1]))
    handles, handle_labels = zip(*pairs)
    fs.legend_strip(fig, (list(handles), list(handle_labels)), side="above")

    fs.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    render(a.data, a.out)


if __name__ == "__main__":
    main()
