"""
@file plot_mapping_waypoints.py
@brief Visualise mapping drive waypoints from mapping_drive.py.

Reads mapping waypoints from get_mapping_waypoints() and overlays them
on the layout geometry. Shows both symmetric and asymmetric variants
side-by-side.

Usage::

    make plot-mapping-waypoints LAYOUT=rectangle

@author Antonio Galdes
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.mapping.mapping_drive import get_mapping_waypoints


def main() -> None:
    """
    @brief Entry point: load layout YAML, get waypoints from mapping_drive.py, render PNG.
    """
    parser = argparse.ArgumentParser(description="Plot mapping waypoints.")
    parser.add_argument(
        "--layout", type=str, default="rectangle", help="Layout name."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/maps/2d/rectangle_waypoints.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    layout_file = f"configs/layouts/{args.layout}.yaml"
    with open(layout_file) as f:
        layout: Dict[str, Any] = yaml.safe_load(f)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MPoly
    except ImportError:
        print("ERROR: matplotlib not available.")
        return

    corners = [(c["x"], c["y"]) for c in layout["corners"]]
    has_asym = "asymmetric_corners" in layout
    asym = (
        [(c["x"], c["y"]) for c in layout["asymmetric_corners"]]
        if has_asym
        else []
    )

    # Get waypoints from mapping_drive.py with dynamic inset
    wall_inset = 3.0  # Match the inset used in mapping_drive.py main()
    wps_rect = get_mapping_waypoints(
        args.layout, False, layout_corners=layout.get("corners"), wall_inset=wall_inset
    )
    wps_asym = (
        get_mapping_waypoints(
            args.layout, True, layout_corners=layout.get("corners"), wall_inset=wall_inset
        )
        if has_asym
        else []
    )

    ncols = 2 if has_asym else 1
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 8))
    if ncols == 1:
        axes = [axes]

    panels = [
        (axes[0], corners, wps_rect, "Rectangle (asymmetric_corner=false)"),
    ]
    if has_asym:
        panels.append(
            (axes[1], asym, wps_asym, "Asymmetric (asymmetric_corner=true)")
        )

    for ax, corner_pts, wps, title in panels:
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_facecolor("#E8E8E8")

        lot = MPoly(
            corner_pts,
            closed=True,
            facecolor="#DDDDDD",
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(lot)

        if wps:
            wx = [w[0] for w in wps] + [wps[0][0]]
            wy = [w[1] for w in wps] + [wps[0][1]]
            ax.plot(wx, wy, "r--", linewidth=1.5, alpha=0.7, label="Mapping path")
            for i, (x, y) in enumerate(wps):
                ax.plot(x, y, "ro", markersize=6)
                # Adjust label offset to keep labels visible on edges
                offset_x = 8 if x > (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2 else -15
                offset_y = 8 if y > (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2 else -12
                ax.annotate(
                    f"{i + 1}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(offset_x, offset_y),
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                )

        sp = layout.get("spawn_transform", {})
        if sp:
            ax.plot(sp["x"], sp["y"], "c^", markersize=12, label="Spawn 1")

        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
