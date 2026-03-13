"""
@file generate_layouts.py
@brief Orchestrator script for parking lot layout generation.

Generates pre-computed world-frame lot geometry (corners, bay positions, spawn
transform, patrol waypoints, pedestrian zones) and writes layout YAML files
consumed by CARLAParkingEnv. Optionally produces bird's-eye PNG plots.

Three floor plan shapes are supported:

  rectangle   -- Standard axis-aligned rectangle (60x45 m). Training layout.
  trapezoid   -- Wider at entrance, narrower at rear (front=48, rear=30, depth=44 m).
                 Training layout.
  irregular_a -- Nine-sided polygon with diagonal top wall and bottom notch
                 (OOD, held out from training).

Each shape is defined in its own module under scripts/layouts/:
  scripts/layouts/rectangle.py
  scripts/layouts/trapezoid.py
  scripts/layouts/irregular_a.py

Usage::

  # Generate all three layouts:
  make generate-layouts

  # Generate one layout only:
  make generate-layouts LAYOUT=trapezoid

  # Override output directories:
  python scripts/generate_layouts.py \
      --output-dir configs/layouts --plot-dir outputs/layouts

  # Generate one layout with custom origin:
  python scripts/generate_layouts.py --layout rectangle --origin -200 0 0.3 --heading 0
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow importing scripts/layouts as a package when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.layouts import irregular_a, rectangle, trapezoid  # noqa: E402
from scripts.layouts.common import (  # noqa: E402
    plot_layout,
    to_world_frame,
    write_layout_yaml,
)

_LAYOUTS = {
    "rectangle": rectangle,
    "trapezoid": trapezoid,
    "irregular_a": irregular_a,
}


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse command-line arguments.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate parking lot layout YAML files and optional bird's-eye PNGs.\n"
            "Omit --layout to generate all three layouts at once."
        )
    )
    parser.add_argument(
        "--layout",
        choices=list(_LAYOUTS.keys()),
        default=None,
        help="Layout name. Omit to generate all.",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help=(
            "World-frame origin of the lot (x y z). "
            "Overrides the default origin in the layout module. "
            "Only valid when --layout is also specified."
        ),
    )
    parser.add_argument(
        "--heading",
        type=float,
        default=None,
        help=(
            "Lot heading in world frame (degrees). "
            "Overrides the default heading in the layout module. "
            "Only valid when --layout is also specified."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs/layouts",
        help="Output directory for YAML files. Default: configs/layouts.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="outputs/layouts",
        help="Output directory for PNG files. Default: outputs/layouts.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation.",
    )
    return parser.parse_args()


def _generate_one(
    name: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    heading_deg: float,
    output_path: Path,
    plot_path: Optional[Path],
    ood: bool,
) -> None:
    """
    @brief Generate one floor plan layout and write to disk.

    @param name: Layout name (rectangle, trapezoid, irregular_a).
    @param origin_x: World x of lot origin.
    @param origin_y: World y of lot origin.
    @param origin_z: World z of lot origin.
    @param heading_deg: Lot heading in world frame (degrees).
    @param output_path: YAML output path.
    @param plot_path: PNG output path, or None to skip.
    @param ood: Whether this layout is held out for OOD evaluation.
    """
    module = _LAYOUTS[name]
    local_layout = module.generate()
    world_layout = to_world_frame(
        local_layout, origin_x, origin_y, origin_z, heading_deg
    )
    write_layout_yaml(
        name, origin_x, origin_y, origin_z, heading_deg, world_layout, output_path, ood
    )
    if plot_path is not None:
        plot_layout(name, world_layout, plot_path)


def main() -> None:
    """
    @brief Entry point: generate one or all floor plan layouts.
    """
    args = _parse_args()
    out_dir = Path(args.output_dir)
    plot_dir = Path(args.plot_dir)

    if args.layout is not None:
        # Single layout mode.
        module = _LAYOUTS[args.layout]
        ox = args.origin[0] if args.origin else module.ORIGIN_X
        oy = args.origin[1] if args.origin else module.ORIGIN_Y
        oz = args.origin[2] if args.origin else module.ORIGIN_Z
        hdg = args.heading if args.heading is not None else module.HEADING_DEG
        print(
            f"Generating {args.layout} layout "
            f"(origin={ox},{oy},{oz}, heading={hdg} deg)"
        )
        _generate_one(
            name=args.layout,
            origin_x=ox,
            origin_y=oy,
            origin_z=oz,
            heading_deg=hdg,
            output_path=out_dir / f"{args.layout}.yaml",
            plot_path=None if args.no_plot else plot_dir / f"{args.layout}.png",
            ood=module.OOD,
        )
    else:
        # Multi-layout mode: generate all layouts using each module's defaults.
        if args.origin is not None or args.heading is not None:
            print(
                "WARNING: --origin and --heading are ignored in multi-layout mode. "
                "Edit the ORIGIN_X/Y/Z and HEADING_DEG constants in each layout module."
            )
        print("Generating all floor plan layouts...")
        print(f"  YAMLs -> {out_dir}/")
        print(f"  PNGs  -> {plot_dir}/")
        print()
        print("NOTE: Origins are approximate placeholders.")
        print("      Run 'make docker-inspect INSPECT_LAYOUT=<floor_plan>'")
        print("      to verify coordinates visually, then re-run this script.")
        print()
        for name, module in _LAYOUTS.items():
            _generate_one(
                name=name,
                origin_x=module.ORIGIN_X,
                origin_y=module.ORIGIN_Y,
                origin_z=module.ORIGIN_Z,
                heading_deg=module.HEADING_DEG,
                output_path=out_dir / f"{name}.yaml",
                plot_path=None if args.no_plot else plot_dir / f"{name}.png",
                ood=module.OOD,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
