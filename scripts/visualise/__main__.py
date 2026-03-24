"""
@file __main__.py
@brief CLI entry point for the 2D bird's-eye visualiser.

Usage:
    python -m scripts.visualise [--history-file PATH]
"""

import os

import matplotlib
# Must set backend before any pyplot import (including from visualiser.py)
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

from scripts.visualise.visualiser import LiveVisualiser  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse CLI arguments for the visualiser.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="2D bird's-eye visualiser for CARLA parking."
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help=(
            "Path to vis_history.jsonl "
            "(default: outputs/vis_history.jsonl)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    @brief Main entry point. Launches the live visualiser.
    """
    args = _parse_args()
    vis = LiveVisualiser(history_file=args.history_file)
    vis.run()


if __name__ == "__main__":
    main()
