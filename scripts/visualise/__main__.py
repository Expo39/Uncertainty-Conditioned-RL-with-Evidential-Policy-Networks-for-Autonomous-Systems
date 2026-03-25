"""
@file __main__.py
@brief CLI entry point for the 2D bird's-eye Pygame visualiser.

Usage:
    python -m scripts.visualise [--history-file PATH]
"""

import argparse
from pathlib import Path

from scripts.visualise.visualiser import LiveVisualiser


def _parse_args() -> argparse.Namespace:
    """
    @brief Parse CLI arguments for the visualiser.
    @return Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="2D bird's-eye Pygame visualiser for CARLA parking."
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Path to vis_history.jsonl (default: outputs/vis_history.jsonl).",
    )
    return parser.parse_args()


def main() -> None:
    """@brief Main entry point. Launches the live visualiser."""
    args = _parse_args()
    vis = LiveVisualiser(history_file=args.history_file)
    vis.run()


if __name__ == "__main__":
    main()
