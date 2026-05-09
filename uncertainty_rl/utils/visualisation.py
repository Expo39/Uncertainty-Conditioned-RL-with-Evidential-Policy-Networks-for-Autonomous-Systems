"""
@file visualisation.py
@brief Atomic JSON writer for the detachable 2D bird's-eye visualiser.

VisStateWriter streams environment state to outputs/vis_history.jsonl every
step. The detachable visualiser (scripts/visualise/visualiser.py) tails that
file and renders each frame via Pygame.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VisStateWriter:
    """
    @class VisStateWriter
    @brief Writes vis_state.json every step for the detachable 2D visualiser.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, output_path: Path) -> None:
        """
        @brief Initialise the writer.
        @param output_path: Destination path for vis_state.json.
        """
        self._output_path = output_path
        self._tmp_path = output_path.with_suffix(".tmp")
        self._output_path_str = str(output_path)
        self._tmp_path_str = str(self._tmp_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        ego_transform: Dict[str, Any],
        actor_transforms: List[Dict[str, Any]],
        target_bay: Dict[str, Any],
        episode_info: Dict[str, Any],
        trajectory: Optional[List[Tuple[float, float]]] = None,
        bays: Optional[List[Dict[str, Any]]] = None,
        corners: Optional[List[Dict[str, Any]]] = None,
        pedestrians: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        @brief Serialise and atomically write the visualisation state.
        @param ego_transform: Dict with keys x, y, yaw.
        @param actor_transforms: List of dicts with x, y, yaw, type ('npc'/'static').
        @param target_bay: Dict with x, y, yaw, bay_type, width, depth.
        @param episode_info: Dict with episode metadata (step, floor_plan, etc.).
        @param trajectory: List of (x, y) tuples for the ego trail.
        @param bays: Full list of bay dicts from the floor plan layout.
        @param corners: Perimeter corner dicts from the floor plan layout.
        @param pedestrians: List of dicts with x, y for pedestrian positions.
        """
        state: Dict[str, Any] = {
            "ego": ego_transform,
            "actors": actor_transforms,
            "target_bay": target_bay,
            "episode_info": episode_info,
            "trajectory": trajectory or [],
            "bays": bays or [],
            "corners": corners or [],
            "pedestrians": pedestrians or [],
        }

        try:
            json_str = json.dumps(state)
            self._tmp_path.write_text(json_str)
            os.replace(self._tmp_path_str, self._output_path_str)
        except Exception as exc:
            logger.debug(f"VisStateWriter: could not write state: {exc}")
