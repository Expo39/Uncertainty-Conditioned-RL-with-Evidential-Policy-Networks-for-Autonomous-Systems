"""
@file bay_success.py
@brief Per-bay success accounting for training and evaluation.

Accumulates terminal-episode outcomes keyed by target bay id and writes a CSV
(one row per bay: attempts, successes, success rate) plus a run_info.txt header.
Used by both the training callback (uncertainty_rl.training.train_ppo) and the
evaluation loop (uncertainty_rl.evaluation.evaluate) so the on-disk format is
identical for the training and eval sinks under outputs/bay_successes/.

Pure-Python: no torch, CARLA, or ROS 2 imports, so it is unit-testable on the
host without the Docker stack.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BaySuccessTracker:
    """
    @class BaySuccessTracker
    @brief Accumulates per-bay attempts and successes and dumps them to CSV.

    One instance covers one run (training or eval). Call record() once per
    terminated episode with the target bay id and the success flag, then dump()
    to flush the running counts to disk. dump() is idempotent and may be called
    periodically during training as well as once at the end.
    """

    def __init__(self) -> None:
        """
        @brief Initialise empty per-bay accumulators.
        """
        # bay_id -> (bay_type, attempts, successes). bay_type is retained from
        # the first sighting so the CSV can report it without a layout lookup.
        self._counts: Dict[str, Tuple[str, int, int]] = {}

    def record(self, bay_id: str, success: bool, bay_type: str = "") -> None:
        """
        @brief Record one terminated episode against its target bay.
        @param bay_id: Identifier of the bay targeted this episode.
        @param success: Whether the episode ended in a successful park.
        @param bay_type: Bay type (e.g. "angled"); kept from first sighting.
        """
        if not bay_id:
            return
        prev_type, attempts, successes = self._counts.get(bay_id, (bay_type, 0, 0))
        # Keep the first non-empty type seen for this bay.
        kept_type = prev_type or bay_type
        self._counts[bay_id] = (
            kept_type,
            attempts + 1,
            successes + (1 if success else 0),
        )

    @property
    def total_attempts(self) -> int:
        """
        @brief Total recorded episodes across all bays.
        @return Sum of per-bay attempt counts.
        """
        return sum(attempts for _, attempts, _ in self._counts.values())

    def _sorted_rows(self) -> List[Tuple[str, str, int, int, float]]:
        """
        @brief Build CSV rows sorted by bay id (numeric suffix where present).
        @return List of (bay_id, bay_type, attempts, successes, success_rate).
        """

        def sort_key(item: Tuple[str, Tuple[str, int, int]]) -> Tuple[str, int]:
            # Sort by type then trailing integer so angled_2 precedes angled_13.
            bay_id = item[0]
            prefix, _, suffix = bay_id.rpartition("_")
            try:
                return (prefix or bay_id, int(suffix))
            except ValueError:
                return (bay_id, -1)

        rows: List[Tuple[str, str, int, int, float]] = []
        for bay_id, (bay_type, attempts, successes) in sorted(
            self._counts.items(), key=sort_key
        ):
            rate = successes / attempts if attempts else 0.0
            rows.append((bay_id, bay_type, attempts, successes, rate))
        return rows

    def dump(
        self,
        output_dir: Path,
        run_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        @brief Write per-bay counts to <output_dir>/bay_successes.csv plus a
               run_info.txt header.
        @param output_dir: Directory to write into (created if absent).
        @param run_info: Optional ordered key/value pairs written to
               run_info.txt (e.g. run_name, seed, checkpoint, total_timesteps).
        @return Path to the written CSV.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "bay_successes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["bay_id", "bay_type", "attempts", "successes", "success_rate"]
            )
            for bay_id, bay_type, attempts, successes, rate in self._sorted_rows():
                writer.writerow([bay_id, bay_type, attempts, successes, f"{rate:.4f}"])

        if run_info is not None:
            info_path = output_dir / "run_info.txt"
            with open(info_path, "w") as f:
                for key, value in run_info.items():
                    f.write(f"{key}: {value}\n")

        return csv_path
