"""
@file test_bay_success.py
@brief Unit tests for the per-bay success tracker.

Covers accumulation, success-rate computation, id-aware sorting, and the CSV +
run_info.txt dump. Pure-Python, so these run on the host without the Docker
stack (no torch / CARLA / ROS 2 dependency).
"""

import csv
from pathlib import Path

from uncertainty_rl.utils.bay_success import BaySuccessTracker


def _read_csv(path: Path):
    """
    @brief Read a bay_successes.csv into a list of dict rows.
    @param path: Path to the CSV file.
    @return List of row dicts keyed by header name.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_record_accumulates_attempts_and_successes() -> None:
    """
    @brief Attempts and successes accumulate per bay across record() calls.
    """
    tracker = BaySuccessTracker()
    tracker.record("angled_3", success=True, bay_type="angled")
    tracker.record("angled_3", success=False, bay_type="angled")
    tracker.record("angled_3", success=True, bay_type="angled")

    assert tracker.total_attempts == 3
    rows = tracker._sorted_rows()
    assert rows == [("angled_3", "angled", 3, 2, 2 / 3)]


def test_empty_bay_id_is_ignored() -> None:
    """
    @brief A blank bay id does not create a row or count an attempt.
    """
    tracker = BaySuccessTracker()
    tracker.record("", success=True)
    assert tracker.total_attempts == 0
    assert tracker._sorted_rows() == []


def test_bay_type_kept_from_first_sighting() -> None:
    """
    @brief The first non-empty bay type seen is retained for later records.
    """
    tracker = BaySuccessTracker()
    tracker.record("perpendicular_7", success=False, bay_type="perpendicular")
    tracker.record("perpendicular_7", success=True, bay_type="")
    rows = tracker._sorted_rows()
    assert rows[0][1] == "perpendicular"


def test_rows_sorted_by_numeric_suffix() -> None:
    """
    @brief angled_2 sorts before angled_13 (numeric suffix, not lexical).
    """
    tracker = BaySuccessTracker()
    for bay_id in ("angled_13", "angled_2", "perpendicular_1"):
        tracker.record(bay_id, success=True, bay_type=bay_id.split("_")[0])
    ordered = [row[0] for row in tracker._sorted_rows()]
    assert ordered == ["angled_2", "angled_13", "perpendicular_1"]


def test_dump_writes_csv_and_run_info(tmp_path: Path) -> None:
    """
    @brief dump() writes bay_successes.csv and run_info.txt with given fields.
    """
    tracker = BaySuccessTracker()
    tracker.record("angled_1", success=True, bay_type="angled")
    tracker.record("angled_1", success=False, bay_type="angled")

    out_dir = tmp_path / "training" / "run_x"
    csv_path = tracker.dump(out_dir, run_info={"run_name": "run_x", "seed": 42})

    assert csv_path == out_dir / "bay_successes.csv"
    rows = _read_csv(csv_path)
    assert rows[0]["bay_id"] == "angled_1"
    assert rows[0]["attempts"] == "2"
    assert rows[0]["successes"] == "1"
    assert rows[0]["success_rate"] == "0.5000"

    info_text = (out_dir / "run_info.txt").read_text()
    assert "run_name: run_x" in info_text
    assert "seed: 42" in info_text


def test_dump_without_run_info_skips_info_file(tmp_path: Path) -> None:
    """
    @brief When run_info is None, only the CSV is written (no run_info.txt).
    """
    tracker = BaySuccessTracker()
    tracker.record("angled_1", success=True, bay_type="angled")
    out_dir = tmp_path / "eval" / "cond"
    tracker.dump(out_dir, run_info=None)
    assert (out_dir / "bay_successes.csv").exists()
    assert not (out_dir / "run_info.txt").exists()
