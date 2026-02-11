"""
@file test_logging.py
@brief Tests for logging and uncertainty tracking utilities.
"""
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from uncertainty_rl.utils.logging import MetricsLogger, UncertaintyTracker


class TestMetricsLogger:
    """
    @class TestMetricsLogger
    @brief Tests for the MetricsLogger utility.
    """

    def test_log_and_retrieve(self, tmp_path: Path) -> None:
        """
        @brief Logged values should be retrievable by name.
        """
        logger = MetricsLogger(log_dir=str(tmp_path))
        logger.log(step=0, metrics={"reward": 10.0, "loss": 0.5})
        logger.log(step=1, metrics={"reward": 15.0, "loss": 0.3})

        assert logger.get_metric("reward") == [10.0, 15.0]
        assert logger.get_metric("loss") == [0.5, 0.3]
        assert logger.get_metric("step") == [0, 1]

    def test_get_missing_metric_returns_none(self, tmp_path: Path) -> None:
        """
        @brief Requesting a non-existent metric returns None.
        """
        logger = MetricsLogger(log_dir=str(tmp_path))
        assert logger.get_metric("nonexistent") is None

    def test_save_csv(self, tmp_path: Path) -> None:
        """
        @brief CSV file should be created and non-empty.
        """
        logger = MetricsLogger(log_dir=str(tmp_path), prefix="test")
        logger.log(step=0, metrics={"reward": 1.0})
        logger.save_csv()

        csv_path = tmp_path / "test.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

    def test_save_json(self, tmp_path: Path) -> None:
        """
        @brief JSON file should be valid and contain logged data.
        """
        logger = MetricsLogger(log_dir=str(tmp_path), prefix="test")
        logger.log(step=0, metrics={"reward": 42.0})
        logger.save_json()

        json_path = tmp_path / "test.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["reward"] == [42.0]

    def test_compute_statistics(self, tmp_path: Path) -> None:
        """
        @brief Statistics should include mean, std, min, max.
        """
        logger = MetricsLogger(log_dir=str(tmp_path))
        for i in range(10):
            logger.log(step=i, metrics={"value": float(i)})

        stats = logger.compute_statistics("value")
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["min"] == 0.0
        assert stats["max"] == 9.0

    def test_compute_statistics_empty(self, tmp_path: Path) -> None:
        """
        @brief Statistics on missing metric returns empty dict.
        """
        logger = MetricsLogger(log_dir=str(tmp_path))
        assert logger.compute_statistics("missing") == {}

    def test_creates_directory(self, tmp_path: Path) -> None:
        """
        @brief Logger should create the log directory if it doesn't exist.
        """
        nested = tmp_path / "a" / "b" / "c"
        logger = MetricsLogger(log_dir=str(nested))
        assert nested.exists()


class TestUncertaintyTracker:
    """
    @class TestUncertaintyTracker
    @brief Tests for the sliding-window uncertainty tracker.
    """

    def test_update_and_statistics(self) -> None:
        """
        @brief Statistics should reflect tracked values.
        """
        tracker = UncertaintyTracker(window_size=100)
        for i in range(50):
            tracker.update(epistemic=float(i), aleatoric=float(i * 2))

        stats = tracker.get_statistics()
        assert "epistemic" in stats
        assert "aleatoric" in stats
        assert stats["epistemic"]["min"] == 0.0
        assert stats["epistemic"]["max"] == 49.0
        assert stats["aleatoric"]["max"] == 98.0

    def test_window_size_limit(self) -> None:
        """
        @brief Tracker should only keep window_size most recent values.
        """
        tracker = UncertaintyTracker(window_size=10)
        for i in range(100):
            tracker.update(epistemic=float(i), aleatoric=float(i))

        assert len(tracker.epistemic_values) == 10
        # Should only have the last 10 values (90-99)
        assert tracker.epistemic_values[0] == 90.0

    def test_reset(self) -> None:
        """
        @brief Reset should clear all tracked values.
        """
        tracker = UncertaintyTracker()
        tracker.update(epistemic=1.0, aleatoric=2.0)
        tracker.reset()

        assert len(tracker.epistemic_values) == 0
        assert len(tracker.aleatoric_values) == 0

    def test_empty_statistics(self) -> None:
        """
        @brief Statistics on empty tracker returns empty dict.
        """
        tracker = UncertaintyTracker()
        assert tracker.get_statistics() == {}
