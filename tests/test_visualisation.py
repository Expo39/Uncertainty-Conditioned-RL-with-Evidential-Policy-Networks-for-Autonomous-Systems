"""
@file test_visualisation.py
@brief Tests for visualisation utility functions.

Validates that plot functions execute without errors and produce output files.
Uses matplotlib's non-interactive backend so no display is required.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for CI/headless testing

from uncertainty_rl.utils import (  # noqa: E402
    plot_training_curves,
    plot_trajectory,
    plot_uncertainty_evolution,
)


class TestPlotUncertaintyEvolution:
    """
    @class TestPlotUncertaintyEvolution
    @brief Tests for the uncertainty evolution plot.
    """

    def test_saves_file(self, tmp_path: Path) -> None:
        """
        @brief Plot should save a PNG when save_path is provided.
        """
        epistemic = [float(i) * 0.01 for i in range(50)]
        aleatoric = [float(i) * 0.02 for i in range(50)]

        save_path = str(tmp_path / "uncertainty.png")
        plot_uncertainty_evolution(epistemic, aleatoric, save_path=save_path)

        assert Path(save_path).exists()
        assert Path(save_path).stat().st_size > 0

    def test_handles_single_point(self, tmp_path: Path) -> None:
        """
        @brief Should not crash with a single data point.
        """
        save_path = str(tmp_path / "single.png")
        plot_uncertainty_evolution([0.5], [0.3], save_path=save_path)
        assert Path(save_path).exists()


class TestPlotTrajectory:
    """
    @class TestPlotTrajectory
    @brief Tests for the vehicle trajectory plot.
    """

    def test_saves_file(self, tmp_path: Path) -> None:
        """
        @brief Trajectory plot should save a PNG.
        """
        positions = np.random.randn(20, 2)
        target = np.array([0.0, 0.0, 0.0])

        save_path = str(tmp_path / "trajectory.png")
        plot_trajectory(positions, target, save_path=save_path)

        assert Path(save_path).exists()

    def test_with_uncertainty_ellipses(self, tmp_path: Path) -> None:
        """
        @brief Should render uncertainty ellipses when covariances provided.
        """
        n = 15
        positions = np.random.randn(n, 2)
        target = np.array([0.0, 0.0, 0.0])
        # Generate positive-definite 2x2 covariances
        uncertainties = np.array([np.eye(2) * 0.1 for _ in range(n)])

        save_path = str(tmp_path / "trajectory_unc.png")
        plot_trajectory(
            positions, target, uncertainties=uncertainties, save_path=save_path
        )

        assert Path(save_path).exists()


class TestPlotTrainingCurves:
    """
    @class TestPlotTrainingCurves
    @brief Tests for the multi-metric training curves plot.
    """

    def test_saves_file(self, tmp_path: Path) -> None:
        """
        @brief Training curves plot should save a PNG.
        """
        metrics: Dict[str, List[float]] = {
            "reward": [float(i) for i in range(100)],
            "loss": [1.0 / (i + 1) for i in range(100)],
        }

        save_path = str(tmp_path / "training.png")
        plot_training_curves(metrics, save_path=save_path)

        assert Path(save_path).exists()

    def test_single_metric(self, tmp_path: Path) -> None:
        """
        @brief Should work with just one metric.
        """
        metrics = {"reward": [1.0, 2.0, 3.0]}

        save_path = str(tmp_path / "single_metric.png")
        plot_training_curves(metrics, save_path=save_path)

        assert Path(save_path).exists()

    def test_short_data_skips_smoothing(self, tmp_path: Path) -> None:
        """
        @brief Data shorter than smoothing window should not crash.
        """
        metrics = {"reward": [1.0, 2.0, 3.0]}

        save_path = str(tmp_path / "short.png")
        plot_training_curves(metrics, save_path=save_path)

        assert Path(save_path).exists()
