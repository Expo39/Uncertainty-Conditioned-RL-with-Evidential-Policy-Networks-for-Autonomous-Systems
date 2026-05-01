"""
@file visualisation.py
@brief Visualisation utilities for uncertainty and performance metrics.

This module provides visualisation tools for understanding agent behaviour
and uncertainty evolution during training and evaluation. It also contains
VisStateWriter, which streams environment state to a detachable 2D bird's-eye
visualiser via an atomically-written JSON file.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class VisStateWriter:
    """
    @class VisStateWriter
    @brief Writes vis_state.json every step for the detachable 2D visualiser.

    Atomic write via tmp file + os.replace -- the visualiser process never reads
    partial data. Training writes to outputs/vis_state.json; the visualiser polls
    that file every 100 ms and redraws on change.

    Usage in training:
        writer = VisStateWriter(Path("outputs/vis_state.json"))
        writer.write(ego_transform, actor_transforms, target_bay, episode_info)

    Visualiser reads outputs/vis_state.json.
    Close the visualiser window at any time -- training is unaffected.
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
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._output_path.with_suffix(".tmp")
            tmp_path.write_text(json_str)
            os.replace(str(tmp_path), str(self._output_path))
        except Exception as exc:
            logger.debug(f"VisStateWriter: could not write state: {exc}")


def plot_uncertainty_evolution(
    epistemic: List[float],
    aleatoric: List[float],
    save_path: Optional[str] = None,
    title: str = "Uncertainty Evolution",
) -> None:
    """
    @brief Plot the evolution of uncertainties over time.
    @param epistemic: List of epistemic uncertainty values.
    @param aleatoric: List of aleatoric uncertainty values.
    @param save_path: Path to save the plot.
    @param title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    steps = np.arange(len(epistemic))

    # Epistemic uncertainty
    ax1.plot(steps, epistemic, label="Epistemic", color="blue", alpha=0.7)
    ax1.fill_between(steps, 0, epistemic, alpha=0.3, color="blue")
    ax1.set_ylabel("Epistemic Uncertainty", fontsize=12)
    ax1.set_title(f"{title} - Epistemic", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Aleatoric uncertainty
    ax2.plot(steps, aleatoric, label="Aleatoric", color="red", alpha=0.7)
    ax2.fill_between(steps, 0, aleatoric, alpha=0.3, color="red")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Aleatoric Uncertainty", fontsize=12)
    ax2.set_title(f"{title} - Aleatoric", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_trajectory(
    positions: np.ndarray,
    target: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Vehicle Trajectory",
) -> None:
    """
    @brief Plot vehicle trajectory with uncertainty ellipses.
    @param positions: Array of (x, y) positions, shape (N, 2).
    @param target: Target position (x, y, yaw).
    @param uncertainties: Optional uncertainty covariances, shape (N, 2, 2).
    @param save_path: Path to save the plot.
    @param title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], "b-", linewidth=2, label="Trajectory")
    ax.plot(positions[0, 0], positions[0, 1], "go", markersize=15, label="Start")
    ax.plot(positions[-1, 0], positions[-1, 1], "ro", markersize=15, label="End")

    # Plot target
    ax.plot(target[0], target[1], "r*", markersize=20, label="Target")

    # Draw target orientation
    arrow_length = 2.0
    dx = arrow_length * np.cos(target[2])
    dy = arrow_length * np.sin(target[2])
    ax.arrow(
        target[0],
        target[1],
        dx,
        dy,
        head_width=0.5,
        head_length=0.3,
        fc="red",
        ec="red",
    )

    # Plot uncertainty ellipses if provided
    if uncertainties is not None:
        from matplotlib.patches import Ellipse

        # Sample points for ellipses
        sample_indices = np.linspace(
            0, len(positions) - 1, min(10, len(positions)), dtype=int
        )

        for idx in sample_indices:
            pos = positions[idx]
            cov = uncertainties[idx]

            # Eigenvalue decomposition (eigvalsh for symmetric matrices -- real,
            # sorted eigenvalues guaranteed)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # 95% confidence ellipse (chi-square with 2 DOF)
            width, height = 2 * np.sqrt(5.991 * eigenvalues)

            ellipse = Ellipse(
                pos,
                width,
                height,
                angle=angle,
                facecolor="blue",
                alpha=0.1,
                edgecolor="blue",
                linewidth=1,
            )
            ax.add_patch(ellipse)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves",
) -> None:
    """
    @brief Plot training curves for multiple metrics.
    @param metrics: Dictionary mapping metric names to lists of values.
    @param save_path: Path to save the plot.
    @param title: Plot title.
    """
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for idx, (name, values) in enumerate(metrics.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        steps = np.arange(len(values))

        ax.plot(steps, values, linewidth=2)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add smoothed curve
        if len(values) > 10:
            window_size = min(50, len(values) // 10)
            smoothed = np.convolve(
                values, np.ones(window_size) / window_size, mode="valid"
            )
            ax.plot(
                steps[window_size - 1 :],
                smoothed,
                linewidth=2,
                alpha=0.7,
                label="Smoothed",
            )
            ax.legend()

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()
