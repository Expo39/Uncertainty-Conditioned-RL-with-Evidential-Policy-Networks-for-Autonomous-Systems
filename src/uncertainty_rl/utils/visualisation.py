## @file visualisation.py
#  @brief Visualisation utilities for uncertainty and performance metrics.
#
#  This module provides visualisation tools for understanding agent behaviour
#  and uncertainty evolution during training and evaluation.
from typing import List, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


## @brief Plot the evolution of uncertainties over time.
#  @param epistemic: List of epistemic uncertainty values.
#  @param aleatoric: List of aleatoric uncertainty values.
#  @param save_path: Path to save the plot.
#  @param title: Plot title.
def plot_uncertainty_evolution(
    epistemic: List[float],
    aleatoric: List[float],
    save_path: Optional[str] = None,
    title: str = "Uncertainty Evolution"
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = np.arange(len(epistemic))
    
    # Epistemic uncertainty
    ax1.plot(steps, epistemic, label='Epistemic', color='blue', alpha=0.7)
    ax1.fill_between(steps, 0, epistemic, alpha=0.3, color='blue')
    ax1.set_ylabel('Epistemic Uncertainty', fontsize=12)
    ax1.set_title(f'{title} - Epistemic', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Aleatoric uncertainty
    ax2.plot(steps, aleatoric, label='Aleatoric', color='red', alpha=0.7)
    ax2.fill_between(steps, 0, aleatoric, alpha=0.3, color='red')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Aleatoric Uncertainty', fontsize=12)
    ax2.set_title(f'{title} - Aleatoric', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


## @brief Plot vehicle trajectory with uncertainty ellipses.
#  @param positions: Array of (x, y) positions, shape (N, 2).
#  @param target: Target position (x, y, yaw).
#  @param uncertainties: Optional uncertainty covariances, shape (N, 2, 2).
#  @param save_path: Path to save the plot.
#  @param title: Plot title.
def plot_trajectory(
    positions: np.ndarray,
    target: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Vehicle Trajectory"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=15, label='Start')
    ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=15, label='End')
    
    # Plot target
    ax.plot(target[0], target[1], 'r*', markersize=20, label='Target')
    
    # Draw target orientation
    arrow_length = 2.0
    dx = arrow_length * np.cos(target[2])
    dy = arrow_length * np.sin(target[2])
    ax.arrow(target[0], target[1], dx, dy, 
            head_width=0.5, head_length=0.3, fc='red', ec='red')
    
    # Plot uncertainty ellipses if provided
    if uncertainties is not None:
        from matplotlib.patches import Ellipse
        
        # Sample points for ellipses
        sample_indices = np.linspace(0, len(positions)-1, min(10, len(positions)), dtype=int)
        
        for idx in sample_indices:
            pos = positions[idx]
            cov = uncertainties[idx]
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # 95% confidence ellipse (chi-square with 2 DOF)
            width, height = 2 * np.sqrt(5.991 * eigenvalues)
            
            ellipse = Ellipse(pos, width, height, angle=angle,
                            facecolor='blue', alpha=0.1, edgecolor='blue', linewidth=1)
            ax.add_patch(ellipse)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()


## @brief Plot training curves for multiple metrics.
#  @param metrics: Dictionary mapping metric names to lists of values.
#  @param save_path: Path to save the plot.
#  @param title: Plot title.
def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> None:
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (name, values) in enumerate(metrics.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        steps = np.arange(len(values))
        
        ax.plot(steps, values, linewidth=2)
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add smoothed curve
        if len(values) > 10:
            window_size = min(50, len(values) // 10)
            smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            ax.plot(steps[window_size-1:], smoothed, linewidth=2, alpha=0.7, label='Smoothed')
            ax.legend()
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()
