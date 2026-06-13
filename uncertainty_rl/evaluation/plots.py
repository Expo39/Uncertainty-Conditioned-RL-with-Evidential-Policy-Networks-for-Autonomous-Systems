"""
@file plots.py
@brief Seaborn/matplotlib figures for the evaluation condition sweep.

Renders the two evaluation figures from the per-condition results DataFrame:
the 2x2 summary panel (success / reward / steps / policy uncertainty) and the
stacked failure-mode breakdown ("degrades gracefully"). Split out of evaluate.py
so the plotting code - the only part that pulls in matplotlib/seaborn - is
isolated from the evaluation loop.
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("uncertainty_rl.evaluation")


def plot_evaluation_results(
    df: pd.DataFrame,
    output_dir: str = "./evaluation_results",
) -> None:
    """
    @brief Create visualisations of evaluation results across conditions.
    @param df: DataFrame with evaluation results.
    @param output_dir: Directory to save plots.
    """
    sns.set_style("whitegrid")

    conditions = df["condition"].tolist()
    x_arr = np.arange(len(conditions))
    x_list = x_arr.tolist()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plots 1-3: single-series bar charts, data-driven
    bar_specs = [
        (
            axes[0, 0],
            "success_rate",
            "steelblue",
            "Success Rate (%)",
            "Success Rate vs Condition",
        ),
        (
            axes[0, 1],
            "average_reward",
            "forestgreen",
            "Average Reward",
            "Average Reward vs Condition",
        ),
        (
            axes[1, 0],
            "average_steps",
            "firebrick",
            "Average Steps",
            "Average Steps to Termination vs Condition",
        ),
    ]
    for ax, col, colour, ylabel, title in bar_specs:
        ax.bar(x_list, df[col], color=colour, alpha=0.8)
        ax.set_xticks(x_list)
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Policy uncertainty estimates (grouped bars). Only the evidential
    # head emits per-state epistemic/aleatoric values; standard Gaussian arms
    # leave these columns at zero (the extraction path is gated on
    # policy_type=evidential). Detect that case and annotate, so a blank panel
    # is never confused with an evidential head that happened to output zero.
    axes[1, 1].set_title("Policy Uncertainty Estimates", fontsize=14)
    has_uncertainty = "mean_epistemic_uncertainty" in df.columns and bool(
        (
            df["mean_epistemic_uncertainty"].abs()
            + df["mean_aleatoric_uncertainty"].abs()
        )
        .gt(0.0)
        .any()
    )
    if has_uncertainty:
        bar_width = 0.35
        axes[1, 1].bar(
            x_arr - bar_width / 2,
            df["mean_epistemic_uncertainty"],
            bar_width,
            label="Epistemic",
            alpha=0.8,
        )
        axes[1, 1].bar(
            x_arr + bar_width / 2,
            df["mean_aleatoric_uncertainty"],
            bar_width,
            label="Aleatoric",
            alpha=0.8,
        )
        axes[1, 1].set_xticks(x_list)
        axes[1, 1].set_xticklabels(conditions, rotation=45, ha="right")
        axes[1, 1].set_ylabel("Uncertainty", fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis="y")
    else:
        # Standard Gaussian head: no evidential uncertainty signal exists.
        axes[1, 1].text(
            0.5,
            0.5,
            "N/A - standard head\n(no evidential uncertainty output)",
            ha="center",
            va="center",
            fontsize=12,
            color="dimgrey",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Plots saved to %s", plot_path)

    plt.close()

    # Failure-mode breakdown: stacked outcome shares per condition. Success at
    # the base, then the failure taxonomy - the "degrades gracefully" figure.
    outcome_specs = [
        ("success_rate", "Success", "steelblue"),
        ("near_miss_rate", "Near miss", "gold"),
        ("stuck_rate", "Stuck", "darkorange"),
        ("handoff_rate", "Handoff", "slategrey"),
        ("out_of_bounds_rate", "Out of bounds", "sienna"),
        ("collision_rate", "Collision", "firebrick"),
    ]
    present = [s for s in outcome_specs if s[0] in df.columns]
    if present:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(conditions))
        for col, label, colour in present:
            values = df[col].to_numpy(dtype=float)
            ax2.bar(x_list, values, bottom=bottom, label=label, color=colour)
            bottom += values
        ax2.set_xticks(x_list)
        ax2.set_xticklabels(conditions, rotation=45, ha="right")
        ax2.set_ylabel("Share of Episodes (%)", fontsize=12)
        ax2.set_title("Episode Outcomes vs Condition", fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        outcome_path = os.path.join(output_dir, "failure_modes.png")
        plt.savefig(outcome_path, dpi=300, bbox_inches="tight")
        logger.info("Failure-mode plot saved to %s", outcome_path)
        plt.close()
