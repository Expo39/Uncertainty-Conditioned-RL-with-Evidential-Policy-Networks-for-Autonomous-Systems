"""
@file build.py
@brief Single entry point for every pooled figure.

    python scripts/analysis/figures/build.py --all
    python scripts/analysis/figures/build.py --only gate_roc ekf_calibration

The split this file sits on: the analysis modules (scripts/analysis/*.py)
compute statistics and write CSVs; this file reads those CSVs and draws them;
scripts/figure_style.py holds the one house style. Computing a number and
drawing it change for different reasons, so they stay apart.

Everything here is presentation. No figure recomputes a statistic, re-bins,
re-sorts or filters beyond the reported scope, so any figure can be redrawn at
any time without moving a reported value.

Figure inventory, id -> source:

  ekf_sawtooth          a logged demo trace
  lot_layouts           the generated layout YAMLs
  training_curves       ported TensorBoard scalars
  ablation_by_condition pooled_condition_summary.csv
  degradation_tiers     raw per-episode + per-step records
  behaviour_by_std      raw per-episode records, three varying conditions
  covariance_probe      probe output (not part of the reported set)
  ekf_calibration       raw calibration records, vanilla arm
  seed_robustness       seed_robustness.csv
  gate_roc              raw episode + per-step records
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List

# Repo root on the path (NOT scripts/, which contains an `inspect` package
# that would shadow the stdlib module matplotlib imports).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import figure_style as fs  # noqa: E402

# Default locations. Override on the command line if a tree moves.
FROZEN = Path("outputs/cross_seed_analysis/all_seeds/stage6")
RAW = Path("outputs/evaluation_results")
OUT = Path("outputs/figures")

# Conditions held out of the pooled ROC, matching the five reported conditions.
_HELD_TIER = ["gnss_fixed", "gnss_degraded"]


# --- helpers ---------------------------------------------------------------
def _stage6_leaf(raw: Path, seed: str, arm: str) -> Path:
    """The stage-6 leaf for an arm/seed, preferring the no-wrapper variant."""
    d = raw / seed / arm
    st6 = sorted(p for p in d.iterdir() if p.name.startswith("6_"))
    if not st6:
        raise FileNotFoundError(f"no stage-6 checkpoint under {d}")
    return st6[0] / "without_wrapper"


def _pooled_max_total(raw: Path, seeds: List[str], arm: str) -> pd.DataFrame:
    """Per-episode max of the per-step total predictive uncertainty.

    The deployed safety layer thresholds epistemic + aleatoric at each step, so
    the gate score must be the max of that sum over the episode. Summing the
    two per-episode maxima instead would not give the same quantity, since the
    two channels need not peak on the same step.
    """
    frames = []
    for seed in seeds:
        f = _stage6_leaf(raw, seed, arm) / "per_step_records.csv"
        if not f.exists():
            continue
        ps = pd.read_csv(f, usecols=["condition", "episode", "epistemic", "aleatoric"])
        ps["max_total"] = ps["epistemic"] + ps["aleatoric"]
        g = ps.groupby(["condition", "episode"])["max_total"].max().reset_index()
        g["seed"] = seed
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["condition", "episode", "seed", "max_total"])
    return pd.concat(frames, ignore_index=True)


def _pooled_episodes(raw: Path, seeds: List[str], arms: List[str]) -> pd.DataFrame:
    """Concatenate every arm/seed's per-episode records into one frame."""
    frames = []
    for arm in arms:
        for seed in seeds:
            f = _stage6_leaf(raw, seed, arm) / "episode_records.csv"
            if not f.exists():
                continue
            ep = pd.read_csv(f)
            ep["arm"] = arm
            ep["seed"] = seed
            frames.append(ep)
    if not frames:
        raise FileNotFoundError(f"no episode_records.csv under {raw}")
    return pd.concat(frames, ignore_index=True)


# --- Success and error per condition ----------------------------------------
def ablation_by_condition(args) -> None:
    """Success and final position error per condition, grouped by arm."""
    summary = pd.read_csv(args.frozen / "pooled_condition_summary.csv")
    conditions = list(pd.unique(summary["condition"]))

    fs.apply()
    fig, axes = plt.subplots(2, 1, figsize=fs.STACK_2, sharex=True)
    handles, labels = [], []
    for ax, metric, ylabel in (
        (axes[0], "success_rate", "success rate (%)"),
        (axes[1], "mean_pos_error_m", "mean final position error (m)"),
    ):
        for i, arm in enumerate(fs.ARM_ORDER):
            a = summary[summary["arm"] == arm]
            if a.empty:
                continue
            xs = [conditions.index(c) + (i - 1.5) * 0.2 for c in a["condition"]]
            ax.bar(
                xs,
                a[metric],
                width=0.2,
                color=fs.ARM_COLOUR[arm],
                label=fs.arm_label(arm),
            )
        ax.set_ylabel(ylabel)
        fs.grid(ax, axis="y")
        if not handles:
            handles, labels = ax.get_legend_handles_labels()
    fs.condition_ticks(axes[1], conditions)
    fs.legend_strip(fig, (handles, labels), side="above")
    fs.save(fig, args.out / "ablation_by_condition")


# --- Caution proxies against EKF std -----------------------------------------
def behaviour_by_std(args) -> None:
    """Four caution proxies against binned EKF position std, on a log axis."""
    from matplotlib.ticker import FixedLocator, FuncFormatter

    df = pd.read_csv(args.frozen / "pooled_behaviour_by_std.csv")
    panels = [
        ("mean_mean_brake_cmd", "Mean brake command"),
        ("mean_mean_speed_moving_ms", "Mean speed while moving (m/s)"),
        ("mean_mean_abs_vyaw_rads", r"Mean $|$yaw rate$|$ (rad/s)"),
        ("mean_mean_action_jerk", "Mean action jerk"),
    ]
    xticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    fs.apply()
    fig, axes = plt.subplots(2, 2, figsize=fs.GRID_2X2, sharex=True)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, panels):
        for arm, disp, colour, marker, ls in fs.ARMS:
            a = df[df["arm"] == arm].sort_values("std_mid")
            if a.empty or metric not in a.columns:
                continue
            ax.plot(
                a["std_mid"],
                a[metric],
                marker=marker,
                linestyle=ls,
                color=colour,
                lw=fs.LW,
                ms=fs.MS,
                label=disp,
                clip_on=False,
            )
        ax.set_xscale("log")
        ax.set_ylabel(label, fontsize=fs.FS_LABEL)
        fs.grid(ax)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for ax in axes[2:]:
        ax.set_xlabel(
            "EKF position std bin midpoint (m, log scale)", fontsize=fs.FS_LABEL
        )
    fs.legend_strip(fig, axes[0], side="above")
    fs.save(fig, args.out / "behaviour_by_std", pad=0.5, rect=(0, 0, 1, 0.94))


# --- EKF calibration ---------------------------------------------------------
# The scatter backdrop is pooled over the three seeds, matching the pooled
# curve drawn over it. An earlier version paired a pooled curve with a
# single-seed cloud, which the caption then described as one seed.
_OOD_CONDITION = "ood_irregular_rtk_fixed"

# Held tiers pin one fix state all episode, so their std barely moves; the
# frozen binned CSV excludes them and the backdrop must match that scope.
_CAL_HELD = ["gnss_fixed", "gnss_degraded"]


def _calibration_scatter(args) -> "pd.DataFrame | None":
    """Per-step records pooled across seeds, for the calibration backdrop.

    Scope matches the frozen binned CSV: one arm (the EKF is identical across
    arms, so the arm is immaterial to calibration) over every seed, with the
    held tiers dropped. An explicit --scatter-src still wins.
    """
    if args.scatter_src and Path(args.scatter_src).exists():
        frames = [pd.read_csv(args.scatter_src)]
    else:
        found = sorted(
            Path(args.raw).glob(
                "seed_*/full_method/6_*/without_wrapper/calibration_records.csv"
            )
        )
        if not found:
            return None
        frames = [pd.read_csv(p) for p in found]
    scatter = pd.concat(frames, ignore_index=True)
    if "condition" in scatter.columns:
        scatter = scatter[~scatter["condition"].isin(_CAL_HELD)]
    if {"std_x", "std_y"} <= set(scatter.columns):
        scatter["std_pos"] = scatter[["std_x", "std_y"]].mean(axis=1)
    return scatter


def ekf_calibration(args) -> None:
    """Binned mean error against predicted std, over a per-step scatter.

    The position x-axis is logarithmic: half the steps sit on the RTK-fixed
    floor of the reported std, which a linear axis stacks into one column.
    The position error is bimodal, the upper mode being the OOD layout's
    localisation failure, so the points are split by regime and the y-axis is
    logarithmic to hold both modes. The heading panel keeps linear axes: its
    errors span no decades and a log axis would stretch near-zero values
    across empty space.
    """
    binned = pd.read_csv(args.frozen / "pooled_calibration_binned.csv")
    scatter = _calibration_scatter(args)

    corr = None
    corr_path = args.frozen / "pooled_calibration_correlations.csv"
    if corr_path.exists():
        corr = pd.read_csv(corr_path)
        corr = corr[corr["scope"] == "varying_pooled"].set_index("axis")

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=fs.SIDE_2)
    panels = [
        ("position", "std_pos", "abs_err_pos", True),
        ("heading", "std_yaw", "abs_err_yaw", False),
    ]
    handles: list = []
    labels: list[str] = []
    for ax, (axis, std_col, err_col, log_axes) in zip(axes, panels):
        if scatter is not None and {std_col, err_col} <= set(scatter.columns):
            cols = [std_col, err_col]
            has_cond = "condition" in scatter.columns
            if has_cond:
                cols = cols + ["condition"]
            pair = scatter[cols].dropna()
            if log_axes:
                # Positive-only, so a log axis keeps every drawn point.
                pair = pair[(pair[std_col] > 0) & (pair[err_col] > 0)]
            groups = [("in-distribution", pair, fs.SCATTER)]
            if has_cond:
                is_ood = pair["condition"] == _OOD_CONDITION
                groups = [
                    ("in-distribution", pair[~is_ood], fs.SCATTER),
                    (
                        fs.condition_label(_OOD_CONDITION).replace("\n", " "),
                        pair[is_ood],
                        fs.ARM_COLOUR["input_uncertainty"],
                    ),
                ]
            for label, part, colour in groups:
                if part.empty:
                    continue
                # Equal sample per group, so the sparser one stays visible.
                if len(part) > 4000:
                    part = part.sample(4000, random_state=0)
                ax.scatter(
                    part[std_col],
                    part[err_col],
                    s=3,
                    alpha=0.18,
                    color=colour,
                    edgecolors="none",
                    zorder=1,
                )
                if label not in labels:
                    handles.append(
                        plt.Line2D(
                            [],
                            [],
                            linestyle="none",
                            marker="o",
                            ms=fs.MS,
                            color=colour,
                            alpha=0.9,
                        )
                    )
                    labels.append(label)
        b = binned[binned["axis"] == axis]
        if not b.empty:
            # Geometric bin centres, to sit correctly on a log x-axis.
            centres = (
                np.sqrt(b["std_low"] * b["std_high"])
                if log_axes
                else (b["std_low"] + b["std_high"]) / 2.0
            )
            (line,) = ax.plot(
                centres,
                b["mean_abs_error"],
                color=fs.ACCENT,
                marker="o",
                linestyle="-",
                lw=fs.LW,
                ms=fs.MS,
                zorder=3,
            )
            lab = "mean error per std bin"
            if lab not in labels:
                handles.append(line)
                labels.append(lab)
        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(fs.axis_label(std_col))
        ax.set_ylabel(fs.axis_label(err_col))
        if corr is not None and axis in corr.index:
            # Low-left in the position panel: the OOD band occupies the top.
            fs.note(
                ax,
                f"Spearman {corr.loc[axis, 'spearman']:.2f}",
                corner="left",
                y=0.10 if log_axes else 0.94,
            )
        fs.grid(ax)
    if handles:
        fs.legend_strip(fig, (handles, labels), side="below")
    fs.save(fig, args.out / "ekf_calibration")


# --- Cross-seed robustness ---------------------------------------------------
def seed_robustness(args) -> None:
    """Per-arm mean success with whiskers spanning the seed range."""
    rob = pd.read_csv(args.frozen / "seed_robustness.csv")
    conditions = sorted(rob["condition"].unique())
    x_index = {c: i for i, c in enumerate(conditions)}
    arms = [a for a in fs.ARM_ORDER if a in set(rob["arm"].astype(str))]

    fs.apply()
    fig, ax = plt.subplots(figsize=fs.WIDE)
    n_arms = max(len(arms), 1)
    width = 0.8 / n_arms
    for j, arm in enumerate(arms):
        a = rob[rob["arm"].astype(str) == arm]
        xs = [x_index[c] + (j - (n_arms - 1) / 2.0) * width for c in a["condition"]]
        mean = a["success_mean_pct"].to_numpy()
        lower = mean - a["success_min_pct"].to_numpy()
        upper = a["success_max_pct"].to_numpy() - mean
        ax.errorbar(
            xs,
            mean,
            yerr=[lower, upper],
            fmt=fs.ARM_MARKER.get(arm, "o"),
            color=fs.ARM_COLOUR.get(arm, fs.MUTED),
            ms=fs.MS,
            lw=fs.LW,
            elinewidth=fs.LW,
            capsize=3,
            linestyle="none",
            label=fs.arm_label(arm),
        )
    fs.condition_ticks(ax, conditions)
    ax.set_ylabel("success rate (%)")
    fs.grid(ax, axis="y")
    fs.legend_strip(fig, ax, side="above")
    fs.save(fig, args.out / "seed_robustness")


# --- Safety-gate ROC ---------------------------------------------------------
def gate_roc(args) -> None:
    """ROC of each candidate gate signal for predicting episode failure."""
    from scripts.analysis.gate_roc import _roc_curve

    records = _pooled_episodes(args.raw, args.seeds, fs.ARM_ORDER)
    records = records[~records["condition"].isin(_HELD_TIER)]
    # Drop the unseen layout: no arm parks on it, so it contributes only
    # failures and cannot be ranked within-condition. Matches the exclusion
    # the EKF calibration statistics already apply.
    records = records[records["condition"] != "ood_irregular_rtk_fixed"]
    labels_arr = (records["success"].astype(float) == 0).astype(int)
    records = records.assign(is_failure=labels_arr)

    # Score the deployed signal: the per-step epistemic + aleatoric sum, maxed
    # over the episode, merged in from the per-step records.
    records["max_total"] = np.nan
    for arm in ("full_method", "output_uncertainty"):
        tot = _pooled_max_total(args.raw, args.seeds, arm)
        if tot.empty:
            continue
        idx = records.index[records["arm"] == arm]
        merged = records.loc[idx].merge(
            tot, on=["condition", "episode", "seed"], how="left", suffixes=("_drop", "")
        )
        records.loc[idx, "max_total"] = merged["max_total"].to_numpy()

    signals = {
        "ekf_std_pos_max_m": None,
        "max_total": ["output_uncertainty", "full_method"],
    }
    signal_dash = {"ekf_std_pos_max_m": "-", "max_total": "--"}
    fs.apply()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for arm in fs.ARM_ORDER:
        a = records[records["arm"] == arm]
        if a.empty:
            continue
        lab = a["is_failure"].to_numpy()
        for signal, allowed in signals.items():
            if allowed is not None and arm not in allowed:
                continue
            if signal not in a.columns:
                continue
            scores = a[signal].to_numpy(dtype=float)
            if np.isnan(scores).all():
                continue
            fpr, tpr, auc = _roc_curve(scores, lab)
            if np.isnan(auc):
                continue
            ax.plot(
                fpr,
                tpr,
                color=fs.ARM_COLOUR.get(arm, fs.MUTED),
                linestyle=signal_dash.get(signal, "-"),
                lw=fs.LW,
                label=f"{fs.arm_label(arm)} / {fs.signal_label(signal)} "
                f"(AUC {auc:.2f})",
            )
    ax.plot([0, 1], [0, 1], color=fs.MUTED, linestyle=":", lw=fs.LW, label="chance")
    ax.set_xlabel("false abort rate (successes needlessly aborted)")
    ax.set_ylabel("failure catch rate (failures pre-empted)")
    ax.set_aspect("equal", adjustable="box")
    fs.grid(ax)
    fs.legend_strip(fig, ax, side="below", ncol=2)
    fs.save(fig, args.out / "gate_roc")


# --- Figures delegating to their own modules ---------------------------------
def ekf_sawtooth(args) -> None:
    """EKF sawtooth: a single logged demo trace with fix-state bands."""
    from scripts.analysis.figures.ekf_sawtooth import plot as _plot

    _plot(args.trace, args.out / "ekf_sawtooth.png")


def training_curves(args) -> None:
    """Training curves across the curriculum, four arms, two panels."""
    from scripts.analysis.figures.training_curves import render

    render(args.training_data, args.out / "training_curves")


def degradation_tiers(args) -> None:
    """Success against realised localisation difficulty, in terciles."""
    from scripts.analysis.figures.degradation_tiers import plot as _plot

    _plot(args.raw, args.out / "degradation_tiers.png")


def covariance_probe(args) -> None:
    """Causal covariance probe: action response to a swept covariance."""
    from scripts.analysis.figures.covariance_probe import render

    render(args.out / "covariance_probe")


def lot_layouts(args) -> None:
    """Both parking lot layouts, drawn from the generated layout YAMLs."""
    from scripts.analysis.figures.lot_layouts import render

    render(args.out / "lot_layouts")


# Figure id -> renderer. The id is also the output filename stem.
FIGURES: Dict[str, Callable] = {
    "ablation_by_condition": ablation_by_condition,
    "behaviour_by_std": behaviour_by_std,
    "covariance_probe": covariance_probe,
    "degradation_tiers": degradation_tiers,
    "ekf_calibration": ekf_calibration,
    "ekf_sawtooth": ekf_sawtooth,
    "gate_roc": gate_roc,
    "lot_layouts": lot_layouts,
    "seed_robustness": seed_robustness,
    "training_curves": training_curves,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate the analysed-data figures in the house style."
    )
    ap.add_argument("--all", action="store_true", help="Draw every figure.")
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="FIG",
        help=f"Draw a subset: {' '.join(sorted(FIGURES))}",
    )
    ap.add_argument(
        "--frozen", type=Path, default=FROZEN, help="Pooled cross-seed CSVs."
    )
    ap.add_argument(
        "--raw", type=Path, default=RAW, help="Per-episode evaluation tree."
    )
    ap.add_argument("--out", type=Path, default=OUT, help="Output directory.")
    ap.add_argument("--seeds", nargs="+", default=["seed_42", "seed_123", "seed_7"])
    ap.add_argument(
        "--scatter-src",
        type=Path,
        default=None,
        help="calibration_records.csv for the calibration backdrop.",
    )
    ap.add_argument(
        "--trace", type=Path, default=None, help="Demo-trace CSV for the EKF sawtooth."
    )
    ap.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Ported TensorBoard series for the training curves.",
    )
    args = ap.parse_args()

    if not args.all and not args.only:
        ap.error("choose --all or --only")
    wanted = sorted(FIGURES) if args.all else args.only
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s): {' '.join(unknown)}")

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in wanted:
        try:
            FIGURES[name](args)
        except Exception as exc:  # keep going: one missing input is not fatal
            print(f"  !! {name} skipped: {type(exc).__name__}: {exc}")
            failed.append(name)
    drawn = [w for w in wanted if w not in failed]
    print(f"\ndrew {len(drawn)}/{len(wanted)}: {' '.join(drawn)}")
    if failed:
        print(f"skipped: {' '.join(failed)}")


if __name__ == "__main__":
    main()
