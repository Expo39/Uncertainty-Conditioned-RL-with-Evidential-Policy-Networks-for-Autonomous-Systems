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
  training_curves       exported TensorBoard scalars (make training-curves)
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
from typing import Any, Callable, Dict, List, Optional

# Repo root on the path (NOT scripts/, which contains an `inspect` package
# that would shadow the stdlib module matplotlib imports).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts import figure_style as fs  # noqa: E402
from scripts.analysis._discovery import pooled_frame  # noqa: E402
from scripts.analysis.ablation import drop_unreported, keep_varying  # noqa: E402

# Default locations. Override on the command line if a tree moves.
FROZEN = Path("outputs/cross_seed_analysis/all_seeds/stage6")
RAW = Path("outputs/evaluation_results")
OUT = Path("outputs/figures")

# Demo trace behind the sawtooth. One logged episode under the live chain; any
# trace carrying a gnss_tier column works, this one is simply the committed one.
DEFAULT_TRACE = Path(
    "outputs/demo_traces/full_method/6_42_22062026-1502"
    "/15-07-2026-143405/episode_107.csv"
)

# Arms carrying an evidential head, so a total-predictive score exists.
EVIDENTIAL_ARMS = ["output_uncertainty", "full_method"]

# Behavioural proxies binned against the reported std, and their axis labels.
_PROXIES = (
    ("mean_brake_cmd", "Mean brake command"),
    ("mean_speed_moving_ms", "Mean speed while moving (m/s)"),
    ("mean_abs_vyaw_rads", r"Mean $|$yaw rate$|$ (rad/s)"),
    ("mean_action_jerk", "Mean action jerk"),
)

# Equal-population bands cut within each condition.
_N_BANDS = 5


# --- helpers ---------------------------------------------------------------
def _pooled_max_total(raw: Path, seeds: List[str], arm: str) -> pd.DataFrame:
    """
    @brief Per-episode max of the per-step total predictive uncertainty.
    @param raw: Evaluation results root.
    @param seeds: Seed sub-roots to pool.
    @param arm: Baseline name.
    @return Frame of condition / episode / seed / max_total.

    The deployed safety layer thresholds epistemic + aleatoric at each step, so
    the gate score is the max of that SUM over the episode. Summing the two
    per-episode maxima is a different quantity: the channels need not peak on
    the same step.
    """
    try:
        per_step = pooled_frame(
            raw,
            seeds,
            [arm],
            "per_step_records.csv",
            usecols=["condition", "episode", "epistemic", "aleatoric"],
        )
    except FileNotFoundError:
        return pd.DataFrame(columns=["condition", "episode", "seed", "max_total"])
    per_step["max_total"] = per_step["epistemic"] + per_step["aleatoric"]
    return (
        per_step.groupby(["condition", "episode", "seed"])["max_total"]
        .max()
        .reset_index()
    )


def _banded_proxies(records: pd.DataFrame) -> pd.DataFrame:
    """
    @brief Average each behavioural proxy over equal-population std bands.
    @param records: Pooled per-episode frame, already restricted to the
           varying conditions.
    @return One row per arm and band: band index, std midpoint, proxy means.

    Bands are cut WITHIN each condition and then averaged across them. Cutting
    the pooled frame instead lets the condition mix swing along the sweep - the
    obstacle-free condition concentrates in the top band, where the absence of
    parked vehicles raises the mean speed on its own - which would confound
    behaviour with composition. The band x position is the arithmetic midpoint
    of its observed std range.
    """
    metrics = [m for m, _ in _PROXIES]
    per_condition: List[pd.DataFrame] = []
    for (arm, condition), group in records.groupby(["arm", "condition"]):
        std = group["ekf_std_pos_mean_m"]
        # Equal-population cut; duplicates="drop" tolerates a tied std floor.
        bands = pd.qcut(std.rank(method="first"), _N_BANDS, labels=False)
        block = group.assign(band=bands)
        agg = block.groupby("band").agg(
            **{m: (m, "mean") for m in metrics},
            std_low=("ekf_std_pos_mean_m", "min"),
            std_high=("ekf_std_pos_mean_m", "max"),
        )
        agg["arm"] = arm
        agg["condition"] = condition
        per_condition.append(agg.reset_index())
    if not per_condition:
        return pd.DataFrame()
    stacked = pd.concat(per_condition, ignore_index=True)
    out = stacked.groupby(["arm", "band"]).agg(
        **{m: (m, "mean") for m in metrics},
        std_low=("std_low", "mean"),
        std_high=("std_high", "mean"),
    )
    out["std_mid"] = (out["std_low"] + out["std_high"]) / 2.0
    return out.reset_index()


# --- Success and error per condition ----------------------------------------
def ablation_by_condition(args) -> None:
    """
    @brief Success and final position error per condition, grouped by arm.
    @param args: Parsed CLI namespace.
    """
    summary = drop_unreported(pd.read_csv(args.frozen / "pooled_condition_summary.csv"))
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
    """
    @brief Four caution proxies against banded EKF position std, on a log axis.
    @param args: Parsed CLI namespace.

    Built from the raw per-episode records rather than the pooled CSV: the
    pooled file carries no condition column, so it cannot be restricted to the
    conditions whose tier actually varies in-episode.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter

    records = keep_varying(
        pooled_frame(args.raw, args.seeds, fs.ARM_ORDER, "episode_records.csv")
    )
    df = _banded_proxies(records)
    xticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    fs.apply()
    fig, axes = plt.subplots(2, 2, figsize=fs.GRID_2X2, sharex=True)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, _PROXIES):
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
# One arm characterises the filter: the EKF is identical across arms, so the
# arm is immaterial to calibration. Mixing arms is NOT equivalent - the pooled
# bin means then dip at bin 2 and lose the monotone rise the reading rests on.
CALIBRATION_ARM = "vanilla_ppo"

# Equal-count bins over the reported std.
_N_CAL_BINS = 5


def _calibration_records(args) -> Optional[pd.DataFrame]:
    """
    @brief Per-step calibration pairs for the reported scope.
    @param args: Parsed CLI namespace; --scatter-src overrides discovery.
    @return Frame with std_pos added, or None when no records are found.

    One arm over every seed, restricted to the conditions whose tier varies
    in-episode: a pinned tier holds the std at its floor, giving nothing to
    correlate error against.
    """
    if args.scatter_src and Path(args.scatter_src).exists():
        scatter = pd.read_csv(args.scatter_src)
    else:
        try:
            scatter = pooled_frame(
                args.raw, args.seeds, [CALIBRATION_ARM], "calibration_records.csv"
            )
        except FileNotFoundError:
            return None
    scatter = keep_varying(scatter)
    if {"std_x", "std_y"} <= set(scatter.columns):
        scatter["std_pos"] = scatter[["std_x", "std_y"]].mean(axis=1)
    return scatter


def _calibration_bins(
    scatter: pd.DataFrame, std_col: str, err_col: str
) -> pd.DataFrame:
    """
    @brief Mean absolute error within equal-count bins of the reported std.
    @param scatter: Per-step calibration frame.
    @param std_col: Reported-std column to bin on.
    @param err_col: Error column to average.
    @return Frame of std_low / std_high / mean_abs_error, one row per bin.

    Equal COUNT, not equal range: roughly half the steps sit on the RTK-fixed
    floor, so equal-range bins would put nearly everything in the first bin.
    duplicates="drop" tolerates the ties that floor creates.
    """
    pair = scatter[[std_col, err_col]].dropna()
    if pair.empty:
        return pd.DataFrame(columns=["std_low", "std_high", "mean_abs_error"])
    bins = pd.qcut(pair[std_col].rank(method="first"), _N_CAL_BINS, labels=False)
    grouped = pair.assign(bin=bins).groupby("bin")
    return pd.DataFrame(
        {
            "std_low": grouped[std_col].min(),
            "std_high": grouped[std_col].max(),
            "mean_abs_error": grouped[err_col].mean(),
        }
    ).reset_index(drop=True)


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """
    @brief Spearman rank correlation, via ranks so no scipy dependency.
    @param x: First series.
    @param y: Second series.
    @return The rank correlation, or nan when either side is constant.
    """
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return float("nan")
    return float(pair.iloc[:, 0].rank().corr(pair.iloc[:, 1].rank()))


def ekf_calibration(args) -> None:
    """
    @brief Binned mean error against predicted std, over a per-step scatter.
    @param args: Parsed CLI namespace.

    The position panel is log-log: roughly half the steps sit on the RTK-fixed
    floor of the reported std, which a linear axis stacks into one column, and
    the errors span decades. The heading panel stays linear - its errors span
    no decades, and a log axis would stretch near-zero values across empty
    space.
    """
    scatter = _calibration_records(args)
    if scatter is None:
        raise FileNotFoundError("no calibration_records.csv found")

    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=fs.SIDE_2)
    panels = [
        ("std_pos", "abs_err_pos", True),
        ("std_yaw", "abs_err_yaw", False),
    ]
    handles: List[Any] = []
    labels: List[str] = []
    for ax, (std_col, err_col, log_axes) in zip(axes, panels):
        pair = scatter[[std_col, err_col]].dropna()
        if log_axes:
            # Positive-only, so a log axis keeps every drawn point.
            pair = pair[(pair[std_col] > 0) & (pair[err_col] > 0)]
        if not pair.empty:
            drawn = pair.sample(4000, random_state=0) if len(pair) > 4000 else pair
            ax.scatter(
                drawn[std_col],
                drawn[err_col],
                s=3,
                alpha=0.18,
                color=fs.SCATTER,
                edgecolors="none",
                zorder=1,
            )
            if "in-distribution" not in labels:
                handles.append(
                    plt.Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        ms=fs.MS,
                        color=fs.SCATTER,
                        alpha=0.9,
                    )
                )
                labels.append("in-distribution")

        binned = _calibration_bins(scatter, std_col, err_col)
        if not binned.empty:
            # Geometric centres on a log axis, arithmetic on a linear one.
            centres = (
                np.sqrt(binned["std_low"] * binned["std_high"])
                if log_axes
                else (binned["std_low"] + binned["std_high"]) / 2.0
            )
            (line,) = ax.plot(
                centres,
                binned["mean_abs_error"],
                color=fs.ACCENT,
                marker="o",
                linestyle="-",
                lw=fs.LW,
                ms=fs.MS,
                zorder=3,
            )
            if "mean error per std bin" not in labels:
                handles.append(line)
                labels.append("mean error per std bin")

        if log_axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(fs.axis_label(std_col))
        ax.set_ylabel(fs.axis_label(err_col))
        rho = _spearman(scatter[std_col], scatter[err_col])
        if rho == rho:  # not nan
            fs.note(
                ax, f"Spearman {rho:.2f}", corner="left", y=0.10 if log_axes else 0.94
            )
        fs.grid(ax)
    if handles:
        fs.legend_strip(fig, (handles, labels), side="below")
    fs.save(fig, args.out / "ekf_calibration")


# --- Cross-seed robustness ---------------------------------------------------
def seed_robustness(args) -> None:
    """Per-arm mean success with whiskers spanning the seed range."""
    rob = drop_unreported(pd.read_csv(args.frozen / "seed_robustness.csv"))
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
    """
    @brief ROC of each candidate gate signal for predicting episode failure.
    @param args: Parsed CLI namespace.

    Restricted to the evidential arms: neither standard-head arm carries an
    evidential head, so neither contributes a total-predictive curve, and an
    EKF-std curve for them would compare unlike sets of signals.
    """
    from scripts.analysis.gate_roc import _roc_curve

    records = keep_varying(
        pooled_frame(args.raw, args.seeds, EVIDENTIAL_ARMS, "episode_records.csv")
    )
    records = records.assign(
        is_failure=(records["success"].astype(float) == 0).astype(int)
    )

    # Score the deployed signal: the per-step epistemic + aleatoric sum, maxed
    # over the episode, merged in from the per-step records.
    records["max_total"] = np.nan
    for arm in EVIDENTIAL_ARMS:
        tot = _pooled_max_total(args.raw, args.seeds, arm)
        if tot.empty:
            continue
        idx = records.index[records["arm"] == arm]
        merged = records.loc[idx].merge(
            tot, on=["condition", "episode", "seed"], how="left", suffixes=("_drop", "")
        )
        records.loc[idx, "max_total"] = merged["max_total"].to_numpy()

    signal_dash = {"ekf_std_pos_max_m": "-", "max_total": "--"}
    fs.apply()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for arm in EVIDENTIAL_ARMS:
        a = records[records["arm"] == arm]
        if a.empty:
            continue
        lab = a["is_failure"].to_numpy()
        for signal in signal_dash:
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
    """
    @brief Training curves across the curriculum, four arms, two panels.
    @param args: Parsed CLI namespace.
    """
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
        "--trace",
        type=Path,
        default=DEFAULT_TRACE,
        help="Demo-trace CSV for the EKF sawtooth.",
    )
    ap.add_argument(
        "--training-data",
        type=Path,
        default=None,
        help="Directory holding the exported training-curve CSVs.",
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
