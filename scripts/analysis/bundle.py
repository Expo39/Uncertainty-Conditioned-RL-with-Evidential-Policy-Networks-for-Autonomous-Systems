"""
@file bundle.py
@brief Assemble the curated analysis bundle from the raw outputs tree.

`outputs/` holds everything an evaluation and its analyses produce, which is
far more than the headline set: five conditions where four are analysed, every
arm where one characterises the filter, plus per-run panels and superseded
trees. This module gathers only what is in the headline set into one directory:

    outputs/main_analysis/
      figures/   the headline figures (written by `make figures`)
      summaries/ one tidy CSV per derived summary
      values/    the pooled CSVs the headline metrics are read from
      MANIFEST.md

Summaries are RECOMPUTED here through the shared scope filters rather than copied,
so every cell traces back to the per-episode records. Anything computed but not
reported - the lidar_degraded condition, the covariance probe, the per-run
panels - stays in the raw tree and is not copied.

@note Read-only over the raw tree. Pure CPU, no torch and no CARLA.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from scripts.analysis._discovery import pooled_frame  # noqa: E402
from scripts.analysis.ablation import (  # noqa: E402
    _ARM_ORDER,
    drop_unreported,
    keep_varying,
)
from scripts.analysis.figures.build import (  # noqa: E402
    CALIBRATION_ARM,
    EVIDENTIAL_ARMS,
    _banded_proxies,
    _calibration_bins,
    _pooled_max_total,
    _spearman,
)
from scripts.analysis.gate_roc import _roc_curve  # noqa: E402

# Default roots.
FROZEN = Path("outputs/raw_derived/cross_seed_analysis/all_seeds/stage6")
RAW = Path("outputs/raw/evaluation_results")
OUT = Path("outputs/main_analysis")
SEEDS = ["seed_42", "seed_123", "seed_7"]

# Pooled CSVs copied verbatim as the values behind the headline metrics.
REPORTED_VALUES: List[str] = [
    "per_seed_summary.csv",
    "pooled_behaviour_by_std.csv",
    "pooled_calibration_binned.csv",
    "pooled_calibration_correlations.csv",
    "pooled_caution_contrast.csv",
    "pooled_caution_levels.csv",
    "pooled_caution_slopes.csv",
    "pooled_condition_summary.csv",
    "pooled_covariance_contrasts.csv",
    "pooled_gate_auc.csv",
    "pooled_handover_timing.csv",
    "seed_robustness.csv",
    "seed_robustness_contrasts.csv",
]

# Behavioural proxies, in the order the conditioning summary lists them.
_PROXY_COLUMNS = [
    "mean_brake_cmd",
    "mean_speed_moving_ms",
    "mean_abs_vyaw_rads",
    "mean_action_jerk",
]


def _success_by_condition(frozen: Path) -> pd.DataFrame:
    """
    @brief Success rate per condition and arm, headline scope.
    @param frozen: Pooled cross-seed directory.
    @return Conditions as rows, arms as columns.
    """
    summary = drop_unreported(pd.read_csv(frozen / "pooled_condition_summary.csv"))
    return summary.pivot(
        index="condition", columns="arm", values="success_rate"
    ).reindex(columns=_ARM_ORDER)


def _position_error(frozen: Path, raw: Path, seeds: List[str]) -> pd.DataFrame:
    """
    @brief Mean and median final position error per condition and arm.
    @param frozen: Pooled cross-seed directory.
    @param raw: Per-episode evaluation tree.
    @param seeds: Seed sub-roots to pool.
    @return One row per condition, mean_<arm> and median_<arm> columns.

    The mean comes from the pooled summary; the median is recomputed from the
    per-episode records, which is the only place it exists. They disagree where
    an arm parks more often and its surviving failures end further out, so both
    are analysed rather than one standing for the other.
    """
    summary = drop_unreported(pd.read_csv(frozen / "pooled_condition_summary.csv"))
    means = summary.pivot(
        index="condition", columns="arm", values="mean_pos_error_m"
    ).reindex(columns=_ARM_ORDER)
    means.columns = [f"mean_{c}" for c in means.columns]

    episodes = drop_unreported(
        pooled_frame(raw, seeds, _ARM_ORDER, "episode_records.csv")
    )
    medians = (
        episodes.groupby(["condition", "arm"])["final_pos_error_m"]
        .median()
        .unstack("arm")
        .reindex(columns=_ARM_ORDER)
    )
    medians.columns = [f"median_{c}" for c in medians.columns]
    return means.join(medians)


def _covariance_contrasts(frozen: Path) -> pd.DataFrame:
    """
    @brief Success-rate contrasts with their bootstrap intervals.
    @param frozen: Pooled cross-seed directory.
    @return The reported-scope rows of the pooled contrast CSV.
    """
    contrasts = drop_unreported(pd.read_csv(frozen / "pooled_covariance_contrasts.csv"))
    return contrasts[
        [
            "pair",
            "treatment",
            "control",
            "condition",
            "success_delta_pp",
            "success_ci_low",
            "success_ci_high",
            "success_significant",
        ]
    ]


def _conditioning_analysis(raw: Path, seeds: List[str]) -> pd.DataFrame:
    """
    @brief Rank correlations and operating levels over the varying conditions.
    @param raw: Per-episode evaluation tree.
    @param seeds: Seed sub-roots to pool.
    @return One row per metric, one column per arm.

    Correlations are cut WITHIN each condition and averaged over the three, so
    a differing condition mix cannot masquerade as a behavioural response. The
    operating levels need no such treatment: each condition contributes the
    same episode count per arm, so the pool is balanced by construction.
    """
    episodes = keep_varying(pooled_frame(raw, seeds, _ARM_ORDER, "episode_records.csv"))
    rows: Dict[str, Dict[str, float]] = {}

    for proxy in _PROXY_COLUMNS:
        per_arm: Dict[str, float] = {}
        for arm in _ARM_ORDER:
            block = episodes[episodes["arm"] == arm]
            within = [
                _spearman(part["ekf_std_pos_mean_m"], part[proxy])
                for _, part in block.groupby("condition")
            ]
            within = [v for v in within if v == v]
            per_arm[arm] = sum(within) / len(within) if within else float("nan")
        rows[f"spearman_{proxy}"] = per_arm

    for proxy in ("mean_speed_moving_ms", "mean_brake_cmd"):
        rows[f"level_{proxy}"] = {
            arm: float(episodes[episodes["arm"] == arm][proxy].mean())
            for arm in _ARM_ORDER
        }
    rows["median_pos_error_m"] = {
        arm: float(episodes[episodes["arm"] == arm]["final_pos_error_m"].median())
        for arm in _ARM_ORDER
    }
    rows["success_pct"] = {
        arm: float(episodes[episodes["arm"] == arm]["success"].mean() * 100.0)
        for arm in _ARM_ORDER
    }
    return pd.DataFrame(rows).T.reindex(columns=_ARM_ORDER)


def _seed_unanimity(frozen: Path) -> pd.DataFrame:
    """
    @brief Per-seed success spread and the matched-seed contrast.
    @param frozen: Pooled cross-seed directory.
    @return One row per condition for the evidential-head pair.
    """
    rob = keep_varying(drop_unreported(pd.read_csv(frozen / "seed_robustness.csv")))
    spread = rob[rob["arm"].isin(EVIDENTIAL_ARMS)].pivot(
        index="condition",
        columns="arm",
        values=["success_min_pct", "success_max_pct"],
    )
    spread.columns = [f"{stat}_{arm}" for stat, arm in spread.columns]

    contrasts = keep_varying(
        drop_unreported(pd.read_csv(frozen / "seed_robustness_contrasts.csv"))
    )
    evidential = contrasts[contrasts["pair"] == "evidential_head"].set_index(
        "condition"
    )
    matched = evidential[
        ["success_delta_min_pp", "success_delta_max_pp", "success_delta_mean_pp"]
    ]
    return spread.join(matched)


def _gate_auc(raw: Path, seeds: List[str]) -> pd.DataFrame:
    """
    @brief Failure-prediction AUC per arm and gate signal.
    @param raw: Per-episode evaluation tree.
    @param seeds: Seed sub-roots to pool.
    @return One row per arm and signal.
    """
    records = keep_varying(
        pooled_frame(raw, seeds, EVIDENTIAL_ARMS, "episode_records.csv")
    )
    records = records.assign(
        is_failure=(records["success"].astype(float) == 0).astype(int)
    )
    records["max_total"] = float("nan")
    for arm in EVIDENTIAL_ARMS:
        totals = _pooled_max_total(raw, seeds, arm)
        if totals.empty:
            continue
        idx = records.index[records["arm"] == arm]
        merged = records.loc[idx].merge(
            totals,
            on=["condition", "episode", "seed"],
            how="left",
            suffixes=("_drop", ""),
        )
        records.loc[idx, "max_total"] = merged["max_total"].to_numpy()

    rows: List[Dict[str, object]] = []
    for arm in EVIDENTIAL_ARMS:
        block = records[records["arm"] == arm]
        labels = block["is_failure"].to_numpy()
        for signal in ("ekf_std_pos_max_m", "max_total"):
            _, _, auc = _roc_curve(block[signal].to_numpy(dtype=float), labels)
            rows.append(
                {"arm": arm, "signal": signal, "auc": auc, "n_episodes": len(block)}
            )
    return pd.DataFrame(rows)


def _calibration(raw: Path, seeds: List[str]) -> pd.DataFrame:
    """
    @brief Rank correlation and binned mean error for the headline scope.
    @param raw: Per-episode evaluation tree.
    @param seeds: Seed sub-roots to pool.
    @return One row per bin, with the pooled correlations repeated alongside.
    """
    records = keep_varying(
        pooled_frame(raw, seeds, [CALIBRATION_ARM], "calibration_records.csv")
    )
    records["std_pos"] = records[["std_x", "std_y"]].mean(axis=1)
    binned = _calibration_bins(records, "std_pos", "abs_err_pos")
    binned.insert(0, "bin", range(1, len(binned) + 1))
    binned["spearman_position"] = _spearman(records["std_pos"], records["abs_err_pos"])
    binned["spearman_heading"] = _spearman(records["std_yaw"], records["abs_err_yaw"])
    binned["n_steps"] = len(records)
    binned["arm"] = CALIBRATION_ARM
    return binned


def _behaviour_bands(raw: Path, seeds: List[str]) -> pd.DataFrame:
    """
    @brief Behavioural proxies per arm and uncertainty band.
    @param raw: Per-episode evaluation tree.
    @param seeds: Seed sub-roots to pool.
    @return The banded proxy summary behind the behaviour figure.
    """
    episodes = keep_varying(pooled_frame(raw, seeds, _ARM_ORDER, "episode_records.csv"))
    bands = _banded_proxies(episodes)
    bands["band"] = bands["band"] + 1
    return bands


def _write_manifest(out: Path, summaries: Dict[str, str], figures: List[str]) -> None:
    """
    @brief Record where each bundled artefact comes from.
    @param out: Bundle root.
    @param summaries: Summary file name -> one-line description.
    @param figures: Figure names already present in the bundle.
    """
    lines = [
        "# Analysis bundle",
        "",
        "Curated from the raw `outputs/` tree by `make analysis-bundle`.",
        "Tables are recomputed from the per-episode records through the shared",
        "scope filters, so every cell traces back to raw data. Figures are",
        "written straight into this directory by `make figures`.",
        "",
        "## Scope",
        "",
        "- Reported conditions drop `lidar_degraded`: it corrupts only the",
        "  obstacle channel, and the EKF never consumes LiDAR, so the",
        "  localisation std stays pinned at its floor.",
        "- The held-tier pair (`gnss_fixed`, `gnss_degraded`) pins one fix state",
        "  for a whole episode, so nothing degrades within one.",
        "- Behaviour and calibration readings use the three conditions whose",
        "  tier varies in-episode.",
        "",
        "## Tables",
        "",
    ]
    lines += [
        f"- `summaries/{name}` - {desc}" for name, desc in sorted(summaries.items())
    ]
    lines += ["", "## Figures", ""]
    lines += [f"- `figures/{name}.png`" for name in figures]
    lines += [
        "",
        "## Values",
        "",
        "`values/` holds the pooled cross-seed CSVs verbatim, as written by",
        "`make analyse-cross-seed STAGE=6`.",
        "",
    ]
    (out / "MANIFEST.md").write_text("\n".join(lines))


def build(frozen: Path, raw: Path, out: Path, seeds: List[str]) -> None:
    """
    @brief Assemble the whole bundle.
    @param frozen: Pooled cross-seed directory.
    @param raw: Per-episode evaluation tree.
    @param out: Bundle root; recreated from scratch each run.
    @param seeds: Seed sub-roots to pool.
    """
    # figures/ is written directly by `make figures`; only the derived
    # summaries and copied values are rebuilt here.
    for sub in ("summaries", "values"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for sub in ("summaries", "values"):
        (out / sub).mkdir(parents=True)

    summaries = {
        "success_by_condition.csv": "success rate per condition and arm",
        "position_error.csv": "mean and median final position error",
        "covariance_contrasts.csv": "contrasts with 95% bootstrap intervals",
        "conditioning_analysis.csv": "rank correlations and operating levels",
        "seed_unanimity.csv": "per-seed spread and matched-seed contrast",
        "gate_auc.csv": "failure-prediction AUC per arm and signal",
        "calibration.csv": "rank correlation and binned mean error",
        "behaviour_bands.csv": "behavioural proxies per arm and band",
        "per_seed_summary.csv": "per-seed success and error, one row per run",
    }

    built = {
        "success_by_condition.csv": _success_by_condition(frozen),
        "position_error.csv": _position_error(frozen, raw, seeds),
        "covariance_contrasts.csv": _covariance_contrasts(frozen),
        "conditioning_analysis.csv": _conditioning_analysis(raw, seeds),
        "seed_unanimity.csv": _seed_unanimity(frozen),
        "gate_auc.csv": _gate_auc(raw, seeds),
        "calibration.csv": _calibration(raw, seeds),
        "behaviour_bands.csv": _behaviour_bands(raw, seeds),
        "per_seed_summary.csv": drop_unreported(
            pd.read_csv(frozen / "per_seed_summary.csv")
        ),
    }
    for name, frame in built.items():
        index = not isinstance(frame.index, pd.RangeIndex)
        frame.to_csv(out / "summaries" / name, index=index)
        print(f"  summary {name} ({len(frame)} rows)")

    for name in REPORTED_VALUES:
        source = frozen / name
        if source.exists():
            shutil.copy2(source, out / "values" / name)
    print(f"  values  {len(list((out / 'values').iterdir()))} copied")

    figures = sorted(p.stem for p in (out / "figures").glob("*.png"))
    _write_manifest(out, summaries, figures)
    print(f"\nbundle written to {out}")


def main() -> None:
    """@brief CLI: assemble the curated analysis bundle."""
    parser = argparse.ArgumentParser(
        description="Assemble the curated analysis bundle from the raw outputs."
    )
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    args = parser.parse_args()
    build(args.frozen, args.raw, args.output_dir, args.seeds)


if __name__ == "__main__":
    main()
