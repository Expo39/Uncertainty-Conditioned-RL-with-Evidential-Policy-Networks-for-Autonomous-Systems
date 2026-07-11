"""
@file test_cross_seed.py
@brief Tests for the cross-seed pooled aggregator (scripts/evaluation/cross_seed.py).

Builds tiny synthetic episode_records.csv files under the nested
seed_<N>/<arm>/<leaf>/<variant>/ layout and checks that cross_seed pools every
seed into one sample, that the pooled contrast frame keeps the expected schema and
the ordered arm Categorical, that the pooled episode count is the sum across seeds,
that the per-seed robustness mean/min/max match hand computation, and that a single
seed (pool of one) still produces non-empty pooled tables with a zero range.
"""

from pathlib import Path
from typing import List

import pandas as pd

from scripts.evaluation import cross_seed
from scripts.evaluation.ablation_analyser import _ARM_ORDER

# The two GNSS-tier endpoints the degradation slope and many tests key on.
_CLEAN = "gnss_fixed"
_DEGRADED = "gnss_degraded"

# The episode_records columns the analysers read (the schema evaluate.py writes).
_COLUMNS: List[str] = [
    "condition",
    "success",
    "final_pos_error_m",
    "final_speed_ms",
    "outcome",
    "ekf_std_pos_mean_m",
    "ekf_std_pos_max_m",
    "max_epistemic",
    "mean_speed_moving_ms",
    "mean_brake_cmd",
    "mean_abs_vyaw_rads",
    "mean_action_jerk",
    "handoff_step",
    "degraded_onset_step",
]


def _write_episode_csv(
    root: Path,
    seed: int,
    arm: str,
    n_per_condition: int,
    success_rate: float,
    variant: str = "without_wrapper",
) -> None:
    """
    @brief Write a synthetic episode_records.csv for one seed/arm at stage 6.
    @param root: The parent results root (holds seed_<N>/...).
    @param seed: Seed integer (the seed_<N> layer).
    @param arm: Baseline (arm) name.
    @param n_per_condition: Episodes per GNSS condition.
    @param success_rate: Fraction of episodes marked success (deterministic split).
    @param variant: Wrapper variant directory (without_wrapper or with_wrapper).

    Two conditions (clean, degraded) so the degradation slope is computable; the
    success split is deterministic (first ceil(rate*n) succeed) so per-seed and
    pooled statistics are hand-checkable. EKF std rises with degradation so the
    caution/behaviour reads have a real gradient.
    """
    leaf = f"6_{seed}_27062026-0000"
    out_dir = root / f"seed_{seed}" / arm / leaf / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition, std in ((_CLEAN, 0.02), (_DEGRADED, 1.0)):
        n_success = round(success_rate * n_per_condition)
        for i in range(n_per_condition):
            is_success = i < n_success
            rows.append(
                {
                    "condition": condition,
                    "success": 1 if is_success else 0,
                    "final_pos_error_m": 0.1 + std + (0.0 if is_success else 0.5),
                    "final_speed_ms": 0.0,
                    "outcome": "success" if is_success else "collision",
                    "ekf_std_pos_mean_m": std,
                    "ekf_std_pos_max_m": std * 1.5,
                    "max_epistemic": std * 0.5,
                    "mean_speed_moving_ms": 2.0 - std,
                    "mean_brake_cmd": 0.1 + std * 0.1,
                    "mean_abs_vyaw_rads": 0.2,
                    "mean_action_jerk": 0.05,
                    "handoff_step": float("nan"),
                    "degraded_onset_step": float("nan"),
                }
            )
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(
        out_dir / "episode_records.csv", index=False
    )


def _build_three_seed_tree(tmp_path: Path) -> Path:
    """
    @brief Build a results root with all four arms across three seeds (42/123/7).
    @param tmp_path: pytest temp dir.
    @return The results root to pass as --results-root.

    Each arm gets a distinct success rate so the cross-arm contrast is non-trivial,
    and the rate is identical across seeds so the per-seed range is a known value
    (here exactly 0 for success; the test that needs a non-zero range sets it).
    """
    root = tmp_path / "evaluation_results"
    rates = {
        "vanilla_ppo": 0.4,
        "input_uncertainty": 0.6,
        "output_uncertainty": 0.5,
        "full_method": 0.8,
    }
    for seed in (42, 123, 7):
        for arm, rate in rates.items():
            _write_episode_csv(root, seed, arm, n_per_condition=10, success_rate=rate)
    return root


class TestCrossSeedPooling:
    """
    @class TestCrossSeedPooling
    @brief Tests for the pooled-frame construction and pooled statistics.
    """

    def test_pooled_n_is_sum_across_seeds(self, tmp_path: Path) -> None:
        """
        @brief The pooled frame holds every seed's episodes (sum, not max).
        """
        root = _build_three_seed_tree(tmp_path)
        pooled = cross_seed._pool_episodes(root, stage="6")
        # 3 seeds x 4 arms x 2 conditions x 10 episodes = 240.
        assert len(pooled) == 3 * 4 * 2 * 10
        assert sorted(pooled["seed"].unique()) == [7, 42, 123]

    def test_pooled_arm_is_ordered_categorical(self, tmp_path: Path) -> None:
        """
        @brief The pooled arm column keeps the ordered _ARM_ORDER Categorical.
        """
        root = _build_three_seed_tree(tmp_path)
        pooled = cross_seed._pool_episodes(root, stage="6")
        assert isinstance(pooled["arm"].dtype, pd.CategoricalDtype)
        assert pooled["arm"].cat.ordered
        assert list(pooled["arm"].cat.categories) == _ARM_ORDER

    def test_contrast_frame_schema(self, tmp_path: Path) -> None:
        """
        @brief The pooled contrast frame exposes the expected columns (guards the
               imported _contrast_table private signature against drift).
        """
        root = _build_three_seed_tree(tmp_path)
        pooled = cross_seed._pool_episodes(root, stage="6")
        from scripts.evaluation.ablation_analyser import _contrast_table

        contrasts = _contrast_table(pooled, seed=42)
        expected = {
            "pair",
            "treatment",
            "control",
            "condition",
            "success_delta_pp",
            "success_ci_low",
            "success_ci_high",
            "success_significant",
            "pos_error_delta_m",
            "pos_error_ci_low",
            "pos_error_ci_high",
            "pos_error_significant",
        }
        assert set(contrasts.columns) == expected
        # full_method (0.8) - output_uncertainty (0.5) is a positive success delta.
        evidential = contrasts[contrasts["pair"] == "evidential_head"]
        assert (evidential["success_delta_pp"] > 0).all()


class TestSeedRobustness:
    """
    @class TestSeedRobustness
    @brief Tests for the per-seed mean/range robustness tables.
    """

    def test_robustness_mean_and_range(self, tmp_path: Path) -> None:
        """
        @brief Cross-seed mean/min/max match hand computation when seeds differ.
        """
        root = tmp_path / "evaluation_results"
        # full_method success differs by seed: 0.4, 0.6, 0.8 on the clean tier.
        _write_episode_csv(root, 42, "full_method", 10, 0.4)
        _write_episode_csv(root, 123, "full_method", 10, 0.6)
        _write_episode_csv(root, 7, "full_method", 10, 0.8)
        pooled = cross_seed._pool_episodes(root, stage="6")
        per_seed = cross_seed._per_seed_summary(pooled)
        robustness = cross_seed._seed_robustness(per_seed)

        clean = robustness[
            (robustness["arm"].astype(str) == "full_method")
            & (robustness["condition"] == _CLEAN)
        ].iloc[0]
        # Success rates 40, 60, 80 -> mean 60, min 40, max 80, range 40.
        assert clean["n_seeds"] == 3
        assert abs(clean["success_mean_pct"] - 60.0) < 1e-6
        assert abs(clean["success_min_pct"] - 40.0) < 1e-6
        assert abs(clean["success_max_pct"] - 80.0) < 1e-6
        assert abs(clean["success_range_pp"] - 40.0) < 1e-6


class TestPoolOfOne:
    """
    @class TestPoolOfOne
    @brief The aggregator must run on a single seed (range 0) and write tables.
    """

    def test_single_seed_writes_pooled_tables(self, tmp_path: Path) -> None:
        """
        @brief With only seed_42 present, analyse writes non-empty pooled tables
               and the robustness range is zero.
        """
        root = tmp_path / "evaluation_results"
        for arm, rate in (
            ("vanilla_ppo", 0.4),
            ("input_uncertainty", 0.6),
            ("output_uncertainty", 0.5),
            ("full_method", 0.8),
        ):
            _write_episode_csv(root, 42, arm, n_per_condition=10, success_rate=rate)
        out_dir = tmp_path / "cross_seed_analysis"

        cross_seed.analyse(
            results_root=root,
            out_dir=out_dir,
            seed=42,
            stage="6",
            slope_clean=_CLEAN,
            slope_degraded=_DEGRADED,
        )

        stage_dir = out_dir / "all_seeds" / "stage6"
        contrasts = pd.read_csv(stage_dir / "pooled_covariance_contrasts.csv")
        robustness = pd.read_csv(stage_dir / "seed_robustness.csv")
        assert not contrasts.empty
        assert not robustness.empty
        assert (robustness["n_seeds"] == 1).all()
        # A pool of one has zero cross-seed range everywhere.
        assert (robustness["success_range_pp"] == 0.0).all()
        # The headline figure and pooled summary were written too.
        assert (stage_dir / "seed_robustness.png").exists()
        assert (stage_dir / "pooled_condition_summary.csv").exists()

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        """
        @brief An empty root raises FileNotFoundError naming the expected layout.
        """
        empty = tmp_path / "evaluation_results"
        empty.mkdir()
        try:
            cross_seed._pool_episodes(empty, stage="6")
        except FileNotFoundError as exc:
            assert "episode_records.csv" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError on an empty root")
