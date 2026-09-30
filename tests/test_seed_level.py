"""
@file test_seed_level.py
@brief Tests for the seed-level analysis (scripts/analysis/seed_level.py).

Synthetic episode_records.csv files under the nested seed_<N>/ layout check the
two-level bootstrap, the exact seed permutation test and the written tables.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import seed_level

_SEEDS = ["seed_42", "seed_123", "seed_7"]
_CONDITION = "anchor_deployment"


def _write_arm(root: Path, seed: str, arm: str, success_rate: float) -> None:
    """
    @brief Write 20 anchor_deployment episodes for one seed/arm at stage 6.
    @param root: The parent results root.
    @param seed: Seed sub-root name.
    @param arm: Arm (baseline) name.
    @param success_rate: Fraction of episodes marked success (deterministic).
    """
    out_dir = root / seed / arm / f"6_{seed[5:]}_27062026-0000" / "without_wrapper"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 20
    std = np.linspace(0.02, 1.0, n)
    pd.DataFrame(
        {
            "condition": _CONDITION,
            "success": (np.arange(n) < round(success_rate * n)).astype(int),
            "final_pos_error_m": 0.1 + std,
            "steps": 300,
            "ekf_std_pos_mean_m": std,
            "mean_brake_cmd": 0.1 + std,
        }
    ).to_csv(out_dir / "episode_records.csv", index=False)


def _build_tree(tmp_path: Path, rates: Dict[str, float]) -> Path:
    """
    @brief Write every arm in rates for all three seeds.
    @return The results root.
    """
    root = tmp_path / "evaluation_results"
    for seed in _SEEDS:
        for arm, rate in rates.items():
            _write_arm(root, seed, arm, rate)
    return root


class TestTwoLevelMeans:
    """
    @class TestTwoLevelMeans
    @brief The two-level bootstrap of a mean.
    """

    def test_constant_groups_give_constant_draws(self) -> None:
        """
        @brief Identical constant seeds leave no resampling variance.
        """
        rng = np.random.default_rng(0)
        groups = [np.full(10, 0.3) for _ in range(3)]
        draws = seed_level._two_level_means(groups, rng, 100)
        np.testing.assert_allclose(draws, 0.3)

    def test_empty_groups_give_nan(self) -> None:
        """
        @brief An arm with no values yields NaN rather than raising.
        """
        rng = np.random.default_rng(0)
        draws = seed_level._two_level_means([np.array([])], rng, 10)
        assert np.isnan(draws).all()


class TestAnalyse:
    """
    @class TestAnalyse
    @brief End-to-end run on a synthetic six-arm tree.
    """

    def test_writes_tables_and_separates_seeds(self, tmp_path: Path) -> None:
        """
        @brief Every covariance contrast is estimated, and a covariance arm above
               its blind arm on every seed gets the minimum p of 1/20.
        """
        rates = {
            "vanilla_ppo": 0.2,
            "input_uncertainty": 0.25,
            "heteroscedastic": 0.2,
            "heteroscedastic_input": 0.4,
            "output_uncertainty": 0.2,
            "full_method": 0.45,
        }
        root = _build_tree(tmp_path, rates)
        out = tmp_path / "seed_level"
        seed_level.analyse(root, out, _SEEDS, "6", bootstrap_seed=1, n_resamples=200)

        estimates = pd.read_csv(out / "estimates.csv")
        primary = estimates[
            estimates["estimate"] == "covariance_heteroscedastic_head"
        ].iloc[0]
        assert primary["point_pp"] == pytest.approx(20.0)
        assert set(estimates["estimate"]) == set(seed_level._covariance_estimates())

        permutation = pd.read_csv(out / "permutation.csv")
        het = permutation[permutation["pair"] == "heteroscedastic_head"].iloc[0]
        assert het["arrangements"] == 20
        assert het["p_one_sided"] == pytest.approx(1 / 20)

        brake = pd.read_csv(out / "brake_spearman.csv")
        assert len(brake) == len(rates)
        assert (brake["pooled"] > 0.99).all()

    def test_missing_arms_are_skipped(self, tmp_path: Path) -> None:
        """
        @brief With only the four original arms, estimates needing the
               heteroscedastic arms are skipped rather than failing.
        """
        root = _build_tree(
            tmp_path,
            {
                "vanilla_ppo": 0.2,
                "input_uncertainty": 0.25,
                "output_uncertainty": 0.2,
                "full_method": 0.45,
            },
        )
        out = tmp_path / "seed_level"
        seed_level.analyse(root, out, _SEEDS, "6", bootstrap_seed=1, n_resamples=50)
        names = set(pd.read_csv(out / "estimates.csv")["estimate"])
        assert "covariance_evidential_head" in names
        assert "covariance_heteroscedastic_head" not in names
