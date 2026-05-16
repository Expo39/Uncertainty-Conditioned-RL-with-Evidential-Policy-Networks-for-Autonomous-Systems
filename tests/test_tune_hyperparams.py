"""
@file test_tune_hyperparams.py
@brief Tests for Optuna hyperparameter tuning infrastructure.

CPU-only unit tests for sample_hyperparams, apply_best_params, and
TrialEvalCallback. No CARLA, ROS 2, or GPU required.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

# Skip entire module if optuna is not available (CI without training deps)
optuna = pytest.importorskip("optuna")

from uncertainty_rl.training.tune_hyperparams import (  # noqa: E402
    TrialEvalCallback,
    apply_best_params,
    sample_hyperparams,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def tuning_config() -> Dict[str, Any]:
    """
    @brief Standard tuning configuration with search space bounds.
    """
    return {
        "search_space": {
            "learning_rate": [1e-5, 1e-3],
            "n_steps": [1024, 2048, 4096],
            "batch_size": [64, 128, 256],
            "n_epochs": [3, 5, 10],
            "gamma": [0.98, 0.999],
            "ent_coef": [1e-6, 0.01],
            "lambda_reg": [1e-5, 0.01],
            "lambda_reg_warmup_steps": [10000, 100000],
        },
    }


@pytest.fixture
def train_config_template() -> Dict[str, Any]:
    """
    @brief Template training config for testing apply_best_params.
    """
    return {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.005,
        "evidential": {
            "lambda_reg": 0.001,
            "lambda_reg_warmup_steps": 50000,
        },
        "net_arch": [256, 256],
    }


# ===========================================================================
# sample_hyperparams Tests
# ===========================================================================


class TestSampleHyperparams:
    """
    @class TestSampleHyperparams
    @brief Tests for the hyperparameter sampling function.
    """

    def test_sample_hyperparams_returns_expected_keys(
        self, tuning_config: Dict[str, Any]
    ) -> None:
        """
        @brief Sampled params contain all expected keys.
        """
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage="sqlite:///:memory:",
            load_if_exists=False,
        )
        trial = study.ask()
        params = sample_hyperparams(trial, tuning_config)

        expected_keys = {
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "ent_coef",
            "evidential",
        }
        assert expected_keys.issubset(
            params.keys()
        ), f"Missing keys: {expected_keys - params.keys()}"

    def test_sample_hyperparams_batch_size_le_n_steps(
        self, tuning_config: Dict[str, Any]
    ) -> None:
        """
        @brief batch_size <= n_steps constraint is always satisfied.
        """
        for seed in range(20):
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(),
                storage="sqlite:///:memory:",
                load_if_exists=False,
            )
            trial = study.ask()

            params = sample_hyperparams(trial, tuning_config)

            assert (
                params["batch_size"] <= params["n_steps"]
            ), f"batch_size ({params['batch_size']}) > n_steps ({params['n_steps']})"

    def test_sample_hyperparams_gamma_in_range(
        self, tuning_config: Dict[str, Any]
    ) -> None:
        """
        @brief Gamma is within [0.97, 0.999].
        """
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage="sqlite:///:memory:",
            load_if_exists=False,
        )
        trial = study.ask()
        params = sample_hyperparams(trial, tuning_config)

        assert (
            0.98 <= params["gamma"] <= 0.999
        ), f"Gamma {params['gamma']} outside [0.98, 0.999]"

    def test_sample_hyperparams_lambda_reg_always_positive(
        self, tuning_config: Dict[str, Any]
    ) -> None:
        """
        @brief lambda_reg is always positive (never disabled).
        """
        for seed in range(20):
            study = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(),
                storage="sqlite:///:memory:",
                load_if_exists=False,
            )
            trial = study.ask()

            params = sample_hyperparams(trial, tuning_config)

            assert (
                params["evidential"]["lambda_reg"] > 0.0
            ), f"lambda_reg should always be positive, got {params['evidential']['lambda_reg']}"


# ===========================================================================
# apply_best_params Tests
# ===========================================================================


class TestApplyBestParams:
    """
    @class TestApplyBestParams
    @brief Tests for writing best params back to train_config.yaml.
    """

    def test_apply_best_params_updates_top_level_key(
        self, train_config_template: Dict[str, Any]
    ) -> None:
        """
        @brief Top-level keys are updated correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_config.yaml"

            # Write template
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(train_config_template, f)

            # Apply update
            best_params = {"learning_rate": 1e-4}
            apply_best_params(best_params, str(config_path))

            # Verify
            with open(config_path) as f:
                updated = yaml.safe_load(f)

            assert updated["learning_rate"] == 1e-4
            # Other keys should be unchanged
            assert updated["n_steps"] == 2048
            assert updated["gamma"] == 0.99

    def test_apply_best_params_updates_nested_key(
        self, train_config_template: Dict[str, Any]
    ) -> None:
        """
        @brief Nested keys (e.g. evidential.lambda_reg) are updated.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_config.yaml"

            # Write template
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(train_config_template, f)

            # Apply nested update
            best_params = {"evidential": {"lambda_reg": 0.005}}
            apply_best_params(best_params, str(config_path))

            # Verify
            with open(config_path) as f:
                updated = yaml.safe_load(f)

            assert updated["evidential"]["lambda_reg"] == 0.005
            # Other evidential keys should be unchanged
            assert updated["evidential"]["lambda_reg_warmup_steps"] == 50000

    def test_apply_best_params_creates_backup(
        self, train_config_template: Dict[str, Any]
    ) -> None:
        """
        @brief A timestamped backup is created in logs/tuning/backups/.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "train_config.yaml"
            backup_dir = Path(tmpdir) / "logs" / "tuning" / "backups"

            # Write template
            import yaml

            with open(config_path, "w") as f:
                yaml.dump(train_config_template, f)

            # Patch the apply_best_params to use our temp backup dir
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Apply update
                best_params = {"learning_rate": 1e-4}
                apply_best_params(best_params, str(config_path))

                # Verify backup dir was created
                assert backup_dir.exists(), f"Backup dir not created at {backup_dir}"

                # Verify at least one backup file exists (with timestamp pattern)
                backup_files = list(backup_dir.glob("train_config_*.yaml.bak"))
                assert len(backup_files) > 0, "No timestamped backup found"

                # Verify backup contains original config
                with open(backup_files[0]) as f:
                    backup = yaml.safe_load(f)

                assert backup["learning_rate"] == 0.0003  # Original value
            finally:
                os.chdir(original_cwd)


# ===========================================================================
# TrialEvalCallback Tests
# ===========================================================================


class TestTrialEvalCallback:
    """
    @class TestTrialEvalCallback
    @brief Tests for the trial evaluation callback.
    """

    def test_trial_eval_callback_reports_progress_tiebreaker(self) -> None:
        """
        @brief Before any success is observed, TrialEvalCallback reports the
               progress-based tiebreaker (_TIEBREAK_SCALE * mean_progress_reward).
        """
        from uncertainty_rl.training.tune_hyperparams import _TIEBREAK_SCALE

        # Mock trial and model logger
        trial = MagicMock()
        trial.should_prune.return_value = False  # Don't prune
        callback = TrialEvalCallback(trial)

        # Mock model and logger - no success_rate yet, only progress reward.
        callback.model = MagicMock()
        callback.model.logger = MagicMock()
        callback.model.logger.name_to_value = {"env/mean_progress_reward": 0.05}
        callback.num_timesteps = 10000

        # Simulate rollout end
        callback._on_rollout_end()

        # Verify trial.report was called with the composite objective. With no
        # success, the objective is the scaled progress tiebreaker.
        trial.report.assert_called_once()
        args, kwargs = trial.report.call_args
        assert args[0] == pytest.approx(_TIEBREAK_SCALE * 0.05)
        assert kwargs["step"] == 10000  # num_timesteps as keyword arg

    def test_trial_eval_callback_handles_missing_metric(self) -> None:
        """
        @brief TrialEvalCallback falls back gracefully if metric is missing.
        """
        trial = MagicMock()
        trial.should_prune.return_value = False
        callback = TrialEvalCallback(trial)

        callback.model = MagicMock()
        callback.model.logger = MagicMock()
        # Neither primary nor fallback metric present
        callback.model.logger.name_to_value = {}
        callback.num_timesteps = 10000

        # Should not raise; should use fallback value 0.0
        callback._on_rollout_end()

        trial.report.assert_called_once()
        args = trial.report.call_args[0]
        assert args[0] == 0.0  # Fallback to 0.0

    def test_trial_eval_callback_checks_pruning(self) -> None:
        """
        @brief TrialEvalCallback raises TrialPruned if trial.should_prune().
        """
        trial = MagicMock()
        trial.should_prune.return_value = True  # Signal pruning
        callback = TrialEvalCallback(trial)

        callback.model = MagicMock()
        callback.model.logger = MagicMock()
        callback.model.logger.name_to_value = {"env/mean_progress_reward": 0.05}
        callback.num_timesteps = 10000

        # Should raise TrialPruned
        with pytest.raises(optuna.TrialPruned):
            callback._on_rollout_end()
