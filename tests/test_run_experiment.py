"""
@file test_run_experiment.py
@brief Tests for the experiment orchestration script.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add scripts/ to path so we can import run_experiment
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiment import (  # noqa: E402
    _assign_carla_port,
    load_config,
    merge_configs,
    run_single,
)


class TestRunExperiment:
    """
    @class TestRunExperiment
    @brief Tests for orchestration script functions.
    """

    def test_load_config(self, tmp_path: Path) -> None:
        """
        @brief load_config loads a YAML file correctly.
        """
        config_path = tmp_path / "test.yaml"
        config_path.write_text("baseline_name: test\nseed: 42\n")
        config: Dict[str, Any] = load_config(str(config_path))
        assert config["baseline_name"] == "test"
        assert config["seed"] == 42

    def test_dry_run_does_not_import_train(self) -> None:
        """
        @brief dry_run=True should complete without importing training code.
        """
        config: Dict[str, Any] = {
            "baseline_name": "test",
            "seed": 42,
            "policy_type": "standard",
            "include_covariance": False,
        }
        # Should complete without error and without importing torch/SB3
        run_single(config, seed=0, dry_run=True)

    def test_dry_run_sets_seed_specific_dirs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        @brief dry_run output should show seed-specific directories.
        """
        config: Dict[str, Any] = {
            "baseline_name": "vanilla_ppo",
            "seed": 42,
            "policy_type": "standard",
            "include_covariance": False,
        }
        run_single(config, seed=3, dry_run=True)
        captured = capsys.readouterr()
        assert "seed_3" in captured.out
        assert "vanilla_ppo" in captured.out

    def test_load_real_baseline_config(self) -> None:
        """
        @brief Can load one of the real baseline override configs.
        """
        config_path = (
            Path(__file__).parent.parent / "configs" / "baselines" / "vanilla_ppo.yaml"
        )
        config = load_config(str(config_path))
        assert config["baseline_name"] == "vanilla_ppo"
        assert config["policy_type"] == "standard"


class TestMergeConfigs:
    """
    @class TestMergeConfigs
    @brief Tests for the deep-merge utility function.
    """

    def test_override_replaces_scalar(self) -> None:
        """
        @brief Scalar values in overrides replace base values.
        """
        base: Dict[str, Any] = {"a": 1, "b": 2}
        overrides: Dict[str, Any] = {"a": 10}
        merged = merge_configs(base, overrides)
        assert merged["a"] == 10
        assert merged["b"] == 2

    def test_override_adds_new_keys(self) -> None:
        """
        @brief Keys in overrides not in base are added.
        """
        base: Dict[str, Any] = {"a": 1}
        overrides: Dict[str, Any] = {"b": 2}
        merged = merge_configs(base, overrides)
        assert merged["a"] == 1
        assert merged["b"] == 2

    def test_nested_dicts_merge_recursively(self) -> None:
        """
        @brief Nested dicts are merged, not replaced.
        """
        base: Dict[str, Any] = {"x": {"a": 1, "b": 2}}
        overrides: Dict[str, Any] = {"x": {"b": 99}}
        merged = merge_configs(base, overrides)
        assert merged["x"]["a"] == 1
        assert merged["x"]["b"] == 99

    def test_base_not_mutated(self) -> None:
        """
        @brief merge_configs does not modify the base dict.
        """
        base: Dict[str, Any] = {"a": 1, "b": {"c": 2}}
        overrides: Dict[str, Any] = {"a": 10, "b": {"c": 99}}
        merge_configs(base, overrides)
        assert base["a"] == 1
        assert base["b"]["c"] == 2

    def test_real_baseline_merge(self) -> None:
        """
        @brief Merging a real baseline onto train_config produces a
               complete config.
        """
        base_path = Path(__file__).parent.parent / "configs" / "train_config.yaml"
        override_path = (
            Path(__file__).parent.parent / "configs" / "baselines" / "vanilla_ppo.yaml"
        )
        base = load_config(str(base_path))
        overrides = load_config(str(override_path))
        merged = merge_configs(base, overrides)

        # Override keys applied
        assert merged["baseline_name"] == "vanilla_ppo"
        assert merged["include_covariance"] is False
        assert merged["policy_type"] == "standard"

        # Base keys preserved
        assert merged["learning_rate"] == 0.0003
        assert merged["n_steps"] == 2048
        assert merged["gamma"] == 0.99
        assert "carla_sensors" in merged
        assert "carla_conditions" in merged


class TestAssignCarlaPort:
    """
    @class TestAssignCarlaPort
    @brief Tests for CARLA port assignment for parallel workers.
    """

    def test_worker_zero_keeps_base_port(self) -> None:
        """
        @brief Worker 0 should use the base port unchanged.
        """
        config: Dict[str, Any] = {"carla_port": 2000}
        result = _assign_carla_port(config, worker_index=0)
        assert result["carla_port"] == 2000

    def test_worker_one_offsets_by_three(self) -> None:
        """
        @brief Worker 1 should use base_port + 3.
        """
        config: Dict[str, Any] = {"carla_port": 2000}
        result = _assign_carla_port(config, worker_index=1)
        assert result["carla_port"] == 2003

    def test_worker_two_offsets_by_six(self) -> None:
        """
        @brief Worker 2 should use base_port + 6.
        """
        config: Dict[str, Any] = {"carla_port": 2000}
        result = _assign_carla_port(config, worker_index=2)
        assert result["carla_port"] == 2006

    def test_does_not_mutate_original(self) -> None:
        """
        @brief _assign_carla_port should not modify the input config.
        """
        config: Dict[str, Any] = {"carla_port": 2000, "seed": 42}
        _assign_carla_port(config, worker_index=3)
        assert config["carla_port"] == 2000

    def test_default_port_when_missing(self) -> None:
        """
        @brief Uses default port 2000 when carla_port is not in config.
        """
        config: Dict[str, Any] = {"seed": 42}
        result = _assign_carla_port(config, worker_index=1)
        assert result["carla_port"] == 2003

    def test_dry_run_shows_port(self, capsys: pytest.CaptureFixture[str]) -> None:
        """
        @brief Dry run output should include the assigned CARLA port.
        """
        config: Dict[str, Any] = {
            "baseline_name": "test",
            "carla_port": 2006,
            "policy_type": "standard",
            "include_covariance": False,
        }
        run_single(config, seed=0, dry_run=True)
        captured = capsys.readouterr()
        assert "carla_port=2006" in captured.out
