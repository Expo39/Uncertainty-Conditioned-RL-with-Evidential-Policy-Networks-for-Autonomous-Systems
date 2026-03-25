"""
@file test_baseline_configs.py
@brief Validate baseline YAML override configs and their merge with train_config.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# Add scripts/ to path so we can import merge_configs
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from run_experiment import load_config, merge_configs  # noqa: E402
except ModuleNotFoundError:
    pytest.skip(
        "run_experiment.py not yet implemented (Task 4 pending)",
        allow_module_level=True,
    )

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
BASELINES_DIR = CONFIGS_DIR / "baselines"
BASE_CONFIG_PATH = CONFIGS_DIR / "train_config.yaml"

BASELINE_FILES = [
    "vanilla_ppo.yaml",
    "input_uncertainty.yaml",
    "output_uncertainty.yaml",
    "full_method.yaml",
]

# Keys that every baseline override file must contain
REQUIRED_OVERRIDE_KEYS = [
    "baseline_name",
    "include_covariance",
    "policy_type",
    "log_dir",
    "checkpoint_dir",
]

# Keys that must exist in the merged config (from train_config.yaml)
REQUIRED_MERGED_KEYS = [
    "baseline_name",
    "include_covariance",
    "policy_type",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "total_timesteps",
    "seed",
    "random_seeds",
    "net_arch",
    "log_dir",
    "checkpoint_dir",
]


@pytest.fixture()
def base_config() -> Dict[str, Any]:
    """
    @brief Load the shared base training config.
    """
    return load_config(str(BASE_CONFIG_PATH))


@pytest.fixture(params=BASELINE_FILES)
def baseline_override(request: pytest.FixtureRequest) -> Dict[str, Any]:
    """
    @brief Load each baseline override config file.
    """
    path = BASELINES_DIR / request.param
    with open(path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    return config


@pytest.fixture(params=BASELINE_FILES)
def merged_config(
    request: pytest.FixtureRequest, base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    @brief Load and merge a baseline override onto the base config.
    """
    path = BASELINES_DIR / request.param
    overrides = load_config(str(path))
    return merge_configs(base_config, overrides)


class TestBaselineOverrides:
    """
    @class TestBaselineOverrides
    @brief Validate baseline override files contain only the expected keys.
    """

    def test_loads_without_error(self, baseline_override: Dict[str, Any]) -> None:
        """
        @brief Override config loads as a valid dict.
        """
        assert isinstance(baseline_override, dict)

    def test_has_required_override_keys(
        self, baseline_override: Dict[str, Any]
    ) -> None:
        """
        @brief Override config has all required keys.
        """
        for key in REQUIRED_OVERRIDE_KEYS:
            assert key in baseline_override, f"Missing key: {key}"

    def test_policy_type_valid(self, baseline_override: Dict[str, Any]) -> None:
        """
        @brief policy_type must be 'standard' or 'evidential'.
        """
        assert baseline_override["policy_type"] in (
            "standard",
            "evidential",
        )

    def test_include_covariance_is_bool(
        self, baseline_override: Dict[str, Any]
    ) -> None:
        """
        @brief include_covariance must be a boolean.
        """
        assert isinstance(baseline_override["include_covariance"], bool)

    def test_output_dirs_contain_baseline_name(
        self, baseline_override: Dict[str, Any]
    ) -> None:
        """
        @brief Output directories should include the baseline name.
        """
        name = baseline_override["baseline_name"]
        assert name in baseline_override["log_dir"]
        assert name in baseline_override["checkpoint_dir"]


class TestMergedConfigs:
    """
    @class TestMergedConfigs
    @brief Validate merged configs (base + override) are complete.
    """

    def test_merged_has_all_required_keys(self, merged_config: Dict[str, Any]) -> None:
        """
        @brief Merged config has all keys needed by train().
        """
        for key in REQUIRED_MERGED_KEYS:
            assert key in merged_config, f"Missing key: {key}"

    def test_ppo_hyperparams_from_base(self, merged_config: Dict[str, Any]) -> None:
        """
        @brief PPO hyperparameters come from the shared base config.
        """
        assert merged_config["learning_rate"] == 0.0003
        assert merged_config["n_steps"] == 2048
        assert merged_config["batch_size"] == 256
        assert merged_config["n_epochs"] == 5
        assert merged_config["gamma"] == 0.99

    def test_random_seeds_is_list(self, merged_config: Dict[str, Any]) -> None:
        """
        @brief random_seeds must be a list of integers.
        """
        seeds = merged_config["random_seeds"]
        assert isinstance(seeds, list)
        assert all(isinstance(s, int) for s in seeds)

    def test_evidential_section_present_when_needed(
        self, merged_config: Dict[str, Any]
    ) -> None:
        """
        @brief Evidential configs must have the evidential section.
        """
        if merged_config["policy_type"] == "evidential":
            assert "evidential" in merged_config
            assert "lambda_reg" in merged_config["evidential"]


class TestBaselineAblationMatrix:
    """
    @class TestBaselineAblationMatrix
    @brief Verify the 4 baselines cover all combinations of the 2x2 ablation.
    """

    def test_all_four_combinations_covered(self) -> None:
        """
        @brief The 4 configs must cover all (include_covariance, policy_type)
               pairs.
        """
        combinations = set()
        for filename in BASELINE_FILES:
            path = BASELINES_DIR / filename
            with open(path, "r") as f:
                config: Dict[str, Any] = yaml.safe_load(f)
            key = (config["include_covariance"], config["policy_type"])
            combinations.add(key)

        expected = {
            (False, "standard"),
            (True, "standard"),
            (False, "evidential"),
            (True, "evidential"),
        }
        assert combinations == expected
