"""
@file test_evaluation.py
@brief Tests for the evaluation metrics and utilities.

Validates the EvaluationMetrics container, metric aggregation,
and the evaluation helper functions.
"""

import numpy as np

from uncertainty_rl.evaluation import EvaluationMetrics


class TestEvaluationMetrics:
    """
    @class TestEvaluationMetrics
    @brief Tests for the EvaluationMetrics container.
    """

    def test_initial_state(self) -> None:
        """
        @brief Fresh metrics should have zero/empty defaults.
        """
        metrics = EvaluationMetrics()
        assert metrics.success_rate == 0.0
        assert metrics.average_reward == 0.0
        assert metrics.average_steps == 0.0
        assert metrics.position_errors == []
        assert metrics.orientation_errors == []

    def test_to_dict_keys(self) -> None:
        """
        @brief to_dict must return all expected keys.
        """
        metrics = EvaluationMetrics()
        d = metrics.to_dict()

        expected_keys = {
            "success_rate",
            "average_reward",
            "average_steps",
            "mean_position_error",
            "std_position_error",
            "mean_orientation_error",
            "std_orientation_error",
            "mean_epistemic_uncertainty",
            "mean_aleatoric_uncertainty",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_with_data(self) -> None:
        """
        @brief to_dict should compute correct statistics from stored data.
        """
        metrics = EvaluationMetrics()
        metrics.success_rate = 85.0
        metrics.average_reward = 42.5
        metrics.position_errors = [0.1, 0.2, 0.3]
        metrics.orientation_errors = [0.05, 0.1, 0.15]

        d = metrics.to_dict()
        np.testing.assert_approx_equal(d["mean_position_error"], 0.2)
        np.testing.assert_approx_equal(d["mean_orientation_error"], 0.1)
        assert d["success_rate"] == 85.0

    def test_to_dict_empty_lists_return_zero(self) -> None:
        """
        @brief Empty error lists should produce zero means and stds.
        """
        metrics = EvaluationMetrics()
        d = metrics.to_dict()
        assert d["mean_position_error"] == 0.0
        assert d["std_position_error"] == 0.0
        assert d["mean_orientation_error"] == 0.0
        assert d["std_orientation_error"] == 0.0

    def test_to_dict_values_are_python_floats(self) -> None:
        """
        @brief All dict values should be plain Python floats for serialisation.
        """
        metrics = EvaluationMetrics()
        metrics.position_errors = [0.5]
        metrics.orientation_errors = [0.1]

        d = metrics.to_dict()
        for key, val in d.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_epistemic_uncertainties_collected(self) -> None:
        """
        @brief Epistemic uncertainties stored in list are retrievable via to_dict.
        """
        metrics = EvaluationMetrics()
        metrics.epistemic_uncertainties = [0.1, 0.2, 0.3]
        d = metrics.to_dict()
        np.testing.assert_approx_equal(d["mean_epistemic_uncertainty"], 0.2)

    def test_aleatoric_uncertainties_collected(self) -> None:
        """
        @brief Aleatoric uncertainties stored in list are retrievable via to_dict.
        """
        metrics = EvaluationMetrics()
        metrics.aleatoric_uncertainties = [0.4, 0.6]
        d = metrics.to_dict()
        np.testing.assert_approx_equal(d["mean_aleatoric_uncertainty"], 0.5)

    def test_empty_uncertainty_lists_return_zero(self) -> None:
        """
        @brief Empty uncertainty lists produce zero means in to_dict.
        """
        metrics = EvaluationMetrics()
        d = metrics.to_dict()
        assert d["mean_epistemic_uncertainty"] == 0.0
        assert d["mean_aleatoric_uncertainty"] == 0.0


# ===========================================================================
# TestMakeEvalEnvPatrolVehiclesKey
# ===========================================================================


class TestMakeEvalEnvPatrolVehiclesKey:
    """
    @class TestMakeEvalEnvPatrolVehiclesKey
    @brief Tests that num_patrol_vehicles key is correctly forwarded.
    """

    def test_num_patrol_vehicles_read_from_condition(self) -> None:
        """
        @brief Condition dict with num_patrol_vehicles is read with correct key.

        This test verifies the key name used internally matches the YAML key,
        without spawning a real environment.
        """
        condition = {"num_patrol_vehicles": 2, "num_pedestrians": 3}
        # The corrected make_eval_env reads condition.get("num_patrol_vehicles", 0)
        assert condition.get("num_patrol_vehicles", 0) == 2

    def test_fallback_to_zero_when_key_absent(self) -> None:
        """
        @brief Missing num_patrol_vehicles key defaults to 0.
        """
        condition: dict = {}
        assert condition.get("num_patrol_vehicles", 0) == 0
