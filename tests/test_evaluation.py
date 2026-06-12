"""
@file test_evaluation.py
@brief Tests for the evaluation metrics and utilities.

Validates the EvaluationMetrics container, metric aggregation,
and the evaluation helper functions.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from uncertainty_rl.evaluation import EvaluationMetrics
from uncertainty_rl.evaluation.evaluate import (
    _classify_outcome,
    _scale_sensor_noise,
    evaluate_agent,
)
from uncertainty_rl.utils.bay_success import BaySuccessTracker


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
            "max_epistemic_uncertainty",
            "max_aleatoric_uncertainty",
            "collision_rate",
            "out_of_bounds_rate",
            "handoff_rate",
            "near_miss_rate",
            "stuck_rate",
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
# TestClassifyOutcome
# ===========================================================================


class TestClassifyOutcome:
    """
    @class TestClassifyOutcome
    @brief Tests for the episode failure-mode taxonomy.
    """

    def test_success(self) -> None:
        """
        @brief A successful episode is classed success regardless of flags.
        """
        info = {"success": True, "collision": True, "pos_error": 0.1}
        assert _classify_outcome(info, 1.5) == "success"

    def test_collision_outranks_handoff(self) -> None:
        """
        @brief A collision on the handoff step is still a collision.
        """
        info = {"success": False, "collision": True, "safety_handoff": True}
        assert _classify_outcome(info, 1.5) == "collision"

    def test_out_of_bounds(self) -> None:
        """
        @brief An OOB termination is classed out_of_bounds.
        """
        info = {"success": False, "collision": False, "oob": True}
        assert _classify_outcome(info, 1.5) == "out_of_bounds"

    def test_handoff(self) -> None:
        """
        @brief A SafetyWrapper truncation is classed handoff.
        """
        info = {"success": False, "safety_handoff": True, "pos_error": 0.5}
        assert _classify_outcome(info, 1.5) == "handoff"

    def test_near_miss_below_threshold(self) -> None:
        """
        @brief A timeout close to the bay is a near miss.
        """
        info = {"success": False, "pos_error": 0.8}
        assert _classify_outcome(info, 1.5) == "near_miss"

    def test_stuck_at_or_above_threshold(self) -> None:
        """
        @brief A timeout far from the bay is stuck.
        """
        info = {"success": False, "pos_error": 1.5}
        assert _classify_outcome(info, 1.5) == "stuck"

    def test_missing_pos_error_defaults_to_stuck(self) -> None:
        """
        @brief Without a final position error the episode cannot be a near
               miss, so it falls through to stuck.
        """
        info = {"success": False}
        assert _classify_outcome(info, 1.5) == "stuck"


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


# ===========================================================================
# TestScaleSensorNoise
# ===========================================================================


class TestScaleSensorNoise:
    """
    @class TestScaleSensorNoise
    @brief Tests for the _scale_sensor_noise() helper.
    """

    def _base_sensors(self) -> dict:
        """
        @brief Return a representative sensor config mirroring train_config.yaml.
        """
        return {
            "imu": {
                "accel_stddev": 0.1,
                "gyro_stddev": 0.05,
                "accel_bias": 0.001,
                "noise_seed": 42,
            },
            "lidar": {
                "channels": 1,
                "range": 30.0,
            },
        }

    def test_stddev_keys_scaled_by_multiplier(self) -> None:
        """
        @brief Keys containing 'stddev' in the IMU section are multiplied.
        """
        sensors = self._base_sensors()
        result = _scale_sensor_noise(sensors, imu_multiplier=2.0)

        assert result["imu"]["accel_stddev"] == pytest.approx(0.2)
        assert result["imu"]["gyro_stddev"] == pytest.approx(0.1)

    def test_non_stddev_keys_unchanged(self) -> None:
        """
        @brief Keys without 'stddev' in their name are not modified.
        """
        sensors = self._base_sensors()
        result = _scale_sensor_noise(sensors, imu_multiplier=5.0)

        assert result["imu"]["accel_bias"] == pytest.approx(0.001)
        assert result["imu"]["noise_seed"] == 42

    def test_multiplier_one_leaves_values_unchanged(self) -> None:
        """
        @brief imu_multiplier=1.0 should be a no-op on all values.
        """
        sensors = self._base_sensors()
        result = _scale_sensor_noise(sensors, imu_multiplier=1.0)

        assert result["imu"]["accel_stddev"] == pytest.approx(0.1)
        assert result["imu"]["gyro_stddev"] == pytest.approx(0.05)

    def test_multiplier_zero_zeros_all_stddevs(self) -> None:
        """
        @brief imu_multiplier=0.0 should set all stddev values to zero.
        """
        sensors = self._base_sensors()
        result = _scale_sensor_noise(sensors, imu_multiplier=0.0)

        assert result["imu"]["accel_stddev"] == pytest.approx(0.0)
        assert result["imu"]["gyro_stddev"] == pytest.approx(0.0)

    def test_does_not_mutate_original(self) -> None:
        """
        @brief The original sensor config dict must not be modified (deepcopy).
        """
        sensors = self._base_sensors()
        original_accel = sensors["imu"]["accel_stddev"]

        _scale_sensor_noise(sensors, imu_multiplier=10.0)

        assert sensors["imu"]["accel_stddev"] == pytest.approx(original_accel)

    def test_lidar_section_untouched(self) -> None:
        """
        @brief Non-IMU sections are copied unchanged even with a large multiplier.
        """
        sensors = self._base_sensors()
        result = _scale_sensor_noise(sensors, imu_multiplier=100.0)

        assert result["lidar"]["channels"] == 1
        assert result["lidar"]["range"] == pytest.approx(30.0)


class _StubModel:
    """
    @class _StubModel
    @brief Minimal model stub returning a fixed action for evaluate_agent.

    Not an EvidentialPPO, so evaluate_agent takes the deterministic
    model.predict() path and never touches torch or a real policy.
    """

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        """
        @brief Return a zero action and no recurrent state.
        @param obs: Observation batch (ignored).
        @param deterministic: Unused; present for signature parity.
        @return Tuple of (action, None).
        """
        return np.zeros((1, 3), dtype=np.float32), None


class _StubVecEnv:
    """
    @class _StubVecEnv
    @brief One-step DummyVecEnv stand-in that terminates with a chosen info.

    Each episode runs a single step that returns done=True and the supplied
    per-episode info dict, so evaluate_agent's terminal-step recording path is
    exercised without CARLA. Mirrors DummyVecEnv's auto-reset contract: the
    info returned on the terminal step IS that episode's terminal info.
    """

    def __init__(self, episode_infos: List[Dict[str, Any]]) -> None:
        """
        @brief Store the queued per-episode terminal infos.
        @param episode_infos: One info dict per episode, returned in order.
        """
        self._episode_infos = episode_infos
        self._episode = -1

    def reset(self) -> np.ndarray:
        """
        @brief Advance to the next queued episode and return a dummy obs.
        @return Observation batch of shape (1, 1).
        """
        self._episode += 1
        return np.zeros((1, 1), dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        @brief Terminate immediately with this episode's queued info.
        @param action: Action from the model (ignored).
        @return Tuple of (obs, rewards, dones, infos) with done=True.
        """
        info = self._episode_infos[self._episode]
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([True]),
            [info],
        )


class TestEvaluateAgentBayRecording:
    """
    @class TestEvaluateAgentBayRecording
    @brief evaluate_agent must record each terminated episode against its bay.
    """

    def test_records_nested_target_bay(self) -> None:
        """
        @brief A nested target_bay dict in the terminal info is recorded.
        """
        infos = [
            {"success": True, "target_bay": {"bay_id": "perpendicular_2"}},
            {"success": False, "target_bay": {"bay_id": "perpendicular_2"}},
        ]
        tracker = BaySuccessTracker()
        evaluate_agent(
            model=_StubModel(),  # type: ignore[arg-type]
            env=_StubVecEnv(infos),  # type: ignore[arg-type]
            n_episodes=2,
            bay_tracker=tracker,
        )

        assert tracker.total_attempts == 2
        _, attempts, successes = tracker._counts["perpendicular_2"]
        assert (attempts, successes) == (2, 1)

    def test_falls_back_to_flat_target_bay_id(self) -> None:
        """
        @brief The flat target_bay_id is used when the nested dict is absent.
        """
        infos = [{"success": True, "target_bay_id": "perpendicular_5"}]
        tracker = BaySuccessTracker()
        evaluate_agent(
            model=_StubModel(),  # type: ignore[arg-type]
            env=_StubVecEnv(infos),  # type: ignore[arg-type]
            n_episodes=1,
            bay_tracker=tracker,
        )

        assert tracker.total_attempts == 1
        assert "perpendicular_5" in tracker._counts
