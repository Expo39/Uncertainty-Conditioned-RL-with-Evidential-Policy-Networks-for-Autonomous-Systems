"""
@file test_train_ppo.py
@brief Unit tests for training script helpers.

Tests the pure helper functions in train_ppo.py that do not require CARLA,
ROS 2, or a GPU. The main training entry point is excluded (requires full
Docker stack).
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from uncertainty_rl.training.train_ppo import EnvDiagnosticsCallback, linear_schedule


# ===========================================================================
# TestLinearSchedule
# ===========================================================================


class TestLinearSchedule:
    """
    @class TestLinearSchedule
    @brief Tests for the linear learning-rate schedule helper.
    """

    def test_returns_callable(self) -> None:
        """
        @brief linear_schedule() must return a callable.
        """
        schedule = linear_schedule(3e-4)
        assert callable(schedule)

    def test_progress_one_gives_initial_value(self) -> None:
        """
        @brief progress_remaining=1.0 is the start of training -- LR equals initial.
        """
        initial = 3e-4
        schedule = linear_schedule(initial)
        assert schedule(1.0) == pytest.approx(initial)

    def test_progress_zero_gives_zero(self) -> None:
        """
        @brief progress_remaining=0.0 is end of training -- LR decays to zero.
        """
        schedule = linear_schedule(1e-3)
        assert schedule(0.0) == pytest.approx(0.0)

    def test_progress_half_gives_half_initial(self) -> None:
        """
        @brief progress_remaining=0.5 gives half the initial learning rate.
        """
        initial = 2e-4
        schedule = linear_schedule(initial)
        assert schedule(0.5) == pytest.approx(initial * 0.5)

    def test_linear_interpolation(self) -> None:
        """
        @brief LR decreases linearly: value at p equals initial * p.
        """
        initial = 1.0
        schedule = linear_schedule(initial)

        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            assert schedule(p) == pytest.approx(initial * p, rel=1e-6)

    def test_independent_schedules_dont_interfere(self) -> None:
        """
        @brief Two schedules with different initial values are independent.
        """
        s1 = linear_schedule(1e-3)
        s2 = linear_schedule(5e-4)

        assert s1(0.5) == pytest.approx(5e-4)
        assert s2(0.5) == pytest.approx(2.5e-4)


# ===========================================================================
# TestEnvDiagnosticsCallback
# ===========================================================================


class TestEnvDiagnosticsCallback:
    """
    @class TestEnvDiagnosticsCallback
    @brief Tests for the EnvDiagnosticsCallback TensorBoard helper.

    SB3's BaseCallback.logger is a read-only property, so tests use
    patch.object to inject a MagicMock logger rather than direct assignment.
    """

    def test_on_step_accumulates_pos_error(self) -> None:
        """
        @brief _on_step() appends pos_error values from each info dict.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [{"pos_error": 3.0}, {"pos_error": 7.0}]}
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: MagicMock())):
            cb._on_step()
        assert cb._ep_pos_errors == [3.0, 7.0]

    def test_on_rollout_end_records_mean_pos_error(self) -> None:
        """
        @brief _on_rollout_end() calls logger.record with the rolling mean pos_error.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [
            {"pos_error": 4.0, "orientation_error": 0.1, "speed": 1.0,
             "progress_reward": 0.1},
            {"pos_error": 6.0, "orientation_error": 0.2, "speed": 2.0,
             "progress_reward": 0.2},
        ]}
        mock_logger = MagicMock()
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)):
            cb._on_step()
            cb._on_rollout_end()

        recorded_keys = [call.args[0] for call in mock_logger.record.call_args_list]
        assert "env/mean_pos_error_m" in recorded_keys
        for call in mock_logger.record.call_args_list:
            if call.args[0] == "env/mean_pos_error_m":
                assert call.args[1] == pytest.approx(5.0)

    def test_on_rollout_end_clears_accumulators(self) -> None:
        """
        @brief After _on_rollout_end(), all accumulators are empty.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [
            {"pos_error": 1.0, "orientation_error": 0.0, "speed": 0.0,
             "progress_reward": 0.0},
        ]}
        mock_logger = MagicMock()
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)):
            cb._on_step()
            cb._on_rollout_end()

        assert cb._ep_pos_errors == []
        assert cb._ep_orientation_errors == []
        assert cb._ep_speeds == []
        assert cb._ep_progress_rewards == []

    def test_episode_outcome_rates_logged_on_terminal_step(self) -> None:
        """
        @brief success_rate, collision_rate, timeout_rate are recorded when an
               episode ends (success, collision, or timeout flag is set).
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [
            {"pos_error": 0.1, "orientation_error": 0.0, "speed": 0.0,
             "progress_reward": 0.0, "success": True, "collision": False,
             "timeout": False},
        ]}
        mock_logger = MagicMock()
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)):
            cb._on_step()
            cb._on_rollout_end()

        recorded_keys = [call.args[0] for call in mock_logger.record.call_args_list]
        assert "env/success_rate" in recorded_keys
        assert "env/collision_rate" in recorded_keys
        assert "env/timeout_rate" in recorded_keys

    def test_success_rate_one_when_all_success(self) -> None:
        """
        @brief success_rate == 1.0 when every terminal step was a success.
        """
        cb = EnvDiagnosticsCallback()
        cb.locals = {"infos": [
            {"pos_error": 0.0, "orientation_error": 0.0, "speed": 0.0,
             "progress_reward": 0.0, "success": True, "collision": False,
             "timeout": False},
            {"pos_error": 0.0, "orientation_error": 0.0, "speed": 0.0,
             "progress_reward": 0.0, "success": True, "collision": False,
             "timeout": False},
        ]}
        mock_logger = MagicMock()
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)):
            cb._on_step()
            cb._on_rollout_end()

        for call in mock_logger.record.call_args_list:
            if call.args[0] == "env/success_rate":
                assert call.args[1] == pytest.approx(1.0)

    def test_no_record_when_no_steps_accumulated(self) -> None:
        """
        @brief If _on_rollout_end() is called with empty accumulators, nothing
               is recorded (no KeyError, no spurious log entries).
        """
        cb = EnvDiagnosticsCallback()
        mock_logger = MagicMock()
        with patch.object(type(cb), "logger", new_callable=lambda: property(lambda self: mock_logger)):
            cb._on_rollout_end()
        mock_logger.record.assert_not_called()
