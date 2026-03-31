"""
@file test_train_ppo.py
@brief Unit tests for training script helpers.

Tests the pure helper functions in train_ppo.py that do not require CARLA,
ROS 2, or a GPU. The main training entry point is excluded (requires full
Docker stack).
"""

import pytest

from uncertainty_rl.training.train_ppo import linear_schedule


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
