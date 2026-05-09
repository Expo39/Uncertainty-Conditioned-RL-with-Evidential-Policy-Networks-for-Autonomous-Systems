"""
@file test_safety_wrapper.py
@brief Unit tests for the SafetyWrapper Gymnasium wrapper.

CPU-only, no CARLA or ROS 2 required. The underlying env is mocked so
all tests run without a live simulation.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from uncertainty_rl.envs.safety_wrapper import SafetyWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_env(obs_shape: int = 12) -> MagicMock:
    """Return a minimal mock Gymnasium env for wrapping."""
    env = MagicMock()
    env.observation_space = MagicMock()
    env.action_space = MagicMock()
    obs = np.zeros(obs_shape, dtype=np.float32)
    env.reset.return_value = (obs, {})
    env.step.return_value = (obs, 0.5, False, False, {})
    return env


def _make_action(
    steer: float = 0.0, lon: float = 0.5, brake: float = 0.0
) -> np.ndarray:
    return np.array([steer, lon, brake], dtype=np.float32)


# ---------------------------------------------------------------------------
# TestSafetyWrapperApply - pure static method, no env needed
# ---------------------------------------------------------------------------


class TestSafetyWrapperApply:
    """
    @class TestSafetyWrapperApply
    @brief Tests for the static SafetyWrapper.apply() logic.
    """

    def test_no_uncertainty_passes_action_unchanged(self) -> None:
        """
        @brief Zero uncertainty should leave the action unchanged.
        """
        action = _make_action(steer=0.3, lon=0.8)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            epistemic=0.0,
            aleatoric=0.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert modulated[0] == pytest.approx(0.3)
        assert modulated[1] == pytest.approx(0.8)

    def test_aleatoric_caps_drive(self) -> None:
        """
        @brief High aleatoric uncertainty caps the drive component.
        """
        action = _make_action(lon=1.0)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            epistemic=0.0,
            aleatoric=4.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        # aleatoric_scale = 1 / (1 + 0.5*4) = 1/3 ~= 0.333
        assert not handoff
        assert modulated[1] < 1.0
        assert modulated[1] == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_aleatoric_does_not_cap_steering(self) -> None:
        """
        @brief Aleatoric uncertainty must not change the steering component.
        """
        action = _make_action(steer=0.9, lon=0.5)
        modulated, _, _s = SafetyWrapper.apply(
            action,
            epistemic=0.0,
            aleatoric=10.0,
            aleatoric_scaling=1.0,
            handoff_threshold=5.0,
        )
        assert modulated[0] == pytest.approx(0.9)

    def test_epistemic_above_threshold_triggers_handoff(self) -> None:
        """
        @brief Epistemic >= handoff_threshold must zero the action and return handoff=True.
        """
        action = _make_action(steer=0.5, lon=0.9)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            epistemic=5.0,
            aleatoric=0.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        assert handoff
        np.testing.assert_array_equal(modulated, np.zeros(3))

    def test_epistemic_below_threshold_no_handoff(self) -> None:
        """
        @brief Epistemic just below threshold must not trigger handoff.
        """
        action = _make_action(lon=0.8)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            epistemic=4.99,
            aleatoric=0.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert modulated[1] == pytest.approx(0.8)

    def test_original_action_not_mutated(self) -> None:
        """
        @brief apply() must return a copy, not modify the input array in place.
        """
        action = _make_action(steer=0.3, lon=0.7)
        original = action.copy()
        SafetyWrapper.apply(
            action,
            epistemic=10.0,
            aleatoric=5.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        np.testing.assert_array_equal(action, original)

    def test_negative_drive_clipped_to_minus_one(self) -> None:
        """
        @brief Large reverse drive commands are still clamped to -1.0.
        """
        action = np.array([0.0, -2.0, 0.0], dtype=np.float32)
        modulated, _, _s = SafetyWrapper.apply(
            action,
            epistemic=0.0,
            aleatoric=0.0,
            aleatoric_scaling=0.5,
            handoff_threshold=5.0,
        )
        assert modulated[1] >= -1.0


# ---------------------------------------------------------------------------
# TestSafetyWrapperStep
# ---------------------------------------------------------------------------


class TestSafetyWrapperStep:
    """
    @class TestSafetyWrapperStep
    @brief Tests for SafetyWrapper.step() integration with mock env.
    """

    def test_step_returns_gymnasium_tuple(self) -> None:
        """
        @brief step() must return (obs, reward, terminated, truncated, info).
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=0.0)
        result = wrapper.step(_make_action())
        assert len(result) == 5

    def test_step_info_contains_safety_keys(self) -> None:
        """
        @brief step() info dict must contain aleatoric_scale, safety_handoff,
               epistemic, and aleatoric.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=1.0)
        _, _, _, _, info = wrapper.step(_make_action())
        assert "aleatoric_scale" in info
        assert "safety_handoff" in info
        assert "epistemic" in info
        assert "aleatoric" in info

    def test_handoff_sets_truncated_true(self) -> None:
        """
        @brief When epistemic >= threshold, step() must set truncated=True.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert truncated is True
        assert info["safety_handoff"] is True

    def test_no_handoff_does_not_force_truncated(self) -> None:
        """
        @brief When epistemic < threshold the underlying truncated value is preserved.
        """
        env = _make_mock_env()
        env.step.return_value = (np.zeros(12), 0.0, False, False, {})
        wrapper = SafetyWrapper(env, handoff_threshold=5.0)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=0.0)
        _, _, terminated, truncated, _ = wrapper.step(_make_action())
        assert not terminated
        assert not truncated


# ---------------------------------------------------------------------------
# TestSafetyWrapperReset
# ---------------------------------------------------------------------------


class TestSafetyWrapperReset:
    """
    @class TestSafetyWrapperReset
    @brief Tests for SafetyWrapper.reset() counter clearing.
    """

    def test_reset_clears_episode_counters(self) -> None:
        """
        @brief After calling reset(), safety stats should reflect a fresh episode.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        # Accumulate some state
        wrapper.set_uncertainty(epistemic=5.0, aleatoric=2.0)
        wrapper.step(_make_action())
        # Reset and check counters cleared
        wrapper.reset()
        stats = wrapper.get_episode_safety_stats()
        assert stats["handoff_count"] == 0.0
        assert stats["total_steps"] == 0.0
        assert stats["avg_aleatoric_scale"] == pytest.approx(1.0)

    def test_reset_clears_uncertainty_state(self) -> None:
        """
        @brief reset() must clear the cached epistemic and aleatoric values to zero.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=3.0, aleatoric=2.0)
        wrapper.reset()
        # After reset, step with no set_uncertainty call - should use defaults (0.0)
        _, _, _, _, info = wrapper.step(_make_action())
        assert info["epistemic"] == pytest.approx(0.0)
        assert info["aleatoric"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestSafetyWrapperStats
# ---------------------------------------------------------------------------


class TestSafetyWrapperStats:
    """
    @class TestSafetyWrapperStats
    @brief Tests for get_episode_safety_stats() aggregation.
    """

    def test_stats_count_handoffs(self) -> None:
        """
        @brief handoff_count must equal the number of handoff-triggering steps.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        for _ in range(3):
            wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)
            wrapper.step(_make_action())
        stats = wrapper.get_episode_safety_stats()
        assert stats["handoff_count"] == pytest.approx(3.0)
        assert stats["total_steps"] == pytest.approx(3.0)

    def test_avg_aleatoric_scale_zero_steps(self) -> None:
        """
        @brief avg_aleatoric_scale should be 1.0 (no scaling) when no steps taken.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        stats = wrapper.get_episode_safety_stats()
        assert stats["avg_aleatoric_scale"] == pytest.approx(1.0)

    def test_set_uncertainty_persists_until_next_set(self) -> None:
        """
        @brief set_uncertainty() value persists across multiple steps until changed.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=10.0)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=4.0)
        # Two steps with the same uncertainty
        _, _, _, _, info1 = wrapper.step(_make_action())
        _, _, _, _, info2 = wrapper.step(_make_action())
        assert info1["aleatoric"] == pytest.approx(4.0)
        assert info2["aleatoric"] == pytest.approx(4.0)
