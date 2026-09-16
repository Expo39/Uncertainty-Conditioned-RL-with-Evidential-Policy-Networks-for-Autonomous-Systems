"""
@file test_safety_wrapper.py
@brief Unit tests for the SafetyWrapper Gymnasium wrapper.

CPU-only with a mocked env, so these run without a live simulation.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest

from uncertainty_rl.envs.safety_wrapper import SafetyWrapper


class _StubEnv(gym.Env):
    """
    @class _StubEnv
    @brief Minimal real Gymnasium env for wrapping in SafetyWrapper tests.

    gymnasium.Wrapper asserts the wrapped env is a genuine gymnasium.Env, so a
    MagicMock cannot be used directly. This stub returns fixed reset()/step()
    values without requiring CARLA or ROS 2.
    """

    def __init__(self, obs_shape: int = 12) -> None:
        self._obs = np.zeros(obs_shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._obs.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        return self._obs.copy(), 0.5, False, False, {}


def _make_mock_env(obs_shape: int = 12) -> _StubEnv:
    """Return a minimal real Gymnasium env for wrapping."""
    return _StubEnv(obs_shape)


def _make_action(
    steer: float = 0.0, throttle: float = 0.5, brake: float = 0.0
) -> np.ndarray:
    return np.array([steer, throttle, brake], dtype=np.float32)


class TestSafetyWrapperApply:
    """
    @class TestSafetyWrapperApply
    @brief Tests for the static SafetyWrapper.apply() logic (single total signal).
    """

    def test_below_threshold_passes_action_unchanged(self) -> None:
        """
        @brief Total uncertainty below the threshold leaves the action unchanged.
        """
        action = _make_action(steer=0.3, throttle=0.8)
        modulated, handoff = SafetyWrapper.apply(
            action,
            total_uncertainty=1.0,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert modulated[0] == pytest.approx(0.3)
        assert modulated[1] == pytest.approx(0.8)

    def test_at_threshold_hands_off_full_stop(self) -> None:
        """
        @brief Total >= threshold returns a full stop (zero steer/throttle, full brake).
        """
        action = _make_action(steer=0.5, throttle=0.9)
        modulated, handoff = SafetyWrapper.apply(
            action,
            total_uncertainty=5.0,
            handoff_threshold=5.0,
        )
        assert handoff
        assert modulated[0] == pytest.approx(0.0)
        assert modulated[1] == pytest.approx(0.0)
        assert modulated[2] == pytest.approx(1.0)

    def test_just_below_threshold_no_handoff(self) -> None:
        """
        @brief Total just below the threshold must not hand off.
        """
        action = _make_action(throttle=0.8)
        modulated, handoff = SafetyWrapper.apply(
            action,
            total_uncertainty=4.99,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert modulated[1] == pytest.approx(0.8)

    def test_original_action_not_mutated(self) -> None:
        """
        @brief apply() must return a copy, not modify the input array in place.
        """
        action = _make_action(steer=0.3, throttle=0.7)
        original = action.copy()
        SafetyWrapper.apply(
            action,
            total_uncertainty=15.0,
            handoff_threshold=5.0,
        )
        np.testing.assert_array_equal(action, original)


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
        @brief step() info dict must contain safety_handoff, epistemic, aleatoric,
               and total_uncertainty.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=1.0)
        _, _, _, _, info = wrapper.step(_make_action())
        assert "safety_handoff" in info
        assert "epistemic" in info
        assert "aleatoric" in info
        assert "total_uncertainty" in info

    def test_handoff_sets_truncated_true(self) -> None:
        """
        @brief When total >= handoff_threshold, step() sets truncated=True and the
               handoff flag.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert truncated is True
        assert info["safety_handoff"] is True

    def test_handoff_uses_total_not_just_epistemic(self) -> None:
        """
        @brief The threshold is keyed on the TOTAL: epistemic+aleatoric crossing it
               triggers a handoff even when epistemic alone is below it.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        # epistemic 0.6 < 1.0, but total 0.6 + 0.6 = 1.2 >= 1.0.
        wrapper.set_uncertainty(epistemic=0.6, aleatoric=0.6)
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert truncated is True
        assert info["safety_handoff"] is True

    def test_no_handoff_does_not_force_truncated(self) -> None:
        """
        @brief When total < handoff_threshold the underlying truncated value is preserved.
        """
        # _StubEnv.step() returns truncated=False; the wrapper must preserve it.
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=5.0)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=0.0)
        _, _, terminated, truncated, _ = wrapper.step(_make_action())
        assert not terminated
        assert not truncated


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
        # Accumulate some state.
        wrapper.set_uncertainty(epistemic=5.0, aleatoric=2.0)
        wrapper.step(_make_action())
        # Reset and check counters cleared.
        wrapper.reset()
        stats = wrapper.get_episode_safety_stats()
        assert stats["handoff_count"] == 0.0
        assert stats["total_steps"] == 0.0

    def test_reset_clears_uncertainty_state(self) -> None:
        """
        @brief reset() must clear the cached epistemic and aleatoric values to zero.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=3.0, aleatoric=2.0)
        wrapper.reset()
        # After reset, step with no set_uncertainty call - should use defaults (0.0).
        _, _, _, _, info = wrapper.step(_make_action())
        assert info["epistemic"] == pytest.approx(0.0)
        assert info["aleatoric"] == pytest.approx(0.0)


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

    def test_set_uncertainty_persists_until_next_set(self) -> None:
        """
        @brief set_uncertainty() value persists across multiple steps until changed.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=10.0)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=4.0)
        # Two steps with the same uncertainty.
        _, _, _, _, info1 = wrapper.step(_make_action())
        _, _, _, _, info2 = wrapper.step(_make_action())
        assert info1["aleatoric"] == pytest.approx(4.0)
        assert info2["aleatoric"] == pytest.approx(4.0)
