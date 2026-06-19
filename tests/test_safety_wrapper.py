"""
@file test_safety_wrapper.py
@brief Unit tests for the SafetyWrapper Gymnasium wrapper.

CPU-only, no CARLA or ROS 2 required. The underlying env is mocked so
all tests run without a live simulation. The wrapper keys both responses
(throttle cap, full-stop handoff) off the TOTAL predictive uncertainty
(epistemic + aleatoric); see
documentation/detailed_notes/epistemic_aleatoric_disentanglement.md.
"""

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pytest

from uncertainty_rl.envs.safety_wrapper import SafetyWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestSafetyWrapperApply - pure static method, no env needed
# ---------------------------------------------------------------------------


class TestSafetyWrapperApply:
    """
    @class TestSafetyWrapperApply
    @brief Tests for the static SafetyWrapper.apply() logic (single total signal).
    """

    def test_no_uncertainty_passes_action_unchanged(self) -> None:
        """
        @brief Zero total uncertainty should leave the action unchanged.
        """
        action = _make_action(steer=0.3, throttle=0.8)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            total_uncertainty=0.0,
            caution_gain=0.5,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert modulated[0] == pytest.approx(0.3)
        assert modulated[1] == pytest.approx(0.8)

    def test_moderate_uncertainty_caps_throttle(self) -> None:
        """
        @brief Total uncertainty below the handoff threshold caps the throttle.
        """
        action = _make_action(throttle=1.0)
        modulated, handoff, _ = SafetyWrapper.apply(
            action,
            total_uncertainty=4.0,
            caution_gain=0.5,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        # throttle_cap = 1 / (1 + 0.5*4) = 1/3 ~= 0.333
        assert not handoff
        assert modulated[1] < 1.0
        assert modulated[1] == pytest.approx(1.0 / 3.0, rel=1e-4)

    def test_below_slow_threshold_passes_throttle_through(self) -> None:
        """
        @brief Total uncertainty below slow_threshold leaves the throttle uncapped.
        """
        action = _make_action(throttle=0.9)
        modulated, handoff, cap = SafetyWrapper.apply(
            action,
            total_uncertainty=0.1,
            caution_gain=2.0,
            slow_threshold=0.5,
            handoff_threshold=5.0,
        )
        assert not handoff
        assert cap == pytest.approx(1.0)
        assert modulated[1] == pytest.approx(0.9)

    def test_caution_does_not_cap_steering(self) -> None:
        """
        @brief The throttle cap must not change the steering component.
        """
        action = _make_action(steer=0.9, throttle=0.5)
        modulated, _, _s = SafetyWrapper.apply(
            action,
            total_uncertainty=10.0,
            caution_gain=1.0,
            slow_threshold=0.0,
            handoff_threshold=50.0,
        )
        assert modulated[0] == pytest.approx(0.9)

    def test_total_above_threshold_flags_over_not_stop(self) -> None:
        """
        @brief Total >= handoff_threshold raises the over_threshold flag but
               returns a DRIVING (throttle-capped) action, not a full stop. The
               full stop is the caller's job once the debounce confirms a handoff,
               so a single over-threshold step cannot truncate the episode.
        """
        action = _make_action(steer=0.5, throttle=0.9)
        modulated, over_threshold, _ = SafetyWrapper.apply(
            action,
            total_uncertainty=5.0,
            caution_gain=0.5,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        assert over_threshold
        # Not a stop: steering preserved, throttle capped (not zeroed), brake untouched.
        assert modulated[0] == pytest.approx(0.5)
        assert 0.0 < modulated[1] <= 0.9

    def test_total_below_threshold_no_crossing(self) -> None:
        """
        @brief Total just below threshold must not raise the over_threshold flag.
        """
        action = _make_action(throttle=0.8)
        modulated, over_threshold, _ = SafetyWrapper.apply(
            action,
            total_uncertainty=4.99,
            caution_gain=0.0,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        assert not over_threshold
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
            caution_gain=0.5,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        np.testing.assert_array_equal(action, original)

    def test_brake_passes_through_unchanged(self) -> None:
        """
        @brief apply() only caps the throttle; the brake axis passes through
               untouched (the cap reduces speed, it does not suppress braking).
        """
        action = _make_action(throttle=0.0, brake=0.7)
        modulated, _, _s = SafetyWrapper.apply(
            action,
            total_uncertainty=4.0,
            caution_gain=0.5,
            slow_threshold=0.0,
            handoff_threshold=5.0,
        )
        assert modulated[2] == pytest.approx(0.7)


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
        @brief step() info dict must contain throttle_cap, safety_handoff,
               epistemic, aleatoric, and total_uncertainty.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=1.0)
        _, _, _, _, info = wrapper.step(_make_action())
        assert "throttle_cap" in info
        assert "safety_handoff" in info
        assert "epistemic" in info
        assert "aleatoric" in info
        assert "total_uncertainty" in info

    def test_handoff_sets_truncated_true(self) -> None:
        """
        @brief When total >= threshold, step() must set truncated=True.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0)
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert truncated is True
        assert info["safety_handoff"] is True

    def test_handoff_uses_total_not_just_epistemic(self) -> None:
        """
        @brief Handoff is keyed on the TOTAL: epistemic+aleatoric crossing the
               threshold triggers it even when epistemic alone is below it.
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
        @brief When total < threshold the underlying truncated value is preserved.
        """
        # _StubEnv.step() returns truncated=False; the wrapper must preserve it.
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=5.0)
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=0.0)
        _, _, terminated, truncated, _ = wrapper.step(_make_action())
        assert not terminated
        assert not truncated

    def test_debounce_single_crossing_does_not_handoff(self) -> None:
        """
        @brief With handoff_consecutive_steps > 1, a single over-threshold step
               must NOT hand off - the debounce guards against a momentary spike.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0, handoff_consecutive_steps=3)
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)  # over threshold
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert info["safety_handoff"] is False
        assert truncated is False

    def test_debounce_handoff_after_n_consecutive(self) -> None:
        """
        @brief Handoff fires only once total has been over threshold for
               handoff_consecutive_steps decisions in a row; a sub-threshold step
               resets the counter.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env, handoff_threshold=1.0, handoff_consecutive_steps=3)
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)  # over threshold
        # Two crossings: not yet (need 3).
        wrapper.step(_make_action())
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert info["safety_handoff"] is False
        # A sub-threshold step resets the counter.
        wrapper.set_uncertainty(epistemic=0.0, aleatoric=0.0)
        wrapper.step(_make_action())
        # Now three fresh consecutive crossings -> handoff on the third.
        wrapper.set_uncertainty(epistemic=2.0, aleatoric=0.0)
        wrapper.step(_make_action())
        wrapper.step(_make_action())
        _, _, _, truncated, info = wrapper.step(_make_action())
        assert info["safety_handoff"] is True
        assert truncated is True


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
        assert stats["avg_throttle_cap"] == pytest.approx(1.0)

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

    def test_avg_throttle_cap_zero_steps(self) -> None:
        """
        @brief avg_throttle_cap should be 1.0 (no cap) when no steps taken.
        """
        env = _make_mock_env()
        wrapper = SafetyWrapper(env)
        stats = wrapper.get_episode_safety_stats()
        assert stats["avg_throttle_cap"] == pytest.approx(1.0)

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
