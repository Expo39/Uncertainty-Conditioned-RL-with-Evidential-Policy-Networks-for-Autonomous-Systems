"""
@file safety_wrapper.py
@brief Gymnasium wrapper that hands off control when policy uncertainty is high.

Sits around CARLAParkingEnv at evaluation/deployment time and stops the vehicle
(full brake, hand off to a human) when the evidential head's TOTAL predictive
uncertainty crosses a threshold. Training runs WITHOUT the wrapper (the policy
learns freely); only evaluation and deployment install it.

One signal, one threshold:
- total = epistemic + aleatoric; total >= handoff_threshold -> full stop and hand off.

The total predictive uncertainty is the single defensible quantity on a single-head
NIG actor: epistemic = aleatoric / nu, so the two are one signal under two names and
gating on either alone double-counts. @see
documentation/detailed_notes/epistemic_aleatoric_disentanglement.md for the
disentanglement argument and the handoff's status as a negative gate result.

Actions are [steering, throttle, brake]: steering in [-1, 1], throttle and brake
non-negative in [0, 1].
"""

import logging
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger("uncertainty_rl.envs.safety_wrapper")


class SafetyWrapper(gym.Wrapper):
    """
    @class SafetyWrapper
    @brief Hands off control when total evidential uncertainty crosses a threshold.

    The wrapper does NOT call the policy itself. The evaluation loop sets the current
    uncertainty estimates via set_uncertainty() before each step; the wrapper sums
    them into the total signal it thresholds.

    @note Only evaluation/deployment uses this wrapper; training runs without it.
    """

    def __init__(
        self,
        env: gym.Env,
        handoff_threshold: float = 1.2,
    ) -> None:
        """
        @brief Initialise the safety wrapper.
        @param env: The underlying CARLAParkingEnv instance.
        @param handoff_threshold: Total predictive uncertainty (epistemic + aleatoric)
            at/above which the wrapper stops the vehicle and hands off. Calibrated per
            checkpoint to the calm-condition total upper tail (config-supplied).
        """
        super().__init__(env)
        self._handoff_threshold = float(handoff_threshold)

        # Per-step uncertainty estimates, set by the evaluation loop before each step.
        self._current_epistemic: float = 0.0
        self._current_aleatoric: float = 0.0

        # Per-episode counters, reset in reset().
        self._handoff_count: int = 0
        self._step_count: int = 0

    def set_uncertainty(self, epistemic: float, aleatoric: float) -> None:
        """
        @brief Set the current step's uncertainty estimates (summed to the total signal).
        @param epistemic: Mean epistemic uncertainty from the evidential actor.
        @param aleatoric: Mean aleatoric uncertainty from the evidential actor.
        @return None.

        Must be called before each step() during evaluation.
        """
        self._current_epistemic = epistemic
        self._current_aleatoric = aleatoric

    @staticmethod
    def apply(
        action: np.ndarray,
        total_uncertainty: float,
        handoff_threshold: float,
    ) -> Tuple[np.ndarray, bool]:
        """
        @brief Apply single-threshold safety interception to a raw policy action.
        @param action: Raw policy action [steering, throttle, brake].
        @param total_uncertainty: Epistemic + aleatoric from the evidential actor.
        @param handoff_threshold: Total-uncertainty level at/above which the vehicle
            is stopped and control handed off.
        @return Tuple (modulated_action, handoff). On a handoff the action is a full
            stop (zero steer/throttle, full brake); otherwise the raw action passes
            through unchanged.

        Static so the sim wrapper (step()) and the real deployment loop
        (RealWorldInferenceLoop) share one interception implementation.
        """
        handoff = total_uncertainty >= handoff_threshold
        if handoff:
            # Full stop: zero steer/throttle, full brake. Brake is action index 2.
            modulated = np.zeros_like(action)
            modulated[2] = 1.0
        else:
            modulated = action.copy()
        return modulated, handoff

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Hand off (full stop) if total uncertainty crosses the threshold, then step.
        @param action: Raw action from the policy.
        @return Standard Gymnasium tuple (obs, reward, terminated, truncated, info).
            A handoff sets info["safety_handoff"] and truncates the episode.
        """
        self._step_count += 1
        total = self._current_epistemic + self._current_aleatoric

        modulated_action, handoff = SafetyWrapper.apply(
            action,
            total_uncertainty=total,
            handoff_threshold=self._handoff_threshold,
        )

        if handoff:
            self._handoff_count += 1
            logger.info(
                "Safety handoff triggered (total=%.3f >= threshold=%.3f)",
                total,
                self._handoff_threshold,
            )

        obs, reward, terminated, truncated, info = self.env.step(modulated_action)

        # Surface the uncertainty signal and the handoff flag for the outcome
        # taxonomy and per-step trace logging in the evaluation loop.
        info["safety_handoff"] = handoff
        info["epistemic"] = self._current_epistemic
        info["aleatoric"] = self._current_aleatoric
        info["total_uncertainty"] = total

        # A handoff ends the episode: the policy has yielded to the human.
        if handoff:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment and clear the per-episode counters.
        @param kwargs: Forwarded to the underlying env reset().
        @return Standard Gymnasium tuple (obs, info).
        """
        self._handoff_count = 0
        self._step_count = 0
        self._current_epistemic = 0.0
        self._current_aleatoric = 0.0
        obs, info = self.env.reset(**kwargs)
        return (np.asarray(obs), info)

    def get_episode_safety_stats(self) -> Dict[str, float]:
        """
        @brief Safety statistics for the completed episode.
        @return Dict with handoff_count and total_steps.
        """
        return {
            "handoff_count": float(self._handoff_count),
            "total_steps": float(self._step_count),
        }
