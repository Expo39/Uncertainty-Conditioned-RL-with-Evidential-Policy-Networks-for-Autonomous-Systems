"""
@file safety_wrapper.py
@brief Gymnasium wrapper that intercepts actions based on policy uncertainty.

Sits around CARLAParkingEnv at evaluation/deployment time and modulates the policy
action from the evidential head's total predictive uncertainty. Training runs WITHOUT
the wrapper (the policy learns freely).

Single signal, two thresholds. Both responses key off the TOTAL predictive uncertainty
(epistemic + aleatoric), by MAGNITUDE:
- total >= slow_threshold: cap the throttle (slower, more cautious driving);
- total >= handoff_threshold: full stop and hand off.

On a single-head NIG actor epistemic = aleatoric / nu, so the two channels are one
signal scaled by nu and cannot disagree per state. Keying separate responses to the
epistemic-vs-aleatoric SPLIT would leave one branch unable to fire, so the controller
uses the total instead - which gives the same caution-then-handoff behaviour.
@see documentation/detailed_notes/epistemic_aleatoric_disentanglement.md

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
    @brief Intercepts actions based on total evidential uncertainty at eval time.

    The wrapper does NOT call the policy itself. The evaluation loop sets the current
    uncertainty estimates via set_uncertainty() before each step; the wrapper sums them
    into the total signal it thresholds.

    @note Only evaluation/deployment uses this wrapper; training runs without it.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        env: gym.Env,
        caution_gain: float = 2.0,
        slow_threshold: float = 0.0,
        handoff_threshold: float = 0.02,
        handoff_consecutive_steps: int = 1,
    ) -> None:
        """
        @brief Initialise the safety wrapper.
        @param env: The underlying CARLAParkingEnv instance.
        @param caution_gain: How hard the throttle is cut as total uncertainty rises
            above slow_threshold (higher = more conservative).
        @param slow_threshold: Total-uncertainty level at/above which throttle is capped.
        @param handoff_threshold: Total-uncertainty level that triggers a full stop + handoff.
        @param handoff_consecutive_steps: Total must stay at/above handoff_threshold
            for this many CONSECUTIVE decisions before a handoff fires (debounce).
            A momentary uncertainty spike then cannot truncate the episode; the
            counter resets on any step below the threshold. 1 = fire on first crossing.
        """
        super().__init__(env)
        self._caution_gain = caution_gain
        self._slow_threshold = slow_threshold
        self._handoff_threshold = handoff_threshold
        self._handoff_consecutive_steps = max(1, int(handoff_consecutive_steps))

        # Per-step uncertainty estimates, set by the evaluation loop.
        self._current_epistemic: float = 0.0
        self._current_aleatoric: float = 0.0

        # Episode-level counters. _consecutive_over tracks how many decisions in a
        # row total has been at/above handoff_threshold (the debounce counter).
        self._handoff_count: int = 0
        self._throttle_cap_sum: float = 0.0
        self._step_count: int = 0
        self._consecutive_over: int = 0

    def set_uncertainty(self, epistemic: float, aleatoric: float) -> None:
        """
        @brief Set the current step's uncertainty estimates (summed to the total signal).
        @param epistemic: Mean epistemic uncertainty from the evidential actor.
        @param aleatoric: Mean aleatoric uncertainty from the evidential actor.

        Must be called before each step() during evaluation.
        """
        self._current_epistemic = epistemic
        self._current_aleatoric = aleatoric

    @staticmethod
    def apply(
        action: np.ndarray,
        total_uncertainty: float,
        caution_gain: float,
        slow_threshold: float,
        handoff_threshold: float,
    ) -> Tuple[np.ndarray, bool, float]:
        """
        @brief Apply single-signal safety interception to a raw policy action.

        Static so SafetyWrapper.step() (sim eval) and RealWorldInferenceLoop (real
        deployment) share one implementation.

        @param action: Raw policy action [steering, throttle, brake].
        @param total_uncertainty: Epistemic + aleatoric from the evidential actor.
        @param caution_gain: Throttle-cut aggressiveness above slow_threshold.
        @param slow_threshold: Total-uncertainty level at/above which throttle is capped.
        @param handoff_threshold: Total-uncertainty level above which the over-threshold
            flag is raised (the caller debounces this into an actual handoff).
        @return Tuple (modulated_action, over_threshold, throttle_cap). over_threshold
            is the RAW per-step crossing of handoff_threshold; the caller decides
            whether a sustained run of crossings constitutes a handoff (debounce).
            The action returned is always a DRIVING action (throttle-capped) - the
            full-stop is the caller's job once a handoff is confirmed, so a single
            spike does not stop the car.
        """
        over_threshold = total_uncertainty >= handoff_threshold
        # Throttle cap: above slow_threshold the cap tightens with uncertainty;
        # below it the action passes through (cap 1.0). The over-threshold step
        # still drives (hard-capped) until the caller confirms a handoff.
        if total_uncertainty >= slow_threshold:
            throttle_cap = 1.0 / (1.0 + caution_gain * total_uncertainty)
        else:
            throttle_cap = 1.0
        modulated = action.copy()
        modulated[1] = min(float(modulated[1]), throttle_cap)

        return modulated, over_threshold, throttle_cap

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Modulate the action by total uncertainty, then step the env.
        @param action: Raw action from the policy.
        @return Standard Gymnasium (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1
        total = self._current_epistemic + self._current_aleatoric

        # apply() reports the RAW per-step crossing (over_threshold); the debounce
        # below decides the actual handoff. The throttle cap is applied every step
        # regardless (it is a soft, reversible response, unlike the handoff).
        modulated_action, over_threshold, throttle_cap = SafetyWrapper.apply(
            action,
            total_uncertainty=total,
            caution_gain=self._caution_gain,
            slow_threshold=self._slow_threshold,
            handoff_threshold=self._handoff_threshold,
        )
        self._throttle_cap_sum += throttle_cap

        # Debounce: hand off only after total stays over the threshold for
        # handoff_consecutive_steps decisions in a row. A single spike (clean total
        # spikes to ~1.3 even on RTK-fixed) resets the counter and does not truncate.
        if over_threshold:
            self._consecutive_over += 1
        else:
            self._consecutive_over = 0
        handoff = self._consecutive_over >= self._handoff_consecutive_steps

        if handoff:
            # On a confirmed handoff, deliver the full-stop command apply() built
            # for the over-threshold step (zero steer/throttle, full brake).
            modulated_action = np.zeros_like(action)
            modulated_action[2] = 1.0
            self._handoff_count += 1
            logger.info(
                "Safety handoff triggered (total uncertainty=%.3f >= threshold=%.3f "
                "for %d consecutive steps)",
                total,
                self._handoff_threshold,
                self._consecutive_over,
            )

        obs, reward, terminated, truncated, info = self.env.step(modulated_action)

        info["throttle_cap"] = throttle_cap
        info["safety_handoff"] = handoff
        info["epistemic"] = self._current_epistemic
        info["aleatoric"] = self._current_aleatoric
        info["total_uncertainty"] = total

        if handoff:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset the environment and clear episode counters.
        @return Standard Gymnasium (obs, info).
        """
        self._handoff_count = 0
        self._throttle_cap_sum = 0.0
        self._step_count = 0
        self._consecutive_over = 0
        self._current_epistemic = 0.0
        self._current_aleatoric = 0.0
        obs, info = self.env.reset(**kwargs)
        return (np.asarray(obs), info)

    def get_episode_safety_stats(self) -> Dict[str, float]:
        """
        @brief Safety statistics for the completed episode.
        @return Dict with handoff_count, avg_throttle_cap, total_steps.
        """
        avg_cap = (
            self._throttle_cap_sum / self._step_count if self._step_count > 0 else 1.0
        )
        return {
            "handoff_count": float(self._handoff_count),
            "avg_throttle_cap": avg_cap,
            "total_steps": float(self._step_count),
        }
