"""
@file safety_wrapper.py
@brief Gymnasium wrapper that intercepts actions based on policy uncertainty.

Sits around CARLAParkingEnv and modulates actions at inference/eval time
based on the evidential policy's epistemic and aleatoric uncertainty
outputs. Training runs WITHOUT the wrapper (the policy learns freely).
Evaluation runs WITH the wrapper (safety layer active).

Two uncertainty types produce two distinct responses:
- Aleatoric (outcome noise): scale down action magnitude (gentler driving).
- Epistemic (novelty/ignorance): graduated from caution to full handoff.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger("uncertainty_rl.envs.safety_wrapper")


class SafetyWrapper(gym.Wrapper):
    """
    @class SafetyWrapper
    @brief Intercepts actions based on evidential uncertainty at eval time.

    The wrapper does NOT call the policy itself. Instead, the evaluation
    loop must set the current uncertainty estimates via set_uncertainty()
    before each step. The wrapper reads these estimates and modulates
    the action accordingly.

    @note Training runs without this wrapper. Only evaluation/deployment
          uses it, so the policy learns freely during training.
    """

    def __init__(
        self,
        env: gym.Env,
        aleatoric_scaling: float = 0.5,
        handoff_threshold: float = 5.0,
    ) -> None:
        """
        @brief Initialise the safety wrapper.
        @param env: The underlying CARLAParkingEnv instance.
        @param aleatoric_scaling: Controls how aggressively aleatoric
            uncertainty scales actions. Higher = more conservative.
            action *= 1.0 / (1.0 + aleatoric_scaling * aleatoric).
        @param handoff_threshold: Epistemic uncertainty level above which
            the wrapper triggers a full safety handoff (zero action).
            Set high -- only extreme cases should trigger this.
        """
        super().__init__(env)
        self._aleatoric_scaling = aleatoric_scaling
        self._handoff_threshold = handoff_threshold

        # Per-step uncertainty estimates, set by evaluation loop.
        self._current_epistemic: float = 0.0
        self._current_aleatoric: float = 0.0

        # Episode-level counters.
        self._handoff_count: int = 0
        self._aleatoric_scale_sum: float = 0.0
        self._step_count: int = 0

    def set_uncertainty(
        self,
        epistemic: float,
        aleatoric: float,
    ) -> None:
        """
        @brief Set the current step's uncertainty estimates.
        @param epistemic: Mean epistemic uncertainty from evidential actor.
        @param aleatoric: Mean aleatoric uncertainty from evidential actor.

        Must be called before each step() during evaluation.
        """
        self._current_epistemic = epistemic
        self._current_aleatoric = aleatoric

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        @brief Modulate action based on uncertainty, then step the env.
        @param action: Raw action from policy.
        @return Standard Gymnasium (obs, reward, terminated, truncated, info).
        """
        self._step_count += 1
        info_extra: Dict[str, Any] = {}

        # 1. Aleatoric: cap longitudinal (throttle) only -- not steering.
        #    High aleatoric means action outcomes are unpredictable (e.g. pedestrian
        #    cutting across, NPC braking suddenly). Reducing speed lowers collision
        #    risk without compromising directional control.
        #    Steering is intentionally left uncapped: the policy must retain full
        #    authority to steer away from obstacles even under high uncertainty.
        modulated_action = action.copy()
        aleatoric_scale = 1.0 / (
            1.0 + self._aleatoric_scaling * self._current_aleatoric
        )
        # Clamp the upper bound of longitudinal -- braking is always unrestricted.
        modulated_action[1] = np.clip(modulated_action[1], -1.0, aleatoric_scale)
        self._aleatoric_scale_sum += aleatoric_scale

        info_extra["aleatoric_scale"] = aleatoric_scale

        # 2. Epistemic: safety handoff if extremely uncertain.
        if self._current_epistemic >= self._handoff_threshold:
            modulated_action = np.zeros_like(action)
            info_extra["safety_handoff"] = True
            self._handoff_count += 1
            logger.info(
                "Safety handoff triggered (epistemic=%.3f >= threshold=%.3f)",
                self._current_epistemic,
                self._handoff_threshold,
            )
        else:
            info_extra["safety_handoff"] = False

        obs, reward, terminated, truncated, info = self.env.step(modulated_action)

        # Merge safety info into step info.
        info.update(info_extra)
        info["epistemic"] = self._current_epistemic
        info["aleatoric"] = self._current_aleatoric

        # On handoff, truncate the episode (correct action when uncertain).
        # No reward penalty -- handoff is the RIGHT thing to do.
        if info_extra.get("safety_handoff", False):
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        @brief Reset environment and clear episode counters.
        @return Standard Gymnasium (obs, info).
        """
        self._handoff_count = 0
        self._aleatoric_scale_sum = 0.0
        self._step_count = 0
        self._current_epistemic = 0.0
        self._current_aleatoric = 0.0
        return self.env.reset(**kwargs)

    def get_episode_safety_stats(self) -> Dict[str, float]:
        """
        @brief Get safety statistics for the completed episode.
        @return Dict with handoff_count, avg_aleatoric_scale, total_steps.
        """
        avg_scale = (
            self._aleatoric_scale_sum / self._step_count
            if self._step_count > 0
            else 1.0
        )
        return {
            "handoff_count": float(self._handoff_count),
            "avg_aleatoric_scale": avg_scale,
            "total_steps": float(self._step_count),
        }
