"""
@file test_debug_logger.py
@brief Unit tests for DebugLogger.

Verifies the zero-overhead no-op contract when debug=False, and the
correct population of the step debug dict when debug=True.
"""

import math
from typing import Any, Dict

import numpy as np
import pytest

from uncertainty_rl.utils.logging import DebugLogger


class TestDebugLoggerDisabled:
    """
    @class TestDebugLoggerDisabled
    @brief Verifies no-op behaviour when debug=False.
    """

    def test_enabled_false_by_default(self) -> None:
        """
        @brief DebugLogger() with no args must default to debug=False.
        """
        logger = DebugLogger()
        assert logger.enabled is False

    def test_step_debug_dict_empty_when_disabled(self) -> None:
        """
        @brief step_debug_dict() must return an empty dict when debug=False.
        """
        logger = DebugLogger(debug=False)
        result = logger.step_debug_dict()
        assert result == {}

    def test_step_debug_dict_empty_before_first_step(self) -> None:
        """
        @brief step_debug_dict() must be empty even before log_step() is called
               (debug=True path, no step yet).
        """
        logger = DebugLogger(debug=True)
        result = logger.step_debug_dict()
        assert result == {}

    def test_log_step_no_op_when_disabled(self) -> None:
        """
        @brief log_step() must not populate the dict when debug=False.
        """
        logger = DebugLogger(debug=False)
        logger.log_step(
            step=1,
            reward=0.5,
            pos_error=1.0,
            yaw_error=0.1,
            speed=0.2,
            action=np.array([0.0, 0.5]),
            uncertainty=np.array([0.1, 0.1, 0.05]),
            obstacle_dist=3.0,
        )
        assert logger.step_debug_dict() == {}

    def test_log_reset_no_op_when_disabled(self) -> None:
        """
        @brief log_reset() must not raise and must have no visible side effects.
        """
        logger = DebugLogger(debug=False)
        logger.log_reset("rectangle", "bay_01", 1.0, 2.0)
        # No assertion needed beyond no exception raised

    def test_log_actors_no_op_when_disabled(self) -> None:
        """
        @brief log_actors() must not raise and must have no visible side effects.
        """
        logger = DebugLogger(debug=False)
        logger.log_actors(n_static=2, n_patrol=1, n_peds=3, n_cones=12)


class TestDebugLoggerEnabled:
    """
    @class TestDebugLoggerEnabled
    @brief Verifies dict population when debug=True.
    """

    def _call_log_step(
        self,
        logger: DebugLogger,
        step: int = 5,
        reward: float = 0.3,
        pos_error: float = 2.0,
        yaw_error: float = 0.5,
        speed: float = 0.8,
        action: "np.ndarray | None" = None,
        uncertainty: "np.ndarray | None" = None,
        obstacle_dist: float = 4.0,
    ) -> Dict[str, Any]:
        if action is None:
            action = np.array([0.1, 0.6, 0.0])
        logger.log_step(
            step=step,
            reward=reward,
            pos_error=pos_error,
            yaw_error=yaw_error,
            speed=speed,
            action=action,
            uncertainty=uncertainty,
            obstacle_dist=obstacle_dist,
        )
        return logger.step_debug_dict()

    def test_enabled_true(self) -> None:
        """
        @brief DebugLogger(debug=True) must report enabled=True.
        """
        logger = DebugLogger(debug=True)
        assert logger.enabled is True

    def test_step_dict_has_all_expected_keys(self) -> None:
        """
        @brief step_debug_dict() must contain all documented keys after log_step().
        """
        logger = DebugLogger(debug=True)
        d = self._call_log_step(logger)
        expected_keys = {
            "pos_err",
            "yaw_err_deg",
            "speed",
            "reward",
            "cov_rms",
            "obs_dist",
            "ekf_drift",
            "lidar_pts",
            "steer",
            "throttle",
            "brake",
        }
        assert expected_keys.issubset(d.keys())

    def test_pos_err_matches_input(self) -> None:
        """
        @brief pos_err in the dict must match the pos_error argument.
        """
        logger = DebugLogger(debug=True)
        d = self._call_log_step(logger, pos_error=3.14)
        assert d["pos_err"] == pytest.approx(3.14, abs=1e-3)

    def test_yaw_err_deg_converted_from_radians(self) -> None:
        """
        @brief yaw_err_deg must be degrees, not radians.
        """
        logger = DebugLogger(debug=True)
        d = self._call_log_step(logger, yaw_error=math.pi / 4)
        assert d["yaw_err_deg"] == pytest.approx(45.0, abs=0.1)

    def test_cov_rms_zero_when_uncertainty_none(self) -> None:
        """
        @brief cov_rms must be 0.0 when uncertainty is None.
        """
        logger = DebugLogger(debug=True)
        d = self._call_log_step(logger, uncertainty=None)
        assert d["cov_rms"] == pytest.approx(0.0)

    def test_cov_rms_nonzero_with_uncertainty(self) -> None:
        """
        @brief cov_rms must be > 0 when a non-zero uncertainty vector is passed.
        """
        logger = DebugLogger(debug=True)
        unc = np.array([0.2, 0.3, 0.1])
        d = self._call_log_step(logger, uncertainty=unc)
        expected_rms = float(np.sqrt(np.mean(np.square(unc))))
        assert d["cov_rms"] == pytest.approx(expected_rms, rel=1e-3)

    def test_action_components_recorded(self) -> None:
        """
        @brief steer, throttle and brake must match the action elements.
        """
        logger = DebugLogger(debug=True)
        action = np.array([0.25, 0.75, 0.4])
        d = self._call_log_step(logger, action=action)
        assert d["steer"] == pytest.approx(0.25, abs=1e-3)
        assert d["throttle"] == pytest.approx(0.75, abs=1e-3)
        assert d["brake"] == pytest.approx(0.4, abs=1e-3)

    def test_step_dict_returns_copy(self) -> None:
        """
        @brief step_debug_dict() must return a copy; mutating it must not affect
               the logger's internal state.
        """
        logger = DebugLogger(debug=True)
        self._call_log_step(logger, reward=0.1)
        d = logger.step_debug_dict()
        d["reward"] = 999.0
        assert logger.step_debug_dict()["reward"] != pytest.approx(999.0)

    def test_log_reset_and_log_actors_do_not_raise(self) -> None:
        """
        @brief log_reset() and log_actors() must not raise when debug=True.
        """
        logger = DebugLogger(debug=True)
        logger.log_reset("trapezoid", "bay_03", -2.5, 10.0)
        logger.log_actors(n_static=3, n_patrol=2, n_peds=1, n_cones=20)
