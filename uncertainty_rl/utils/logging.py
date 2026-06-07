"""
@file logging.py
@brief Per-step debug diagnostics for CARLAParkingEnv.

Provides DebugLogger, a zero-overhead debug helper that emits structured
per-step log lines and a compact debug dict for the visualiser HUD.
All methods are no-ops when debug=False so there is no cost during normal
training runs. SB3's built-in logger handles training metrics.
"""

import logging
import math
from typing import Any, Dict, Optional

import numpy as np


class DebugLogger:
    """
    @class DebugLogger
    @brief Per-step debug diagnostics for CARLAParkingEnv.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, debug: bool = False) -> None:
        """
        @brief Construct a DebugLogger.
        @param debug: If True, emit per-step diagnostics. If False, all methods
                      are no-ops (zero overhead).
        """
        self._debug = debug
        self._logger = logging.getLogger("uncertainty_rl.envs.carla_parking")
        self._last_dict: Dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        """@brief True when debug mode is active."""
        return self._debug

    def log_step(
        self,
        step: int,
        reward: float,
        pos_error: float,
        yaw_error: float,
        speed: float,
        action: "np.ndarray",
        uncertainty: Optional["np.ndarray"],
        obstacle_dist: float,
        ekf_drift: float = 0.0,
        lidar_points: int = 0,
    ) -> None:
        """
        @brief Emit a structured DEBUG log line and cache the debug dict.

        @param step: Current episode step number.
        @param reward: Scalar reward returned this step.
        @param pos_error: Distance from ego to target bay centre (metres).
        @param yaw_error: Heading error to target (radians).
        @param speed: Ego speed (m/s).
        @param action: 2-element action array [steer, drive]. drive is bipolar:
                       positive = throttle, negative = brake.
        @param uncertainty: 9-element EKF covariance feature vector (log1p-
                            transformed), or None when covariance is disabled.
        @param obstacle_dist: Distance to nearest obstacle from LiDAR (metres).
        @param ekf_drift: Distance between EKF filtered position and CARLA
                          ground truth (metres). Non-zero indicates localisation
                          error - key signal for sim-to-real debugging.
        @param lidar_points: Number of points in the latest LiDAR scan. Zero
                             indicates the sensor has not ticked yet or returned
                             no returns (e.g. open area, sensor failure).
        """
        if not self._debug:
            return

        cov_mag = 0.0
        if uncertainty is not None and len(uncertainty) >= 3:
            # Summarise as RMS of the three diagonal std elements (indices 0-2)
            cov_mag = float(np.sqrt(np.mean(uncertainty[:3] ** 2)))

        steer = float(action[0]) if len(action) > 0 else 0.0
        drive = float(action[1]) if len(action) > 1 else 0.0

        yaw_deg = math.degrees(yaw_error)
        self._last_dict = {
            "pos_err": round(pos_error, 3),
            "yaw_err_deg": round(yaw_deg, 1),
            "speed": round(speed, 3),
            "reward": round(reward, 4),
            "cov_rms": round(cov_mag, 4),
            "obs_dist": round(obstacle_dist, 2),
            "ekf_drift": round(ekf_drift, 3),
            "lidar_pts": lidar_points,
            "steer": round(steer, 3),
            "drive": round(drive, 3),
        }

        self._logger.debug(
            "[step %4d] "
            "pos_err=%.2fm  yaw=%.1fdeg  spd=%.2fm/s  "
            "rwd=%.4f  cov_rms=%.4f  "
            "obs=%.2fm  ekf_drift=%.3fm  lidar=%dpts  "
            "act=[%.2f %.2f]",
            step,
            pos_error,
            yaw_deg,
            speed,
            reward,
            cov_mag,
            obstacle_dist,
            ekf_drift,
            lidar_points,
            steer,
            drive,
        )

    def step_debug_dict(self) -> Dict[str, Any]:
        """
        @brief Return the debug dict from the most recent log_step() call.

        Returns an empty dict when debug=False or before the first step so that
        consumers can safely do ``frame.get('debug', {})``.

        @return Dict with keys: pos_err, yaw_err_deg, speed, reward, cov_rms,
                obs_dist, ekf_drift, lidar_pts, steer, drive.
        """
        if not self._debug:
            return {}
        return dict(self._last_dict)

    def log_reset(
        self, floor_plan: str, target_bay_id: str, spawn_x: float, spawn_y: float
    ) -> None:
        """
        @brief Emit a DEBUG line summarising the episode reset.

        @param floor_plan: Name of the floor plan selected for this episode.
        @param target_bay_id: ID of the target bay.
        @param spawn_x: Ego spawn x coordinate (metres).
        @param spawn_y: Ego spawn y coordinate (metres).
        """
        if not self._debug:
            return
        self._logger.debug(
            "[reset] floor=%s  bay=%s  spawn=(%.1f, %.1f)",
            floor_plan,
            target_bay_id,
            spawn_x,
            spawn_y,
        )

    def log_actors(
        self,
        n_static: int,
        n_patrol: int,
        n_peds: int,
        n_cones: int,
    ) -> None:
        """
        @brief Emit a DEBUG line summarising the spawned actors for this episode.

        @param n_static: Number of static parked vehicles.
        @param n_patrol: Number of patrol NPC vehicles.
        @param n_peds: Number of pedestrians.
        @param n_cones: Number of perimeter cone markers.
        """
        if not self._debug:
            return
        self._logger.debug(
            "[actors] static=%d  patrol=%d  peds=%d  cones=%d",
            n_static,
            n_patrol,
            n_peds,
            n_cones,
        )
