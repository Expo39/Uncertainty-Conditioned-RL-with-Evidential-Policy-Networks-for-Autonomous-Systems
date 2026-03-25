"""
@file logging.py
@brief Logging utilities for uncertainty-conditioned RL.

This module provides logging utilities for tracking training progress,
uncertainty metrics, and per-step environment debug diagnostics.
"""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsLogger:
    """
    @class MetricsLogger
    @brief Logger for tracking training and evaluation metrics.
    """

    def __init__(self, log_dir: str, prefix: str = "metrics") -> None:
        """
        @brief Constructor for MetricsLogger.
        @param log_dir: Directory to save logs.
        @param prefix: Prefix for log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

        self.metrics: Dict[str, List[Any]] = {}

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """
        @brief Log metrics for a given step.
        @param step: Training step or episode number.
        @param metrics: Dictionary of metric names to values.
        """
        if "step" not in self.metrics:
            self.metrics["step"] = []
        self.metrics["step"].append(step)

        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def save_csv(self, filename: Optional[str] = None) -> None:
        """
        @brief Save metrics to CSV file.
        @param filename: Optional custom filename.
        """
        if filename is None:
            filename = f"{self.prefix}.csv"

        filepath = self.log_dir / filename

        if not self.metrics:
            return

        # Get all keys
        keys = list(self.metrics.keys())

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

            # Write rows
            num_rows = len(self.metrics[keys[0]])
            for i in range(num_rows):
                row = {key: self.metrics[key][i] for key in keys}
                writer.writerow(row)

    def save_json(self, filename: Optional[str] = None) -> None:
        """
        @brief Save metrics to JSON file.
        @param filename: Optional custom filename.
        """
        if filename is None:
            filename = f"{self.prefix}.json"

        filepath = self.log_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def get_metric(self, name: str) -> Optional[List[Any]]:
        """
        @brief Get logged values for a specific metric.
        @param name: Metric name.
        @return List of logged values or None if metric doesn't exist.
        """
        return self.metrics.get(name)

    def compute_statistics(self, name: str) -> Dict[str, float]:
        """
        @brief Compute statistics for a metric.
        @param name: Metric name.
        @return Dictionary with mean, std, min, max.
        """
        values = self.get_metric(name)
        if values is None:
            return {}

        values_array = np.array(values)
        return {
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }


class UncertaintyTracker:
    """
    @class UncertaintyTracker
    @brief Tracker for epistemic and aleatoric uncertainty during training.
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        @brief Constructor for UncertaintyTracker.
        @param window_size: Size of sliding window for computing statistics.
        """
        self.window_size = window_size
        self.epistemic_values: List[float] = []
        self.aleatoric_values: List[float] = []

    def update(self, epistemic: float, aleatoric: float) -> None:
        """
        @brief Update tracker with new uncertainty values.
        @param epistemic: Epistemic uncertainty value.
        @param aleatoric: Aleatoric uncertainty value.
        """
        self.epistemic_values.append(epistemic)
        self.aleatoric_values.append(aleatoric)

        # Keep only recent values
        if len(self.epistemic_values) > self.window_size:
            self.epistemic_values.pop(0)
            self.aleatoric_values.pop(0)

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        @brief Get statistics for tracked uncertainties.
        @return Dictionary with statistics for epistemic and aleatoric uncertainty.
        """
        if not self.epistemic_values:
            return {}

        epistemic_array = np.array(self.epistemic_values)
        aleatoric_array = np.array(self.aleatoric_values)

        return {
            "epistemic": {
                "mean": float(np.mean(epistemic_array)),
                "std": float(np.std(epistemic_array)),
                "min": float(np.min(epistemic_array)),
                "max": float(np.max(epistemic_array)),
            },
            "aleatoric": {
                "mean": float(np.mean(aleatoric_array)),
                "std": float(np.std(aleatoric_array)),
                "min": float(np.min(aleatoric_array)),
                "max": float(np.max(aleatoric_array)),
            },
        }

    def reset(self) -> None:
        """
        @brief Reset the tracker.
        """
        self.epistemic_values.clear()
        self.aleatoric_values.clear()


class DebugLogger:
    """
    @class DebugLogger
    @brief Per-step debug diagnostics for CARLAParkingEnv.

    When debug=True, emits structured lines at DEBUG level every step covering
    reward breakdown, EKF covariance stats, obstacle proximity, action, and pose
    error. Also produces a compact dict via step_debug_dict() that is embedded in
    vis_history.jsonl frames so the visualiser can display a second HUD line.

    When debug=False every method is a no-op so there is zero overhead during
    normal training runs.

    Usage in the environment::

        self._debug_logger = DebugLogger(debug=config.get("debug", False))
        # inside step():
        self._debug_logger.log_step(reward, obs, uncertainty, action, pos_error)
        # inside _write_vis_state():
        state["debug"] = self._debug_logger.step_debug_dict()
    """

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
        obstacle_type: float = 0.0,
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
        @param action: 3-element action array [steer, throttle, brake].
        @param uncertainty: 9-element EKF covariance feature vector (log1p-
                            transformed), or None when covariance is disabled.
        @param obstacle_dist: Distance to nearest obstacle from LiDAR (metres).
        @param obstacle_type: 0.0=static, 1.0=dynamic nearest obstacle.
        @param ekf_drift: Distance between EKF filtered position and CARLA
                          ground truth (metres). Non-zero indicates localisation
                          error -- key signal for sim-to-real debugging.
        @param lidar_points: Number of points in the latest LiDAR scan. Zero
                             indicates the sensor has not ticked yet or returned
                             no returns (e.g. open area, sensor failure).
        """
        if not self._debug:
            return

        cov_mag = 0.0
        if uncertainty is not None and len(uncertainty) >= 3:
            # Summarise as RMS of the three diagonal std elements (indices 0-2)
            cov_mag = float(np.sqrt(np.mean(np.square(uncertainty[:3]))))

        steer = float(action[0]) if len(action) > 0 else 0.0
        throttle = float(action[1]) if len(action) > 1 else 0.0
        brake = float(action[2]) if len(action) > 2 else 0.0

        yaw_deg = math.degrees(yaw_error)
        obs_label = "dyn" if obstacle_type > 0.5 else "sta"

        self._last_dict = {
            "pos_err": round(pos_error, 3),
            "yaw_err_deg": round(yaw_deg, 1),
            "speed": round(speed, 3),
            "reward": round(reward, 4),
            "cov_rms": round(cov_mag, 4),
            "obs_dist": round(obstacle_dist, 2),
            "obs_type": obs_label,
            "ekf_drift": round(ekf_drift, 3),
            "lidar_pts": lidar_points,
            "steer": round(steer, 3),
            "throttle": round(throttle, 3),
            "brake": round(brake, 3),
        }

        self._logger.debug(
            "[step %4d] "
            "pos_err=%.2fm  yaw=%.1fdeg  spd=%.2fm/s  "
            "rwd=%.4f  cov_rms=%.4f  "
            "obs=%.2fm(%s)  ekf_drift=%.3fm  lidar=%dpts  "
            "act=[%.2f %.2f %.2f]",
            step,
            pos_error,
            yaw_deg,
            speed,
            reward,
            cov_mag,
            obstacle_dist,
            obs_label,
            ekf_drift,
            lidar_points,
            steer,
            throttle,
            brake,
        )

    def step_debug_dict(self) -> Dict[str, Any]:
        """
        @brief Return the debug dict from the most recent log_step() call.

        Returns an empty dict when debug=False or before the first step so that
        consumers can safely do ``frame.get('debug', {})``.

        @return Dict with keys: pos_err, yaw_err_deg, speed, reward, cov_rms,
                obs_dist, steer, throttle, brake.
        """
        if not self._debug:
            return {}
        return dict(self._last_dict)

    def log_reset(self, floor_plan: str, target_bay_id: str, spawn_x: float,
                  spawn_y: float) -> None:
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
