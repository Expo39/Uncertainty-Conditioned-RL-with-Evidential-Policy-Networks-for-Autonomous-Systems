"""
@file _parking_core.py
@brief Shared parking environment logic, independent of CARLA and physical hardware.

Pure functions used by both the CARLA simulation environment (envs/sim/) and
the real-world deployment environment (envs/real/). No CARLA imports, no
ROS 2 imports, no hardware dependencies.

Covers:
  - Observation vector construction from EKF pose, covariance, and LiDAR.
  - Hemispheric LiDAR obstacle feature extraction.
  - Floor plan loading from layout YAML.
  - EKF readiness wait loop (hardware-agnostic, caller provides tick function).

NOT included here (sim-only):
  - compute_parking_reward  -- requires GT position, training/sim only.
  - sample_target_bay       -- random bay selection, sim only; real world
                               receives a target bay from dispatch via
                               RealWorldDeployment.set_target_bay().

@author Antonio Galdes
"""

import logging
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml

from uncertainty_rl.utils.constants import (
    COVARIANCE_FEATURES_DIM,
    OBSTACLE_FEATURES_DIM,
    TARGET_POSE_DIM,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.geometry import _compute_relative_target_pose

logger = logging.getLogger(__name__)

# Shared layout cache: keyed by resolved absolute path string so multiple env
# instances in the same process share the parsed YAML without re-reading disk.
_layout_cache: Dict[str, Any] = {}


def compute_obs_dim(
    include_covariance: bool,
    include_obstacle_obs: bool,
) -> int:
    """
    @brief Compute observation vector length from active feature flags.
    @param include_covariance: Whether EKF covariance features are included.
    @param include_obstacle_obs: Whether hemispheric LiDAR features are included.
    @return Integer observation dimension.

    Base: VEHICLE_STATE_DIM (1) + TARGET_POSE_DIM (3) = 4
    With include_covariance: +COVARIANCE_FEATURES_DIM (3) -> 7
    With include_obstacle_obs: +OBSTACLE_FEATURES_DIM (5) -> 12 (or 9 without cov)
    """
    dim = VEHICLE_STATE_DIM + TARGET_POSE_DIM
    if include_covariance:
        dim += COVARIANCE_FEATURES_DIM
    if include_obstacle_obs:
        dim += OBSTACLE_FEATURES_DIM
    return dim


def build_observation(
    ekf_pose: Optional[np.ndarray],
    uncertainty: Optional[np.ndarray],
    target_bay: Dict[str, Any],
    obstacle_features: np.ndarray,
    include_covariance: bool,
    include_obstacle_obs: bool,
    obs_buffer: np.ndarray,
) -> np.ndarray:
    """
    @brief Construct the policy observation vector.

    Fills obs_buffer in-place and returns a copy. The caller is responsible
    for providing a pre-allocated buffer of the correct shape.

    When ekf_pose is None (EKF not yet initialised), all pose-derived dims
    (velocity, target) are set to zero. When uncertainty is None or all-zero,
    covariance dims are zeroed and a debug log is emitted.

    Layout (include_covariance=True, include_obstacle_obs=True, 12-dim):
      [0]     EKF yaw rate (vyaw)
      [1-3]   EKF std devs (std_x, std_y, std_yaw)
      [4-6]   target bay in ego body frame (dx, dy, dyaw)
      [7-11]  hemispheric LiDAR clearance (left_dist, left_bearing,
              right_dist, right_bearing, forward_dist)

    @param ekf_pose: 4-element array [x, y, yaw, vyaw] in world frame,
                     or None if EKF is not yet available.
    @param uncertainty: COVARIANCE_FEATURES_DIM-element array of covariance
                        features, or None if unavailable.
    @param target_bay: Dict with keys 'x', 'y', 'yaw' in world frame.
    @param obstacle_features: OBSTACLE_FEATURES_DIM-element array from
                              extract_obstacle_features().
    @param include_covariance: Whether to write covariance dims.
    @param include_obstacle_obs: Whether to write obstacle dims.
    @param obs_buffer: Pre-allocated float32 buffer of the correct length.
    @return Copy of obs_buffer populated with current observation.
    """
    obs_buffer[:] = 0.0

    if ekf_pose is not None:
        x = float(ekf_pose[0])
        y = float(ekf_pose[1])
        yaw = float(ekf_pose[2])
        vyaw = float(np.clip(ekf_pose[3], -math.pi, math.pi))

        dx, dy, dyaw = _compute_relative_target_pose(
            x, y, yaw,
            float(target_bay["x"]),
            float(target_bay["y"]),
            float(target_bay["yaw"]),
        )
    else:
        vyaw = 0.0
        dx = dy = dyaw = 0.0

    if not include_covariance:
        obs_buffer[0] = vyaw
        obs_buffer[1] = dx
        obs_buffer[2] = dy
        obs_buffer[3] = dyaw
        if include_obstacle_obs:
            obs_buffer[4:4 + OBSTACLE_FEATURES_DIM] = obstacle_features
        return obs_buffer.copy()

    # With covariance: [vyaw(1), cov(6), target(3), obstacle(5)]
    obs_buffer[0] = vyaw

    if uncertainty is not None:
        unc = uncertainty.astype(np.float32)
        if not np.any(unc):
            logger.debug("[obs] EKF covariance all-zeros -- policy sees no uncertainty")
        obs_buffer[1:1 + COVARIANCE_FEATURES_DIM] = unc
    # else: covariance dims remain zero (EKF not yet publishing)

    cov_end = 1 + COVARIANCE_FEATURES_DIM  # index 7
    obs_buffer[cov_end] = dx
    obs_buffer[cov_end + 1] = dy
    obs_buffer[cov_end + 2] = dyaw

    if include_obstacle_obs:
        tgt_end = cov_end + TARGET_POSE_DIM  # index 10
        obs_buffer[tgt_end:tgt_end + OBSTACLE_FEATURES_DIM] = obstacle_features

    return obs_buffer.copy()


def extract_obstacle_features(
    scan: Optional[np.ndarray],
    out: np.ndarray,
) -> np.ndarray:
    """
    @brief Extract hemispheric obstacle clearance features from a 2D LiDAR scan.

    Fills the pre-allocated output buffer in-place and returns it.

    @param scan: Nx2 float32 array of (x, y) point returns in ego body frame,
                 or None / empty if the sensor has not ticked yet.
    @param out: Pre-allocated float32 buffer of length OBSTACLE_FEATURES_DIM (5).
    @return out filled with [left_dist, left_bearing, right_dist, right_bearing,
            forward_dist]. Any hemisphere with no valid returns gets 0.0.

    @note Self-returns closer than 1.0 m and the rear hemisphere (x <= 0) are
          discarded -- matching the ~270 deg FOV of a front-bumper-mounted LiDAR.
    @note When a scan produces no valid returns (empty tick or all-rear points),
          the buffer is left unchanged so the previous step's values are held.
          This prevents spurious zero spikes at the 15 Hz / 20 Hz scan boundary.
    """
    if scan is None or len(scan) == 0:
        return out

    dists = np.sqrt(scan[:, 0] ** 2 + scan[:, 1] ** 2)

    # Strip self-returns and rear hemisphere
    valid = (dists >= 1.0) & (scan[:, 0] > 0.0)
    if not np.any(valid):
        return out

    # Only zero the buffer once we know we have valid returns to write
    out[:] = 0.0

    dists = dists[valid]
    scan = scan[valid]
    bearings = np.arctan2(scan[:, 1], scan[:, 0])

    _15 = math.radians(15.0)

    # Non-overlapping sectors: left (+15,+90], forward (-15,+15), right (-90,-15)
    left_mask = bearings > _15
    if np.any(left_mask):
        idx = int(np.argmin(dists[left_mask]))
        out[0] = float(dists[left_mask][idx])
        out[1] = float(bearings[left_mask][idx])

    right_mask = bearings < -_15
    if np.any(right_mask):
        idx = int(np.argmin(dists[right_mask]))
        out[2] = float(dists[right_mask][idx])
        out[3] = float(bearings[right_mask][idx])

    # Forward: nearest return within +-15 deg of straight ahead
    forward_mask = np.abs(bearings) <= _15
    if np.any(forward_mask):
        out[4] = float(np.min(dists[forward_mask]))

    return out


def load_floor_plan(
    floor_plans_config: Dict[str, Any],
    eval_mode: bool,
    layout_cache: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """
    @brief Select and load a floor plan layout YAML for one episode.

    During training (eval_mode=False), only floor plans with ood=false are
    eligible. During evaluation (eval_mode=True), all plans are eligible.

    Uses the shared layout_cache to avoid re-reading the same YAML on every
    episode reset. Pass the module-level _layout_cache or a per-class dict.

    @param floor_plans_config: Dict mapping plan names to config dicts, from
                               parking_scenarios.floor_plans in train_config.yaml.
    @param eval_mode: If True, OOD plans are included.
    @param layout_cache: Mutable dict used as a read-through cache.
    @return Tuple (plan_name, layout_dict).
    @raises RuntimeError if no eligible plans are configured.
    @raises FileNotFoundError if the chosen layout YAML does not exist.
    """
    eligible = {
        name: cfg
        for name, cfg in floor_plans_config.items()
        if eval_mode or not cfg.get("ood", False)
    }

    if not eligible:
        raise RuntimeError(
            "No eligible floor plans found. "
            "Check parking_scenarios.floor_plans in train_config.yaml."
        )

    name = random.choice(list(eligible.keys()))
    layout_file = eligible[name].get("layout_file", "")
    layout_path = Path(layout_file)

    if not layout_path.exists():
        raise FileNotFoundError(
            f"Floor plan layout file not found: {layout_path}. "
            "Run 'make generate-layouts' to create it."
        )

    cache_key = str(layout_path.resolve())
    if cache_key not in layout_cache:
        with open(layout_path, "r") as fh:
            layout_cache[cache_key] = yaml.safe_load(fh)
        logger.debug("Cached floor plan layout: %s", layout_path)

    return name, layout_cache[cache_key]


def wait_for_ekf(
    has_lidar: Callable[[], bool],
    has_ekf: Callable[[], bool],
    timeout: float,
    tick_fn: Optional[Callable[[], None]] = None,
    tick_interval: float = 0.05,
) -> None:
    """
    @brief Block until LiDAR and EKF data are both available.

    Hardware-agnostic: the caller provides lambdas for the readiness checks
    and an optional tick function (e.g. world.tick() in sim, no-op in real).

    @param has_lidar: Returns True when at least one LiDAR scan has arrived.
    @param has_ekf: Returns True when the EKF state file has been written.
    @param timeout: Maximum wait time in seconds.
    @param tick_fn: Optional callable invoked each iteration (e.g. CARLA tick).
    @param tick_interval: Sleep duration between poll iterations (seconds).
    @raises RuntimeError listing missing inputs if timeout is exceeded.
    """
    import time

    start = time.monotonic()
    while True:
        lidar_ready = has_lidar()
        ekf_ready = has_ekf()

        if lidar_ready and ekf_ready:
            logger.debug(
                "EKF inputs ready after %.2fs (lidar + covariance).",
                time.monotonic() - start,
            )
            return

        elapsed = time.monotonic() - start
        if elapsed > timeout:
            missing = []
            if not lidar_ready:
                missing.append(
                    "LiDAR scan (check sensor spawned and bridge is publishing)"
                )
            if not ekf_ready:
                missing.append(
                    "EKF covariance (check CovarianceExtractorNode is writing "
                    "ekf_state.json)"
                )
            raise RuntimeError(
                f"EKF inputs not ready after {timeout:.0f}s. "
                f"Missing: {'; '.join(missing)}"
            )

        if tick_fn is not None:
            tick_fn()
        time.sleep(tick_interval)
