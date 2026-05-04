"""
@file _parking_core.py
@brief Shared parking environment logic, independent of CARLA and physical hardware.

Pure functions used by both the CARLA simulation environment (envs/sim/) and
the real-world deployment environment (envs/real/). No CARLA imports, no
ROS 2 imports, no hardware dependencies.
"""

import logging
import math
import random
import time
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

# Sector boundary for hemispheric LiDAR feature extraction (radians).
# Left: bearing > _SECTOR_BOUNDARY; forward: |bearing| <= _SECTOR_BOUNDARY; right: < -_SECTOR_BOUNDARY.
_SECTOR_BOUNDARY: float = math.radians(15.0)

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
    (velocity, target) are set to zero. When uncertainty is None, covariance
    dims remain zero. When uncertainty is present but all-zero, a debug log
    is emitted (EKF may still be initialising).

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
    if ekf_pose is not None:
        x = float(ekf_pose[0])
        y = float(ekf_pose[1])
        yaw = float(ekf_pose[2])
        # Python min/max avoids a numpy scalar allocation
        raw_vyaw = float(ekf_pose[3])
        vyaw = raw_vyaw if -math.pi <= raw_vyaw <= math.pi else max(-math.pi, min(math.pi, raw_vyaw))

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
        else:
            obs_buffer[4:4 + OBSTACLE_FEATURES_DIM] = 0.0
        return obs_buffer.copy()

    # With covariance: [vyaw(1), cov(3), target(3), obstacle(5)]
    obs_buffer[0] = vyaw

    if uncertainty is not None:
        # asarray avoids a copy when already float32
        unc = np.asarray(uncertainty, dtype=np.float32)
        if not np.any(unc):
            logger.debug("[obs] EKF covariance all-zeros - policy sees no uncertainty")
        obs_buffer[1:1 + COVARIANCE_FEATURES_DIM] = unc
    else:
        # Covariance dims remain zero (EKF not yet publishing)
        obs_buffer[1:1 + COVARIANCE_FEATURES_DIM] = 0.0

    cov_end = 1 + COVARIANCE_FEATURES_DIM  # index 4
    obs_buffer[cov_end] = dx
    obs_buffer[cov_end + 1] = dy
    obs_buffer[cov_end + 2] = dyaw

    if include_obstacle_obs:
        tgt_end = cov_end + TARGET_POSE_DIM  # index 7
        obs_buffer[tgt_end:tgt_end + OBSTACLE_FEATURES_DIM] = obstacle_features
    else:
        tgt_end = cov_end + TARGET_POSE_DIM
        obs_buffer[tgt_end:tgt_end + OBSTACLE_FEATURES_DIM] = 0.0

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
          discarded - matching the ~270 deg FOV of a front-bumper-mounted LiDAR.
    """
    if scan is None or len(scan) == 0:
        return out

    x = scan[:, 0]
    y = scan[:, 1]

    # Filter using squared distance to avoid full-array sqrt
    sq = x * x + y * y
    valid = (sq >= 1.0) & (x > 0.0)
    if not valid.any():
        return out

    # Only zero the buffer once we know we have valid returns to write
    out[:] = 0.0

    x = x[valid]
    y = y[valid]
    sq = sq[valid]
    bearings = np.arctan2(y, x)

    _INF = np.inf

    # Non-overlapping sectors: left (> +15 deg), forward (+/-15 deg), right (< -15 deg).
    left_mask = bearings > _SECTOR_BOUNDARY
    if left_mask.any():
        # sq_left keeps squared distances only in the left sector; elsewhere +inf
        sq_left = np.where(left_mask, sq, _INF)
        idx = int(sq_left.argmin())
        out[0] = math.sqrt(float(sq[idx]))
        out[1] = float(bearings[idx])

    right_mask = bearings < -_SECTOR_BOUNDARY
    if right_mask.any():
        sq_right = np.where(right_mask, sq, _INF)
        idx = int(sq_right.argmin())
        out[2] = math.sqrt(float(sq[idx]))
        out[3] = float(bearings[idx])

    fwd_mask = np.abs(bearings) <= _SECTOR_BOUNDARY
    if fwd_mask.any():
        sq_fwd = np.where(fwd_mask, sq, _INF)
        out[4] = math.sqrt(float(sq_fwd.min()))

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


def calibrate_ekf_frame_offset(
    world_x: float,
    world_y: float,
    world_yaw: float,
    get_pose: Callable[[], Optional[np.ndarray]],
    timeout: float,
    tick_fn: Optional[Callable[[], None]] = None,
    tick_interval: float = 0.05,
    pos_stable_threshold: float = 0.5,
    yaw_stable_threshold: float = 0.1,
    min_stable_readings: int = 3,
) -> Tuple[float, float, float, float, float]:
    """
    @brief Compute the odom-to-world 2D rigid body transform by waiting for
           the EKF pose to converge, then returning (tx, ty, cos_r, sin_r, r).

    @param world_x: Known world-frame x of the reference point (metres).
    @param world_y: Known world-frame y of the reference point (metres).
    @param world_yaw: Known world-frame yaw at the reference point (radians).
    @param get_pose: Returns current EKF pose array or None if unavailable.
    @param timeout: Maximum wait time in seconds.
    @param tick_fn: Optional callable invoked each poll iteration (e.g. CARLA tick).
    @param tick_interval: Sleep duration between iterations (seconds).
    @param pos_stable_threshold: Max tx/ty change (m) to count as stable.
    @param yaw_stable_threshold: Max rotation change (rad) to count as stable.
    @param min_stable_readings: Consecutive stable readings required.
    @return (tx, ty, cos_r, sin_r, r) offset tuple, or identity if timed out
            before any pose was received.

    @note See documentation/detailed_notes/ekf_pipeline.md for derivation.
    """
    start = time.monotonic()
    prev_tx: Optional[float] = None
    prev_ty: Optional[float] = None
    prev_r: Optional[float] = None
    stable_count = 0
    last_offset: Optional[Tuple[float, float, float, float, float]] = None

    while True:
        ekf_pose = get_pose()
        elapsed = time.monotonic() - start

        if ekf_pose is None:
            if elapsed > timeout:
                logger.warning(
                    "EKF pose unavailable after %.0fs - odom transform will be identity.",
                    timeout,
                )
                return (0.0, 0.0, 1.0, 0.0, 0.0)
            if tick_fn is not None:
                tick_fn()
            time.sleep(tick_interval)
            continue

        # ekf_state.json y is negated relative to CARLA world y (ROS REP-103
        # vs CARLA left-handed axes). Negate here to match _get_state().
        ekf_x = float(ekf_pose[0])
        ekf_y = -float(ekf_pose[1])
        ekf_yaw = float(ekf_pose[2])

        r = math.atan2(
            math.sin(world_yaw - ekf_yaw),
            math.cos(world_yaw - ekf_yaw),
        )
        cos_r = math.cos(r)
        sin_r = math.sin(r)
        tx = world_x - (cos_r * ekf_x - sin_r * ekf_y)
        ty = world_y - (sin_r * ekf_x + cos_r * ekf_y)
        last_offset = (tx, ty, cos_r, sin_r, r)

        if prev_tx is not None:
            dtx = abs(tx - prev_tx)
            dty = abs(ty - prev_ty)  # type: ignore[operator]
            dr = abs(math.atan2(
                math.sin(r - prev_r),  # type: ignore[arg-type]
                math.cos(r - prev_r),  # type: ignore[arg-type]
            ))
            stable_count = stable_count + 1 if (
                dtx < pos_stable_threshold
                and dty < pos_stable_threshold
                and dr < yaw_stable_threshold
            ) else 0

        prev_tx, prev_ty, prev_r = tx, ty, r

        if stable_count >= min_stable_readings:
            logger.info(
                "EKF converged after %.2fs: rotation=%.1fdeg tx=%.3fm ty=%.3fm"
                " (ref=(%.2f,%.2f) EKF odom=(%.2f,%.2f))",
                elapsed, math.degrees(r), tx, ty, world_x, world_y, ekf_x, ekf_y,
            )
            return last_offset

        if elapsed > timeout:
            logger.warning(
                "EKF convergence timeout (%.0fs) - transform still unstable"
                " after %d stable readings. Using best available transform.",
                timeout, stable_count,
            )
            return last_offset  # type: ignore[return-value]

        logger.debug(
            "Waiting for EKF convergence: stable=%d/%d elapsed=%.1fs",
            stable_count, min_stable_readings, elapsed,
        )
        if tick_fn is not None:
            tick_fn()
        time.sleep(tick_interval)


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
    and an optional tick function.

    @param has_lidar: Returns True when at least one LiDAR scan has arrived.
    @param has_ekf: Returns True when the EKF state file has been written.
    @param timeout: Maximum wait time in seconds.
    @param tick_fn: Optional callable invoked each iteration (e.g. CARLA tick).
    @param tick_interval: Sleep duration between poll iterations (seconds).
    @raises RuntimeError listing missing inputs if timeout is exceeded.
    """
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
