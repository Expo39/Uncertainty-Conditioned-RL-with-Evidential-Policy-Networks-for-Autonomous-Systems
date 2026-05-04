"""
@file __init__.py
@brief Shared logging, metrics, visualisation, constants, and covariance tools.
"""

from uncertainty_rl.utils.constants import (
    ACTION_DIM,
    CLEARANCE_THRESHOLD,
    COVARIANCE_FEATURES_DIM,
    MAX_PARKING_SPEED,
    OBSTACLE_FEATURES_DIM,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_VELOCITY,
    TARGET_POSE_DIM,
    TOTAL_OBS_DIM,
    VEHICLE_STATE_DIM,
)
from uncertainty_rl.utils.covariance_utils import (
    extract_2d_covariance_features,
    get_covariance_dimension,
    validate_covariance_matrix,
)
from uncertainty_rl.utils.geometry import (
    _compute_relative_target_pose,
    _interpolate_cone_positions,
    point_in_polygon,
    wrap_angle_symmetric,
    zone_bbox,
)
from uncertainty_rl.utils.actuation_calibration import ActuationCalibration
from uncertainty_rl.utils.logging import DebugLogger
from uncertainty_rl.utils.visualisation import VisStateWriter

__all__ = [
    # Actuation calibration
    "ActuationCalibration",
    # Logging
    "DebugLogger",
    # Visualisation
    "VisStateWriter",
    # Constants
    "ACTION_DIM",
    "CLEARANCE_THRESHOLD",
    "COVARIANCE_FEATURES_DIM",
    "MAX_PARKING_SPEED",
    "OBSTACLE_FEATURES_DIM",
    "OUT_OF_BOUNDS_THRESHOLD",
    "SUCCESS_THRESHOLD_ORIENTATION",
    "SUCCESS_THRESHOLD_POSITION",
    "SUCCESS_THRESHOLD_VELOCITY",
    "TARGET_POSE_DIM",
    "TOTAL_OBS_DIM",
    "VEHICLE_STATE_DIM",
    # Covariance utilities
    "extract_2d_covariance_features",
    "get_covariance_dimension",
    "validate_covariance_matrix",
    # Geometry utilities (private helpers re-exported for internal package use)
    "zone_bbox",
    "point_in_polygon",
    "wrap_angle_symmetric",
    "_interpolate_cone_positions",
    "_compute_relative_target_pose",
]
