"""
@file __init__.py
@brief Shared logging, metrics, visualisation, constants, and covariance tools.
"""

from uncertainty_rl.utils.actuation_calibration import ActuationCalibration
from uncertainty_rl.utils.constants import (
    ACTION_DIM,
    APPROACH_INNER_ALIGNMENT_CUTOFF,
    APPROACH_INNER_RADIUS,
    COVARIANCE_FEATURES_DIM,
    EVAL_BAY_MARGIN,
    OBSTACLE_FEATURES_DIM,
    OOB_INFLATION_MARGIN,
    OOB_STEP_PENALTY,
    OOB_TERMINATION_PENALTY_LIMIT,
    OUT_OF_BOUNDS_THRESHOLD,
    SUCCESS_THRESHOLD_VELOCITY,
    TARGET_POSE_DIM,
    TIMEOUT_PENALTY_FLOOR,
    TOTAL_OBS_DIM,
    TRAINING_BAY_MARGIN,
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
    bay_containment_fraction,
    car_fully_inside_bay,
    inflate_polygon,
    point_in_polygon,
    wrap_angle_symmetric,
    zone_bbox,
)
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
    "APPROACH_INNER_ALIGNMENT_CUTOFF",
    "APPROACH_INNER_RADIUS",
    "COVARIANCE_FEATURES_DIM",
    "EVAL_BAY_MARGIN",
    "OBSTACLE_FEATURES_DIM",
    "OOB_INFLATION_MARGIN",
    "OOB_STEP_PENALTY",
    "OOB_TERMINATION_PENALTY_LIMIT",
    "OUT_OF_BOUNDS_THRESHOLD",
    "SUCCESS_THRESHOLD_VELOCITY",
    "TARGET_POSE_DIM",
    "TIMEOUT_PENALTY_FLOOR",
    "TOTAL_OBS_DIM",
    "TRAINING_BAY_MARGIN",
    "VEHICLE_STATE_DIM",
    # Covariance utilities
    "extract_2d_covariance_features",
    "get_covariance_dimension",
    "validate_covariance_matrix",
    # Geometry utilities (private helpers re-exported for internal package use)
    "bay_containment_fraction",
    "car_fully_inside_bay",
    "inflate_polygon",
    "zone_bbox",
    "point_in_polygon",
    "wrap_angle_symmetric",
    "_interpolate_cone_positions",
    "_compute_relative_target_pose",
]
