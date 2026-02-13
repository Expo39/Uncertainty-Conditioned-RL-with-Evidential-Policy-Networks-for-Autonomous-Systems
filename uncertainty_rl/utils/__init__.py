"""
@file __init__.py
@brief Shared logging, metrics, visualisation utilities, constants, and covariance tools.
"""

from uncertainty_rl.utils.constants import (
    SUCCESS_THRESHOLD_POSITION,
    SUCCESS_THRESHOLD_ORIENTATION,
    SUCCESS_THRESHOLD_VELOCITY,
    VEHICLE_STATE_DIM,
    COVARIANCE_FEATURES_DIM,
    TOTAL_OBS_DIM,
    ACTION_DIM,
)
from uncertainty_rl.utils.covariance_utils import (
    extract_2d_covariance_features,
    get_covariance_dimension,
    validate_covariance_matrix,
)
from uncertainty_rl.utils.logging import MetricsLogger, UncertaintyTracker
from uncertainty_rl.utils.visualisation import (
    plot_training_curves,
    plot_trajectory,
    plot_uncertainty_evolution,
)

__all__ = [
    # Logging and metrics
    "MetricsLogger",
    "UncertaintyTracker",
    # Visualisation
    "plot_uncertainty_evolution",
    "plot_trajectory",
    "plot_training_curves",
    # Constants
    "SUCCESS_THRESHOLD_POSITION",
    "SUCCESS_THRESHOLD_ORIENTATION",
    "SUCCESS_THRESHOLD_VELOCITY",
    "VEHICLE_STATE_DIM",
    "COVARIANCE_FEATURES_DIM",
    "TOTAL_OBS_DIM",
    "ACTION_DIM",
    # Covariance utilities
    "extract_2d_covariance_features",
    "get_covariance_dimension",
    "validate_covariance_matrix",
]

