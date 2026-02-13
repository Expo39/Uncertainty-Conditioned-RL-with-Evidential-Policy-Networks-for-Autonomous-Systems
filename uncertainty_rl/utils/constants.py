"""
@file constants.py
@brief Structural constants for the uncertainty RL project.

Defines dimensions and thresholds that are fixed by the system architecture.
These are not tuneable hyperparameters - changing them requires coordinated
updates across networks, environments, and evaluation code.

Tuneable values (timesteps, seeds, noise levels, etc.) belong in YAML configs.
"""

import numpy as np

# ============================================================================
# Success Criteria for Parking Manoeuvres
# ============================================================================

# Position error threshold for successful parking (metres)
SUCCESS_THRESHOLD_POSITION = 0.5

# Orientation error threshold for successful parking (radians)
# Equivalent to 10 degrees
SUCCESS_THRESHOLD_ORIENTATION = np.deg2rad(10)

# Velocity threshold for successful parking (m/s)
SUCCESS_THRESHOLD_VELOCITY = 0.1

# ============================================================================
# State Space Dimensions
# ============================================================================

# Core vehicle state: [x, y, yaw, vx, vy, vyaw]
VEHICLE_STATE_DIM = 6

# SLAM uncertainty features: [std_x, std_y, std_yaw, cov_xx, cov_yy,
# cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
COVARIANCE_FEATURES_DIM = 9

# Total observation dimension (with uncertainty conditioning)
TOTAL_OBS_DIM = VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM  # 15

# ============================================================================
# Action Space Dimensions
# ============================================================================

# Continuous action: [steering, throttle, brake]
ACTION_DIM = 3
