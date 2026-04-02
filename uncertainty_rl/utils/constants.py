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

# Core vehicle state: [vx, vy, vyaw]
# x, y, yaw removed: absolute odom position accumulates across episodes and
# carries no consistent signal for the policy. Navigation intent is fully
# encoded by dx/dy/dyaw (relative target pose).
VEHICLE_STATE_DIM = 3

# EKF localisation uncertainty features: [std_x, std_y, std_yaw, cov_xx, cov_yy,
# cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
COVARIANCE_FEATURES_DIM = 9

# Relative target pose features: [dx, dy, dyaw] in ego body frame
TARGET_POSE_DIM = 3

# Hemispheric obstacle clearance features:
# [left_dist, left_bearing, right_dist, right_bearing, forward_dist]
# Replaces the old 2-dim nearest-only features with symmetrical left/right
# clearance for bay entry guidance.
# Appended to the observation when include_obstacle_obs=True.
OBSTACLE_FEATURES_DIM = 5

# Total observation dimension (with uncertainty conditioning and obstacle obs)
# Indices  0-2:  velocity (vx, vy, vyaw)
# Indices  3-11: EKF covariance features (when include_covariance=True)
# Indices 12-14: relative target pose (dx, dy, dyaw)
# Indices 15-19: hemispheric obstacle clearance (when include_obstacle_obs=True)
TOTAL_OBS_DIM = (
    VEHICLE_STATE_DIM
    + COVARIANCE_FEATURES_DIM
    + TARGET_POSE_DIM
    + OBSTACLE_FEATURES_DIM
)  # 20

# ============================================================================
# Action Space Dimensions
# ============================================================================

# Continuous action: [steering, throttle, brake]
ACTION_DIM = 3

# ============================================================================
# Environment Safety and Termination Thresholds
# ============================================================================

# Maximum physically plausible vehicle speed in a parking lot (m/s).
# Used to clamp EKF velocity observations during the IMU initialisation
# transient at episode reset, where integrated IMU noise can produce
# unrealistic velocity spikes before the first scan-match correction.
MAX_PARKING_SPEED = 15.0

# Minimum clearance to any obstacle before episode terminates (metres)
CLEARANCE_THRESHOLD = 0.8

# Maximum distance from target bay before out-of-bounds termination (metres)
OUT_OF_BOUNDS_THRESHOLD = 20.0
