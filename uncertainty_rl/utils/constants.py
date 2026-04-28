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

# Core vehicle state: [vyaw]
# Absolute position (x, y, yaw) and linear velocity (vx, vy) are excluded.
# Navigation intent is encoded by dx/dy/dyaw. vyaw is from the IMU gyro
# directly and is reliable; linear velocity has no correction source.
VEHICLE_STATE_DIM = 1

# EKF localisation uncertainty features: [std_x, std_y, std_yaw,
# cov_xy, cov_xyaw, cov_yyaw]
# Diagonal variances (cov_xx, cov_yy, cov_yawyaw) are redundant given std devs
# above (std = sqrt(var)); dropped to reduce obs dimensionality.
COVARIANCE_FEATURES_DIM = 6

# Relative target pose features: [dx, dy, dyaw] in ego body frame
TARGET_POSE_DIM = 3

# Hemispheric obstacle clearance features:
# [left_dist, left_bearing, right_dist, right_bearing, forward_dist]
# Distances and bearings to nearest LiDAR return in left/right hemispheres
# and nearest return in the forward cone.
# Appended to the observation when include_obstacle_obs=True.
OBSTACLE_FEATURES_DIM = 5

# Total observation dimension (with uncertainty conditioning and obstacle obs)
# Index   0:     yaw rate (vyaw)
# Indices 1-6:   EKF covariance features (when include_covariance=True)
# Indices 7-9:   relative target pose (dx, dy, dyaw)
# Indices 10-14: hemispheric obstacle clearance (when include_obstacle_obs=True)
TOTAL_OBS_DIM = (
    VEHICLE_STATE_DIM
    + COVARIANCE_FEATURES_DIM
    + TARGET_POSE_DIM
    + OBSTACLE_FEATURES_DIM
)  # 17

# ============================================================================
# Action Space Dimensions
# ============================================================================

# Continuous action: [steering, longitudinal]
# steering     : [-1, 1]  left to right
# longitudinal : [-1, 1]  negative = brake, positive = throttle
# The env maps longitudinal to CARLA throttle/brake internally.
# Real-world deployment: drive-by-wire controllers accept the same signed
# longitudinal command and handle the throttle/brake split in hardware.
ACTION_DIM = 2

# ============================================================================
# Environment Safety and Termination Thresholds
# ============================================================================

# Maximum physically plausible vehicle speed in a parking lot (m/s).
# Used to clamp EKF velocity observations during the IMU initialisation
# transient at episode reset, where integrated IMU noise can produce
# unrealistic velocity spikes before the first GNSS correction.
MAX_PARKING_SPEED = 15.0

# Minimum clearance to any obstacle before episode terminates (metres)
CLEARANCE_THRESHOLD = 0.8

# Maximum distance from target bay before out-of-bounds termination (metres)
OUT_OF_BOUNDS_THRESHOLD = 20.0
