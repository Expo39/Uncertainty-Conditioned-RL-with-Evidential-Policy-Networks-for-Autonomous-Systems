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

# EKF localisation uncertainty features: [std_x, std_y, std_yaw, cov_xx, cov_yy,
# cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
COVARIANCE_FEATURES_DIM = 9

# Relative target pose features: [dx, dy, dyaw] in ego body frame
TARGET_POSE_DIM = 3

# Obstacle awareness features: [nearest_dist_m, nearest_bearing_rad]
# obstacle_type removed: classification relied on privileged CARLA actor list
# (not replicable on a real robot without a separate tracking system).
# Appended to the observation when include_obstacle_obs=True. Easily removable:
# set include_obstacle_obs: false in carla/env_config.yaml to restore 18-dim obs.
OBSTACLE_FEATURES_DIM = 2

# Total observation dimension (with uncertainty conditioning and target pose)
# Indices  0-5:  EKF vehicle state (x, y, yaw, vx, vy, vyaw)
# Indices  6-14: EKF covariance features
# Indices 15-17: relative target pose
# Indices 18-19: obstacle awareness (nearest dist, bearing)
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
