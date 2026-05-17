"""
@file constants.py
@brief Structural constants for the uncertainty RL project.

Defines dimensions and thresholds that are fixed by the system architecture.
These are not tuneable hyperparameters - changing them requires coordinated
updates across networks, environments, and evaluation code.

Tuneable values (timesteps, seeds, noise levels, etc.) belong in YAML configs.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Success Criteria for Parking Manoeuvres
# ---------------------------------------------------------------------------

# Position error threshold for successful parking (metres).
# STAGE 1 CURRICULUM: relaxed from 0.75 to 1.5. The 0.75 m window gave PPO
# essentially no success signal from random init (8+ runs, ~0 successes), so
# the +50 terminal reward never propagated back. 1.5 m makes success reachable
# so the agent can learn what winning looks like. RESTORE TO 0.75 before
# evaluation - 0.75 m is the real-world parking standard.
SUCCESS_THRESHOLD_POSITION = 1.5

# Orientation error threshold for successful parking (radians)
# STAGE 1 CURRICULUM: relaxed from 10 to 25 degrees - see note above.
# RESTORE TO np.deg2rad(10) before evaluation.
SUCCESS_THRESHOLD_ORIENTATION = np.deg2rad(25)

# Velocity threshold for successful parking (m/s)
# STAGE 1 CURRICULUM: relaxed from 0.1 to 0.3 - see note above.
# RESTORE TO 0.1 before evaluation.
SUCCESS_THRESHOLD_VELOCITY = 0.3

# ---------------------------------------------------------------------------
# State Space Dimensions
# ---------------------------------------------------------------------------

# Core vehicle state: [vyaw]
VEHICLE_STATE_DIM = 1

# EKF localisation uncertainty features: [std_x, std_y, std_yaw]
COVARIANCE_FEATURES_DIM = 3

# Relative target pose features: [dx, dy, dyaw] in ego body frame
TARGET_POSE_DIM = 3

# Hemispheric obstacle clearance features:
# [left_dist, left_bearing, right_dist, right_bearing, forward_dist]
OBSTACLE_FEATURES_DIM = 5

# Total observation dimension (with uncertainty conditioning and obstacle obs)
# Index   0:     yaw rate (vyaw)
# Indices 1-3:   EKF covariance features (std_x, std_y, std_yaw) (when include_covariance=True)
# Indices 4-6:   relative target pose (dx, dy, dyaw)
# Indices 7-11:  hemispheric obstacle clearance (when include_obstacle_obs=True)
TOTAL_OBS_DIM = (
    VEHICLE_STATE_DIM
    + COVARIANCE_FEATURES_DIM
    + TARGET_POSE_DIM
    + OBSTACLE_FEATURES_DIM
)  # 12

# ---------------------------------------------------------------------------
# Action Space Dimensions
# ---------------------------------------------------------------------------

# Continuous action: [steering, drive]
# steering : [-1, 1]  left to right
# drive    : [-1, 1]  negative = brake (friction), positive = forward throttle.
#                     No reverse gear: forward perpendicular bay parking only.
ACTION_DIM = 2

# ---------------------------------------------------------------------------
# Environment Safety and Termination Thresholds
# ---------------------------------------------------------------------------

# Maximum physically plausible vehicle speed in a parking lot (m/s).
MAX_PARKING_SPEED = 15.0

# Minimum clearance to any obstacle before episode terminates (metres)
CLEARANCE_THRESHOLD = 0.8

# Maximum distance from target bay before out-of-bounds termination (metres)
OUT_OF_BOUNDS_THRESHOLD = 20.0
