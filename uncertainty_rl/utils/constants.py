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

# Velocity threshold for successful parking (m/s). Combined with the
# geometric in-bay check (car_fully_inside_bay) to define a parked vehicle.
SUCCESS_THRESHOLD_VELOCITY = 0.1

# Inward bay margin applied to the polygon-fit success check (metres).
# Zero = "inside or on the line"; positive shrinks the bay; negative inflates
# it to allow corners to overhang the bay edge by `-margin` metres.
SUCCESS_BAY_MARGIN = -0.175

# ---------------------------------------------------------------------------
# Approach Reward Shaping (not success criteria)
# ---------------------------------------------------------------------------

# Position error (metres) below which the inner-annulus approach shaping
# fires. Decoupled from the success gate so this ring can be tuned
# independently of what counts as a park.
APPROACH_INNER_RADIUS = 1.5

# Orientation error (radians) at which the inner alignment factor saturates.
APPROACH_INNER_ALIGNMENT_CUTOFF = np.pi / 4

# ---------------------------------------------------------------------------
# State Space Dimensions
# ---------------------------------------------------------------------------

# Core vehicle state: [speed, vyaw]. speed is signed body-frame longitudinal
# velocity (m/s) from the EKF.
VEHICLE_STATE_DIM = 2

# EKF localisation uncertainty features: [std_x, std_y, std_yaw]
COVARIANCE_FEATURES_DIM = 3

# Relative target pose features: [dx, dy, dyaw] in ego body frame
TARGET_POSE_DIM = 3

# Hemispheric obstacle clearance features:
# [left_dist, left_bearing, right_dist, right_bearing, forward_dist]
OBSTACLE_FEATURES_DIM = 5

# Total observation dimension (with uncertainty conditioning and obstacle obs)
# Index   0:      signed body-frame speed (m/s)
# Index   1:      yaw rate (vyaw, rad/s)
# Indices 2-4:    EKF covariance features (std_x, std_y, std_yaw) (when include_covariance=True)
# Indices 5-7:    relative target pose (dx, dy, dyaw)
# Indices 8-12:   hemispheric obstacle clearance (when include_obstacle_obs=True)
TOTAL_OBS_DIM = (
    VEHICLE_STATE_DIM
    + COVARIANCE_FEATURES_DIM
    + TARGET_POSE_DIM
    + OBSTACLE_FEATURES_DIM
)  # 13

# ---------------------------------------------------------------------------
# Action Space Dimensions
# ---------------------------------------------------------------------------

# Continuous action: [steering, throttle, brake]
# steering : [-1, 1]  left to right
# throttle : [ 0, 1]  forward throttle (no reverse gear: forward perpendicular
#                     bay parking only)
# brake    : [ 0, 1]  friction brake
# Throttle and brake are separate non-negative axes so a held stop
# (throttle = 0, brake > 0) is a stable region of the action space.
ACTION_DIM = 3

# ---------------------------------------------------------------------------
# Environment Safety and Termination Thresholds
# ---------------------------------------------------------------------------

# Maximum physically plausible vehicle speed in a parking lot (m/s).
MAX_PARKING_SPEED = 15.0

# Minimum clearance to any obstacle before episode terminates (metres)
CLEARANCE_THRESHOLD = 0.8

# Maximum distance from target bay before out-of-bounds termination (metres)
OUT_OF_BOUNDS_THRESHOLD = 20.0
