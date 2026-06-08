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

# Consecutive steps the success conditions (in-bay polygon fit + velocity below
# threshold) must hold before the episode terminates as a park. Prevents a
# fly-through from counting as a success.
SUCCESS_DWELL_STEPS = 5

# Strict inward bay margin (metres) for the polygon-fit success check - the
# published parking criterion used by EVALUATION, the demo driver, and the lot
# inspector. Negative inflates the acceptance box outward so corners may overhang
# by |margin| metres.
#
# TRAINING does NOT use this constant: training (and hyperparameter tuning) read
# `bay_margin` from configs/deployment/sim/env_config.yaml, relaxed per stage by
# the curriculum files in configs/deployment/sim/curriculum/. The training default
# in env_config equals this strict value, and the curriculum tightens back to it.
STRICT_BAY_MARGIN = -0.25

# ---------------------------------------------------------------------------
# Corridor Reward Shaping (not success criteria)
# ---------------------------------------------------------------------------
#
# The reward is shaped in the BAY FRAME, not as a radial distance to the bay
# centre: the bay depth axis defines a centreline and the car is rewarded for
# getting onto it (cross-track -> 0), squaring to it (heading, 180-deg symmetric),
# and advancing to the parked depth (along-track -> 0). Bays are open / back-to-
# back, so the shaping uses magnitudes and 180-deg heading symmetry. These live
# here, not in YAML, because reward changes are code, not data.

# Cross-track reference (metres): half-width of the approach corridor. The
# `on_line` factor is 1 on the centreline and ramps to 0 at this offset.
CORRIDOR_HALF_WIDTH = 2.0

# Along-track reference (metres): depth scale over which the `near_depth` factor
# ramps from 1 (at the parked depth) to 0, covering the full approach run-in.
ALONG_TRACK_SCALE = 6.0

# Orientation error (radians) at which the alignment factor saturates (45 deg).
# Reused by the corridor `aligned` factor.
APPROACH_INNER_ALIGNMENT_CUTOFF = np.pi / 4

# Corridor potential weights: phi = -(W_ALONG*|along| + W_CROSS*|cross| +
# W_HEAD*heading_err). Cross-track and heading outweigh along-track so the gradient
# pulls the car onto the centreline and square before advancing in depth; W_HEAD is
# the alignment lever (at 3.0 a 20 deg heading error costs ~1.05 m of along-track).
CORRIDOR_W_ALONG = 1.0
CORRIDOR_W_CROSS = 2.0
CORRIDOR_W_HEAD = 3.0

# Held-stop finisher on top of the corridor potential:
# endgame = (ENDGAME_MOVE_COEF + ENDGAME_HOLD_COEF*stopped) * on_line * aligned
# * near_depth. MOVE applies while moving, +HOLD once stopped on the line. Bounded
# so the most an episode can accrue stays below the discounted +50 terminal, hence
# completing the park always dominates hovering near the goal.
ENDGAME_MOVE_COEF = 0.008
ENDGAME_HOLD_COEF = 0.006

# ---------------------------------------------------------------------------
# Obstacle Clearance Shaping (safety nudge, not a success criterion)
# ---------------------------------------------------------------------------
#
# Smooth penalty for drifting toward a neighbouring parked car during a crooked
# approach. SAFE sits below the ~0.98 m side gap a square-parked ego leaves beside
# an occupied neighbour, so a correct park pays ~0; the reward also gates this off
# once the car is square on the centreline.
OBSTACLE_CLEARANCE_SAFE = 0.8
OBSTACLE_CLEARANCE_DANGER = 0.3

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
# Observation Normalisation Scales
# ---------------------------------------------------------------------------
# Fixed physical-range divisors for the observation, applied in build_observation():
# each component is divided by its scale and clipped to +/-OBS_NORM_CLIP. Stage- and
# layout-invariant: identical in training, evaluation, every layout, and on the real
# vehicle. Generous physical bounds; the clip guards outliers, not the working range.
OBS_SPEED_SCALE = 5.0  # m/s - forward parking speed cap (speed is non-negative)
OBS_YAW_RATE_SCALE = np.pi  # rad/s - vyaw is wrapped to [-pi, pi] in build_observation
OBS_STD_POS_SCALE = 5.0  # m - EKF position-std ceiling (degraded GNSS tier ~5 m)
OBS_STD_YAW_SCALE = 0.5  # rad - EKF heading-std ceiling (~29 deg)
OBS_TARGET_POS_SCALE = (
    40.0  # m - relative target offset scale over the lot (65 x 42.5 m)
)
OBS_TARGET_YAW_SCALE = (
    np.pi / 2
)  # rad - dyaw wrapped to [-pi/2, pi/2] (180-deg symmetry)
OBS_OBSTACLE_DIST_SCALE = (
    25.0  # m - 2D LiDAR max range (configs/deployment/sensor_config.yaml)
)
OBS_OBSTACLE_BEARING_SCALE = np.pi / 2  # rad - forward-hemisphere bearing bound

# Clip magnitude applied after scaling. Wide enough that only outliers are clipped.
OBS_NORM_CLIP = 5.0

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

# Radial distance from target bay before out-of-bounds termination (metres).
# Used on the real-world inference path only; the sim path uses the soft
# polygon boundary below.
OUT_OF_BOUNDS_THRESHOLD = 20.0

# ---------------------------------------------------------------------------
# Dense reward scale
# ---------------------------------------------------------------------------
# Per-episode progress target. progress = (phi(curr) - phi(prev)) /
# max(|phi(start)|, PHI_NORM_FLOOR) * PROGRESS_TARGET. The /|phi(start)| factor
# equalises bays so a far bay offers no larger shaping pool than a near one;
# PROGRESS_TARGET sets the dense magnitude. Set to the stage-1 |phi(start)| (~39 m)
# so for the fixed stage-1 bay the /|phi(start)| factor is ~1 and the dense reward
# equals the raw potential progress (phi(curr) - phi(prev)) the stage-1 reward was
# designed around; for stage 2+ the same factor equalises varied bays. Terminals
# (+50 / -25 / -10) are NOT scaled by it.
PROGRESS_TARGET = 39.0

# Floor on the per-episode start potential |phi(start)| used as the reward
# normaliser. Progress and the graded timeout penalty are divided by
# max(|phi(start)|, PHI_NORM_FLOOR) so a spawn very close to its bay
# (|phi(start)| -> 0) cannot blow the normalised reward up.
PHI_NORM_FLOOR = 5.0

# Graded timeout penalty coefficients. The penalty at truncation is
# -(TIMEOUT_POS_COEF * final_pos_error + TIMEOUT_YAW_COEF * final_orientation_error),
# normalised by the per-episode start potential and clamped to
# TIMEOUT_PENALTY_FLOOR_NORM. yaw is weighted above pos so a square near-miss is
# cheaper than a far-short freeze.
TIMEOUT_POS_COEF = 1.5
TIMEOUT_YAW_COEF = 2.5

# Floor of the graded timeout penalty. Sized below the worst graded value but above
# the ego collision penalty so the ordering success(+50) > timeout > ego
# collision(-25) holds and the policy never crashes deliberately to escape a timeout.
TIMEOUT_PENALTY_FLOOR_NORM = -24.0

# ---------------------------------------------------------------------------
# Soft Out-of-Bounds Boundary (sim training)
# ---------------------------------------------------------------------------
# The drivable boundary is the lot polygon inflated outward by this margin
# (metres), forming a run-off skirt beyond the lot edge that softens the boundary
# penalty during early learning.
OOB_INFLATION_MARGIN = 5.0

# Reward applied each policy decision the ego centre is outside the inflated
# polygon. Applied RAW (never divided by the per-episode normaliser): the lot edge is
# a bay-independent world boundary, so its cost must not shrink for far bays. It
# accumulates so a sustained run-out terminates the episode.
OOB_STEP_PENALTY = -0.5

# Accumulated out-of-bounds cost (sum of |OOB_STEP_PENALTY|) at which the episode
# terminates with no extra crash penalty - the accrued per-step penalties are the
# cost. At OOB_STEP_PENALTY = -0.5 this is reached after ~20 consecutive outside
# decisions, so a committed run-out terminates while a momentary clip does not. Sized
# below the ego collision penalty so leaving the lot is never punished harder than a
# real collision.
OOB_TERMINATION_PENALTY_LIMIT = 10.0
