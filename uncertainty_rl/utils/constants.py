"""
@file constants.py
@brief Structural constants for the uncertainty RL project.

These are fixed by the system architecture, not tuneable: changing them requires
coordinated updates across networks, environments, and evaluation code. Tuneable
values (timesteps, seeds, noise levels) belong in YAML configs.

Observation layout (TOTAL_OBS_DIM = 13, both ablation flags on):

    Index   0:      signed body-frame speed (m/s)
    Index   1:      yaw rate (vyaw, rad/s)
    Indices 2-4:    EKF covariance features (std_x, std_y, std_yaw),
                    present when include_covariance=True
    Indices 5-7:    relative target bay pose (dx, dy, dyaw), ego body frame
    Indices 8-12:   hemispheric obstacle clearance (left_dist, left_bearing,
                    right_dist, right_bearing, forward_dist), present when
                    include_obstacle_obs=True

Action layout (ACTION_DIM = 3), all continuous:

    steering : [-1, 1]  left to right
    throttle : [ 0, 1]  forward throttle (no reverse gear: forward
                        perpendicular bay parking only)
    brake    : [ 0, 1]  friction brake

Throttle and brake are separate non-negative axes so a held stop
(throttle = 0, brake > 0) is a stable region of the action space.
"""

import numpy as np

# Velocity threshold for a successful park (m/s), combined with the geometric
# in-bay check (car_fully_inside_bay).
SUCCESS_THRESHOLD_VELOCITY = 0.1

# Consecutive steps the success conditions must hold before the episode
# terminates as a park, so a fly-through cannot count as a success.
SUCCESS_DWELL_STEPS = 5

# Strict inward bay margin (metres) for the polygon-fit success check: the
# published parking criterion, used by evaluation, the demo driver and the lot
# inspector. Negative inflates the acceptance box, so corners may overhang by
# |margin| metres. Training/tuning instead read per-stage `bay_margin`.
STRICT_BAY_MARGIN = -0.25

# Corridor reward shaping (not success criteria), expressed in the BAY FRAME
# rather than as a radial distance: the bay depth axis is a centreline the car is
# rewarded for joining (cross-track -> 0), squaring to (heading, 180-deg
# symmetric because bays are open / back-to-back) and advancing along.

# Cross-track reference (metres): half-width of the approach corridor. The
# `on_line` factor is 1 on the centreline and ramps to 0 at this offset.
CORRIDOR_HALF_WIDTH = 2.0

# Along-track reference (metres): depth scale over which the `near_depth` factor
# ramps from 1 (at the parked depth) to 0, covering the full approach run-in.
ALONG_TRACK_SCALE = 6.0

# Orientation error (radians) at which the alignment factor saturates (45 deg).
APPROACH_INNER_ALIGNMENT_CUTOFF = np.pi / 4

# Corridor potential weights:
# phi = -(W_ALONG*|along| + W_CROSS*|cross| + W_HEAD*heading_err)
CORRIDOR_W_ALONG = 1.0  # depth term, the unit the other two are priced against
# Above along-track, so the gradient pulls onto the centreline before advancing.
CORRIDOR_W_CROSS = 2.0
# Alignment lever: at 3.0 a 20 deg heading error costs ~1.05 m of along-track.
# Kept moderate - larger values make every S-correction locally expensive (heading
# is paid before cross-track pays back), stalling a forward-only car when aligned
# but offset.
CORRIDOR_W_HEAD = 3.0

# Held-stop finisher on top of the corridor potential:
# endgame = (MOVE_COEF + HOLD_COEF*stopped) * on_line * aligned * near_depth
# Both are small so a full episode of hovering near the goal accrues less than the
# discounted +50 terminal, and completing the park always dominates.
ENDGAME_MOVE_COEF = 0.008
ENDGAME_HOLD_COEF = 0.006

# Obstacle clearance shaping (safety nudge, not a success criterion): a smooth
# penalty for drifting toward a neighbouring parked car during a crooked approach.
# SAFE sits below the ~0.98 m side gap a square-parked ego leaves beside an
# occupied neighbour, so a correct park pays ~0.
OBSTACLE_CLEARANCE_SAFE = 0.8
OBSTACLE_CLEARANCE_DANGER = 0.3

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

# Total observation dimension, both ablation flags on. Index table in the module
# docstring; the live dim comes from _compute_obs_dim(), not this constant.
TOTAL_OBS_DIM = (
    VEHICLE_STATE_DIM
    + COVARIANCE_FEATURES_DIM
    + TARGET_POSE_DIM
    + OBSTACLE_FEATURES_DIM
)  # 13

# Fixed physical-range divisors for the observation, applied in build_observation():
# each component is divided by its scale and clipped to +/-OBS_NORM_CLIP. Stage- and
# layout-invariant: identical in training, evaluation, every layout, and on the real
# vehicle. Generous physical bounds; the clip guards outliers, not the working range.
OBS_SPEED_SCALE = 5.0  # m/s - forward parking speed cap (speed is non-negative)
OBS_YAW_RATE_SCALE = np.pi  # rad/s - vyaw is wrapped to [-pi, pi] in build_observation
OBS_STD_POS_SCALE = 1.0  # m - EKF position-std ceiling (RTK-float/standalone band)
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

# Continuous action: [steering, throttle, brake]. Ranges and the reason throttle
# and brake are separate axes are in the module docstring.
ACTION_DIM = 3

# Radial distance from target bay before out-of-bounds termination (metres).
# Used on the real-world inference path only; the sim path uses the soft
# polygon boundary below.
OUT_OF_BOUNDS_THRESHOLD = 20.0

# Consecutive decisions below SUCCESS_THRESHOLD_VELOCITY while outside the
# acceptance box before truncating as a stall (50 = 10 s), else a frozen policy
# floods the buffer with zero-advantage frames. The graded timeout penalty is
# discounted, so an early stall is worse than a late one and cannot be gamed.
STALL_TRUNCATION_DECISIONS = 50

# Live EKF position std (metres, 1-sigma) above which the stall counter is
# SUSPENDED, so the rule governs only a freeze under GOOD localisation and
# waiting out a bad fix is never a stall. Set at the RTK-float/standalone
# boundary: waiting there is the behaviour the input covariance should induce.
STALL_GATE_EKF_STD_M = 0.4

# progress = (phi(curr) - phi(prev)) / max(|phi(start)|, PHI_NORM_FLOOR)
#            * PROGRESS_TARGET
# The divisor equalises bays; 39.0 is the stage-1 |phi(start)|, so there the
# dense reward is the raw potential progress. Terminals are NOT scaled by it.
PROGRESS_TARGET = 39.0

# Floor on the per-episode start potential |phi(start)| used as the reward
# normaliser, so a spawn very close to its bay (|phi(start)| -> 0) cannot blow the
# normalised reward up.
PHI_NORM_FLOOR = 5.0

# Graded timeout penalty, normalised by the per-episode start potential and
# clamped to TIMEOUT_PENALTY_FLOOR_NORM:
# -(TIMEOUT_POS_COEF * final_pos_error + TIMEOUT_YAW_COEF * final_yaw_error)
TIMEOUT_POS_COEF = 1.5
# Weighted above pos, so a square near-miss is cheaper than a far-short freeze.
TIMEOUT_YAW_COEF = 2.5

# Floor of the graded timeout penalty. Sized below the worst graded value but above
# the ego collision penalty so the ordering success(+50) > timeout > ego
# collision(-25) holds and the policy never crashes deliberately to escape a timeout.
TIMEOUT_PENALTY_FLOOR_NORM = -24.0

# The drivable boundary is the lot polygon inflated outward by this margin
# (metres), forming a run-off skirt that softens the boundary penalty during
# early learning.
OOB_INFLATION_MARGIN = 5.0

# Reward applied each policy decision the ego centre is outside the inflated
# polygon. Applied RAW (never divided by the per-episode normaliser): the lot edge
# is a bay-independent world boundary, so its cost must not shrink for far bays.
OOB_STEP_PENALTY = -0.5

# Accumulated |OOB_STEP_PENALTY| at which the episode terminates, with no extra
# crash penalty - the accrued per-step cost is the penalty, and it stays below the
# ego collision penalty so leaving the lot never costs more than a real collision.
# ~20 outside decisions: a committed run-out ends, a momentary clip does not.
OOB_TERMINATION_PENALTY_LIMIT = 10.0
