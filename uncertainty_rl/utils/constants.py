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
# The reward shapes the approach in the BAY FRAME, not as a radial distance to
# the bay centre. The bay's long (depth) axis defines a centreline; the car is
# rewarded for getting onto that centreline (cross-track -> 0), aligning with it
# (heading -> bay axis, 180-deg symmetric), and advancing to the parked depth
# (along-track -> 0). Bays are open / back-to-back, so the centreline extends out
# BOTH ends and the shaping uses magnitudes (abs(along)) and 180-deg heading
# symmetry. This is what produces the natural arc-onto-the-line approach and
# breaks the "arrives crooked at the mouth" failure of the old radial ring.
# These live here, not in YAML, because they shape the reward and reward changes
# are code, not data.

# Cross-track reference (metres): the half-width of the approach corridor. The
# `on_line` factor is 1 on the centreline and ramps to 0 at this offset. Sized so
# the corridor is wide enough to capture realistic approach lanes yet narrow
# enough that being "on the line" means genuinely lined up with the bay.
CORRIDOR_HALF_WIDTH = 2.0

# Along-track reference (metres): the depth scale over which the `near_depth`
# factor ramps from 1 (at the parked depth) to 0. Covers the full approach run-in
# so the endgame terms have a live gradient across the whole final approach.
ALONG_TRACK_SCALE = 6.0

# Orientation error (radians) at which the alignment factor saturates (45 deg).
# Reused by the corridor `aligned` factor.
APPROACH_INNER_ALIGNMENT_CUTOFF = np.pi / 4

# Corridor potential weights: phi = -(W_ALONG*|along| + W_CROSS*|cross| +
# W_HEAD*heading_err). Cross-track and heading are weighted ABOVE along-track so the
# dominant progress gradient pulls the car onto the centreline and SQUARE before
# advancing in depth. W_HEAD is the alignment lever: at 3.0, a 20 deg heading error
# costs ~1.05 m of along-track distance, so the policy "feels" crookedness as
# strongly as distance - the fix for arriving crooked at the bay mouth. The small
# endgame term below only sharpens the held stop; the alignment pull lives here.
CORRIDOR_W_ALONG = 1.0
CORRIDOR_W_CROSS = 2.0
CORRIDOR_W_HEAD = 3.0

# Endgame held-stop term coefficients. One term (replacing the earlier split
# approach/precision/hold terms): endgame = (ENDGAME_MOVE_COEF +
# ENDGAME_HOLD_COEF*stopped) * on_line * aligned * near_depth. MOVE is the value
# while still moving; +HOLD when fully stopped on the line. Kept small - the
# dominant alignment pull is the corridor potential, not this finisher.
ENDGAME_MOVE_COEF = 0.008
ENDGAME_HOLD_COEF = 0.006

# ---------------------------------------------------------------------------
# Obstacle Clearance Shaping (safety nudge, not a success criterion)
# ---------------------------------------------------------------------------
#
# Smooth penalty for drifting toward a neighbouring parked car DURING a crooked
# approach, turning the binary post-impact collision penalty into a gradient that
# discourages clipping occupied adjacent bays. Calibrated to the bay geometry:
# bay centres are 3.1 m apart, and an ego (half-width ~1.06 m) parked square
# beside an occupied neighbour (half-width ~1.06 m) leaves a ~0.98 m side gap. The
# SAFE distance therefore sits BELOW 0.98 m so a CORRECT park pays ~0; only a
# corner swinging in closer than SAFE incurs a cost. In the reward this penalty is
# additionally gated off once the car is square on the centreline, so an expected
# neighbour abeam of a correctly parked car never registers.
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

# Lower bound on the graded timeout penalty. The timeout penalty scales with
# final position and orientation error, and the lot diagonal exceeds 16 m with
# the target bay sampled per episode, so the unclamped penalty can exceed the
# ego-fault collision magnitude (25) for far-target timeouts. Floored here
# strictly above -25 so timing out is always less costly than crashing - the
# policy must never have an incentive to crash deliberately to escape a worse
# timeout. Keeps the terminal ordering success(+50) > timeout > collision(-25).
TIMEOUT_PENALTY_FLOOR = -24.0

# ---------------------------------------------------------------------------
# Soft Out-of-Bounds Boundary (sim training)
# ---------------------------------------------------------------------------
# The drivable boundary is the lot polygon inflated outward by this margin
# (metres), forming a run-off skirt beyond the lot edge. Leaving the lot is a
# lost episode for a forward-only vehicle, so the skirt is sized to soften the
# penalty near the operational boundary during early learning, not to enable
# recovery.
OOB_INFLATION_MARGIN = 5.0

# Reward applied each policy decision the ego centre is outside the inflated
# polygon. Small and negative so a brief excursion is cheap; it accumulates so
# a sustained run-out terminates the episode (see OOB_TERMINATION_PENALTY_LIMIT).
OOB_STEP_PENALTY = -0.5

# Accumulated out-of-bounds cost (sum of |OOB_STEP_PENALTY| over outside steps)
# at which the episode terminates with no extra crash-magnitude penalty - the
# accrued per-step penalties are the cost. Sized below the ego collision
# penalty so leaving the lot is never punished harder than a real collision.
OOB_TERMINATION_PENALTY_LIMIT = 10.0
