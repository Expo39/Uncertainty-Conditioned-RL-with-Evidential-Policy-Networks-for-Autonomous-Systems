# GNSS Fix-State Markov Chain -- Design and Sim-to-Real Rationale

**Author:** Antonio Galdes
**Date:** 8 April 2026
**Status:** Implemented in `gnss_noise_relay.py` (commit on core/19-model-tuning)

---

## Problem

The original training scheme sampled an RTK fix-state tier (fixed, float,
standalone, degraded) once per episode at `reset()` and held it constant for
the full episode duration. This is unrealistic: in a real outdoor parking lot,
RTK fix quality changes continuously as the vehicle moves -- driving past a
parked van blocks satellites, rounding a corner re-exposes the antenna, nearby
structures cause multipath bursts.

If the policy is only trained on episodes with a single constant fix state, it
has never seen a covariance spike that occurs mid-manoeuvre. On the real
vehicle this could happen at any point during the approach or final alignment,
leaving the policy in an out-of-distribution covariance region where its
uncertainty-conditioned behaviour is undefined.

---

## Solution: Discrete-Time Markov Chain

At each GNSS callback (20 Hz) the active fix-state tier can transition to an
adjacent tier with a small probability. The transition is drawn from a 4-state
discrete Markov chain:

```
States:  rtk_fixed (0) <-> rtk_float (1) <-> standalone (2) <-> degraded (3)
```

No direct transitions between non-adjacent tiers (e.g. fixed -> standalone)
are permitted; degradation and recovery progress through neighbouring states.
This matches the physical mechanism: RTK does not jump directly from cm-level
to metre-level accuracy -- it degrades through float first.

### Transition Matrix (per callback, 20 Hz)

```
             to: fixed   float   standalone  degraded
from: fixed      0.9950  0.0050  0.0000      0.0000
from: float      0.0030  0.9920  0.0050      0.0000
from: standalone 0.0000  0.0040  0.9930      0.0030
from: degraded   0.0000  0.0000  0.0050      0.9950
```

### Mean Dwell Times

At 20 Hz with the probabilities above:

| State      | p(leave) per step | Mean dwell (steps) | Mean dwell (seconds) |
|------------|-------------------|--------------------|----------------------|
| rtk_fixed  | 0.0050            | 200                | 10 s                 |
| rtk_float  | 0.0080            | 125                | 6.3 s                |
| standalone | 0.0070            | 143                | 7.1 s                |
| degraded   | 0.0050            | 200                | 10 s                 |

A typical parking manoeuvre runs 45-200 seconds (max_steps=1000 at 5 Hz agent
decision rate). With mean dwell times of 6-10 seconds, the agent will
experience 5-20 tier transitions per episode on average, covering the full
range of covariance magnitudes mid-manoeuvre.

---

## Parameter Choices and Justification

### Transition probabilities

The probabilities were chosen to satisfy three constraints:

1. **Stationarity matches training tier weights.** At stationarity, the
   Markov chain stationary distribution should approximate the per-episode
   sampling weights in `gnss_noise_profiles.yaml`:
   fixed=0.4, float=0.3, standalone=0.2, degraded=0.1.
   The current matrix produces approximate stationarity:
   pi ~ [0.39, 0.31, 0.22, 0.08] -- close enough for training purposes.

2. **Dwell times are physically plausible.** Real RTK fix changes on a scale
   of seconds to tens of seconds (satellite acquisition ~30 s, multipath burst
   ~2-5 s). Mean dwells of 6-10 s are conservative but within range.

3. **Degradation is more likely than recovery.** The float->standalone (0.005)
   rate is higher than standalone->float (0.004), and standalone->degraded
   (0.003) is non-zero. This reflects that once RTK starts degrading it tends
   to continue degrading rather than recovering immediately.

### IMU process noise covariance

CARLA publishes zero IMU covariance, which causes `robot_localization` to
treat IMU as an infinitely reliable sensor. The result is that EKF covariance
stays artificially flat between GNSS fixes -- it never grows during the IMU
prediction step. On real hardware (e.g. VectorNav VN-100), yaw covariance
grows visibly between 20 Hz GNSS updates.

Fix: set `process_noise_covariance` in the EKF config with values derived
from VN-100 datasheet:

- Gyro noise density: ~0.0035 rad/s/sqrt(Hz) at 20 Hz
  -> per-sample variance = (0.0035 * sqrt(20))^2 = 2.45e-4 (rad/s)^2
  -> using 2.5e-4 (slight rounding up for margin)
- Yaw orientation stddev: ~0.5 deg = 0.0087 rad
  -> variance = 7.6e-5 rad^2

These values mean the EKF covariance inflates slightly between each pair of
GNSS fixes and collapses when a fix arrives, producing the realistic
``saw-tooth'' covariance profile the policy should learn to respond to.

On the real vehicle, replace these with the actual IMU datasheet noise floor.

---

## Control Flag

The Markov chain can be disabled via `enable_markov_transitions: false` in
`configs/ros2_config.yaml` (or the ROS 2 node parameter). This is useful for:

- Ablation studies that require a fixed noise tier for the full episode.
- Debugging: isolating the effect of tier transitions from other variables.
- Evaluation runs where a specific fixed condition is desired.

---

## Where This Is Implemented

| File | What changed |
|------|--------------|
| `uncertainty_rl/ros2/uncertainty_rl_ros2/gnss_noise_relay.py` | `_TIER_ORDER`, `_TIER_DEFAULTS`, `_DEFAULT_TRANSITION_MATRIX` constants; `_apply_tier()`, `_step_markov()` methods; `enable_markov_transitions` parameter; `_gnss_callback()` calls `_step_markov()` each tick |
| `configs/ros2_config.yaml` | `gnss_noise_relay.enable_markov_transitions` flag; `ekf.process_noise_covariance` matrix with realistic IMU noise |
| `uncertainty_rl/ros2/launch/carla_bridge.launch.py` | Passes `enable_markov_transitions` to the node |

---

## Limitations and Future Work

- The transition matrix is hand-tuned. For a higher-fidelity model, the
  matrix could be learned from a logged RTK dataset on the real site.
- Off-diagonal covariance terms (from correlated satellite geometry errors)
  are not modelled -- the NavSatFix covariance is always diagonal.
- Weather effects (ionospheric delay in rain) are not modelled because
  FlatPlane does not render weather. This is a known limitation documented
  in CLAUDE.md.
