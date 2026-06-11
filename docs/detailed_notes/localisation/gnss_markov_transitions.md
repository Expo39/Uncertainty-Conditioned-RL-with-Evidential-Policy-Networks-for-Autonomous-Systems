# GNSS Fix-State Markov Chain - Design and Sim-to-Real Rationale

> **Single-phase curriculum.** Mid-episode Markov drift is ON from Stage 1 (master switches
> in `ros2_config.yaml` enabled globally). Each episode STARTS in `rtk_fixed` and the chain
> wanders from there. The chain is a FIXED, stage-invariant process - it is loaded once at
> node startup from `configs/deployment/sim/gnss_noise_profiles.yaml` and is identical in
> every curriculum stage (the GNSS degradation is not a ramped axis). The transition matrix
> shown below is the one in that config.

Extracted from the GNSS noise relay pipeline in `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py`.

## Rationale

In a real outdoor parking lot RTK fix quality changes continuously as the
vehicle moves: driving past a parked van blocks satellites, rounding a corner
re-exposes the antenna, nearby structures cause multipath bursts. The policy
must therefore see covariance spikes that occur mid-manoeuvre, not only a
single constant fix state per episode, so that its uncertainty-conditioned
behaviour is defined across the whole covariance range it will meet on the
real vehicle.

The fix-state tier is therefore modelled as a discrete-time Markov chain that
can transition mid-episode, rather than a constant sampled once at `reset()`.

---

## Discrete-Time Markov Chain

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
from: fixed      0.9920  0.0080  0.0000      0.0000
from: float      0.0400  0.9550  0.0050      0.0000
from: standalone 0.0000  0.0350  0.9600      0.0050
from: degraded   0.0000  0.0000  0.0600      0.9400
```

### Behaviour

The chain is **upward-biased**: at every off-fixed rung the recovery
probability (toward `rtk_fixed`) is far larger than the degradation
probability (deeper). This makes it fixed-dominant and self-recovering.

| Property | Value |
|----------|-------|
| Stationary distribution | fixed ~0.81, float ~0.16, standalone ~0.02, degraded ~0.002 |
| Mean recovery to fixed   | float ~1.4 s, standalone ~3 s, degraded ~3.8 s |
| Leave-fixed rate         | 0.008/step -> a degradation event begins every ~6 s of fixed |

The `_step_markov()` call runs on every GNSS callback (20 Hz); with
`action_repeat=4` that is 4 chain steps per agent decision. Over a typical
~150 s parking approach the agent experiences ~20 excursions from fixed,
reaching standalone in ~90% of approaches and the (rare, brief) degraded tier
in ~25%, covering the full covariance range mid-manoeuvre while the fix is
clean the majority of the time.

---

## Parameter Choices and Justification

### Transition probabilities

The probabilities were chosen to satisfy three constraints:

1. **Fixed-dominant, like a healthy open-sky RTK receiver.** A correctly
   operating RTK rover with sky view holds fix the large majority of the time;
   fix loss is a discrete, transient event (cycle slip, a passing vehicle
   blocking the antenna, a multipath burst), not the baseline. The stationary
   distribution (~81% fixed) reflects this. An earlier matrix that diffused
   freely (~30% fixed, recovery in minutes) modelled a chronically degraded
   urban-canyon receiver and made degraded episodes effectively unwinnable.

2. **Excursions recover within an approach.** Recovery to fixed in a few
   seconds (not minutes) is what makes "wait for the fix to recover" a
   learnable behaviour rather than a frozen, unwinnable episode: the policy can
   hold while uncertain and then commit once the EKF target sharpens.

3. **Recovery outweighs degradation at every rung.** The upward (toward fixed)
   probability is ~7-8x the downward (deeper) probability in each off-fixed row,
   so the chain is pulled back toward fixed. The degraded tier is reachable only
   by several downward steps against this bias, making it rare and brief - the
   abort/handoff regime, present for training but not dominating the episode.

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
  FlatPlane does not render weather. This is a known, documented limitation.
