# COG Heading Fusion - Three Fixes for EKF Yaw Stability

Extracted from the EKF localisation pipeline in `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py`.

## Background

The localisation EKF fuses three measurement streams:

- `odom0`: GNSS-derived flat-earth XY (position correction).
- `pose0`: Course-Over-Ground (COG) heading derived from successive GNSS
  positions (yaw correction).
- `imu0`: IMU angular velocity vyaw (yaw prediction).

`pose0` exists because IMU vyaw integration alone drifts over long episodes
without an absolute heading reference. COG provides the absolute reference:
two successive GNSS fixes give a displacement vector whose direction is the
direction of motion.

A manual dryrun on 9 May 2026 with `enable_gnss_noise: true` (rtk_fixed
tier, sigma = 0.02 m) revealed that the EKF yaw drifted by tens of degrees
over a parking manoeuvre while the EKF claimed only a few degrees of
uncertainty. With `enable_gnss_noise: false` the same manoeuvre tracked
yaw correctly. The defect was therefore specific to the COG fusion path.

Three independent bugs were diagnosed and fixed.

---

## Fix 1 -- COG Heading Variance Formula

### Bug

The published heading covariance was:

```
raw_var = 2 * sigma^2 / speed_ms^2
```

where `sigma` is the per-axis GNSS position noise (metres) and `speed_ms`
is `displacement_m / dt` (metres per second). The numerator has units of
`m^2`; the denominator has units of `m^2 / s^2`. The result has units of
`s^2`, not `rad^2` -- the formula is dimensionally inconsistent.

### Correct Derivation

The COG heading is `theta = atan2(dy, dx)` where `dy`, `dx` are the noisy
displacement components between two successive GNSS fixes. Each fix has
per-axis noise variance `sigma^2`; the displacement noise variance is
therefore `2 * sigma^2` per axis. Linearising `atan2`:

```
d(theta)/d(dx) = -sin(theta) / |d|
d(theta)/d(dy) =  cos(theta) / |d|
Var(theta)     = (sin^2 + cos^2) * 2 * sigma^2 / |d|^2
               = 2 * sigma^2 / displacement^2
```

The denominator must be `displacement^2` (metres squared), not
`speed^2` (metres squared per second squared).

### Magnitude of the Error

Since `displacement = speed * dt`, the buggy formula was off by a factor
of `1 / dt^2`. With `dt = 0.05 s` (20 Hz GNSS) this is a factor of 400.
The published variance was 400 times too tight, so the EKF
over-trusted every COG measurement by that factor.

When GNSS noise was off the COG heading was exact regardless of variance,
so the EKF over-trusting a correct value did no harm. When noise was on,
each COG measurement carried random heading error but a deceptively
small variance, dragging the EKF yaw toward whatever the latest noisy
measurement claimed.

### Fix

`_cog_heading_variance` now takes `displacement_m` directly and uses
`displacement_m^2` in the denominator. The diagnostic log still reports
`speed_ms` for human readability, but it no longer enters the variance
calculation.

---

## Fix 2 -- Adaptive Noise Floor for the COG Gate

### Bug

`cog_min_displacement_m` was a fixed constant (0.05 m). The intent was to
reject pure noise displacements at standstill: with `sigma = 0.02 m` the
expected noise displacement is `sigma * sqrt(2) = 0.028 m`, so 0.05 m is
about 1.77 standard deviations above the noise floor. Roughly 21 percent
of standstill GNSS callbacks therefore exceed the gate by chance, each
triggering a COG publication with a heading determined entirely by the
direction of the noise vector.

Even after Fix 1 (correct variance), these spurious triggers remain --
they just carry larger variance. They still nudge the EKF yaw at every
firing, accumulating error during long stationary or slow-creep phases.

### Fix

The gate is now adaptive to the active noise tier:

```
sigma_now    = self._metric_stddev_m if self._gnss_noise_enabled else 0.02
noise_floor  = 3.0 * sigma_now * sqrt(2)
gate         = max(self._cog_min_displacement_m, noise_floor)
```

Three standard deviations above the noise floor reduces the
false-trigger probability to under 1 percent.

### Effective Gates per Tier

| Tier       | sigma (m) | 3-sigma noise gate (m) | Min speed for COG (m/s) |
|------------|-----------|------------------------|-------------------------|
| rtk_fixed  | 0.020     | 0.085                  | 1.7                     |
| rtk_float  | 0.360     | 1.530                  | 30.6                    |
| standalone | 1.802     | 7.642                  | 152.8                   |
| degraded   | 5.000     | 21.213                 | 424.3                   |

For the higher-noise tiers the gate is so high that COG never fires at
parking speeds. This is intentional: COG derived from noise that large
carries no usable heading information. The EKF then runs on `/set_pose`
plus IMU vyaw integration alone, which is sufficient for parking-length
episodes (a few minutes at most).

---

## Fix 3 -- Reverse-Motion Disambiguation

### Bug

COG is the direction of the velocity vector. For a non-holonomic vehicle
the velocity vector aligns with the heading only when going forward.
When reversing, the velocity vector points 180 degrees away from the
heading. The EKF, which treats COG as a yaw measurement, then receives
a 180-degree-flipped heading every time the vehicle reverses.

During parking manoeuvres the vehicle alternates forward and reverse
several times. Each phase change feeds the EKF a contradictory heading,
and the filter settles somewhere between the two -- worse than either
extreme. Position is unaffected because position has no analogous
ambiguity (one true position regardless of direction of travel).

The 180-degree symmetry of the parking goal (`wrap_angle_symmetric`
folds the orientation error to (-pi/2, pi/2]) means the policy can
park in either heading, but only if the EKF yaw is consistent with
the actual heading. Inconsistent intermediate values make the dyaw
observation point in the wrong direction, leaving the policy unable
to align the vehicle with the bay.

### Fix

The relay now subscribes to `/odometry/filtered` (the EKF's own output)
and caches the latest yaw. On every COG measurement, the raw heading
from `atan2(-dy, dx)` is compared against this reference:

```
diff = wrap_pi(raw_heading - ref_yaw)
if abs(diff) > pi / 2:
    raw_heading = wrap_pi(raw_heading + pi)
```

If the difference exceeds 90 degrees, the vehicle is reversing and
the COG is flipped by pi to recover the true heading. The reference
yaw fall-back chain is:

1. Latest EKF yaw from `/odometry/filtered` (steady state).
2. Spawn-yaw seed from `episode_config.json` (first COG before the
   EKF has published any output).
3. No reference (initial transient with no spawn seed) -- raw COG used
   directly, same behaviour as before the fix.

### Why No Feedback-Loop Instability

The relay does not modify the EKF state directly; it only chooses
which of two valid measurements (raw or flipped) to publish. The EKF
fuses the chosen value normally, so there is no amplification path.
The 50 ms staleness of the cached EKF yaw is irrelevant since vehicle
yaw changes only marginally over that interval.

The detector relies on the EKF being approximately correct at the
moment of comparison. This holds because:

- `/set_pose` snaps the EKF yaw to the spawn heading at episode reset
  with covariance `1e-4 rad^2`.
- IMU vyaw integration drifts only a few degrees over a parking
  episode (gyro bias capped at 5 deg/h, integrated noise sub-degree).
- Each correctly disambiguated COG measurement keeps the EKF pinned.

The detector would fail if the EKF yaw ever drifted by more than 90
degrees from truth. This does not occur in practice for the parking
scenarios in this project.

### Configuration Flag

`enable_cog_reverse_detection` (default `true`) in
`configs/ros2_config.yaml` toggles the feature. Disabling it reverts
to the original COG-as-absolute-heading behaviour.

---

## Validation

Manual dryrun on 9 May 2026, three episodes (two forward parking, one
reverse parking) on `rtk_fixed` tier with `enable_gnss_noise: true`:

| Episode | Direction | Final dyaw (EKF) | Final pos error | Yaw RMSE (mod 180) |
|---------|-----------|------------------|-----------------|---------------------|
| 3       | Forward   | -8.6 deg         | 0.10 m          | 17.97 deg           |
| 4       | Forward   | -7.7 deg         | 0.05 m          | 16.65 deg           |
| 5       | Reverse   | -1.8 deg         | 0.20 m          | 16.18 deg           |

All three ended within the success thresholds (10 deg orientation,
0.75 m position) per CARLA ground truth. The reverse-parked episode
matched the forward episodes in quality, confirming Fix 3.

The yaw RMSE figures (15-18 deg) reflect transient drift during slow
phases of the manoeuvre, not final-pose error. The EKF covariance
reports this honestly: yaw standard deviation grows from about 0.04 rad
during high-speed motion to about 0.20 rad during long stationary
phases. This is the variation the evidential policy is designed to
consume.

---

## Residual Limitation

During slow phases below the COG gate (about 1.7 m/s for rtk_fixed),
the EKF runs on IMU vyaw alone. Gyro bias drift accumulates at up to
5 deg/h plus integrated noise. Over a 60-second slow phase this can
reach 5-8 degrees of yaw error before a high-speed phase resumes and
COG corrects it. This is below the parking success threshold but
contributes most of the mid-episode RMSE.

A future improvement could be online gyro-bias estimation, but it is
not a blocker for training and adds complexity that would obscure
the dissertation's core contribution.

---

## See Also

- `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py`
  (implementation of all three fixes).
- `configs/ros2_config.yaml` (`gnss_noise_relay` block: `cog_min_displacement_m`,
  `enable_cog_reverse_detection`).
- `documentation/extras/design/gnss_markov_transitions.md`
  (per-episode tier sampling that drives `sigma`).
