# sensor_noise_models

Extracted from `uncertainty_rl/envs/sim/helpers/_sensor_manager.py`,
`uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/imu_noise_relay.py`,
`configs/deployment/sim/env_config.yaml`, and `configs/ros2_config.yaml`.

This note documents the derivation of all sensor noise parameters used in simulation,
with explicit references to the source datasheets. It exists here rather than in the
source files to keep inline comments concise.

---

## 1. SICK TiM571 2D LiDAR (P/N 1075091)

### Physical characteristics

| Spec | Datasheet value |
|------|----------------|
| Measurement principle | HDDM (High Definition Distance Measurement) |
| Aperture angle | 270 deg |
| Scanning frequency | 15 Hz |
| Angular resolution | 0.33 deg |
| Working range | 0.05 m - 25 m |
| Scanning range (10% reflectance) | 8 m typical |
| Systematic error | +-60 mm (typical at 90% reflectance, up to max range) |
| Statistical error (90% reflectance) | <20 mm |
| Statistical error (10% reflectance) | <10 mm (up to 6 m range) |

### Noise model and parameter derivation

Three independent error sources are modelled, applied per-callback in
`SensorManager._apply_lidar_noise()`.

#### 1a. Systematic range error (bias)

The datasheet states +-60 mm systematic error. This is a fixed offset that varies
from unit to unit and with mounting conditions but is constant within a single
measurement session. In simulation it is modelled as:

    bias ~ Uniform(-0.060, +0.060)  metres

Resampled once per episode via `SensorManager.sample_lidar_noise_bias()`, called
from `CARLAParkingEnv.reset()`. The bias is reset to 0.0 in `cleanup()` and
`reset_state()` between episodes.

Config key: `carla_sensors.lidar.noise.range_bias_limit_m = 0.060`

#### 1b. Statistical range error (random noise)

The datasheet gives <20 mm at 90% reflectance and <10 mm at 10% reflectance.
A car park environment contains tarmac, concrete, and vehicle surfaces with
reflectance spanning the full range. The conservative (larger) 90% figure is
used as the 1-sigma parameter:

    range_noise ~ N(0, 0.020)  metres  per point per scan

Config key: `carla_sensors.lidar.noise.range_random_stddev_m = 0.020`

Note: using 0.020 m rather than 0.010 m makes the simulation harder (more noise)
and therefore more conservative for the obstacle clearance features. The 10%
reflectance figure (0.010 m) could be used for a best-case bound.

#### 1c. Angular error

The datasheet gives no angular accuracy specification. No angular noise is modelled.
The 0.33 deg angular resolution is a quantisation property of the sensor, not a
noise figure, and no convention for converting it to a noise model is provided by
the manufacturer. Adding an assumed value would introduce an unverifiable parameter.

#### 1d. Implementation notes

Noise is applied in polar coordinates. No angular noise is modelled (see 1c).
The conversion is:

    r = hypot(x, y)
    theta = arctan2(y, x)
    r_noisy = r + bias + N(0, range_stddev)
    x_noisy = r_noisy * cos(theta)
    y_noisy = r_noisy * sin(theta)

After noise, returns below `min_range_m = 0.05 m` are discarded to prevent
sub-minimum artefacts. The z-coordinate (always 0.0 for a 2D scan) is preserved
unchanged. The output is stored in `_latest_lidar_scan` and consumed by
`extract_obstacle_features()` in `_parking_core.py` without modification.

CARLA's native `noise_stddev` blueprint attribute is zeroed in `_spawn_lidar_2d()`
so that all noise is applied in Python and is fully configurable from YAML.

---

## 2. VectorNav VN-100 IMU (Hardware v7.0)

### Physical characteristics

| Spec | Datasheet value |
|------|----------------|
| Gyro range | +-2000 deg/s |
| Gyro noise density | 0.0035 deg/s/sqrt(Hz) |
| Gyro in-run bias stability | 5 deg/hr typical, 10 deg/hr max |
| Gyro bandwidth | 265 Hz |
| Accel range | +-16 g |
| Accel noise density | 0.14 mg/sqrt(Hz) |
| Accel in-run bias stability | 0.04 mg typical |
| Accel bandwidth | 230 Hz |
| Sample rate | 800 Hz (IMU), 400 Hz (AHRS) |

The EKF runs at 20 Hz and CARLA publishes sensor data at 20 Hz
(`carla_sensors.imu.sensor_tick = 0.05 s`).

### Noise model and parameter derivation

Two complementary mechanisms are applied in `ImuNoiseRelayNode`.

#### 2a. Covariance stamping

`robot_localization` reads the covariance fields of `sensor_msgs/Imu` to perform
Mahalanobis gating and uncertainty propagation. CARLA publishes zero covariance
(meaning infinite reliability), so realistic values must be stamped.

The covariance value to stamp is the variance of a single sample at the EKF
publish rate. For a white-noise process with power spectral density S
(units: (unit)^2/Hz), the variance of one sample at rate f is:

    variance = S * f

**Gyro covariance:**

Noise density: 0.0035 deg/s/sqrt(Hz)                               (Table 3)
Convert:       0.0035 * pi / 180 = 6.10865e-5 rad/s/sqrt(Hz)
PSD:           (6.10865e-5)^2    = 3.73157e-9 (rad/s)^2/Hz
Variance at 20 Hz: 3.73157e-9 * 20 = 7.46314e-8 (rad/s)^2

The code computes this exactly: `(0.0035 * pi/180)**2 * 20`.
Config key: `imu_noise_relay.imu_gyro_variance = 7.4631e-8`

Previous value was 1.0e-7 - not derived from the datasheet.

**Accel covariance:**

Noise density: 0.14 mg/sqrt(Hz)                                     (Table 2)
Convert:       0.14e-3 * 9.81 = 1.37340e-3 m/s^2/sqrt(Hz)
PSD:           (1.37340e-3)^2  = 1.88623e-6 (m/s^2)^2/Hz
Variance at 20 Hz: 1.88623e-6 * 20 = 3.77246e-5 (m/s^2)^2

The code computes this exactly: `(0.14e-3 * 9.81)**2 * 20`.
Config key: `imu_noise_relay.imu_accel_variance = 3.77245e-5`

Previous value was 3.76e-5 - a rough approximation.

**Diagonal covariance matrices** (3x3) are pre-built once at node init and
assigned to the published message. `orientation_covariance` uses the -1.0
sentinel in position [0] so `robot_localization` ignores the CARLA identity
quaternion.

#### 2b. Value noise injection

Covariance stamping alone does not make the EKF prediction step noisier - the
measurement values from CARLA are still ground truth. To make the EKF
realistically uncertain between GNSS fixes, Gaussian noise is added to the
`angular_velocity.z` and `linear_acceleration.x/y` fields before publishing.

**Per-sample Gaussian noise:**

At each callback, independent samples are drawn:

    gyro_noise ~ N(0, sqrt(7.46e-8)) = N(0, 2.73e-4)  rad/s
    accel_noise ~ N(0, sqrt(3.77e-5)) = N(0, 6.14e-3)  m/s^2

These are applied after ZUPT clamping so that standstill behaviour is not
corrupted: if `|vyaw| < zupt_threshold`, the angular velocity is already zeroed
before noise is added, and the noise block is skipped for that channel.

**Per-run bias (in-run bias stability):**

The VN-100 specifies in-run bias stability of 5 deg/hr (gyro, typical) and
0.04 mg (accel, typical). This is a slow drift within one continuous session.
In simulation it is modelled as a fixed offset resampled each episode:

    gyro_bias ~ Uniform(-2.424e-5, +2.424e-5)  rad/s
    accel_bias_x ~ Uniform(-3.924e-4, +3.924e-4)  m/s^2
    accel_bias_y ~ Uniform(-3.924e-4, +3.924e-4)  m/s^2

Derivation (both computed exactly in code):

    gyro:  5 * pi/180 / 3600 = 2.42407e-5 rad/s    [5 deg/hr from Table 3]
    accel: 0.04e-3 * 9.81    = 3.92400e-4 m/s^2    [0.04 mg from Table 2]

The bias is held constant for the entire training run since `ImuNoiseRelayNode`
is a long-running process that is not restarted between episodes.

Config keys: `imu_noise_relay.imu_gyro_bias_limit_rad_s = 2.42407e-5`,
             `imu_noise_relay.imu_accel_bias_limit_ms2 = 3.924e-4`

#### 2c. ZUPT (Zero-velocity Update)

When the vehicle is stationary, CARLA's physics engine produces a small residual
yaw rate (~0.012 rad/s) and acceleration (~0.05-0.15 m/s^2) due to floating-point
noise in the physics solver. Without correction, these residuals cause the EKF to
estimate non-zero velocity at standstill, leading to growing position error.

ZUPT clamping zeros `angular_velocity.z` when `|vyaw| < 0.015 rad/s`, and zeros
`linear_acceleration.x/y` when gyro ZUPT is active and `|ax|, |ay| < 0.2 m/s^2`.
This bounds EKF velocity drift at standstill.

Value noise injection is skipped for channels where ZUPT is active (the `if not
zupt_active` guards in `_imu_callback`), so ZUPT and noise injection are consistent.

---

## 3. u-blox ZED-F9P-05B RTK-GNSS

Datasheet: UBXDOC-963802114-12824, Revision R02, 16-Oct-2024.
All position accuracy values are from Table 3 (GPS+GLO+GAL+BDS mode).

### Physical characteristics

| Spec | Datasheet value | Source |
|------|----------------|--------|
| Horizontal accuracy (RTK fixed) | 0.010 m CEP + 1 ppm | Table 3 |
| Horizontal accuracy (PVT standalone) | 1.5 m CEP | Table 3 |
| Horizontal accuracy (SBAS) | 1.0 m CEP | Table 3 |
| Moving-base heading accuracy | 0.4 deg (50%) at 30 m/s | Table 6 |
| Velocity accuracy | 0.05 m/s (50%) | Table 1 |
| RTK convergence time | <10 s | Table 2 |
| Max navigation update rate (RTK) | 5 Hz (all-constellation) | Table 2 |

No RTK float accuracy is specified in the datasheet. Float is a receiver-internal
state; accuracy depends on atmospheric conditions, baseline length, and satellite
geometry. An industry-standard estimate of 0.300 m CEP is used (documented below).

### CEP to 1-sigma conversion

Position accuracy is specified in CEP (Circular Error Probable, 50th percentile
of the 2D position error distribution). For a 2D isotropic zero-mean Gaussian:

    1-sigma = CEP / 0.8326

This conversion is exact for a circularly symmetric Gaussian. The ZED-F9P-05B
datasheet does not state the distribution assumption explicitly, but CEP is the
standard IEEE definition and this conversion is appropriate.

### Noise tier derivations

Four simulation tiers model the RTK fix-state continuum. Noise is applied as
zero-mean Gaussian per GNSS fix in `GnssNoiseRelayNode._gnss_callback()`.

#### 3a. RTK fixed

Datasheet Table 3: horizontal accuracy = 0.010 m CEP + 1 ppm.

    1-sigma = 0.010 / 0.8326 = 0.01202 m

The 1 ppm baseline term: at a parking-lot baseline of 100 m from the base station,
1 ppm = 0.0001 m - negligible. Excluded from the model.

Datasheet footnote 10 states: "Does not account for possible antenna phase centre
offset errors." A conservative 0.020 m 1-sigma is used to cover this and
multipath from parked vehicles. This is the floor variance; EKF RTK fixed is
not limited by the receiver alone.

    metric_stddev_m = 0.020 m (conservative; datasheet floor 0.0120 m)
    lat_stddev_deg  = 0.020 / 111320 = 1.797e-7 deg  (rounded to 2.0e-7)
    lon_stddev_deg  = 0.020 / 111320 = 1.797e-7 deg  (equatorial approximation)

Config key: `tiers.rtk_fixed.metric_stddev_m = 0.020`

#### 3b. RTK float

No datasheet specification. RTK float is a transitional state between
fixed-integer and standalone; its accuracy depends on atmospheric conditions
and is not guaranteed by the manufacturer.

Industry-standard figure from RTK literature (e.g. Trimble, Leica application
notes): approximately 0.300 m CEP at short baselines under good conditions.

    1-sigma = 0.300 / 0.8326 = 0.360 m
    lat_stddev_deg  = 0.360 / 111320 = 3.233e-6 deg  (rounded to 3.2e-6)
    lon_stddev_deg  = 0.360 / 111320 = 3.233e-6 deg

This is an estimate, not a datasheet value. It is conservative relative to the
fixed tier and represents marginal accuracy for a 2.5 m wide parking bay.

Config key: `tiers.rtk_float.metric_stddev_m = 0.360`

#### 3c. Standalone PVT

Datasheet Table 3: horizontal accuracy = 1.5 m CEP (24 h static, all-constellation).

    1-sigma = 1.5 / 0.8326 = 1.802 m
    lat_stddev_deg  = 1.802 / 111320 = 1.619e-5 deg  (rounded to 1.62e-5)
    lon_stddev_deg  = 1.802 / 111320 = 1.619e-5 deg

At this accuracy level, a 2.5 m bay is only 1.4-sigma wide - parking is unsafe.
The reward function's uncertainty penalty approaches zero and the policy learns
to abort or hold position.

Config key: `tiers.standalone.metric_stddev_m = 1.802`

#### 3d. Degraded (simulation parameter)

No datasheet value. This tier models conditions beyond the receiver's specified
operating range: heavy multipath in a covered car park entrance, jamming, or
severe signal obstruction.

5.0 m 1-sigma is a simulation design choice to saturate the uncertainty penalty
and force the policy to learn hard abort / safety-handoff behaviour. It does not
correspond to any published ZED-F9P-05B specification.

    lat_stddev_deg  = 5.0 / 111320 = 4.492e-5 deg  (rounded to 4.49e-5)
    lon_stddev_deg  = 5.0 / 111320 = 4.492e-5 deg

Config key: `tiers.degraded.metric_stddev_m = 5.0`

### Lat/lon conversion

CARLA GNSS noise is added in degrees of latitude and longitude. The flat-earth
conversion uses 111320 m/deg for both axes (equatorial approximation). CARLA
FlatPlane does not assign a real-world latitude to the map, so a
latitude-dependent cos() correction cannot be applied. At mid-latitudes (~51 deg,
representative of a UK deployment site), the longitude scale factor is
`111320 * cos(51 deg) = 70023 m/deg`, giving a ~37% underestimate of longitude
noise in the injected lat/lon values. However, the EKF consumes the metric
Odometry output of `GnssNoiseRelayNode` (flat-earth XY), not the raw lat/lon,
so the covariance stamped on the Odometry message is in metres and is correct
regardless of the lat/lon degree conversion.

### COG heading

The ZED-F9P-05B in moving-base RTK mode achieves 0.4 deg (50th percentile)
heading accuracy at 30 m/s with a short baseline (Table 6). The simulation
does not use moving-base RTK for heading; instead, Course Over Ground is derived
from successive noisy GNSS position fixes in `GnssNoiseRelayNode._gnss_callback()`.
COG heading variance is computed analytically by propagating position noise:

    var_heading = 2 * sigma_xy^2 / speed^2

where sigma_xy is the current tier's metric_stddev_m and speed is the
GNSS-derived displacement / dt. This is always larger than the datasheet
moving-base figure and is the correct model for COG-based heading.

---

## 4. Cross-references

| File | Role |
|------|------|
| `uncertainty_rl/envs/sim/helpers/_sensor_manager.py` | `_apply_lidar_noise()`, `sample_lidar_noise_bias()` |
| `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/imu_noise_relay.py` | `ImuNoiseRelayNode.__init__()`, `_imu_callback()` |
| `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` | `GnssNoiseRelayNode._gnss_callback()`, `_TIER_DEFAULTS` |
| `configs/deployment/sim/env_config.yaml` | `carla_sensors.lidar.noise.*` |
| `configs/deployment/sim/gnss_noise_profiles.yaml` | GNSS tier noise parameters with derivation comments |
| `configs/deployment/sensor_config.yaml` | Physical sensor specs (docs only) |
| `configs/ros2_config.yaml` | `imu_noise_relay.*` |
| `tests/test_lidar_noise.py` | Unit tests for TiM571 noise model |
