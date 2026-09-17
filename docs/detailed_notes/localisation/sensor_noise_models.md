# sensor_noise_models

Extracted from `uncertainty_rl/envs/sim/helpers/_sensor_manager.py`,
`uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/imu_noise_relay.py`,
`configs/deployment/sim/env_config.yaml`, and `configs/ros2_config.yaml`.

This note records the datasheet figures behind the simulated sensor noise and maps
each mechanism to the code that implements it. The derivations that turn these
figures into the injected parameters - the CEP to sigma conversion, the spectral
density to per-sample variance conversion, the per-tier GNSS values and the EKF
process-noise diagonal - are given in the sensor-noise appendix of the dissertation
(`docs/AntonioGaldes_Dissertation.pdf`), which is canonical for all of them.

## Target hardware

- **IMU**: VectorNav VN-100 Rugged (tactical-grade MEMS AHRS)
- **GNSS**: u-blox ZED-F9P-05B (multi-band RTK module)
- **LiDAR**: SICK TiM571 (obstacle detection only; not in the EKF localisation pipeline)

All noise injection in the simulator is gated by master flags (`enable_gnss_noise`,
`enable_imu_noise`). In real deployment both flags are false and every mechanism
becomes a pass-through.

## 1. SICK TiM571 2D LiDAR (P/N 1075091)

Datasheet [1].

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

Two range terms are modelled in `_apply_lidar_noise()`: a per-episode bias sampled
from the systematic figure, and per-point Gaussian noise from the statistical figure.
Returns below the 0.05 m minimum range are discarded, and CARLA's own per-sensor
noise attribute is zeroed at spawn so the injected model is the only corruption.

## 2. VectorNav VN-100 IMU (Hardware v7.0)

Datasheet [2].

| Spec | Datasheet value |
|------|----------------|
| Gyro range | +-2000 deg/s |
| Gyro noise density | 0.0035 deg/s/sqrt(Hz) |
| Gyro in-run bias stability | 5-7 deg/hr typical, < 10 deg/hr (Allan variance) |
| Gyro bandwidth | 256 Hz |
| Accel range | +-16 g |
| Accel noise density | 0.14 mg/sqrt(Hz) |
| Accel in-run bias stability | < 0.04 mg |
| Accel bandwidth | 260 Hz |
| Output rate | up to 800 Hz (IMU), up to 400 Hz (Attitude) |

The EKF runs at 20 Hz and CARLA publishes sensor data at 20 Hz
(`carla_sensors.imu.sensor_tick = 0.05 s`). Each channel is modelled as
`out = scale * truth + bias + noise`, with the scale factor and in-run bias resampled
per episode and the white noise redrawn per sample. The same variances populate the
outgoing covariance field, which CARLA would otherwise leave at zero. A ZUPT guard
clamps near-stationary rates and skips injection on any clamped channel.

## 3. u-blox ZED-F9P RTK-GNSS

Datasheet [3]. All position accuracy values are from Table 3 (GPS+GLO+GAL+BDS mode).

| Spec | Datasheet value | Source |
|------|----------------|--------|
| Horizontal accuracy (RTK fixed) | 0.010 m CEP + 1 ppm | Table 3 |
| Horizontal accuracy (PVT standalone) | 1.5 m CEP | Table 3 |
| Horizontal accuracy (SBAS) | 1.0 m CEP | Table 3 |
| Moving-base heading accuracy | 0.4 deg (50%) at 30 m/s | Table 6 |
| Velocity accuracy | 0.05 m/s (50%) | Table 1 |
| RTK convergence time | <10 s | Table 2 |
| Max navigation update rate (RTK) | 5 Hz (all-constellation) | Table 2 |

No RTK float accuracy is specified: float is a receiver-internal state whose accuracy
depends on atmospheric conditions, baseline length and satellite geometry, so the
float tier is estimated rather than taken from the datasheet.

The per-tier noise values live in `configs/deployment/sim/gnss_noise_profiles.yaml`,
which is their single source of truth. Beyond the tier magnitudes, the relay applies
per-episode east/north anisotropy, per-callback dropout, tier-mapped `NavSatStatus`,
flat-earth lat/lon projection and COG heading derivation.

## 4. Cross-references

| Source | Role |
|--------|------|
| SICK TiM571 datasheet [1] | Systematic error (+-60 mm), statistical error (<20 mm), angular resolution (0.33 deg), range (0.05-25 m) |
| VectorNav VN-100 datasheet [2] | Gyro noise density (0.0035 deg/s/sqrt(Hz)), accel noise density (0.14 mg/sqrt(Hz)), gyro bias (5-7 deg/hr typ.), accel bias (< 0.04 mg) |
| u-blox ZED-F9P datasheet [3] | RTK fixed CEP (Table 3), standalone CEP (Table 3), moving-base heading (Table 6), convergence time (Table 2) |
| `uncertainty_rl/envs/sim/helpers/_sensor_manager.py` | `_apply_lidar_noise()`, `sample_lidar_noise_bias()` |
| `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/imu_noise_relay.py` | `ImuNoiseRelayNode.__init__()`, `_imu_callback()` |
| `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` | `GnssNoiseRelayNode._gnss_callback()`, `_TIER_DEFAULTS` |
| `configs/deployment/sim/env_config.yaml` | `carla_sensors.lidar.noise.*` |
| `configs/deployment/sim/gnss_noise_profiles.yaml` | GNSS tier noise parameters with derivation comments |
| `configs/deployment/sensor_config.yaml` | Physical sensor specs (docs only) |
| `configs/ros2_config.yaml` | `imu_noise_relay.*` |
| `tests/test_lidar_noise.py` | Unit tests for TiM571 noise model |

## References

[1] SICK AG, *TiM571-2050101 2D LiDAR Sensors: Data Sheet*, Waldkirch, Germany, order
    no. 1075091. [Online]. Available:
    https://www.sick.com/media/pdf/4/44/444/dataSheet_TiM571-2050101_1075091_en.pdf

[2] VectorNav Technologies, LLC, *VN-100 Rugged IMU/AHRS: Sensor Datasheet (Hardware
    v7.0)*, Dallas, TX, USA, 2023, doc. DS100-CR-70. [Online]. Available:
    https://www.vectornav.com/resources/detail/vn-100-imu-ahrs

[3] u-blox AG, *ZED-F9P-04B High Precision GNSS Module: Data Sheet*, Thalwil,
    Switzerland, 2024, doc. UBX-21044850, revision R05. [Online]. Available:
    https://content.u-blox.com/sites/default/files/ZED-F9P-04B_DataSheet_UBX-21044850.pdf
