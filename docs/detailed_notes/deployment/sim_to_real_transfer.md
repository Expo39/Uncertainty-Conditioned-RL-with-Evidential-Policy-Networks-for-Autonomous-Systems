# Sim-to-Real Transfer - Known Gaps and Mitigations

Extracted from the real-world deployment pipeline (`envs/real/inference_loop.py`). This document records the sim-to-real transfer analysis for the
uncertainty-conditioned parking system. It covers what has been implemented,
what has been deliberately left as a known limitation, and what must be
completed before real-vehicle testing.

---

## Summary of Changes Implemented (8 April 2026)

| Item | Change | File(s) |
|------|--------|---------|
| Mid-episode RTK transitions | Markov chain in GnssNoiseRelayNode | `gnss_noise_relay.py`, `ros2_config.yaml` |
| IMU process noise | Realistic VN-100 values in EKF config | `ros2_config.yaml` |
| LiDAR spec | Matched to real TiM571 (15 Hz, 12,150 pts/sec) | `env_config.yaml` |
| Sensor TF mounts | `real_world_mounts` section + `real_world_deployment` flag | `env_config.yaml`, `carla_bridge.launch.py` |
| Action repeat | Set to 1 (20 Hz, matching real control loop) | `env_config.yaml` |
| EKF datum calibration | Datum-file path for real-vehicle frame offset | `carla_parking.py`, `real_world_datum.yaml` |
| Actuation calibration | Identity-by-default layer with YAML override | `actuation_calibration.py`, `actuation_calibration.yaml` |

---

## Pre-Deployment Checklist

Before the first real-vehicle test, complete all items marked TODO:

### Sensor mounts (configs/deployment/sim/env_config.yaml -> real_world_mounts)
- [ ] Measure IMU position (rear passenger seat floor) from rear axle: x, y, z
- [ ] Measure GNSS antenna position (roof centre) from rear axle: x, y, z
- [ ] Measure LiDAR mount position (front bumper) from rear axle: x, y, z
- [ ] Set `real_world_deployment: true` in env_config.yaml

### Datum survey (configs/real_world_datum.yaml)
- [ ] Choose a clearly marked reference point in the test lot
- [ ] Drive the vehicle to the point with RTK fixed
- [ ] Record UTM easting/northing from u-blox u-center (or `ros2 topic echo /gnss/fix`)
- [ ] Record vehicle heading (from EKF yaw after convergence)
- [ ] Measure the reference point's position in the lot layout frame (from the PNG/YAML)
- [ ] Fill in `utm_easting`, `utm_northing`, `heading_deg`, `lot_x`, `lot_y`

### Actuation calibration (configs/actuation_calibration.yaml)
- [ ] Perform a calibration run: apply known policy outputs, measure vehicle response
- [ ] Fit gain/deadband/bias for steering and longitudinal actuators
- [ ] Fill in `configs/actuation_calibration.yaml`
- [ ] Document the calibration procedure in `documentation/design/actuation_calibration.md`

### EKF pipeline verification
- [ ] Drive manually in the test lot with the full ROS stack running
- [ ] Confirm `/odometry/filtered` covariance varies with sky visibility
      (open area = low covariance, near a wall = higher covariance)
- [ ] Confirm the odom frame does not drift significantly over a 60 s manual drive

### Target bay verification
- [ ] Confirm `dx/dy/dyaw` (obs indices 12-14) point toward the correct bay
      during a manual drive
- [ ] Check the lot layout YAML matches the physical bay positions to within 0.2 m

### Policy in the loop
- [ ] Only close the loop with the policy after all items above are verified

---

## Known Limitations (Dissertation Chapter 5)

These are acknowledged gaps that do not invalidate the thesis contribution
but should be discussed in the limitations section.

### #9 -- Isotropic GNSS noise model

**What the sim does:** Stamps a scalar diagonal covariance on NavSatFix.
All three axes (lat, lon, alt) have equal uncertainty, equal to the tier
metric stddev.

**Real behaviour:** RTK accuracy is anisotropic. It depends on satellite
geometry (DOP), baseline length to the reference station, and multipath from
nearby structures. The error ellipse is spatially correlated and changes as
the vehicle moves.

**Why it does not invalidate the thesis:** The policy observes EKF covariance
(not raw GNSS accuracy). The EKF covariance magnitude is driven by the
NavSatFix covariance -- so as long as the magnitude varies correctly across
tiers, the policy receives the right signal. The anisotropy affects the
covariance *shape* (off-diagonal terms) but not the diagonal variance which
dominates the policy's input. Indices 3-5 are diagonal stddevs; indices 9-11
are off-diagonal -- the policy will see near-zero off-diagonal terms in both
sim and (for most conditions) reality, since open-sky RTK errors are nearly
isotropic.

**Where it might matter:** In heavily obstructed environments with strong
multipath, the real error ellipse can be significantly elongated. This is
unlikely in an open parking lot but should be mentioned as a limitation.

### #10 -- No IMU gyro bias drift

**What the sim does:** CARLA IMU has no bias walk. The process noise fix
(ros2_config.yaml) adds realistic per-step noise, but the noise is zero-mean
with no cumulative drift.

**Real behaviour:** MEMS gyroscopes have a bias instability (typically
3-10 deg/hr for consumer-grade IMUs). Over a 45 s parking manoeuvre this is
3-10 deg/hr * (45/3600) hr = 0.04-0.12 deg of accumulated heading error.
At 1.5 m from the vehicle centre to a bay edge, 0.1 deg heading error
introduces ~2.6 mm lateral position error -- negligible for a 2.5 m bay.

**Conclusion:** For short-duration parking manoeuvres, IMU bias drift is
below the success threshold (0.5 m position, 10 deg orientation) and can be
safely ignored.

### #11 -- No surface variation (FlatPlane world)

**What the sim does:** FlatPlane is a perfectly flat ground plane with no
surface features. LiDAR returns are from static props (cones, parked cars)
and moving NPCs only.

**Real behaviour:** Real parking lots have road markings, drain covers, speed
bumps, and slight inclines (drainage gradient ~1-2%). These affect:
- LiDAR: minor spurious returns from drain covers and surface markings
  (typically <0.05 m height, below bumper-height scan plane)
- EKF: slight pitch/roll from surface gradient (suppressed by two_d_mode=true)

**Conclusion:** The bumper-height LiDAR mount (z=0.5 m) is above typical
surface features. two_d_mode=true in the EKF suppresses pitch/roll. Surface
variation is not expected to materially affect policy behaviour. Mention as a
limitation but do not spend time fixing it.

---

## Deployment Sequence

1. Verify EKF pipeline in isolation (manual drive, check covariance)
2. Verify target bay pose pipeline (manual drive, check dx/dy/dyaw)
3. Verify actuation calibration (manual drive with known inputs)
4. First closed-loop test in a single bay, RTK fixed, low NPC count
5. Progressive testing across tiers and conditions
