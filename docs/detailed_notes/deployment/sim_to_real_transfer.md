# Sim-to-Real Transfer - Known Gaps and Mitigations

Extracted from the real-world deployment pipeline (`envs/real/inference_loop.py`). This note records the deliberate sim-to-real gaps and why each is acceptable. The pre-deployment checklist and calibration procedure live in [`real_world_deployment.md`](real_world_deployment.md).

---

## Known Limitations

These are acknowledged gaps that do not invalidate the thesis contribution
but should be discussed in the limitations section.

### Isotropic GNSS noise model

**What the sim does:** Stamps a scalar diagonal covariance on NavSatFix.
All three axes (lat, lon, alt) have equal uncertainty, equal to the tier
metric stddev.

**Real behaviour:** RTK accuracy is anisotropic. It depends on satellite
geometry (DOP), baseline length to the reference station, and multipath from
nearby structures. The error ellipse is spatially correlated and changes as
the vehicle moves.

**Why it does not invalidate the thesis:** The policy observes EKF covariance
(not raw GNSS accuracy). The EKF covariance magnitude is driven by the
NavSatFix covariance, so as long as the magnitude varies correctly across
tiers, the policy receives the right signal. The observation carries only the
diagonal standard deviations (std_x, std_y, std_yaw); off-diagonal correlation
is not in the observation, so the anisotropy of the real error ellipse does
not change the policy's input, which is near-isotropic for open-sky RTK.

**Where it might matter:** In heavily obstructed environments with strong
multipath, the real error ellipse can be significantly elongated. This is
unlikely in an open parking lot but should be mentioned as a limitation.

### No IMU gyro bias drift

**What the sim does:** CARLA IMU has no bias walk. The process noise fix
(ros2_config.yaml) adds realistic per-step noise, but the noise is zero-mean
with no cumulative drift.

**Real behaviour:** MEMS gyroscopes have a bias instability (typically
3-10 deg/hr for consumer-grade IMUs). Over a 45 s parking manoeuvre this is
3-10 deg/hr * (45/3600) hr = 0.04-0.12 deg of accumulated heading error.
At 1.5 m from the vehicle centre to a bay edge, 0.1 deg heading error
introduces ~2.6 mm lateral position error - negligible for a 2.5 m bay.

**Conclusion:** For short-duration parking manoeuvres, IMU bias drift is far
below the geometric acceptance tolerance (the ego bounding box must fit inside
the bay polygon, `car_fully_inside_bay()`) and can be safely ignored.

### No surface variation (FlatPlane world)

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
