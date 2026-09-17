# Sim-to-Real Transfer - Known Gaps and Mitigations

Extracted from the real-world deployment pipeline (`envs/real/inference_loop.py`).

The deployment path, code-path parity, frame and actuation calibration and the
pre-deployment sequence are given in Appendix B of the dissertation
(`docs/AntonioGaldes_Dissertation.pdf`); the scope limitation that no result was
obtained on hardware is Section 4.8.3; the five secondary sensor effects that ARE
modelled, including the per-episode east/north anisotropy and why it must reach the
covariance features rather than the position alone, are Section 3.4.

This note records only what those do not: two modelling gaps left in the sensor model,
and why each is tolerable at this task's geometry.

## No within-episode IMU gyro bias drift

Appendix A models in-run bias as a fixed per-episode offset, redrawn at each boundary.
What is not modelled is drift *within* an episode: a real bias walks over time, whereas
the simulated one is constant once drawn.

A real MEMS gyroscope has a bias instability of roughly 3-10 deg/hr for consumer-grade
parts. Over a 45 s parking manoeuvre that accumulates

    3-10 deg/hr * (45/3600) hr = 0.04-0.12 deg

of heading error. At 1.5 m from the vehicle centre to a bay edge, 0.1 deg of heading
error displaces the corner by roughly 2.6 mm, against a 2.5 m bay.

Success is decided geometrically, by the ego bounding box fitting inside the bay
polygon (`car_fully_inside_bay()`), so a millimetre-scale displacement cannot change
the outcome. Modelling the walk would add state to the relay for an effect three orders
of magnitude below the acceptance tolerance, so it is deliberately omitted. The
argument holds only for short manoeuvres: over a multi-minute run the same drift would
need modelling.

## No surface variation

The world is a flat OpenDRIVE surface (Section 3.2), so the ground contributes no LiDAR
returns and no attitude disturbance. Real lots have road markings, drain covers, speed
bumps and a drainage gradient of roughly 1-2 %.

Two consequences, both bounded by existing design choices. Surface features are
typically under 0.05 m in height, below the bumper-height scan plane, so they fall
outside the LiDAR's field of view rather than being filtered out of it. Gradient-induced
pitch and roll are suppressed by `two_d_mode` in the EKF, which does not estimate those
states at all.

Neither is expected to affect policy behaviour materially, and both would be
re-examined on a site with a pronounced gradient.
