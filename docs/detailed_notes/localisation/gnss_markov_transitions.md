# GNSS Fix-State Markov Chain - Implementation Notes

Extracted from `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py`.

Section 3.4.1 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical
for the chain: why fix quality is modelled as a Markov process rather than a per-episode
constant, the ladder-only transition structure, the upward recovery bias, the long-run
occupancy and dwell times, the one-way drift mode used for evaluation, and the local
datum. Figure 3.8 there draws the chain with its probabilities.

This note records only the runtime behaviour around it.

## Configuration and rates

The transition matrix, tier definitions and per-episode start weights live in
`configs/deployment/sim/gnss_noise_profiles.yaml`, which is their single source of
truth. The chain is loaded once at node startup and is identical in every curriculum
stage, so GNSS degradation is not a ramped axis, and mid-episode drift is active from
stage 1 with the master switches in `ros2_config.yaml` enabled globally.

`_step_markov()` runs on every GNSS callback at 20 Hz, so with `action_repeat=4` the
chain takes four steps per agent decision.

Each episode starts in a tier sampled from the per-tier `weight` values, biased toward
the degraded tiers (0.40/0.20/0.20/0.20 over fixed/float/standalone/degraded) so that
most episodes begin under uncertainty, which is the realistic arrival case. Because the
chain recovers up the ladder, the agent meets the full covariance range mid-manoeuvre
while the fix trends clean over the approach. The start distribution is therefore
deliberately not the chain's stationary distribution.

## EKF process noise

CARLA publishes zero IMU covariance, which makes `robot_localization` treat the IMU as
infinitely reliable: the EKF covariance never grows during the prediction step and stays
artificially flat between fixes. On real hardware, yaw covariance grows visibly between
20 Hz GNSS updates.

`process_noise_covariance` in `configs/ros2_config.yaml` corrects this. The active
diagonal entries are 1.0e-2 on x and y, 1.0e-4 on yaw and the two planar velocities, and
1.0e-3 on yaw rate, with 1.0e-6 as a positive-definite floor on the states the planar
mode does not estimate. The relative sizing is derived in the sensor-noise appendix.

These values make the covariance inflate between fixes and collapse when one arrives,
producing the sawtooth profile the policy learns to respond to.

## Control flag

`enable_markov_transitions: false` in `configs/ros2_config.yaml` (or the node parameter)
disables the chain, holding the start tier for the whole episode. This is used for
ablations needing a fixed tier, for isolating tier transitions when debugging, and for
evaluation runs at a specified condition.

## Limitations

- The transition matrix is hand-tuned. A higher-fidelity model could learn it from a
  logged RTK dataset at the deployment site.
- Off-diagonal covariance terms from correlated satellite-geometry errors are not
  modelled: the NavSatFix covariance is always diagonal.
- Weather effects such as ionospheric delay are not modelled, as FlatPlane does not
  render weather.

## Where This Is Implemented

| File | Role |
|------|------|
| `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` | `_TIER_ORDER`, `_TIER_DEFAULTS` and `_DEFAULT_TRANSITION_MATRIX` constants; `_apply_tier()` and `_step_markov()`, the latter called from `_gnss_callback()` each tick |
| `configs/deployment/sim/gnss_noise_profiles.yaml` | Tier definitions, start weights and the transition matrix |
| `configs/ros2_config.yaml` | `gnss_noise_relay.enable_markov_transitions` flag; `ekf.process_noise_covariance` |
| `uncertainty_rl/ros2/launch/carla_bridge.launch.py` | Passes `enable_markov_transitions` to the node |
