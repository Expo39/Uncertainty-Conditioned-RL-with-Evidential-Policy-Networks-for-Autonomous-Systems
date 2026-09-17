# actuator_model

Extracted from `uncertainty_rl/envs/sim/carla_parking.py` (`step()`) and
`configs/deployment/agent_config.yaml` (`actuator_model:` block).

Section 3.3 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the actuator model: the clipping equation, the adopted rate limits, the brake-priority
threshold, and the comparison of the steering rate against the EPS capability ceiling
and the MPC comfort envelope.

This note records two things the dissertation does not: why the constraint is enforced
in the env rather than the reward, and the wider set of measurements the chosen rates
were checked against.

## Why the env and not the reward

A soft penalty on action change (a `-k * (action - prev_action)^2` term per axis) is the
wrong tool, because it conflates "do not change the actuator" with "the actuator cannot
change that quickly". Only the second is physically true, and neither coefficient regime
works:

- A weak penalty costs a few tenths of reward per episode against a large success
  terminal, so the policy ignores it and produces bang-bang outputs.
- A strong symmetric quadratic is minimised by *any* constant action regardless of
  magnitude, so the policy locks the actuators at constants, drives in circles and
  times out.

Enforcing the rate limit in the action mapping leaves the policy free to command
anything while the delivered command stays physically achievable, which keeps the
reward function outcome-only.

The clamped setpoint is held for all `action_repeat` simulation ticks, so CARLA's
physics evolve under a constant control input for one decision interval. That matches a
real ECU running at a fixed cycle with the actuator setpoint updated each cycle.

`prev_*` carries the post-clamp command, so the limit is measured against what the
actuator delivered rather than what the policy asked for.

## Reference measurements

The operative values come from `actuator_model:` in
`configs/deployment/agent_config.yaml`. The `.get()` fallbacks in `carla_parking.py`
differ and apply only if that block is absent.

Steering, against the adopted ~70 deg/s at the road wheel:

| Source | Quoted rate | Role |
|---|---|---|
| Driverless EPS, lock-to-lock target [1] | ~140 deg/s | Actuator capability ceiling |
| MPC path-tracking rate constraint [2] | ~115 deg/s | Comfort and control envelope |

The adopted rate sits below both, so it is physically realisable and remains within the
comfort envelope while still permitting a full lock-to-lock manoeuvre in about two
seconds.

EPS slowdown at standstill is not modelled separately: it is a second-order effect that
adds complexity without affecting whether the car parks in the bay.

Pedals, against the adopted 0.4 s to full application. Production longitudinal actuation
responds well inside that interval, the powertrain delay measured for electric-vehicle
speed tracking [3] and the pressure build-up of an integrated electro-hydraulic
brake [4] both falling below 200 ms. Both pedal limits are therefore deliberately
conservative against the hardware.

The brake figure matters at this task because success requires the speed to stay below
`SUCCESS_THRESHOLD_VELOCITY` for `SUCCESS_DWELL_STEPS` consecutive decisions, so the
policy must be able to commit to braking without delaying the success terminal.

## References

[1] R. Manca, S. Circosta, I. Khan, S. Feraco, S. Luciani, N. Amati, A. Bonfitto, and
    R. Galluzzi, "Performance assessment of an electric power steering system for
    driverless Formula Student vehicles," *Actuators*, vol. 10, no. 7, p. 165, 2021,
    doi: [10.3390/act10070165](https://doi.org/10.3390/act10070165).

[2] A. Domina and V. Tihanyi, "Model predictive controller approach for automated
    vehicle's path tracking," *Sensors*, vol. 23, no. 15, p. 6862, 2023,
    doi: [10.3390/s23156862](https://doi.org/10.3390/s23156862).

[3] J. Lee and K. Jo, "Model predictive control with powertrain delay consideration for
    longitudinal speed tracking of autonomous electric vehicles," *World Electric
    Vehicle Journal*, vol. 15, no. 10, p. 433, 2024. [Online]. Available:
    https://www.mdpi.com/2032-6653/15/10/433

[4] Z. Yu, S. Xu, L. Xiong, and W. Han, "An integrated-electro-hydraulic brake system
    for active safety," SAE Technical Paper 2016-01-1640, 2016,
    doi: [10.4271/2016-01-1640](https://doi.org/10.4271/2016-01-1640).
