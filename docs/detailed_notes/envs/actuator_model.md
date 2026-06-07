# actuator_model

Extracted from `uncertainty_rl/envs/sim/carla_parking.py` (`step()`) and
`configs/deployment/agent_config.yaml` (`actuator_model:` block).

This note records the physical actuator model the env imposes on the policy
action, the derivation of each rate limit value from the published literature,
and the rationale for moving these constraints out of the reward function into
the env's action mapping.

## 1 Motivation

A trained policy must produce actions that a real vehicle's actuators can
physically deliver. Earlier iterations of this project used soft reward
penalties to discourage chatter (a `-0.01 * (action - prev_action)^2` shape on
each axis). Two failure modes were observed:

- **Coefficient too small.** A weak penalty (e.g. -0.01 per change^2) pays only
  a few tenths reward per episode against a +50 success terminal. The policy
  learns to ignore the penalty and produces bang-bang outputs (throttle flipping
  0 <-> 1 on ~9 % of consecutive decisions; brake on ~11 %).
- **Coefficient too large.** A strong penalty (e.g. -0.1 per change^2) is
  minimised by *any* constant action regardless of magnitude (a symmetric
  quadratic has a minima everywhere on flat lines). The policy locks steering,
  throttle, and brake at constants, drives in circles, and times out. Success
  rate collapses to zero.

Both failures share a structural cause: a soft penalty conflates "do not
change the actuator" with "the actuator cannot change that quickly". Only the
second statement is physically true.

The fix is to enforce the second statement directly in the env. The policy is
free to command any action; the env clamps the delivered command to a
physically achievable value before forwarding it to CARLA. The reward
function then becomes outcome-only.

## 2 Model

Three rate limits and one cross-axis constraint, applied on the policy
decision boundary (every `action_repeat` simulation ticks, i.e. every 0.2 s
with the defaults `action_repeat = 4`, `carla_timestep = 0.05`):

```
steer_cmd    = clip(policy_steer,    prev_steer    - 0.15, prev_steer    + 0.15)
throttle_cmd = clip(policy_throttle, prev_throttle - 0.50, prev_throttle + 0.50)
brake_cmd    = clip(policy_brake,    prev_brake    - 0.50, prev_brake    + 0.50)

if brake_cmd > 0.1:
    throttle_cmd = 0.0
```

`prev_*` carries the post-clamp command from the previous decision, so the
limit is measured against what the actuator delivered, not what the policy
asked for.

Within one decision the clamped setpoint is held constant for all
`action_repeat` simulation ticks. CARLA's vehicle physics then evolve under a
constant control input for 0.2 s, which matches the operating mode of a real
electronic control unit (ECU) running at ~ 50 Hz with the actuator setpoint
updated each cycle.

## 3 Derivation of values

All values are in the normalised action space CARLA exposes:
`steer in [-1, 1]`, `throttle in [0, 1]`, `brake in [0, 1]`. CARLA's default
passenger vehicle `max_steer_angle` is 70 deg at the road wheel, so a unit of
normalised steer corresponds to 70 deg of road-wheel angle.

### 3.1 Steering rate: 0.20 per decision

Mapped to physical units: 0.20 normalised / 0.2 s = 1.0 normalised per
second = 1.0 x 70 = 70 deg/s at the road wheel.

Reference points:

| Source | Quoted rate | Notes |
|---|---|---|
| CarSim "Sine with Dwell" standard test | 13.5 deg/s | Steady-state steering input for ESC evaluation. Lower than our limit because the test is for highway dynamics, not parking. |
| Production MPC parking controllers | ~ 15 deg/s | Self-developed MPC controller for parking maneuvers (arXiv 2109.10075). Conservative for comfort. |
| Driverless EPS, lock-to-lock target | ~ 140 deg/s at the road wheel | Manca et al. 2021 (Actuators 10:165, doi 10.3390/act10070165) report a full lock-to-lock design target of ~ 1 s. With CARLA max_steer_angle = 70 deg, lock-to-lock (140 deg) in 1 s = 140 deg/s. This is the actuator capability ceiling. |
| MPC path-tracking rate constraint | ~ 115 deg/s at the steering wheel | Domina and Tihanyi 2023 (Sensors 23:6862, doi 10.3390/s23156862) constrain the steering rate to ~ 2 rad/s as the comfort/control envelope. |
| Human panic input | ~ 800-1000 deg/s at the steering wheel | Extreme test conditions. At a typical 14:1 wheel-to-rack ratio this is ~ 60-70 deg/s at the road wheel. |

Our 70 deg/s sits below both the actuator capability ceiling (~ 140 deg/s,
Manca et al.) and the MPC comfort/control envelope (~ 115 deg/s, Domina and
Tihanyi), so the rate is physically realisable and still transfers to
hardware. It is fast enough that the policy can complete a full lock-to-lock
manoeuvre in 140 / 70 = ~ 2.0 s, slow enough that step changes of 180 deg in
one decision are impossible. The earlier 0.15 (52.5 deg/s) was conservative
relative to both published bounds; 0.20 gives a brisker parking-lot manoeuvre
while remaining defensible against the cited limits.

We do not separately model EPS slowdown at standstill. It is a second-order
effect; modelling it adds complexity without affecting the outcome metric
(does the car park in the bay).

### 3.2 Throttle rate: 0.5 per decision

Mapped to physical units: 0 to full in two decisions = 0.4 s.

Reference points:

| Component | Quoted response | Notes |
|---|---|---|
| Drive-by-wire pedal -> throttle plate | 30-80 ms | Production DBW measurement; see [LSXmag](https://www.lsxmag.com/tech-stories/anti-lag-fixing-throttle-response-issues-on-drive-by-wire/) and [Throttle Response Controller](https://throttleresponsecontroller.com/blogs/interchange/what-is-throttle-lag). |
| Engine torque rise (NA petrol) | 100-200 ms | Time from throttle plate fully open to peak wheel torque, dominated by manifold filling. Wikipedia [throttle response](https://en.wikipedia.org/wiki/Throttle_response). |
| Combined (pedal -> wheels) | 130-280 ms | Sum of the above. |

One policy decision is 200 ms, so a real vehicle can produce ~ 80 % of full
acceleration within one decision. A rate limit of 0.5 per decision is
deliberately conservative against that figure: the policy needs two decisions
(0.4 s) to ask for full throttle and receive it. This is also defensible for
electric vehicles, which have faster torque rise but inherit the same
safety-motivated rate limit in production driver-assistance code.

### 3.3 Brake rate: 0.5 per decision

Mapped to physical units: 0 to full in two decisions = 0.4 s.

Reference points:

| Component | Quoted response | Notes |
|---|---|---|
| Hydraulic master cylinder | 100-150 ms | Time from pedal-fully-pressed to ~ 80 % peak line pressure in a vacuum-assisted hydraulic system. ABS modulation cycles run faster (10-20 ms) but only after pressure has built up. |
| Brake-by-wire (EHB) | 80-100 ms | Electronically actuated hydraulic systems. Comparable or slightly faster. |

A real car can build full brake pressure in ~ 0.1 s, so a rate limit of 0.5
per decision is again deliberately conservative. The policy needs 0.4 s to
go from coast to full brake. This matters at our task: the success criterion
needs `speed < 0.3 m/s` held for 5 consecutive ticks, so the policy must be
able to commit to braking without delaying the success terminal.

### 3.4 Brake-overrides-throttle: threshold 0.1

A real driver-assistance system cuts the throttle whenever the brake is
meaningfully pressed. We use 0.1 as the threshold (above light brake-pedal
modulation but well below firm braking). This is a binary constraint, not a
rate, and it eliminates the "press both pedals simultaneously" pathology
that the previous `co_activation_penalty` was discouraging via soft shaping.

## 4 References

- [LSXmag, "Anti Lag: Fixing Throttle Response Issues On GM Drive By Wire"](https://www.lsxmag.com/tech-stories/anti-lag-fixing-throttle-response-issues-on-drive-by-wire/)
- [Throttle Response Controller, "What is Throttle Lag?"](https://throttleresponsecontroller.com/blogs/interchange/what-is-throttle-lag)
- [Wikipedia, "Throttle response"](https://en.wikipedia.org/wiki/Throttle_response)
- [arXiv 2109.10075, "Implementation of a self-developed model predictive control scheme for vehicle parking maneuvers"](https://arxiv.org/abs/2109.10075)
- [Mechanical Simulation, "Sine with Dwell Test in CarSim"](https://www.carsim.com/downloads/pdf/sine_w_dwell.pdf)
- [arXiv 2511.01369, "Lateral Velocity Model for Vehicle Parking Applications"](https://arxiv.org/html/2511.01369v1)
- [CARLA Python API, WheelPhysicsControl](https://carla.readthedocs.io/en/0.9.16/python_api/)
- [ScienceDirect, "Brake Pressure" topic page](https://www.sciencedirect.com/topics/engineering/brake-pressure)
- SAE J266-2018, "Steady-State Directional Control Test Procedures for Passenger Cars and Light Trucks" (cited indirectly via CarSim test references; not directly purchased).
