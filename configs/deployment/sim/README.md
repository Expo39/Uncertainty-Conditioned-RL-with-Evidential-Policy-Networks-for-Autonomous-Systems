# configs/deployment/sim/

CARLA simulation settings, sensor definitions, and pre-built assets. All files
here are specific to the CARLA simulator and the FlatPlane parking lot world.

## Files

| File | Purpose |
|------|---------|
| `env_config.yaml` | CARLA connection, simulation timing, sensor specs, parking lot scenarios |
| `gnss_noise_profiles.yaml` | RTK fix-state tier definitions and per-episode sampling weights |
| `OpenDriveMap.bin` | Pre-built pedestrian nav mesh for the FlatPlane world |

---

## `env_config.yaml`

Loaded by `CARLAParkingEnv`, `lot_inspector.py`, `demo_drive.py`, and
`carla_bridge.launch.py`. Pass as `--env-config configs/deployment/sim/env_config.yaml`.
Do not mix RL training hyperparameters here - those belong in
[`configs/train_config.yaml`](../../train_config.yaml).

**Config loading chain** (`train_ppo.py`):

```
deployment/sensor_config.yaml   (physical sensor specs)
         +
deployment/agent_config.yaml    (observation flags, safety params)
         +
deployment/sim/env_config.yaml  (CARLA-specific, wins on conflict)
         =
merged env config passed to CARLAParkingEnv
```

### Key sections

The file is organised into these blocks. Read the YAML directly for the live
values; they change as the project iterates and stage-specific overrides
come and go.

| Block | What it controls |
|-------|------------------|
| `carla_host`, `carla_port`, `town` | CARLA connection |
| `carla_timestep`, `max_steps`, `action_repeat` | Simulation timing |
| `no_rendering_mode`, `map_load_sleep` | CARLA runtime behaviour |
| `carla_sensors.imu`, `carla_sensors.gnss`, `carla_sensors.lidar` | Sensor specs (noise injected by relay nodes for IMU and GNSS) |
| `parking_scenarios.fixed_*` | When set, force a named floor plan / bay / GNSS tier every episode (curriculum overrides). Comment out for random sampling. |
| `parking_scenarios.bay_occupancy_*` | Per-episode parked-vehicle density |
| `parking_scenarios.num_patrol_vehicles_max`, `patrol_*` | NPC patrol vehicles |
| `parking_scenarios.pedestrian_*` | NPC pedestrians |
| `parking_scenarios.floor_plans` | Layout-file paths and OOD flags |

Real LiDAR-noise parameters under `carla_sensors.lidar.noise` are documented
in [`docs/detailed_notes/localisation/sensor_noise_models.md`](../../../docs/detailed_notes/localisation/sensor_noise_models.md).

---

## `gnss_noise_profiles.yaml`

Defines the RTK fix-state tiers sampled per episode to vary GNSS noise and
drive EKF covariance variation - the primary uncertainty source in training.

The tiers (e.g. `rtk_fixed`, `rtk_float`, `standalone`, `degraded`), their
position-stddev values, and per-tier sampling weights live in the YAML
itself. The sampling weights change as the curriculum progresses, so read
the live file rather than relying on a snapshot in this README.

At eval time, `gnss_noise_multiplier` in `configs/eval_config.yaml` overrides
the per-episode sampling to fix a specific noise level for each evaluation
condition. The curriculum sets `parking_scenarios.fixed_gnss_tier: rtk_fixed` in
every stage as the per-episode START tier; mid-episode Markov drift (always on) then
wanders from there, scaled by the per-stage `parking_scenarios.drift_scale` in [0,1]
(0 = the start tier holds, 1 = the full realistic chain).

The file also defines a `transition_matrix` block: a per-step Markov chain
over the tiers used by `GnssNoiseRelayNode` when
`gnss_noise_relay.enable_markov_transitions: true` in
[`configs/ros2_config.yaml`](../../ros2_config.yaml). Diagnose the chain
(stationary distribution, mean dwell, time to first contiguous good window)
with `make analyse-markov`.

---

## `OpenDriveMap.bin`

Recast/Detour pedestrian navigation mesh for the FlatPlane OpenDRIVE world.
Pre-built and committed because CARLA segfaults when building the nav mesh
on headless GPU setups (known issue upstream). The ros2-bridge container
copies this file into CARLA's Nav directory before loading the FlatPlane
world.

**To regenerate** (requires a headed CARLA session, not headless):

1. Start CARLA with a display: `-screen` flag (not `-RenderOffScreen`).
2. Load the FlatPlane world via `generate_opendrive_world()`.
3. Copy the resulting `.bin` from `CarlaUE4/Content/Carla/Maps/Nav/OpenDriveMap.bin`.
4. Replace this file and commit.

---

## See also

- [configs/README.md](../../README.md) - full configs directory map and loading chain
- [configs/deployment/README.md](../README.md) - shared sensor and agent configs
- [uncertainty_rl/envs/README.md](../../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes these files
- [docs/detailed_notes/localisation/sensor_noise_models.md](../../../docs/detailed_notes/localisation/sensor_noise_models.md) - LiDAR and IMU noise derivation
