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

`env_config.yaml` sits third in the merge, so it wins over the sensor and agent configs
and is in turn overridden by the selected curriculum stage. The full precedence chain is
documented in [configs/README.md](../../README.md#how-configs-are-loaded).

### Key sections

The file is organised into these blocks. Read the YAML directly for the live
values, which change as the project iterates as stage-specific overrides
come and go.

| Block | What it controls |
|-------|------------------|
| `carla_host`, `carla_port`, `town` | CARLA connection |
| `carla_timestep`, `max_steps`, `action_repeat` | Simulation timing |
| `no_rendering_mode`, `map_load_sleep` | CARLA runtime behaviour |
| `carla_sensors.imu`, `carla_sensors.gnss`, `carla_sensors.lidar` | CARLA spawn keys. Range, channels and field of view come from `sensor_config.yaml`, and IMU and GNSS noise is injected by the relay nodes |
| `parking_scenarios.num_patrol_vehicles_max`, `patrol_*` | NPC patrol vehicles |
| `parking_scenarios.pedestrian_*` | NPC pedestrians |
| `parking_scenarios.floor_plans` | Layout-file paths and OOD flags |
| `gnss_datum_lat`, `gnss_datum_lon` | Flat-earth projection datum for the GNSS relay |
| `ros2.*`, `ros2.carla_recovery` | Shared file paths and the CARLA reconnection policy |
| `inspect.dryrun_action`, `debug` | Inspector dry-run command and per-step debug logging |

The per-stage difficulty knobs (`fixed_floor_plan`, `fixed_target_bay_id`,
`allowed_bay_ids`, `bay_occupancy_min/max` and the top-level `bay_margin`) are **not**
in this file. They are owned by the curriculum stage files.

Real LiDAR-noise parameters under `carla_sensors.lidar.noise` are documented
in [`docs/detailed_notes/localisation/sensor_noise_models.md`](../../../docs/detailed_notes/localisation/sensor_noise_models.md).

---

## `gnss_noise_profiles.yaml`

Defines the RTK fix-state tiers sampled per episode to vary GNSS noise and
drive EKF covariance variation - the primary uncertainty source in training.

The four tiers (`rtk_fixed`, `rtk_float`, `standalone`, `degraded`), their
position-stddev values and their sampling weights live in the YAML itself. Read the live
file rather than relying on a snapshot in this README.

The start tier of each episode is drawn from those weights. `fixed_gnss_tier` is
deliberately omitted from every curriculum stage, so no stage pins the start tier, and a
CI test enforces that omission. The always-on Markov chain then wanders from the sampled
start. The chain is stage-invariant, so the GNSS degradation process is identical in
every stage rather than being a ramped curriculum axis.

At eval time, `held_gnss_tier` in `configs/eval_config.yaml` overrides the sampling to
hold one named tier for the whole episode, and `degrade_one_way` instead forces a
one-way drift that never recovers. The two are alternatives, never combined.

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
