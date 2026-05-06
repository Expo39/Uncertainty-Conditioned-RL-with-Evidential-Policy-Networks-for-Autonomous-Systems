# configs/deployment/sim/

CARLA simulation settings, sensor definitions, and pre-built assets. All files here are
specific to the CARLA simulator and the FlatPlane parking lot world.

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
Do not mix RL training hyperparameters here - those belong in `configs/train_config.yaml`.

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

### Key groups

**Connection**

| Key | Default | Notes |
|-----|---------|-------|
| `carla_host` | `"uncertainty-rl-carla-0"` | Container name for worker 0. Use `"localhost"` for local dev. |
| `carla_port` | `2000` | |
| `town` | `"FlatPlane"` | Custom OpenDRIVE world loaded via `generate_opendrive_world()`. |

**Simulation timing**

| Key | Default | Notes |
|-----|---------|-------|
| `carla_timestep` | `0.05` | 20 Hz. Policy runs at 1/carla_timestep. |
| `max_steps` | `1500` | Episode length. 1500 x 0.05 s = 75 s sim time. |
| `action_repeat` | `1` | Steps per policy call. |
| `no_rendering_mode` | `true` | Disables Unreal rendering (~3-4x speed-up). Cameras are empty; physics is active. |

**Sensors** - noise values match real hardware; see `docs/detailed_notes/sensor_noise_models.md`

| Sensor | Key prefix | Notes |
|--------|-----------|-------|
| IMU | `carla_sensors.imu` | Noise injected by `ImuNoiseRelayNode`, not in CARLA directly. |
| RTK-GNSS | `carla_sensors.gnss` | Noise injected by `GnssNoiseRelayNode` from the per-episode tier (see below). |
| 2D LiDAR | `carla_sensors.lidar` | SICK TiM571 spec: 15 Hz, 25 m range. Obstacle detection only - not used for localisation. |

**Parking lot scenarios**

| Key | Default | Notes |
|-----|---------|-------|
| `parking_scenarios.bay_occupancy_min` | `0.3` | Lower bound for per-episode bay occupancy. |
| `parking_scenarios.bay_occupancy_max` | `0.8` | Upper bound. |
| `parking_scenarios.num_patrol_vehicles_max` | `1` | NPC patrol vehicles per episode. |
| `parking_scenarios.floor_plans` | rectangle, trapezoid, irregular_a | Layout files and OOD flags. `irregular_a` is held out for OOD evaluation. |

---

## `gnss_noise_profiles.yaml`

Defines the RTK fix-state tiers sampled per episode to vary GNSS noise and drive EKF
covariance variation - the primary uncertainty source in training.

| Tier | Approx. stddev | Sampling weight | Interpretation |
|------|---------------|-----------------|----------------|
| `rtk_fixed` | ~2 cm | 0.4 (40%) | Nominal RTK fix - parking is straightforward |
| `rtk_float` | ~36 cm | 0.3 (30%) | Marginal RTK - elevated covariance |
| `standalone` | ~1.8 m | 0.2 (20%) | RTK lost - high covariance, policy must adapt |
| `degraded` | ~5 m | 0.1 (10%) | Severe degradation - policy should be cautious |

At eval time, `gnss_noise_multiplier` in `configs/eval_config.yaml` overrides the per-episode
sampling to fix a specific noise level for each evaluation condition.

---

## `OpenDriveMap.bin`

Recast/Detour pedestrian navigation mesh for the FlatPlane OpenDRIVE world. Pre-built and
committed because CARLA segfaults when building the nav mesh on headless GPU setups (known
issue upstream). The ros2-bridge container copies this file into CARLA's Nav directory before
loading the FlatPlane world.

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
- [docs/detailed_notes/sensor_noise_models.md](../../../docs/detailed_notes/sensor_noise_models.md) - LiDAR and IMU noise derivation
