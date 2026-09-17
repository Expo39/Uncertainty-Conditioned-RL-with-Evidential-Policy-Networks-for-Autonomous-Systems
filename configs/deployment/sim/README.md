# configs/deployment/sim/

CARLA simulation settings, sensor definitions, and pre-built assets. All files
here are specific to the CARLA simulator and the FlatPlane parking lot world.

## Files

| File | Purpose |
|------|---------|
| `env_config.yaml` | CARLA connection, simulation timing, sensor spawn keys, parking lot scenarios |
| `gnss_noise_profiles.yaml` | RTK fix-state tier definitions, start weights and the transition matrix |
| `OpenDriveMap.bin` | Pre-built pedestrian nav mesh for the FlatPlane world (not currently wired in) |
| `curriculum/` | Per-stage difficulty and schedule overrides - see [`curriculum/README.md`](curriculum/README.md) |

---

## `env_config.yaml`

Loaded through `load_env_config()` by `train_ppo.py`, `tune_hyperparams.py`,
`evaluate.py`, `demo_drive.py`, `lot_inspector.py` and `generate_layouts.py`, and read
directly by `carla_bridge.launch.py` for the sensor block. The training, tuning, eval and
demo entry points accept
`--env-config configs/deployment/sim/env_config.yaml`; the inspector and the layout
generator hard-code that same path. Do not mix RL training hyperparameters here - those
belong in [`configs/train_config.yaml`](../../train_config.yaml).

`env_config.yaml` sits third in the one precedence chain
(`sensor_config < agent_config < env_config (+ stage) < train_config (+ baseline)`), so it
wins over the sensor and agent configs. `load_env_config()` deep-merges the selected
curriculum stage onto it first, so a stage key wins over the base value here. The full
chain is documented in [configs/README.md](../../README.md#how-configs-are-loaded).

### Key sections

The file is organised into the blocks below. Read the YAML for the live values - it is
commented with units and carries the rationale for each setting.

| Block | What it controls |
|-------|------------------|
| `carla_host`, `carla_port`, `town` | CARLA connection. `town: FlatPlane` is the generated OpenDRIVE world |
| `carla_timestep`, `max_steps`, `action_repeat` | Simulation timing. `max_steps` counts sim ticks, so an episode is `max_steps / action_repeat` policy decisions |
| `no_rendering_mode`, `map_load_sleep` | CARLA runtime behaviour |
| `use_extra_spawns` | Base value only. Every stage overrides it; it exists here so the layout generator, which applies no stage, honours it too |
| `carla_sensors.imu`, `carla_sensors.gnss`, `carla_sensors.lidar` | CARLA spawn keys only (`sensor_tick`, `points_per_second`, `rotation_frequency`, `upper_fov` / `lower_fov`, `noise.*`). `load_env_config()` injects `mount`, `range` and `channels` from `sensor_config.yaml`; IMU and GNSS noise is injected by the relay nodes, not by CARLA |
| `parking_scenarios.perimeter_*`, `spawn_perimeter_cones` | Perimeter marker blueprint and spacing. Cones are off by default - the soft out-of-bounds penalty enforces the lot edge |
| `parking_scenarios.num_patrol_vehicles_max`, `patrol_*` | NPC patrol vehicles |
| `parking_scenarios.pedestrian_*` | NPC pedestrians |
| `parking_scenarios.floor_plans` | Layout-file paths and OOD flags (`rectangle` trains; `trapezoid` and `irregular_a` are held out) |
| `gnss_noise_profiles` | Path to `gnss_noise_profiles.yaml` |
| `gnss_datum_lat`, `gnss_datum_lon` | Flat-earth projection datum for the GNSS relay. `0.0` auto-latches from the first GNSS reading |
| `ros2.carla_recovery` | CARLA reconnection policy after a headless segfault. Deep-merged onto the `ros2` block from `agent_config.yaml`, which owns the shared topics and file paths |
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
Section 3.4.1 of [`docs/AntonioGaldes_Dissertation.pdf`](../../../docs/AntonioGaldes_Dissertation.pdf)
is the canonical write-up. The YAML is the single source of truth for the numbers; the
figures below are orientation, not a substitute for reading it.

### Tiers

Four tiers, ordered worst-to-best as a ladder. Each carries `lat/lon_stddev_deg`,
`alt_stddev_m`, `metric_stddev_m`, `doppler_stddev_ms`, a start `weight` and a
`description`. `doppler_stddev_ms` scales by the same per-tier factor as
`metric_stddev_m`, so velocity and heading degrade together with the fix state.

| Tier | `metric_stddev_m` | Start `weight` | Parking regime |
|------|------------------|----------------|----------------|
| `rtk_fixed` | 0.020 | 0.40 | Nominal manoeuvring |
| `rtk_float` | 0.360 | 0.20 | Marginal for 2.5 m bays |
| `standalone` | 1.802 | 0.20 | RTK lost; parking unsafe |
| `degraded` | 5.0 | 0.20 | Abort / handoff |

These values are mirrored in `_TIER_DEFAULTS` in `gnss_noise_relay.py`; change both
together.

### Start tier and the Markov chain

- The start tier of each episode is drawn from the `weight` values above. They are
  deliberately biased toward the degraded tiers rather than matching the chain's long-run
  occupancy, so most episodes begin under uncertainty and the chain still recovers
  mid-episode.
- `fixed_gnss_tier` is deliberately omitted from every curriculum stage, so no stage
  pins the start tier. `tests/test_curriculum_invariants.py` asserts that omission under
  `make docker-test-unit`.
- The always-on chain then wanders from the sampled start. It is stage-invariant, so the
  GNSS degradation process is identical in every stage rather than being a ramped
  curriculum axis, and localisation uncertainty is present from stage 1.
- `transition_matrix` holds the per-step transitions applied at the 20 Hz relay callback
  rate by `GnssNoiseRelayNode` when `gnss_noise_relay.enable_markov_transitions: true` in
  [`configs/ros2_config.yaml`](../../ros2_config.yaml). Rows are from-tier, columns
  to-tier, in the key order above.
- The chain is **ladder-only** (a walk never skips a rung, so `rtk_fixed` cannot jump
  straight to `degraded`) and **upward-biased** (recovery probability exceeds degradation
  probability in every non-fixed row). The result is a fixed-dominant chain: the
  stationary occupancy of `rtk_fixed` is about **71%**, with the remaining mass spread
  over the three degraded tiers.
- Diagnose the chain (stationary distribution, mean dwell, time to first contiguous good
  window) with `make analyse-markov`, optionally
  `make analyse-markov N_EPISODES=10000 N_STEPS=1750`.

### Evaluation overrides

At eval time, `held_gnss_tier` in `configs/eval_config.yaml` overrides the sampling to
hold one named tier for the whole episode, and `degrade_one_way` instead forces a
one-way drift that never recovers. The two are alternatives, never combined.

---

## `OpenDriveMap.bin`

Recast/Detour pedestrian navigation mesh for the FlatPlane OpenDRIVE world.
Pre-built and committed because CARLA segfaults when building the nav mesh
on headless GPU setups (known issue upstream).

**Not wired into the running stack.** No compose service mounts it into CARLA's
`Content/Carla/Maps/Nav/` directory and no code path reads it, so it is currently a
retained artefact rather than an active pipeline step. Pedestrian spawning is off by
default anyway (`parking_scenarios.pedestrian_spawn_probability: 0.0`). Copy it into the
CARLA container by hand if a walker-navigation experiment needs it.

**To regenerate** (requires a headed CARLA session, not headless):

1. Start CARLA with a display: `-screen` flag (not `-RenderOffScreen`).
2. Load the FlatPlane world via `generate_opendrive_world()`.
3. Copy the resulting `.bin` from `CarlaUE4/Content/Carla/Maps/Nav/OpenDriveMap.bin`.
4. Replace this file and commit.

---

## See also

- [configs/README.md](../../README.md) - full configs directory map and loading chain
- [configs/deployment/README.md](../README.md) - shared sensor and agent configs
- [configs/deployment/sim/curriculum/README.md](curriculum/README.md) - the six stage files
- [uncertainty_rl/envs/README.md](../../../uncertainty_rl/envs/README.md) - `CARLAParkingEnv` that consumes these files
- [docs/detailed_notes/localisation/sensor_noise_models.md](../../../docs/detailed_notes/localisation/sensor_noise_models.md) - LiDAR and IMU noise derivation
