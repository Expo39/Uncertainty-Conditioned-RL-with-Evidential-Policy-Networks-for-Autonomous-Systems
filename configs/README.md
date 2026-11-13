# configs/

All experiment configuration lives here. No hyperparameters are hardcoded in source files -
every tuneable value is driven by a YAML in this directory tree.

## Directory map

```
configs/
|- train_config.yaml        PPO + evidential training hyperparameters
|- eval_config.yaml         Evaluation condition sweep
|- ros2_config.yaml         ROS 2 node parameters (EKF, GNSS relay, covariance extractor)
|- deployment/
|   |- sensor_config.yaml   Physical sensor specs + mount positions (sim and real)
|   |- agent_config.yaml    Safety params, actuator model, deployed baseline pointer
|   |- sim/                 CARLA-specific settings
|   |   |- env_config.yaml
|   |   |- gnss_noise_profiles.yaml
|   |   |- OpenDriveMap.bin
|   |   `- curriculum/      Per-stage difficulty overrides
|   |       `- stage1.yaml ... stage6.yaml
|   `- real/                Real-vehicle deployment settings
|       |- real_world_datum.yaml.example
|       |- actuation_calibration.yaml
|       `- mission.yaml
|- layouts/                 Pre-computed lot geometry (generated - do not edit)
|   |- rectangle.yaml
|   |- trapezoid.yaml
|   |- irregular_a.yaml
|   `- flat_plane.xodr
|- baselines/               Ablation override configs (3x2 study)
|   |- vanilla_ppo.yaml
|   |- input_uncertainty.yaml
|   |- heteroscedastic.yaml
|   |- heteroscedastic_input.yaml
|   |- output_uncertainty.yaml
|   `- full_method.yaml
`- training/
    `- tuning_config.yaml   Optuna search settings (never run - see below)
```

**No hyperparameter search was run.** Every reported result uses the committed defaults
in `train_config.yaml`; tuning per arm would make the configuration an extra variable and
confound the 3x2 ablation. See [configs/training/README.md](training/README.md).

## How configs are loaded

There is ONE precedence chain, lower loses on conflict:

```
sensor_config < agent_config < env_config (+ stage) < train_config (+ baseline)
```

`load_env_config()` (in `train_ppo.py`) deep-merges the stage override onto
`env_config.yaml`, then merges `sensor_config < agent_config < env_config` into the
unified env config passed to `CARLAParkingEnv`. `merge_configs()` then layers
`train_config.yaml` on top, and `apply_baseline()` overlays the ablation keys last.

| Step | Function | Merge behaviour |
|------|----------|-----------------|
| stage over env | `deep_merge` | Recursive; stage wins per key |
| sensor/agent/env | `deep_merge` | Recursive; env wins per key |
| train over env | `merge_configs` | Shallow `{**env, **train}`; train wins per top-level key |
| baseline last | `apply_baseline` | Only the four `BASELINE_KEYS`; any other key raises `ValueError` |

`BASELINE_KEYS` is a frozenset of exactly four keys: `baseline_name`,
`include_covariance`, `include_obstacle_obs`, `policy_type`.

`load_env_config()` also injects the `mount` / `range` / `channels` spec from
`sensor_config.yaml` into the matching `carla_sensors` entry, and drops
`training_overrides` from the env dict (the stage schedule blocks are applied after the
train/env merge by `_apply_stage_training_overrides()`, not passed to the env).

**One value, one place.** A config value lives in exactly one file - never repeated
down the chain. The two exceptions are deliberate:
- **Curriculum stages** each spell out the *full* difficulty key set (`use_extra_spawns`,
  `bay_margin`, `fixed_floor_plan`, `bay_occupancy_min/max`, `lidar noise.enabled`,
  `allowed_bay_ids`) plus two per-policy schedule blocks
  (`standard_overrides` / `evidential_overrides`). These knobs are NOT in `env_config` -
  they are owned by the stages. `fixed_gnss_tier` is deliberately omitted throughout, so
  the start tier is sampled rather than pinned.
- **Baselines** each set the ablation cell flags (`baseline_name`, `include_covariance`,
  `include_obstacle_obs`, `policy_type`). These are NOT in `train_config` /
  `agent_config` - they are owned by the baselines. Output directories are derived in
  code from `baseline_name` rather than set per baseline.

**Every training run is staged and baselined.** Difficulty lives only in stages and
obs/policy flags only in baselines, so a run must pick one of each:
- `train_ppo.py` and `tune_hyperparams.py` default to `DEFAULT_STAGE = 1` (the curriculum
  head) and `DEFAULT_BASELINE = configs/baselines/full_method.yaml`. A missing
  `stage<N>.yaml` raises `FileNotFoundError` rather than silently skipping the stage.
- `evaluate.py` takes `--baseline` (same full-method default) but has **no** `--stage`:
  the sweep's difficulty is set per condition in `eval_config.yaml`.
- Real deployment names the baseline via `agent_config.yaml`'s `baseline:` pointer.

The merge is implemented once in `uncertainty_rl/utils/config_merge.py`
(`deep_merge`, `apply_baseline`, `BASELINE_KEYS`). Its callers:

| Consumer | Uses |
|----------|------|
| `uncertainty_rl/training/train_ppo.py` | `deep_merge` + `apply_baseline` (owns `load_env_config` / `merge_configs`) |
| `uncertainty_rl/training/tune_hyperparams.py` | `apply_baseline` (reuses `train_ppo`'s loaders) |
| `uncertainty_rl/evaluation/evaluate.py` | `apply_baseline` over `load_env_config()` |
| `scripts/visualise/demo_drive.py` | `apply_baseline` |
| `scripts/inspect/lot_inspector.py` | `apply_baseline` |

`uncertainty_rl/envs/real/inference_loop.py` is the one exception: it reads the baseline
YAML named by `agent_config.yaml`'s `baseline:` pointer directly for the three obs/policy
flags, rather than merging a full training config.

`evaluate.py` reads connection/timing/ROS 2 from the merged env config, so evaluation
runs the same environment training used. Two caveats:
- It calls `load_env_config()` with **no stage**, so eval uses `env_config.yaml`'s own
  `parking_scenarios` values; per-condition difficulty comes from `eval_config.yaml`.
- Its `--train-config` argument is accepted but not read - PPO hyperparameters are
  already baked into the loaded checkpoint.

## Root-level files (three only)

| File | Purpose | Consumed by |
|------|---------|-------------|
| `train_config.yaml` | PPO hyperparameters, `net_arch` / `activation`, `evidential.*`, logging and checkpoint schedule, `parallel_workers`. Per-stage `stage_timesteps` / LR / entropy schedules live in the curriculum files, not here | `train_ppo.py`, `tune_hyperparams.py` |
| `eval_config.yaml` | Evaluation condition sweep and episode count. Connection and timing come from env_config | `evaluate.py` |
| `ros2_config.yaml` | EKF node params, GNSS and IMU relay settings, topic names, QoS, noise master switches | `carla_bridge.launch.py`, `real_vehicle.launch.py`, ROS 2 nodes |

`eval_config.yaml` holds `model_path`, `n_episodes`, `deterministic`, `debug`,
`near_miss_threshold_m` and the `eval_conditions` list. Each condition sets `name` and
`description` plus one varied factor drawn from `held_gnss_tier`, `degrade_one_way` (with
`degrade_rate_scale`), `lidar_noise_multiplier`, `bay_occupancy_rate` and `floor_plan`.

**Never add a new YAML to `configs/` root.** Use an appropriate subfolder.

## Structural constants vs config keys

Tuneable hyperparameters (learning rate, n_steps, noise weights) live in YAML here.
Fixed structural values (observation dim, action dim, success thresholds) live in
`uncertainty_rl/utils/constants.py` and are never duplicated in config files.

## See also

- [configs/baselines/README.md](baselines/README.md) - ablation 3x2 matrix and reproduction
- [configs/layouts/README.md](layouts/README.md) - layout YAML schema and regeneration
- [configs/deployment/README.md](deployment/README.md) - shared sensor and agent configs
- [configs/deployment/sim/README.md](deployment/sim/README.md) - CARLA simulation settings
- [configs/deployment/sim/curriculum/README.md](deployment/sim/curriculum/README.md) - the six curriculum stage files
- [configs/deployment/real/README.md](deployment/real/README.md) - real-vehicle deployment
- [configs/training/README.md](training/README.md) - Optuna tuning settings
