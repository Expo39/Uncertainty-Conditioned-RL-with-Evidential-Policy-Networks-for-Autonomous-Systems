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
|   |- agent_config.yaml    Observation flags, safety params (sim and real)
|   |- sim/                 CARLA-specific settings
|   |   |- env_config.yaml
|   |   |- gnss_noise_profiles.yaml
|   |   `- OpenDriveMap.bin
|   `- real/                Real-vehicle deployment settings
|       |- real_world_datum.yaml.example
|       |- actuation_calibration.yaml
|       `- mission.yaml
|- layouts/                 Pre-computed lot geometry (generated - do not edit)
|   |- rectangle.yaml
|   |- trapezoid.yaml
|   |- irregular_a.yaml
|   `- flat_plane.xodr
|- baselines/               Ablation override configs (2x2 study)
|   |- vanilla_ppo.yaml
|   |- input_uncertainty.yaml
|   |- output_uncertainty.yaml
|   `- full_method.yaml
`- training/
    `- tuning_config.yaml   Optuna hyperparameter search settings
```

## How configs are loaded

There is ONE precedence chain, lower loses on conflict:

```
sensor_config < agent_config < env_config (+ stage) < train_config (+ baseline)
```

`load_env_config()` deep-merges the first three plus the `--stage` override into the
unified env config passed to `CARLAParkingEnv`. `merge_configs()` then layers
`train_config.yaml` on top, and `apply_baseline()` overlays the ablation keys last.

**One value, one place.** A config value lives in exactly one file - never repeated
down the chain. The two exceptions are deliberate:
- **Curriculum stages** each spell out the *full* difficulty key set (`use_extra_spawns`,
  `bay_margin`, `fixed_floor_plan`, `fixed_gnss_tier`, `bay_occupancy_min/max`,
  `lidar noise.enabled`) with their own values. These knobs are NOT in `env_config` -
  they are owned by the stages.
- **Baselines** each set the ablation cell flags (`include_covariance`,
  `include_obstacle_obs`, `policy_type`, dirs). These are NOT in `train_config` /
  `agent_config` - they are owned by the baselines.

**Every run is staged and baselined.** Difficulty lives only in stages and obs/policy
flags only in baselines, so a run must pick one of each. Omitting `--stage` defaults to
stage 1 (the curriculum head); omitting `--baseline` defaults to the full method
(`baselines/full_method.yaml`). No value falls back to a hidden Python default. Real
deployment is the same: `agent_config.yaml` names the deployed `baseline:` to get the
obs flags.

The merge is implemented once in `uncertainty_rl/utils/config_merge.py`
(`deep_merge`, `apply_baseline`, `BASELINE_KEYS`); every consumer (`train_ppo.py`,
`tune_hyperparams.py`, `evaluate.py`, `demo_drive.py`, `lot_inspector.py`,
`inference_loop.py`) calls it - no hand-rolled overlays. `evaluate.py` also reads
connection/timing/ROS 2 from the merged env config, so evaluation runs the exact
environment training used.

## Root-level files (three only)

| File | Purpose | Consumed by |
|------|---------|-------------|
| `train_config.yaml` | PPO hyperparameters, evidential settings, training schedule | `train_ppo.py`, `tune_hyperparams.py` |
| `eval_config.yaml` | Evaluation condition sweep (noise tiers, traffic, OOD layouts) + episode count; connection/timing come from env_config | `evaluate.py` |
| `ros2_config.yaml` | EKF node params, GNSS relay settings, topic names | `carla_bridge.launch.py`, ROS 2 nodes |

**Never add a new YAML to `configs/` root.** Use an appropriate subfolder.

## Structural constants vs config keys

Tuneable hyperparameters (learning rate, n_steps, noise weights) live in YAML here.
Fixed structural values (observation dim, action dim, success thresholds) live in
`uncertainty_rl/utils/constants.py` and are never duplicated in config files.

## See also

- [configs/baselines/README.md](baselines/README.md) - ablation 2x2 matrix and reproduction
- [configs/layouts/README.md](layouts/README.md) - layout YAML schema and regeneration
- [configs/deployment/README.md](deployment/README.md) - shared sensor and agent configs
- [configs/deployment/sim/README.md](deployment/sim/README.md) - CARLA simulation settings
- [configs/deployment/real/README.md](deployment/real/README.md) - real-vehicle deployment
- [configs/training/README.md](training/README.md) - Optuna tuning settings
