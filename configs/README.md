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

`train_ppo.py` takes `--train-config` and `--env-config`. `load_env_config()` automatically
resolves and merges three files (lower wins on conflict):

```
sensor_config.yaml  +  agent_config.yaml  +  env_config.yaml
                                               (env wins)
```

This merged env config is passed to `CARLAParkingEnv`. The RL training keys come from
`train_config.yaml` and are kept separate.

## Root-level files (three only)

| File | Purpose | Consumed by |
|------|---------|-------------|
| `train_config.yaml` | PPO hyperparameters, evidential settings, training schedule | `train_ppo.py`, `tune_hyperparams.py` |
| `eval_config.yaml` | 9 evaluation conditions (noise tiers, traffic, OOD layouts) | `evaluate.py` |
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
