# detailed_notes

Implementation notes extracted from the code: index layouts, parameter provenance,
runtime configuration and design rationale that would otherwise bloat inline comments.

## Scope

These notes do not restate the subpackage READMEs; each records only the implementation
detail the READMEs and inline comments leave out. Where a note and the code ever
disagree, the code is correct and the note is stale.

## Index

### networks/

| File | Extracted from | Records |
|------|---------------|---------|
| `evidential_nig_initialisation.md` | `networks/evidential_policy.py` | NIG bias-block layout and the prior values `EvidentialPPO.train` must track |

### envs/

| File | Extracted from | Records |
|------|---------------|---------|
| `observation_space.md` | `envs/_parking_core.py` | Observation index layout and the constants it maps to |
| `layout.md` | `scripts/layouts/`, `envs/sim/helpers/` | Patrol-path and cone-interpolation geometry per layout; bay exclusions |
| `actuator_model.md` | `envs/sim/carla_parking.py`, `configs/deployment/agent_config.yaml` | Why the limit is in the env not the reward; the wider rate-reference set |

### localisation/

| File | Extracted from | Records |
|------|---------------|---------|
| `ros2_architecture.md` | `envs/covariance_subscriber.py`, `ros2/` | Shared-file paths and per-worker overrides for parallel training |
| `sensor_noise_models.md` | `envs/sim/helpers/_sensor_manager.py`, `ros2/.../sensor_relay/` | Datasheet figures and the code map for each mechanism |
| `gnss_markov_transitions.md` | `ros2/.../sensor_relay/gnss_noise_relay.py` | Chain runtime behaviour, EKF process noise, the disable flag |

### training/

| File | Extracted from | Records |
|------|---------------|---------|
| `hyperparameter_search.md` | `training/tune_hyperparams.py` | The unused search space, sampler and pruner choices |
| `ablation_hpo_methodology.md` | `training/train_ppo.py`, `training/tune_hyperparams.py` | The code-level constraints behind the untuned configuration |
| `heteroscedastic_control_preregistration.md` | `networks/heteroscedastic.py`, `scripts/analysis/seed_level.py` | Pre-registered hypothesis, decision rule and analysis for the heteroscedastic control arms |

### deployment/

| File | Extracted from | Records |
|------|---------------|---------|
| `real_world_deployment.md` | `envs/real/` | Map from each deployment step to the code implementing it |
| `sim_to_real_transfer.md` | `envs/real/inference_loop.py` | The two modelling gaps left in the sensor model |

## Note on hyperparameter tuning

No hyperparameter optimisation was run for any reported result. The tuning pipeline is
retained as future work, and the two training notes describe it in those terms only.
