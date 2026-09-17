# detailed_notes

Implementation notes extracted from the code: index layouts, parameter provenance,
runtime configuration and design rationale that would otherwise bloat inline comments.

## Scope

The dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for the design and
its justification. These notes do not restate it; each one names the section that covers
its topic and records only what that section leaves out. Where the two ever disagree,
the dissertation is correct and the note is stale.

## Index

### networks/

| File | Extracted from | Records | Canonical |
|------|---------------|---------|-----------|
| `evidential_nig_initialisation.md` | `networks/evidential_policy.py` | NIG bias-block layout and the prior values `EvidentialPPO.train` must track | Section 3.5.1 |

### envs/

| File | Extracted from | Records | Canonical |
|------|---------------|---------|-----------|
| `observation_space.md` | `envs/_parking_core.py` | Observation index layout and the constants it maps to | Section 3.3 |
| `layout.md` | `scripts/layouts/`, `envs/sim/helpers/` | Patrol-path and cone-interpolation geometry per layout; bay exclusions | Section 3.2 (occupancy) |
| `actuator_model.md` | `envs/sim/carla_parking.py`, `configs/deployment/agent_config.yaml` | Why the limit is in the env not the reward; the wider rate-reference set | Section 3.3 |

### localisation/

| File | Extracted from | Records | Canonical |
|------|---------------|---------|-----------|
| `ros2_architecture.md` | `envs/covariance_subscriber.py`, `ros2/` | Shared-file paths and per-worker overrides for parallel training | Section 3.1 |
| `sensor_noise_models.md` | `envs/sim/helpers/_sensor_manager.py`, `ros2/.../sensor_relay/` | Datasheet figures and the code map for each mechanism | Appendix A |
| `gnss_markov_transitions.md` | `ros2/.../sensor_relay/gnss_noise_relay.py` | Chain runtime behaviour, EKF process noise, the disable flag | Section 3.4.1 |

### training/

| File | Extracted from | Records | Canonical |
|------|---------------|---------|-----------|
| `hyperparameter_search.md` | `training/tune_hyperparams.py` | The unused search space, sampler and pruner choices | Section 3.8.1 |
| `ablation_hpo_methodology.md` | `training/train_ppo.py`, `training/tune_hyperparams.py` | The code-level constraints behind the untuned configuration | Section 3.8.1 |

### deployment/

| File | Extracted from | Records | Canonical |
|------|---------------|---------|-----------|
| `real_world_deployment.md` | `envs/real/` | Map from each Appendix B step to the code implementing it | Appendix B |
| `sim_to_real_transfer.md` | `envs/real/inference_loop.py` | The two modelling gaps the appendices do not cover | Appendix B, Section 4.8.3 |

## Note on hyperparameter tuning

No hyperparameter optimisation was run for any reported result. The tuning pipeline is
retained as future work, and the two training notes describe it in those terms only.
