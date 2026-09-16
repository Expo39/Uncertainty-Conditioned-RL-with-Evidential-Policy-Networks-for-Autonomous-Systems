# detailed_notes

Technical notes on system design, derivations, and rationale. Organised by subsystem: networks, envs, localisation, training, deployment.

## Cross-reference index

### networks/

| File | Extracted from | Topic |
|------|---------------|-------|
| `evidential_nig_initialisation.md` | `networks/evidential_policy.py` `EvidentialLayer.__init__` | NIG hyperprior bias derivation and ortho_init interaction |

### envs/

| File | Extracted from | Topic |
|------|---------------|-------|
| `observation_space.md` | `envs/_parking_core.py` | Observation layout (13-dim default), LiDAR sector boundaries, covariance features |
| `layout.md` | `scripts/layouts/floor_plans/`, `envs/sim/_npc_controller.py`, `envs/sim/_lot_spawner.py` | Lot geometry derivations, patrol path controller, bay sampling, cone placement |
| `actuator_model.md` | `envs/sim/carla_parking.py` `step()`, `configs/deployment/agent_config.yaml` | Per-axis rate limits, brake-overrides-throttle constraint, derivation from production DBW/EPS/hydraulic literature |

### localisation/

| File | Extracted from | Topic |
|------|---------------|-------|
| `ros2_architecture.md` | `envs/covariance_subscriber.py`, `ros2/` | DDS-bypass via shared JSON, atomicity, sequence-number guard |
| `sensor_noise_models.md` | `envs/sim/helpers/_sensor_manager.py`, `ros2/uncertainty_rl_ros2/sensor_relay/imu_noise_relay.py` | SICK TiM571 LiDAR and VN-100 IMU noise derivations with datasheet sources |
| `gnss_markov_transitions.md` | `ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` | RTK fix-state Markov chain design and sim-to-real rationale |

### training/

| File | Extracted from | Topic |
|------|---------------|-------|
| `hyperparameter_search.md` | `training/tune_hyperparams.py` | Optuna search space design, sampler/pruner rationale, literature references |
| `ablation_hpo_methodology.md` | `training/train_ppo.py`, `training/tune_hyperparams.py` | Ablation study methodology and hyperparameter optimisation rationale |

### deployment/

| File | Extracted from | Topic |
|------|---------------|-------|
| `real_world_deployment.md` | `envs/real/deployment_utils.py`, `envs/real/inference_loop.py` | Sensor data flow, EKF frame calibration, surveyed datum, actuation calibration |
| `sim_to_real_transfer.md` | `envs/real/inference_loop.py` | Known sim-to-real gaps and mitigations |

## The other notes tree

These notes are extracted from the code and are indexed above. A second set of notes
lives under `documentation/detailed_notes/`, covering the methodological and
argumentative material behind the dissertation rather than the implementation. Several
are cited directly from source and from the analysis scripts, among them
`epistemic_aleatoric_disentanglement.md`, `evidential_actor_variance_collapse.md`,
`degraded_gnss_is_not_a_blackout.md`, `uncertainty_input_only.md` and
`curriculum_design_principles.md`.

Where a topic appears in both trees, the `documentation/` copy is canonical.
