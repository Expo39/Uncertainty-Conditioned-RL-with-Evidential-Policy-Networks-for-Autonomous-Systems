# detailed_notes

Long-form explanations, derivations, and design rationale organised by subsystem.

## Purpose

Every `.md` file here covers one topic that was previously embedded as a multi-paragraph comment
inside a `.py` or `.yaml` file. Keeping long explanations here lets the code stay navigable while
preserving the full reasoning for future reference. Notes are grouped by theme: networks, envs,
localisation, training, and deployment.

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
| `cog_heading_fixes.md` | `ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` | Course-Over-Ground heading fusion and three EKF yaw stability fixes |

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
