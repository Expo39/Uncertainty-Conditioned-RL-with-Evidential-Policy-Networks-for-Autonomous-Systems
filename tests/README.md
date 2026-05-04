# tests/

pytest test suite mirroring the `uncertainty_rl/` package structure. Two tiers: CPU-only unit tests (no CARLA, no ROS 2, no GPU) and integration tests that run inside the Docker container.

## Files

| File | Tests |
|------|-------|
| `conftest.py` | Shared fixtures: state tensors, config dicts, uncertainty states |
| `test_evidential_policy.py` | EvidentialLayer, EvidentialPolicyNetwork, UncertaintyConditionedActor, NIG constraints, loss computation |
| `test_carla_parking.py` | Env obs/action shapes, geometry helpers, `point_in_polygon`, `yaw_from_quaternion`, `wrap_angle_symmetric`, `build_observation`, `extract_obstacle_features`, `load_floor_plan`, `wait_for_ekf`, bay sampling, reward, VisStateWriter |
| `test_covariance_utils.py` | Feature extraction, matrix validation, dimension helper, `make_diagonal_covariance` |
| `test_sb3_integration.py` | SB3 + evidential policy integration: policy build, forward pass, action sampling |
| `test_baseline_configs.py` | All 4 baseline YAMLs load correctly and override only permitted keys |
| `test_evaluation.py` | EvaluationMetrics container and aggregation, `_scale_sensor_noise` |
| `test_visualisation.py` | Plot generation (uncertainty evolution, trajectory, training curves) |
| `test_covariance_subscriber.py` | `_CovarianceSubscriber`: file reading, mtime staleness guard, `invalidate()`, `get_latest_uncertainty()`, `get_latest_pose()`, `has_data` |
| `test_train_ppo.py` | `linear_schedule`, `EnvDiagnosticsCallback`, `make_env` helpers |
| `test_tune_hyperparams.py` | Optuna sampling, `apply_best_params`, `TrialEvalCallback` |
| `test_safety_wrapper.py` | `SafetyWrapper.apply()`, `step()`, `reset()`, episode safety stats |
| `test_debug_logger.py` | `DebugLogger`: no-op contract when disabled, dict population when enabled |
| `test_actuation_calibration.py` | `ActuatorMap` (gain, deadband, bias, clamp), `ActuationCalibration` (identity, from_config) |
| `test_ros2_integration.py` | EKF covariance arrival, dimension check, non-zero uncertainty (4 tests, `@pytest.mark.integration`) |

## Running

```bash
# CPU-only (no Docker needed) -same checks run in CI
make test-unit         # Unit tests only (no CARLA/ROS 2/GPU)
make verify            # Unit tests + lint + typecheck + import sanity

# Full stack (Docker + GPU machine)
make docker-test-unit          # Unit tests inside container
make docker-test-integration   # Integration tests (needs CARLA + ROS 2)
make docker-test               # Full suite (unit + integration)
```

## Test Tiers

**Unit tests** (`make test-unit`): Run on any machine without CARLA, ROS 2, or GPU. The env falls back to `carla = None` mode and rclpy is guarded with `_ROS2_AVAILABLE = False`. Returns zero uncertainty features -acceptable for unit tests, not for training.

**Integration tests** (`@pytest.mark.integration`): Run inside the training container via `make docker-test-integration`. Test that covariance arrives from EKF, vehicle spawns in CARLA, actions move the vehicle, etc.

## Key Fixture Constants

- `TOTAL_OBS_DIM = 12` (vyaw + std_x/y/yaw + dx/dy/dyaw + 5 LiDAR obstacle features)
- `ACTION_DIM = 2` (steering, longitudinal)
- `BATCH_SIZE = 8`
- `HIDDEN_DIMS = [64, 64]`
