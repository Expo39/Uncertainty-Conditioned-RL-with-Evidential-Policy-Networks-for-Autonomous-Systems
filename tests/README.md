# tests/

pytest test suite mirroring the `uncertainty_rl/` package structure. Two tiers: CPU-only unit tests (no CARLA, no ROS 2, no GPU) and integration tests that run inside the Docker container.

## Files

| File | Tests |
|------|-------|
| `conftest.py` | Shared fixtures: 21-dim state tensors, 12-dim state tensors, config dicts, uncertainty states |
| `test_evidential_policy.py` | EvidentialLayer, EvidentialPolicyNetwork, UncertaintyConditionedActor, NIG constraints, loss computation |
| `test_carla_parking.py` | Env API (reset/step), geometry helpers, bay sampling, VisStateWriter, obs space shapes, `zone_bbox`, `_compute_reward` (40 tests) |
| `test_covariance_utils.py` | Covariance feature extraction, matrix validation, dimension helper (18 tests) |
| `test_sb3_integration.py` | SB3 + evidential policy integration: policy build, forward pass, action sampling (23 tests) |
| `test_baseline_configs.py` | All 4 baseline YAMLs load correctly and override only permitted keys (10 tests) |
| `test_run_experiment.py` | run_experiment.py orchestration: dry-run, config merging, seed enumeration (15 tests) |
| `test_evaluation.py` | EvaluationMetrics container and aggregation, `_scale_sensor_noise` (13 tests) |
| `test_logging.py` | MetricsLogger and UncertaintyTracker (11 tests) |
| `test_visualisation.py` | Plot generation (uncertainty evolution, trajectory, training curves) (7 tests) |
| `test_covariance_subscriber.py` | `_CovarianceSubscriber`: file reading, mtime staleness guard, `invalidate()`, `get_latest_uncertainty()`, `get_latest_pose()`, `has_data` (16 tests) |
| `test_tf_to_odom.py` | `_yaw_from_quaternion`, `_make_diagonal_covariance`, `TfToOdomNode._compute_covariance_scale` (18 tests) |
| `test_train_ppo.py` | `linear_schedule`: callable return, progress=1.0/0.0/0.5, linear interpolation, independence (6 tests) |
| `test_ros2_integration.py` | EKF covariance arrival, dimension check, non-zero uncertainty (3 tests, `@pytest.mark.integration`) |

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

- `STATE_DIM_FULL = 20` (pose 6 + covariance 9 + target 3 + obstacle 2)
- `STATE_DIM_NO_COV = 11` (pose 6 + target 3 + obstacle 2, when `include_covariance=False`)
- `ACTION_DIM = 2`
- `BATCH_SIZE = 8`
- `HIDDEN_DIMS = [64, 64]`
