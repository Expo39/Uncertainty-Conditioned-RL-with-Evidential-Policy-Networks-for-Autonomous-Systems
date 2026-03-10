# tests/

pytest test suite mirroring the `uncertainty_rl/` package structure. Two tiers: CPU-only unit tests (no CARLA, no ROS 2, no GPU) and integration tests that run inside the Docker container.

## Files

| File | Tests |
|------|-------|
| `conftest.py` | Shared fixtures: 18-dim state tensors, 9-dim state tensors, config dicts, uncertainty states |
| `test_evidential_policy.py` | EvidentialLayer, EvidentialPolicyNetwork, UncertaintyConditionedActor, NIG constraints, loss computation |
| `test_carla_parking.py` | Env API (reset/step), reward function, geometry helpers, bay sampling, VisStateWriter, obs space shapes (26 tests) |
| `test_covariance_utils.py` | Covariance feature extraction, matrix validation, dimension helper |
| `test_evaluation.py` | EvaluationMetrics container and aggregation |
| `test_logging.py` | MetricsLogger and UncertaintyTracker |
| `test_visualisation.py` | Plot generation (uncertainty evolution, trajectory, training curves) |

## Running

```bash
# CPU-only (no Docker needed) — same checks run in CI
make test-unit         # Unit tests only (no CARLA/ROS 2/GPU)
make verify            # Unit tests + lint + typecheck + import sanity

# Full stack (Docker + GPU machine)
make docker-test-unit          # Unit tests inside container
make docker-test-integration   # Integration tests (needs CARLA + ROS 2)
make docker-test               # Full suite (unit + integration)
```

## Test Tiers

**Unit tests** (`make test-unit`): Run on any machine without CARLA, ROS 2, or GPU. The env falls back to `carla = None` mode and rclpy is guarded with `_ROS2_AVAILABLE = False`. Returns zero uncertainty features — acceptable for unit tests, not for training.

**Integration tests** (`@pytest.mark.integration`): Run inside the training container via `make docker-test-integration`. Test that covariance arrives from EKF, vehicle spawns in CARLA, actions move the vehicle, etc.

## Key Fixture Constants

- `STATE_DIM_FULL = 18` (pose 6 + covariance 9 + target 3)
- `STATE_DIM_NO_COV = 9` (pose 6 + target 3, when `include_covariance=False`)
- `ACTION_DIM = 3`
- `BATCH_SIZE = 8`
- `HIDDEN_DIMS = [64, 64]`
