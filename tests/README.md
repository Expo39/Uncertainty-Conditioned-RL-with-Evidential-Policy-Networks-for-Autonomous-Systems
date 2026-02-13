# tests/

pytest test suite mirroring the `uncertainty_rl/` package structure.

## Files

| File | Tests |
|------|-------|
| `conftest.py` | Shared fixtures: state/action tensors, config dicts, uncertainty states |
| `test_evidential_policy.py` | EvidentialLayer, EvidentialPolicyNetwork, UncertaintyConditionedActor, loss computation |
| `test_carla_parking.py` | Environment API, reward function, covariance simulation |
| `test_covariance_utils.py` | Covariance feature extraction, matrix validation, dimension helper |
| `test_evaluation.py` | EvaluationMetrics container and aggregation |
| `test_logging.py` | MetricsLogger and UncertaintyTracker |
| `test_visualisation.py` | Plot generation (uncertainty evolution, trajectory, training curves) |

## Running

```bash
make test              # Full suite
make test-fast         # Skip slow/integration tests
make test-networks     # Evidential policy tests only
make test-env          # Environment tests only
make test-cov          # With coverage report (htmlcov/)
```

## Markers

- `@pytest.mark.slow` - long-running tests
- `@pytest.mark.integration` - requires CARLA or ROS 2

## Constants

- `STATE_DIM = 15`, `ACTION_DIM = 3`, `BATCH_SIZE = 8`, `HIDDEN_DIMS = [64, 64]`
