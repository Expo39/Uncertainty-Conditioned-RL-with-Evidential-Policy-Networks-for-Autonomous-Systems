# configs/

YAML configuration files for all hyperparameters and runtime settings. **No hyperparameters are hardcoded in source** - everything is config-driven.

## Files

| File | Purpose |
|------|---------|
| `train_config.yaml` | Training hyperparameters: learning rate, batch size, buffer size, network architecture, evidential regularisation |
| `eval_config.yaml` | Evaluation settings: noise levels for uncertainty sweep, number of episodes, success criteria thresholds |
| `ros2_config.yaml` | ROS 2 node parameters: odometry topic, covariance topic, publish rate |

## Convention

- All consuming code uses `.get(key, default)` so configs are backwards-compatible.
- Units are commented inline (e.g., `uncertainty_noise_std: 0.1  # metres`).
- New parameters must be added here with sensible defaults before use in code.
