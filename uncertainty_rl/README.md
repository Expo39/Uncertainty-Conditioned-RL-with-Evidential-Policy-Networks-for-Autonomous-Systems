# uncertainty_rl/

Main Python package for uncertainty-conditioned reinforcement learning with evidential policy networks.

## Subpackages

| Directory | Purpose |
|-----------|---------|
| `networks/` | Evidential deep learning policy - NIG distributions, uncertainty quantification |
| `envs/` | CARLA Gymnasium parking environment with SLAM uncertainty in observations |
| `training/` | RL training scripts using Stable-Baselines3 |
| `evaluation/` | Performance evaluation across varying uncertainty levels |
| `ros2/` | ROS 2 bridge to `robot_localisation` EKF covariance |
| `utils/` | Shared logging, metrics tracking, and visualisation |

## Imports

```python
from uncertainty_rl.networks import EvidentialPolicyNetwork
from uncertainty_rl.envs import CARLAParkingEnv
```

## Package Exports

- `EvidentialPolicyNetwork`
- `EvidentialLayer`
- `UncertaintyConditionedActor`
- `CARLAParkingEnv`
