# uncertainty_rl/

Main Python package for uncertainty-conditioned reinforcement learning with evidential policy networks.

## Subpackages

| Directory | Purpose |
|-----------|---------|
| `networks/` | Evidential deep learning policy - NIG distributions, uncertainty quantification |
| `envs/` | CARLA Gymnasium parking environment with EKF localisation uncertainty in observations |
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

**Networks** (`uncertainty_rl.networks`):
- `EvidentialLayer`
- `EvidentialPolicyNetwork`
- `UncertaintyConditionedActor`
- `EvidentialActorCriticPolicy`
- `EvidentialDistribution`
- `EvidentialPPO`

**Environments** (`uncertainty_rl.envs`):
- `CARLAParkingEnv`
- `RealWorldDeployment`
- `SafetyWrapper`

Sub-modules are not eagerly imported at the package root (`uncertainty_rl/__init__.py`) because
`torch`, `gymnasium`, and `carla` are not installed in the local `.venv`. Import directly from
sub-modules when needed.
