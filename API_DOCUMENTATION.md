# API Documentation

## Table of Contents

1. [Networks Module](#networks-module)
2. [Environments Module](#environments-module)
3. [Training Module](#training-module)
4. [Evaluation Module](#evaluation-module)
5. [ROS 2 Module](#ros-2-module)
6. [Utilities Module](#utilities-module)

---

## Networks Module

### `EvidentialPolicyNetwork`

Main policy network with evidential uncertainty quantification.

```python
class EvidentialPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Optional[list] = None,
        activation: str = "relu"
    )
```

**Parameters:**
- `state_dim`: Dimension of state space (including uncertainty features)
- `action_dim`: Dimension of action space
- `hidden_dims`: List of hidden layer dimensions (default: [256, 256])
- `activation`: Activation function ("relu", "tanh", "elu", "leaky_relu")

**Methods:**

#### `forward(state: torch.Tensor) -> Tuple`
Forward pass through network.

**Returns:** `(gamma, nu, alpha, beta)` - evidential parameters

#### `get_action(state: torch.Tensor, deterministic: bool = False) -> Tuple`
Get action with uncertainty estimates.

**Returns:** `(action, uncertainty_dict)` where uncertainty_dict contains:
- `epistemic`: Epistemic (model) uncertainty
- `aleatoric`: Aleatoric (data) uncertainty
- `total`: Total uncertainty
- `gamma`, `nu`, `alpha`, `beta`: Evidential parameters

#### `compute_evidential_loss(...) -> Dict`
Compute evidential regression loss.

**Returns:** Dictionary with `loss`, `nll`, `regularisation`

---

### `UncertaintyConditionedActor`

Actor that explicitly conditions on state uncertainty.

```python
class UncertaintyConditionedActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        uncertainty_dim: int,
        action_dim: int,
        hidden_dims: Optional[list] = None
    )
```

**Parameters:**
- `state_dim`: Dimension of state (excluding uncertainty)
- `uncertainty_dim`: Dimension of uncertainty features
- `action_dim`: Dimension of action space
- `hidden_dims`: Hidden layer dimensions

---

## Environments Module

### `CARLAParkingEnv`

Gymnasium environment for autonomous parking in CARLA.

```python
class CARLAParkingEnv(gym.Env):
    def __init__(
        self,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        town: str = "Town01",
        uncertainty_noise_std: float = 0.1,
        max_steps: int = 500,
        target_parking_spot: Optional[Tuple[float, float, float]] = None,
        render_mode: Optional[str] = None,
    )
```

**Parameters:**
- `carla_host`: CARLA server host
- `carla_port`: CARLA server port
- `town`: CARLA town/map name
- `uncertainty_noise_std`: SLAM uncertainty std deviation (metres)
- `max_steps`: Maximum episode length
- `target_parking_spot`: Target position (x, y, yaw)
- `render_mode`: "human", "rgb_array", or None

**Spaces:**
- **Observation**: Box(15,) - [x, y, yaw, vx, vy, vyaw, σx, σy, σyaw, cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]
- **Action**: Box(3,) - [steering, throttle, brake]

**Methods:**

#### `reset(seed=None, options=None) -> Tuple[np.ndarray, dict]`
Reset environment.

#### `step(action: np.ndarray) -> Tuple`
Execute one step.

**Returns:** `(state, reward, terminated, truncated, info)`

#### `close() -> None`
Clean up resources.

---

## Training Module

### `train()`

Main training function using SAC.

```python
def train(
    config_path: str,
    total_timesteps: int = 1000000,
    log_dir: str = "./logs",
    checkpoint_dir: str = "./checkpoints",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    seed: int = 42,
) -> None
```

**Parameters:**
- `config_path`: Path to YAML configuration file
- `total_timesteps`: Total training timesteps
- `log_dir`: TensorBoard log directory
- `checkpoint_dir`: Model checkpoint directory
- `eval_freq`: Evaluation frequency (timesteps)
- `n_eval_episodes`: Number of evaluation episodes
- `seed`: Random seed

**Configuration File Format:**
See `configs/train_config.yaml` for full example.

---

## Evaluation Module

### `EvaluationMetrics`

Container for evaluation metrics.

```python
class EvaluationMetrics:
    success_rate: float
    average_reward: float
    average_steps: float
    position_errors: List[float]
    orientation_errors: List[float]
    epistemic_uncertainties: List[float]
    aleatoric_uncertainties: List[float]
```

### `evaluate_agent()`

Evaluate agent performance.

```python
def evaluate_agent(
    model: SAC,
    env: DummyVecEnv,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
) -> EvaluationMetrics
```

### `evaluate_across_noise_levels()`

Evaluate across different uncertainty levels.

```python
def evaluate_across_noise_levels(
    model_path: str,
    config_path: str,
    noise_levels: List[float],
    n_episodes: int = 100,
    output_dir: str = "./evaluation_results",
) -> pd.DataFrame
```

**Returns:** DataFrame with evaluation results for each noise level.

---

## ROS 2 Module

### `CovarianceExtractorNode`

ROS 2 node for extracting covariance from robot_localization.

```python
class CovarianceExtractorNode(Node):
    def __init__(self, node_name: str = "covariance_extractor")
```

**Subscribed Topics:**
- `/odometry/filtered` (nav_msgs/Odometry): Input odometry

**Published Topics:**
- `/slam_uncertainty/covariance` (Float64MultiArray): Covariance matrix

**Methods:**

#### `get_uncertainty_state() -> Optional[np.ndarray]`
Get current uncertainty state vector.

**Returns:** Array of [σx, σy, σyaw, cov_xx, cov_yy, cov_yawyaw, cov_xy, cov_xyaw, cov_yyaw]

---

### `CovarianceMonitorNode`

Monitoring node for visualising covariance.

```python
class CovarianceMonitorNode(Node):
    def __init__(self, node_name: str = "covariance_monitor")
```

---

## Utilities Module

### Logging

#### `MetricsLogger`

Logger for tracking metrics.

```python
class MetricsLogger:
    def __init__(self, log_dir: str, prefix: str = "metrics")
    
    def log(self, step: int, metrics: Dict[str, float]) -> None
    def save_csv(self, filename: Optional[str] = None) -> None
    def save_json(self, filename: Optional[str] = None) -> None
    def get_metric(self, name: str) -> Optional[List[Any]]
    def compute_statistics(self, name: str) -> Dict[str, float]
```

#### `UncertaintyTracker`

Track epistemic and aleatoric uncertainty.

```python
class UncertaintyTracker:
    def __init__(self, window_size: int = 100)
    
    def update(self, epistemic: float, aleatoric: float) -> None
    def get_statistics(self) -> Dict[str, Dict[str, float]]
    def reset(self) -> None
```

---

### Visualisation

#### `plot_uncertainty_evolution()`

Plot uncertainty over time.

```python
def plot_uncertainty_evolution(
    epistemic: List[float],
    aleatoric: List[float],
    save_path: Optional[str] = None,
    title: str = "Uncertainty Evolution"
) -> None
```

#### `plot_trajectory()`

Plot vehicle trajectory with uncertainty ellipses.

```python
def plot_trajectory(
    positions: np.ndarray,
    target: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Vehicle Trajectory"
) -> None
```

#### `plot_training_curves()`

Plot training curves.

```python
def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training Curves"
) -> None
```

---

## Configuration Files

### Training Configuration (`train_config.yaml`)

```yaml
# Environment
carla_host: "localhost"
carla_port: 2000
town: "Town01"
uncertainty_noise_std: 0.1
max_steps: 500

# SAC hyperparameters
learning_rate: 0.0003
buffer_size: 1000000
batch_size: 256
tau: 0.005
gamma: 0.99

# Network
net_arch: [256, 256]
activation: "relu"

# Evidential
evidential:
  lambda_reg: 0.01
  use_uncertainty_conditioning: true
```

### Evaluation Configuration (`eval_config.yaml`)

```yaml
model_path: "checkpoints/final_model"
n_episodes: 100
deterministic: true

noise_levels:
  - 0.05
  - 0.1
  - 0.2
  - 0.5
  - 1.0

success_criteria:
  position_threshold: 0.5
  orientation_threshold: 10.0
  velocity_threshold: 0.1
```

### ROS 2 Configuration (`ros2_config.yaml`)

```yaml
odom_topic: "/odometry/filtered"
covariance_topic: "/slam_uncertainty/covariance"
publish_rate: 10.0

qos:
  reliability: "reliable"
  history: "keep_last"
  depth: 10
```

---

## Usage Examples

### Example 1: Basic Training

```python
from uncertainty_rl.training.train_sac import train

train(
    config_path="configs/train_config.yaml",
    total_timesteps=1000000,
    checkpoint_dir="./my_checkpoints"
)
```

### Example 2: Load and Evaluate Model

```python
from stable_baselines3 import SAC
from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

# Load model
model = SAC.load("checkpoints/final_model")

# Create environment
env = DummyVecEnv([lambda: CARLAParkingEnv(uncertainty_noise_std=0.2)])

# Evaluate
state = env.reset()
for _ in range(100):
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    if done:
        break
```

### Example 3: Use Evidential Network Directly

```python
import torch
from uncertainty_rl.networks.evidential_policy import EvidentialPolicyNetwork

# Create network
net = EvidentialPolicyNetwork(state_dim=15, action_dim=3)

# Get action with uncertainty
state = torch.randn(1, 15)
action, unc = net.get_action(state)

print(f"Action: {action}")
print(f"Epistemic uncertainty: {unc['epistemic']}")
print(f"Aleatoric uncertainty: {unc['aleatoric']}")
```

---

## Type Hints Reference

Common types used throughout the codebase:

```python
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import torch

# State and action types
State = np.ndarray  # Shape: (state_dim,)
Action = np.ndarray  # Shape: (action_dim,)

# Uncertainty types
UncertaintyDict = Dict[str, torch.Tensor]  # Contains epistemic, aleatoric, etc.

# Episode return types
StepReturn = Tuple[State, float, bool, bool, Dict[str, Any]]
ResetReturn = Tuple[State, Dict[str, Any]]
```
