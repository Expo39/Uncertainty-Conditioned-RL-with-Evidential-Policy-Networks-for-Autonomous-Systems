# evaluation/

Performance evaluation across varying physical conditions - the centrepiece of the dissertation's experimental chapter. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model and VecNormalize statistics from checkpoint
2. Load base sensor noise from `configs/train_config.yaml`
3. Sweep across `eval_conditions` defined in `configs/eval_config.yaml`
4. For each condition: fix weather/traffic/floor plan, run `n_episodes`
5. Collect metrics, generate plots and CSV

### Evaluation Conditions (from `configs/eval_config.yaml`)

Only ClearNoon and HardRainNoon weather presets are used: fog does not render on the generated OpenDRIVE FlatPlane world. EKF uncertainty variation comes from weather (rain LiDAR scatter), IMU noise multipliers, traffic density, and bay occupancy rate.

Three distinct experiment types, all using the same environment code:

**Performance sweep** (ordered from easiest to hardest):

| Condition | Weather | IMU mult | Patrol | Pedestrians | Notes |
|-----------|---------|----------|--------|-------------|-------|
| `clear_low_noise` | ClearNoon | 0.5x | 0 | 0 | Best-case LiDAR |
| `clear_nominal` | ClearNoon | 1.0x | 1 | 2 | Training distribution |
| `clear_sparse_lot` | ClearNoon | 1.0x | 0 | 0 | Low bay occupancy (0.2) |
| `clear_busy` | ClearNoon | 1.5x | 3 | 4 | Dense traffic, occlusions |
| `rain_nominal` | HardRainNoon | 1.0x | 1 | 2 | Rain LiDAR scatter |
| `rain_degraded` | HardRainNoon | 2.0x | 2 | 3 | Compound degradation |
| `rain_busy` | HardRainNoon | 3.0x | 3 | 4 | Max training-distribution stress |

**OOD calibration** (tests whether epistemic uncertainty rises on novel geometry):

| Condition | Floor plan | Purpose |
|-----------|-----------|---------|
| `ood_nominal` | `irregular_a` | Novel perimeter, nominal conditions |
| `ood_degraded` | `irregular_a` | Novel perimeter + heavy rain |

**Safety handoff demonstration** (conditions far beyond training distribution):

| Condition | Description |
|-----------|-------------|
| `worst_case` | Heavy rain + empty lot (zero parked cars) + max pedestrians + 6.0x IMU noise - LiDAR collapses, EKF covariance spikes, evidential epistemic uncertainty should cross the safety handoff threshold |

### Metrics Collected

- Success rate per condition (position < 0.5 m, orientation < 10 deg, velocity < 0.1 m/s)
- Mean and standard deviation of final position and orientation errors
- Epistemic and aleatoric uncertainty from the evidential policy
- Cumulative reward per episode

### Usage

```bash
# Inside training container (make docker-shell):
python uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --eval-config configs/eval_config.yaml \
    --train-config configs/train_config.yaml
```

Or via `make docker-eval`.

### Output

- `evaluation_results/*.csv` - per-condition metrics
- `evaluation_results/*.png` - seaborn bar charts: success rate vs condition, uncertainty estimates, error distributions
