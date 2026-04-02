# evaluation/

Performance evaluation across varying physical conditions - the centrepiece of the dissertation's experimental chapter. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model and VecNormalize statistics from checkpoint
2. Load base sensor noise from `configs/carla/env_config.yaml`
3. Sweep across `eval_conditions` defined in `configs/eval_config.yaml`
4. For each condition: fix weather/traffic/floor plan, run `n_episodes`
5. Collect metrics, generate plots and CSV

### Evaluation Conditions (from `configs/eval_config.yaml`)

Only ClearNoon and HardRainNoon weather presets are used: fog does not render on the generated OpenDRIVE FlatPlane world. EKF uncertainty variation comes from weather (rain LiDAR scatter), IMU noise multipliers, traffic density, and bay occupancy rate.

Three distinct experiment types, all using the same environment code:

**Performance sweep** (ordered from easiest to hardest):

| Condition | IMU mult | Patrol | Ped. prob | Bay occ. | Notes |
|-----------|----------|--------|-----------|----------|-------|
| `low_noise` | 0.5x | 0 | 0.0 | 0.6 | Best-case LiDAR |
| `nominal` | 1.0x | 1 | 1.0 | 0.6 | Training distribution |
| `sparse_lot` | 1.0x | 0 | 0.0 | 0.3 | Low bay occupancy |
| `full_traffic` | 1.5x | 1 | 1.0 | 0.8 | Dense + dynamic occlusions |
| `high_noise_nominal` | 2.0x | 1 | 1.0 | 0.6 | Elevated sensor degradation |
| `high_noise_degraded` | 2.0x | 1 | 1.0 | 0.3 | Compound degradation |
| `extreme_noise` | 3.0x | 1 | 1.0 | 0.8 | Max training-distribution stress |

**OOD calibration** (tests whether epistemic uncertainty rises on novel geometry):

| Condition | Floor plan | Purpose |
|-----------|-----------|---------|
| `ood_nominal` | `trapezoid` | Novel geometry, nominal conditions (ood: true) |
| `ood_degraded` | `irregular_a` | Novel geometry + 2.0x noise (ood: true) |

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
    --env-config configs/carla/env_config.yaml \
    --train-config configs/train_config.yaml
```

Or via `make docker-eval`.

### Output

- `evaluation_results/*.csv` - per-condition metrics
- `evaluation_results/*.png` - seaborn bar charts: success rate vs condition, uncertainty estimates, error distributions
