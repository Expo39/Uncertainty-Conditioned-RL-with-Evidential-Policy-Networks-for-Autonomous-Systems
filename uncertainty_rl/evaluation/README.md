# evaluation/

Performance evaluation across varying physical conditions — the centrepiece of the dissertation's experimental chapter. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model and VecNormalize statistics from checkpoint
2. Load base sensor noise from `configs/train_config.yaml`
3. Sweep across `eval_conditions` defined in `configs/eval_config.yaml`
4. For each condition: fix weather/fog/traffic/floor plan, run `n_episodes`
5. Collect metrics, generate plots and CSV

### Evaluation Conditions (from `configs/eval_config.yaml`)

Three distinct experiment types, all using the same environment code:

**Performance sweep** (ordered from easiest to hardest):

| Condition | Weather | Fog | IMU mult | NPC vehicles | Pedestrians |
|-----------|---------|-----|----------|--------------|-------------|
| `clear_low_noise` | ClearNoon | 0% | 0.5x | 0 | 0 |
| `clear_nominal` | ClearNoon | 0% | 1.0x | 10 | 5 |
| `cloudy_moderate` | CloudyNoon | 10% | 1.5x | 20 | 10 |
| `rain_degraded` | SoftRainNoon | 20% | 2.0x | 25 | 15 |
| `heavy_rain_noisy` | HardRainNoon | 30% | 3.0x | 40 | 20 |
| `fog_moderate` | CloudyNoon | 50% | 2.0x | 30 | 15 |
| `fog_heavy` | CloudySunset | 75% | 4.0x | 50 | 30 |
| `fog_extreme` | CloudySunset | 95% | 8.0x | 60 | 40 |

**OOD calibration** (tests whether epistemic uncertainty rises on novel geometry):

| Condition | Floor plan | Purpose |
|-----------|-----------|---------|
| `ood_nominal` | `irregular_a` | Novel perimeter, nominal conditions |
| `ood_degraded` | `irregular_a` | Novel perimeter + degraded sensors |

**Safety handoff demonstration** (conditions far beyond training distribution):

| Condition | Description |
|-----------|-------------|
| `worst_case` | Fog 90% + empty lot (zero parked cars) + max pedestrians — LiDAR collapses, EKF covariance spikes, evidential epistemic uncertainty should cross the safety handoff threshold |

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

- `evaluation_results/*.csv` — per-condition metrics
- `evaluation_results/*.png` — seaborn bar charts: success rate vs condition, uncertainty estimates, error distributions
