# evaluation/

Performance evaluation across varying physical conditions — the centrepiece of the dissertation's experimental chapter. Tests whether the uncertainty-conditioned policy degrades more gracefully than baselines as localisation uncertainty increases.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model and VecNormalize statistics from checkpoint.
2. Load base sensor noise from `configs/deployment/sim/env_config.yaml`.
3. Sweep across `eval_conditions` defined in `configs/eval_config.yaml`.
4. For each condition: fix GNSS/IMU noise, traffic, floor plan, run `n_episodes`.
5. Collect metrics, generate plots and CSV.

### Evaluation Conditions (from `configs/eval_config.yaml`)

Primary uncertainty axis: GNSS noise tier (RTK fix state). `gnss_noise_multiplier` scales
the base RTK-fixed noise (0.02 m stddev). No weather variation — FlatPlane does not render
weather effects.

**Performance sweep** (ordered from easiest to hardest):

| Condition | GNSS mult | IMU mult | Patrol | Ped. prob | Bay occ. | Notes |
|-----------|-----------|----------|--------|-----------|----------|-------|
| `nominal_empty` | 1.0x | 1.0x | 0 | 0.0 | 0.6 | RTK fixed, best-case |
| `nominal_busy` | 1.0x | 1.0x | 1 | 1.0 | 0.8 | RTK fixed, full traffic |
| `rtk_float` | 15.0x | 1.0x | 1 | 0.8 | 0.6 | Marginal localisation |
| `rtk_float_busy` | 15.0x | 1.5x | 1 | 1.0 | 0.8 | Compound degradation |
| `rtk_standalone` | 100.0x | 1.0x | 1 | 0.8 | 0.6 | Policy should abort |
| `rtk_lost` | 250.0x | 2.0x | 1 | 1.0 | 0.8 | Safety handoff expected |

**OOD calibration** (tests whether epistemic uncertainty rises on novel geometry):

| Condition | Floor plan | Purpose |
|-----------|-----------|---------|
| `ood_layout` | `irregular_a` | Novel geometry, nominal GNSS |
| `ood_layout_degraded` | `irregular_a` | Novel geometry + RTK float |

**Safety handoff demonstration**:

| Condition | Description |
|-----------|-------------|
| `worst_case` | GNSS 250x + IMU 6x + zero parked cars + full pedestrians — safety handoff expected |

### Metrics Collected

- Success rate per condition (position < 0.5 m, orientation < 10 deg, velocity < 0.1 m/s)
- Mean episode reward and mean steps to termination
- Epistemic and aleatoric uncertainty from the evidential policy

### Usage

```bash
# Inside training container (make docker-shell):
python uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --eval-config configs/eval_config.yaml \
    --env-config configs/deployment/sim/env_config.yaml \
    --train-config configs/train_config.yaml
```

Or via `make docker-eval`.

### Output

- `evaluation_results/*.csv` — per-condition metrics
- `evaluation_results/*.png` — seaborn bar charts (success rate, reward, steps, uncertainty)
