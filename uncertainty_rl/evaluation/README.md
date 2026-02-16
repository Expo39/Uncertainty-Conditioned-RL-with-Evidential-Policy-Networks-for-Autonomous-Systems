# evaluation/

Performance evaluation across varying physical conditions (parking lot scenarios) - the centrepiece of the dissertation's experimental chapter.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model with VecNormalize statistics
2. Load base sensor noise from `configs/train_config.yaml`
3. Sweep across `eval_conditions` defined in `configs/eval_config.yaml`
4. For each condition: scale sensor noise, fix weather/traffic, run N episodes
5. Generate plots and CSV results

### Parking Lot Conditions

Conditions range from "empty clear lot" to "foggy crowded lot at sunset":
- **clear_low_noise**: Perfect conditions, empty lot
- **clear_nominal**: Normal day, light traffic
- **cloudy_moderate** through **heavy_rain_noisy**: Progressively worse weather
- **fog_moderate** through **fog_extreme**: Dense fog severely degrading GNSS

### Metrics Collected

- Success rate per condition
- Mean and standard deviation of position/orientation errors
- Epistemic and aleatoric uncertainty estimates from the policy
- Cumulative reward distribution

### Usage

```bash
python uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --eval-config configs/eval_config.yaml \
    --train-config configs/train_config.yaml
```

Conditions, episode counts, and success criteria are all configured in `configs/eval_config.yaml`.

### Output

- CSV with per-condition metrics
- Seaborn bar charts: success rate vs condition, uncertainty estimates, error distributions
