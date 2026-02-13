# evaluation/

Performance evaluation across varying SLAM uncertainty levels - the centrepiece of the dissertation's experimental chapter.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model with VecNormalize statistics
2. Sweep `uncertainty_noise_std` across `noise_levels` defined in `configs/eval_config.yaml`
3. Run N episodes per level, collecting metrics
4. Generate plots and CSV results

### Metrics Collected

- Success rate per noise level
- Mean and standard deviation of position/orientation errors
- Epistemic and aleatoric uncertainty estimates from the policy
- Cumulative reward distribution

### Usage

```bash
python uncertainty_rl/evaluation/evaluate.py \
    --model-path checkpoints/final_model \
    --config configs/eval_config.yaml
```

Noise levels, episode counts, and success criteria are all configured in `configs/eval_config.yaml`.

### Output

- CSV with per-episode metrics
- Seaborn plots: success rate vs noise, uncertainty evolution, error distributions
