# evaluation/

Performance evaluation across varying SLAM uncertainty levels — the centrepiece of the dissertation's experimental chapter.

## Module: `evaluate.py`

### Evaluation Protocol

1. Load trained model with VecNormalize statistics
2. Sweep `uncertainty_noise_std` across levels (e.g., 0.05 to 1.0 metres)
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
    --config configs/eval_config.yaml \
    --noise-levels 0.05 0.1 0.2 0.5 1.0
```

### Output

- CSV with per-episode metrics
- Seaborn plots: success rate vs noise, uncertainty evolution, error distributions
