# utils/

Shared utilities for logging, metrics tracking, and visualisation.

## Modules

### `logging.py`

| Class | Purpose |
|-------|---------|
| `MetricsLogger` | Logs training and evaluation metrics to CSV and JSON |
| `UncertaintyTracker` | Sliding-window tracker for monitoring uncertainty trends over time |

### `visualisation.py`

Matplotlib/seaborn plotting functions for:

- **Uncertainty evolution** — epistemic and aleatoric uncertainty over training steps
- **Trajectory plots** — vehicle path with uncertainty ellipses
- **Training curves** — reward, success rate, loss over time

### Plot Standards

- Style: seaborn `whitegrid`
- Resolution: 300 DPI
- Font sizes: title 14, axis labels 12, legend 10
- Colours: blue = epistemic, red = aleatoric
- Uncertainty ellipses: 95% confidence (chi-squared = 5.991 for 2 DOF)
- Save with `bbox_inches='tight'`
