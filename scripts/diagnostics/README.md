# scripts/diagnostics/

Standalone diagnostics for inspecting a configured process or a finished training run.
Both tools are host-side, CPU-only and read-only, with no CARLA, ROS 2, Docker or GPU
dependency.

Nothing here reads an evaluation results tree - that is [`analysis/`](../analysis/). These
two answer questions about the GNSS process as configured and about training progress as
logged, neither of which involves an eval sweep.

## Modules

| Module | Purpose |
|--------|---------|
| `markov_analyser.py` | Offline diagnostic for the GNSS tier Markov chain: stationary distribution, mean dwell per tier, and time to a usable localisation window |
| `tb_read.py` | Scalar trajectories from TensorBoard event files: tag selection, sampling, smoothing, tails and CSV export |

## `markov_analyser.py`

Answers "what does the configured chain actually do?" without running an episode. It
loads the per-step transition matrix and the per-episode initial-tier weights from
`configs/deployment/sim/gnss_noise_profiles.yaml` and reports three things:

- **Static chain properties** - the initial weight, stationary distribution and mean
  dwell for each tier, derived from the matrix rather than simulated.
- **Per-episode occupation** across simulated rollouts, which is what an episode of a
  given length actually sees rather than the asymptotic figure.
- **Time to the first contiguous good window** for episodes starting in a bad tier,
  evaluated at several window lengths. This is the practical question: if the vehicle
  arrives under a degraded fix, how long before it has a usable stretch of localisation?

The distinction between the stationary distribution and per-episode occupation matters.
The chain is fixed-dominant in the long run, but episodes start in a tier drawn from
weights biased toward the degraded tiers, so a finite episode is not a sample from the
stationary distribution.

```bash
make analyse-markov [N_EPISODES=10000] [N_STEPS=1750]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--profiles` | `configs/deployment/sim/gnss_noise_profiles.yaml` | Chain definition to analyse |
| `--n-episodes` | `10000` | Rollouts simulated for the occupation and window statistics |
| `--n-steps` | `1750` | Episode length in steps, matching `max_steps` in the env config |
| `--dt` | `0.05` | Per-step duration in seconds, matching `carla_timestep` |
| `--good-tiers` | `rtk_fixed rtk_float` | Tiers counted as usable localisation |
| `--window-seconds` | `3.0 5.0 10.0` | Contiguous good-window lengths to evaluate |
| `--seed` | `0` | Rollout seed |

The Make target forwards only `N_EPISODES` and `N_STEPS`. The remaining flags need a
target of their own before they can be used; add one rather than calling `python`
directly.

## `tb_read.py`

Reads the scalars a training run logged, for checking whether a stage progressed, where a
metric plateaued, or how two runs compare. It prints full-series statistics alongside an
evenly sampled trajectory, so a spiky sparse-positive metric such as success rate stays
visible instead of being flattened into a quantile summary.

Event files are decoded by hand rather than through the `tensorboard` package, which
would pull a whole dashboard (gRPC, Werkzeug, a web server) onto the host just to parse a
file. Only the fields needed are read, and unrecognised fields are skipped.

```bash
make tb-scalars LOG=logs/<baseline>/<leaf> [ARGS="--match success --last 10"]
```

`LOG` is required and accepts several run directories, which is how runs are compared.
Every other flag goes through `ARGS`.

| Flag | Default | Purpose |
|------|---------|---------|
| `--list` | off | List the available scalar tags and exit |
| `--tags` | all | Exact tag names to print, e.g. `env/success_rate` |
| `--match` | none | Case-insensitive substrings; any tag containing one is printed |
| `--points` | `12` | Evenly spaced trajectory samples per tag |
| `--full` | off | Print every logged point instead of a sampled trajectory |
| `--last` | `0` | Also print the last N raw points, for plateau inspection |
| `--smooth` | `0` | Trailing moving-average window applied before printing |
| `--csv` | none | Export the selected raw series as tidy CSV (`run,tag,step,value`) |

Start with `--list` on an unfamiliar run: tag names vary by policy type, since the
evidential arms log the uncertainty and regulariser scalars the standard arms do not.

## See also

- [scripts/README.md](../README.md) - all Make targets and the wider tooling layout
- [scripts/analysis/README.md](../analysis/README.md) - the evaluation-results pipeline
- [configs/deployment/sim/gnss_noise_profiles.yaml](../../configs/deployment/sim/gnss_noise_profiles.yaml) - the chain `markov_analyser.py` reads
- [uncertainty_rl/networks/README.md](../../uncertainty_rl/networks/README.md) - the evidential scalars logged to TensorBoard
