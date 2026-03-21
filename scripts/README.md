# scripts/

Offline tooling for layout generation, CARLA inspection, training visualisation,
and experiment orchestration. None of these scripts are imported by the training
pipeline -- they are invoked exclusively via `make` targets.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| [`layouts/`](layouts/) | Parking lot floor plan modules + `generate_layouts.py` orchestrator |
| [`inspect/`](inspect/) | CARLA debug overlay scripts for layouts and sensor placement |

## Root-Level Scripts

### `run_experiment.py`

Ablation study orchestrator. Enumerates all 4 baseline configs x N seeds, runs them
sequentially or in parallel (each on a separate CARLA port), and writes results to
per-baseline output directories.

```bash
make experiment-dry             # Preview run plan locally (no Docker needed)
make docker-experiment          # Full ablation inside container (GPU required)
make docker-experiment-dry      # Dry-run inside container
```

### `visualise_training.py`

Detachable 2D bird's-eye visualiser. Polls `outputs/vis_state.json` (written
atomically by `VisStateWriter` in `carla_parking.py`) and renders ego vehicle,
NPCs, target bay, trajectory trail, and lot geometry in a Matplotlib window.

Runs on the host without a CARLA connection -- start it any time during training:

```bash
make visualise           # Live window
make visualise-record    # Live window + save MP4 on close
```

## Quick Reference

| Task | Command |
|------|---------|
| Generate all lot YAMLs + PNGs | `make generate-layouts` |
| Generate one layout | `make generate-layouts LAYOUT=trapezoid` |
| Inspect layout in CARLA | `make docker-inspect INSPECT_LAYOUT=rectangle` |
| Inspect sensor placement | `make docker-inspect-sensors INSPECT_SUITE=suite_a` |
| Live training visualiser | `make visualise` |
| Dry-run ablation study | `make experiment-dry` |
