# scripts/

Offline tooling for layout generation, CARLA inspection, training visualisation,
and experiment orchestration. None of these scripts are imported by the training
pipeline -- they are invoked exclusively via `make` targets.

## Subdirectories

| Directory | Contents |
|-----------|----------|
| [`colours/`](colours/) | Centralised colour palette for all visualisations (single source of truth) |
| [`layouts/`](layouts/) | Parking lot floor plan modules + `generate_layouts.py` orchestrator |
| [`inspect/`](inspect/) | CARLA debug overlay scripts for layouts, sensor placement, and dryrun |
| [`training/`](training/) | Training helper scripts invoked inside the container (`train.sh`) |
| [`visualise/`](visualise/) | 2D bird's-eye visualiser + checkpoint demo driver |

## Quick Reference

| Task | Command |
|------|---------|
| Generate all lot YAMLs + PNGs | `make generate-layouts` |
| Generate one layout | `make generate-layouts LAYOUT=trapezoid` |
| Train (full run) | `make docker-train` |
| Train (10k step smoke-test) | `make docker-train-short` |
| Inspect layout in CARLA | `make docker-inspect INSPECT_LAYOUT=rectangle` |
| Inspect sensor placement | `make docker-inspect-sensors` |
| Full pipeline dryrun (windowed) | `make docker-inspect-dryrun` |
| Live training visualiser | `make visualise` |
| Checkpoint demo + 2D viewer | `make eval-visualise-2d` |
| Checkpoint demo + 3D CARLA view | `make docker-eval-visualise-3d` |
