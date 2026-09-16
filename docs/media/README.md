# docs/media/

Visual assets used across the project's READMEs.

| File | Contents | Shown in |
|------|----------|----------|
| `carla_3d.gif` | CARLA chase view of the evidential policy parking across two consecutive episodes, each with a different highlighted target bay | `README.md` |
| `visualiser_2d.gif` | 2D bird's-eye visualiser parking as the GNSS fix state climbs degraded -> standalone -> float -> RTK fixed | `README.md`, `scripts/visualise/README.md` |
| `training_curves.png` | Success and collision rate per arm across the six curriculum stages | `README.md`, `uncertainty_rl/training/README.md` |
| `eval_degradation.png` | Success rate and mean final position error per arm across the reported evaluation conditions | `README.md`, `uncertainty_rl/evaluation/README.md` |
| `inspect_sensors.jpeg` | Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye | `USAGE.md`, `scripts/inspect/README.md` |

## Regenerating an asset

| Asset | Command |
|-------|---------|
| `visualiser_2d.gif` | `make eval-visualise-2d RECORD=true ...` then `make clip START=00:05 END=00:20 FORMAT=gif WIDTH=800 FPS=15` |
| `carla_3d.gif` | `make docker-eval-visualise-3d` + `make record-screen DURATION=30 REGION=800x600 OFFSET=<X>,<Y>` then `make clip` |
| `inspect_sensors.jpeg` | `make docker-inspect-sensors SENSORS_VIEW=birds_eye INSPECT_ZOOM=close` |
| `training_curves.png` | `make training-curves` then `make figures FIG=training_curves` |
| `eval_degradation.png` | `make figures FIG=ablation_by_condition` |
