# docs/media/

Visual assets for the project documentation. This directory seeds the relative paths
used by GIF placeholder blocks across all READMEs so that links do not 404 in repo
browsers before recordings are captured.

---

## GIF Placeholder Convention

Every animated recording is declared with a two-line block:

```markdown
<!-- gif:placeholder name="<short_name>" caption="<one-line caption>" -->
![<alt text> placeholder](docs/media/<short_name>.gif)
```

The comment line records the intended content. The image line is the path that will
resolve once the GIF is added. Paths from nested READMEs use relative `../../docs/media/`
notation.

To replace a placeholder: record the session, export as a GIF, name it `<short_name>.gif`,
and drop it in this directory. The image link resolves automatically.

---

## Registered Placeholders

| Name | Caption | Referenced in |
|------|---------|---------------|
| `training_convergence` | PPO training convergence - episode reward and success rate over 1M steps | `README.md` |
| `vis_2d` | Detachable 2D bird's-eye visualiser during a live training run | `README.md` |
| `carla_3d` | 3D CARLA spectator view - evidential policy navigating the rectangular lot | `README.md` |
| `training_curves` | PPO reward and evidential uncertainty metrics over 1 M steps | `uncertainty_rl/training/README.md` |
| `uncertainty_evolution` | Epistemic and aleatoric uncertainty during a parking episode | `uncertainty_rl/networks/README.md` |
| `parking_episode` | Bird's-eye view of a parking episode under RTK float conditions | `uncertainty_rl/envs/README.md` |
| `eval_degradation` | Success rate and epistemic uncertainty across the 9 evaluation conditions | `uncertainty_rl/evaluation/README.md` |
| `visualiser_2d` | Detachable 2D bird's-eye visualiser during a parking episode | `scripts/visualise/README.md` |
| `inspect_layout` | Layout inspector showing bay outlines, patrol path, and pedestrian zones | `scripts/inspect/README.md` |
| `inspect_sensors` | Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye | `scripts/inspect/README.md` |
| `inspect_dryrun` | Dryrun inspector: manual keyboard drive with EKF covariance output | `scripts/inspect/README.md` |

---

## Adding a New Placeholder

1. Add the two-line block to the relevant README at the point where the GIF will appear.
2. Add a row to the table above with the name, caption, and source README.
3. Do not commit a binary placeholder image - leave the link unresolved until the real
   recording is ready.
