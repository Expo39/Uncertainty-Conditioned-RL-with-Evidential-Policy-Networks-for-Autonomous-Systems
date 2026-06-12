# docs/media/

Visual assets for the project documentation. This directory seeds the relative paths
used by GIF placeholder blocks across all READMEs so that links do not 404 in repo
browsers before recordings are captured.

---

## Placeholder Convention

Every visual asset is declared with a two-line block. Animated recordings use
`gif:placeholder` with a `.gif` path; static plots use `img:placeholder` with a `.png`
path:

```markdown
<!-- gif:placeholder name="<short_name>" caption="<one-line caption>" -->
![<alt text> placeholder](docs/media/<short_name>.gif)
```

The comment line records the intended content. The image line is the path that will
resolve once the asset is added. Paths from nested READMEs use relative
`../../docs/media/` notation.

To replace a placeholder: record the session (or export the plot), name the file
`<short_name>.gif` / `<short_name>.png`, and drop it in this directory. The image link
resolves automatically. A name referenced from several READMEs is one shared file.

---

## Registered Placeholders

### GIFs (animated recordings)

| Name | Caption | Referenced in |
|------|---------|---------------|
| `visualiser_2d` | Detachable 2D bird's-eye visualiser during a parking episode | `README.md`, `scripts/README.md`, `scripts/visualise/README.md` |
| `carla_3d` | 3D CARLA spectator view - evidential policy navigating the rectangular lot | `README.md` |
| `gnss_degradation` | Same bay attempted under RTK fixed vs degraded GNSS - EKF covariance growth side by side | `README.md` |
| `parking_episode` | Bird's-eye view of a parking episode under RTK float conditions | `uncertainty_rl/envs/README.md` |
| `baseline_comparison` | Vanilla PPO vs full method side by side under degraded GNSS | `uncertainty_rl/evaluation/README.md` |
| `inspect_layout` | Layout inspector showing bay outlines, patrol path, and pedestrian zones | `scripts/inspect/README.md` |
| `inspect_sensors` | Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye | `scripts/inspect/README.md` |
| `inspect_live` | Live LiDAR inspector - red scan return dots in the CARLA world from birds-eye | `scripts/inspect/README.md` |
| `inspect_dryrun` | Dryrun inspector: manual keyboard drive with EKF covariance output | `scripts/inspect/README.md` |

### PNGs (static plots)

| Name | Caption | Referenced in |
|------|---------|---------------|
| `training_curves` | PPO training convergence - episode reward, success rate, and evidential uncertainty metrics | `README.md`, `uncertainty_rl/README.md`, `uncertainty_rl/training/README.md` |
| `uncertainty_evolution` | Epistemic and aleatoric uncertainty during a parking episode | `uncertainty_rl/networks/README.md` |
| `eval_degradation` | Success rate and epistemic uncertainty across the 9 evaluation conditions | `uncertainty_rl/evaluation/README.md` |

---

## Adding a New Placeholder

1. Add the two-line block to the relevant README at the point where the asset will appear.
2. Add a row to the matching table above with the name, caption, and source READMEs.
3. Do not commit a binary placeholder image - leave the link unresolved until the real
   recording or plot is ready.
