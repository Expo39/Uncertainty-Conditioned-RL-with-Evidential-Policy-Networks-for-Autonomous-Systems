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
resolves automatically. An asset may be shown in more than one README - reference the same
file rather than committing a second copy of the same bytes.

### Sizing (do not skip)

Markdown image syntax renders at the file's native pixel width, so a print-resolution
figure swamps the page. The figure pipeline writes at **400 DPI**
(`DPI` in `scripts/figure_style.py`), around 2900 px wide - several times the width of a
README's text column.

Two things are needed, and **both** matter:

1. **Scale the file** to about 900 px wide, so the repo is not carrying megabytes of
   pixels no one sees:

   ```bash
   ffmpeg -i outputs/main_analysis/figures/<fig>.png \
     -vf "scale=900:-1:flags=lanczos" docs/media/<short_name>.png
   ```

2. **Embed with an explicit width**, not `![...](...)`. This is what actually constrains
   the rendered size, and it is the step that gets forgotten:

   ```html
   <p align="center">
     <img src="../../docs/media/<short_name>.png" alt="<what it shows>" width="620">
   </p>
   ```

   The `<p align="center">` wrapper centres the figure in the column; a constrained
   image left-aligned against the text reads as a stray screenshot.

   620 px sits comfortably inside a GitHub README column. Near-square figures (the
   two-panel plots here are about 1:1) need the narrower end of that range, since width
   drives height too; a wide, short figure can take 700-760.


Keep the full-resolution original in `outputs/` - that is the one to cite in the
dissertation, where a fixed `\includegraphics` width makes the DPI an asset rather than a
problem. GIFs from `make clip` take `WIDTH=800`, which is already sized for a README, but
still give them an explicit `width` when embedding.

Give every image a lead-in sentence saying what it shows, and write real alt text rather
than "placeholder", so the page reads as prose with a figure in it.

---

### How each asset is produced

Assets fall into three groups. Run `make check-host-deps` first - every route needs the
host `ffmpeg` binary.

**Group A - the 2D viewer records itself.** `visualiser_2d`. The viewer owns its Pygame
surface, so `RECORD=true` (or the `R` key) captures it directly:

```bash
make eval-visualise-2d LAYOUT=rectangle BASELINE=full_method \
  CHECKPOINT=6_42_22062026-1502 STAGE=6 REALTIME=true RECORD=true TRACE=false
make clip START=00:05 END=00:20 FORMAT=gif WIDTH=800 FPS=15
```

`GNSS_TIER=fixed|float|standalone|degraded` pins one fix state for a whole drive, so the
localisation level is a controlled constant; leave it empty for the live Markov drift.

**Group B - screen capture.** `carla_3d`, `inspect_live`. These are drawn by the CARLA
server into its own Unreal window (the inspectors use CARLA's server-side debug API), so
nothing in this repo can record them from the inside. `make record-screen` grabs the
window off the X display:

```bash
xhost +local:docker
make docker-inspect-live                              # or the target for the asset
xwininfo -name CarlaUE4                               # read "Absolute upper-left X/Y"
make record-screen DURATION=30 REGION=800x600 OFFSET=<X>,<Y>
make clip START=00:02 END=00:14
```

The CARLA window is launched at 800x600, which is the default `REGION`.

**Group C - static plots, already generated or a single screen grab.**
`training_curves`, `eval_degradation` come from the figure pipeline, not a recording; copy
the PNG out of `outputs/main_analysis/figures/`. Both are already in place. `inspect_sensors`
is a single frame off the CARLA window (same `xhost`/`xwininfo` setup as Group B, then a
still screenshot instead of `record-screen`).

---

## Registered Placeholders

### GIFs (animated recordings)

| Name | Caption | Shown in |
|------|---------|----------|
| `carla_3d` | **Present.** Chase view across an episode boundary, parking into a different highlighted bay each time | `README.md` |
| `visualiser_2d` | **Present.** Parking as the GNSS fix state climbs degraded -> standalone -> float -> RTK fixed | `README.md`, `scripts/visualise/README.md` |

### PNGs (static plots)

| Name | Caption | Shown in |
|------|---------|----------|
| `training_curves` | Success and collision rate per arm across the six curriculum stages | `uncertainty_rl/training/README.md` |
| `eval_degradation` | Success rate and mean final position error per arm across the reported evaluation conditions | `uncertainty_rl/evaluation/README.md` |
| `inspect_sensors` | **Present.** Sensor inspector showing GNSS, IMU, and LiDAR FOV arc from birds-eye | `scripts/inspect/README.md` |

`training_curves` and `eval_degradation` are copied from `outputs/main_analysis/figures/`
(see each README for the two-command regeneration recipe). `inspect_sensors` is a
screen-grab still.

**Before filling any placeholder, check the caption against the result it implies.** An
asset is not neutral illustration when its caption makes a claim.

---

## Status: what can be generated today

| Asset | Route | Status |
|-------|-------|--------|
| `visualiser_2d` | A: `make eval-visualise-2d RECORD=true` | **Done** - in `docs/media/` |
| `carla_3d` | B: `make docker-eval-visualise-3d` + `make record-screen` | **Done** - in `docs/media/` |
| `inspect_sensors` | C: `make docker-inspect-sensors SENSORS_VIEW=birds_eye INSPECT_ZOOM=close` | **Done** - in `docs/media/` |
| `inspect_live` | B: `make docker-inspect-live` | Ready |
| `training_curves` | C: `make training-curves` then `make figures FIG=training_curves` | **Done** - in `docs/media/` |
| `eval_degradation` | C: `make figures FIG=ablation_by_condition` | **Done** - in `docs/media/` |

### Known gaps in the capture path

- `demo_drive.py --render` is a **no-op**: it calls `env.render()`, which only acts when
  `render_mode == "human"`, and `make_env` never passes `render_mode`
  (`uncertainty_rl/envs/factory.py`). So the CARLA spectator does not auto-follow the ego
  and must be aimed by hand. Passing `render_mode` through the factory would revive the
  birds-eye follow at `carla_parking.py:2514`.
- `make docker-demo MODEL=...` silently ignores `MODEL`: the recipe exports `MODEL` but the
  compose service reads `${CHECKPOINT}`. Use `make docker-eval-visualise-3d` instead, which
  sets `CHECKPOINT` correctly.
- `INSPECT_SENSOR` is dead - `lot_inspector.py` has no `--sensor` argument and live mode is
  always the 2D LiDAR.
- No RGB camera sensor is ever spawned, so CARLA's native `save_to_disk()` frame dump is not
  available; screen capture is the only route for Groups B and C's inspector stills.

---

## Adding a New Placeholder

1. Add the two-line block to the relevant README at the point where the asset will appear.
2. Add a row to the matching table above with the name, caption, and source READMEs.
3. Do not commit a binary placeholder image - leave the link unresolved until the real
   recording or plot is ready.
