# Plan: Implement CARLA Parking Environment (Task 5)

## Context

Task 5 requires building the full CARLA parking environment as specified in the agreed design
(7 March 2026). The current `carla_parking.py` spawns at a random CARLA spawn point and targets
`(0, 0, 0)` -not a realistic parking scenario. This plan implements the full lot geometry, bay
configs, dynamic actors, 18-dim observation space, visualisation, and all downstream propagation.

Several source files contain stale content that must be corrected as part of this task:
- `ENVIRONMENT_DESIGN.md` still specifies 6 floor plans, UK BPA dimensions, 3 bay configs per floor
  plan -all superseded by the agreed design (3 floor plans, German EAR 05, 1 config per floor plan)
- `envs/CLAUDE.md` still says 15-dimensional state, wrong NPC counts, wrong `include_covariance`
  description
- `README.md` still says `15D state` in the architecture diagram, missing new Make targets and
  visualisation sections

---

## Design References (consult both before touching any file)

- [`uncertainty_rl/envs/ENVIRONMENT_DESIGN.md`](uncertainty_rl/envs/ENVIRONMENT_DESIGN.md) -
  agreed design spec. **Must be updated as part of Step 0 before anything else is implemented.**
- [`uncertainty_rl/envs/CLAUDE.md`](uncertainty_rl/envs/CLAUDE.md) -coding rules for the envs
  module. **Must be updated in Step 7.**
- [`TODO.md`](TODO.md) -Task 5 acceptance checklist (ground truth)

---

## Key Design Decisions (agreed, canonical)

### Map -`Town05_Opt` (verify terrain in Pre-Step)

**`Town05_Opt`** is the primary choice. Town05 is a flat urban map -not mountainous like Town04.
The `_Opt` layered map API strips all buildings, foliage, and props via `world.unload_map_layer()`,
leaving just the ground mesh. You place your cone perimeter on any flat patch of that ground -you
are not using CARLA's road network or any pre-existing carpark geometry.

**Total terrain required: 3 lots x ~40m x 35m, spaced ~60m apart.** Arranged in a row this is
~220m x 35m; in an L or triangle ~160m x 100m. This is 1.5-2 city blocks of flat clear ground.
The Pre-Step flythrough confirms Town05_Opt can accommodate this before any code is written.

There is no community flat-plane map worth using. All CARLA parking research repos use existing
towns with custom Python spawn points -exactly what this project does.

**`_Opt` maps and `world.unload_map_layer()` confirmed in CARLA 0.9.16** (feature since 0.9.12).

**Pre-Step verification** (~10 minutes, before writing any code):
```python
import carla
client = carla.Client('carla-server', 2000)
client.set_timeout(10.0)
maps = client.get_available_maps()
print([m for m in maps if 'Town05' in m])  # expect /Game/Carla/Maps/Town05_Opt
# Fly spectator around Town05_Opt -- find 3 flat patches ~40m x 35m each, ~60m apart
```

If Town05_Opt terrain is unsuitable, fallback is any other flat urban `_Opt` town found in the
map list. No custom map needed regardless.

After verification: update `ENVIRONMENT_DESIGN.md` Section 2 and `configs/train_config.yaml`
`town` field with the confirmed map name.

### Floor Plans -3 total (2 training + 1 OOD)
**Replaces the 6-floor-plan spec in ENVIRONMENT_DESIGN.md Section 3.**

| Preset | Shape | Training / OOD |
|--------|-------|----------------|
| `rectangle` | Standard rectangle | Training |
| `trapezoid` | Widened at one end | Training |
| `irregular_a` | Five-sided polygon | OOD only |

Each floor plan: `origin` (x, y, z) + `heading_deg` + shape dimensions. All corners, bay centres,
spawn transforms, and patrol waypoints computed offline by `scripts/generate_lot_layout.py` and
stored in `configs/layouts/<name>.yaml`. **3 CARLA origin clicks total.**

**Why 3 floor plans (not 6):** The agent's relative observation `(dx, dy, dyaw)` makes bay position
invisible -overfitting to bay position is structurally impossible. The real diversity axes are
manoeuvre type (3 types) and lot geometry (floor plan shape). 2 training floor plans give enough
geometric variety; 1 OOD floor plan is the minimum needed for the "epistemic uncertainty rises on
novel geometry" claim.

Minimum lot size: **35 m x 30 m**.
- N-S: 8.0m (parallel depth) + 3.6m (aisle) + 5.0m (perp depth) + 6.0m (perp aisle) + 2m margin = 24.6m -> 30m
- E-W: ~25m for 5 angled bays + aisle + margins -> 35m

### Bay Standards -German EAR 05
**Replaces UK BPA 2016 in ENVIRONMENT_DESIGN.md Section 5.**

| Bay type | Width | Depth | Aisle width |
|----------|-------|-------|-------------|
| Perpendicular (90 deg) | 2.5 m | 5.0 m | 6.0 m |
| Angled (45 deg) | 2.5 m | 5.4 m | 3.6 m |
| Parallel (forward pull-in) | 2.5 m | 8.0 m | 4.0 m |

### Bay Configuration -1 Config Per Floor Plan, 5 Instances Per Type
**Replaces 3-configs-per-floor-plan spec in ENVIRONMENT_DESIGN.md Section 5.**

Each floor plan has **1 config** with exactly 5 bays of each type = **15 bays per floor plan**.

**Stratified target bay sampling:**
1. Sample bay type uniformly: perpendicular / 45-deg / parallel (1/3 each)
2. Sample one bay of that type uniformly from the 5 instances

`bay_type_sampling: stratified`, `min_bays_per_type: 5`.

Parallel bay neighbours: slots immediately in front and behind are `always_empty: true`.
Agent is not told bay type -infers the required manoeuvre from `(dx, dy, dyaw)` and reward.

**Bay count justification (references for methodology chapter):**
- Bengio et al. (2009) "Curriculum Learning" -balanced difficulty distribution (Bengio 2009)
- Florensa et al. (2017) "Reverse Curriculum Generation" -rare scenarios need deliberate exposure
- Cobbe et al. (2019) -general diversity principle only; their 4000-16000 CoinRun count does NOT
  apply here (image-based RL failure mode doesn't exist in vector observation space)
- Leurent 2018 highway-env: 28 same-type slots; this env uses 15 across 3 types + 3 floor plans

### Perimeter
`static.prop.trafficcone01` cones along polygon edges at `perimeter_cone_spacing` (~2 m).
`set_simulate_physics(False)` on all cones.

### Dynamic Actors
- **NPC patrol vehicles**: scripted `apply_control()` proportional controller. No Traffic Manager.
  Count: 0 to `num_patrol_vehicles_max` per episode.
- **Pedestrians**: `WalkerControl` random heading, re-randomised every N steps. No NavMesh needed.
  Count: 0 to `num_pedestrians_max` per episode.

### Observation Space -18-dimensional
**`constants.py` currently has `TOTAL_OBS_DIM = 15`. Must update to 18.**

| Index | Feature | Source |
|-------|---------|--------|
| 0-2 | x, y, yaw | EKF filtered pose |
| 3-5 | vx, vy, vyaw | EKF filtered pose |
| 6-8 | std_x, std_y, std_yaw | EKF covariance diagonal stdevs |
| 9-11 | cov_xx, cov_yy, cov_yawyaw | EKF covariance diagonal |
| 12-14 | cov_xy, cov_xyaw, cov_yyaw | EKF covariance off-diagonal |
| 15-17 | dx, dy, dyaw | Target bay pose in ego body frame |

`(dx, dy, dyaw)` recomputed every step from current EKF pose + fixed world-frame target bay coords:
```
dx  =  cos(yaw_ego) * (x_target - x_ego) + sin(yaw_ego) * (y_target - y_ego)
dy  = -sin(yaw_ego) * (x_target - x_ego) + cos(yaw_ego) * (y_target - y_ego)
dyaw = angle_wrap(yaw_target - yaw_ego)
```

**Ablation flag interaction:** `include_covariance: false` drops indices 6-14, giving a 9-dim obs
(6-dim EKF pose + 3-dim target). The "vanilla PPO" baseline becomes 9-dim (was 6-dim). All 4
baseline `configs/baselines/*.yaml` comment notes must be updated.

### Reward Function
Keep existing formula for now -Task 9 rewrites it. Task 5 only wires correct CARLA ground truth
distance into `_compute_reward()`.

### Visualisation -Three Views

**Training is always headless.** Three independent ways to observe the agent:

```bash
make docker-train          # headless, always

make visualise             # host-side 2D bird's-eye window (detachable, training unaffected)
make visualise-record      # same + saves MP4 on close -> outputs/recordings/<timestamp>.mp4

make docker-demo MODEL=checkpoints/final_model   # 3D windowed CARLA, checkpoint playback
```

#### 2D Bird's-Eye View (matplotlib, host-side)

Shared state file written by training every step:
```
outputs/vis_state.json   # atomic write via os.replace, negligible overhead
```

`make visualise` runs `scripts/visualise_training.py` on the host. The `outputs/` directory is
bind-mounted in `docker-compose.yml`. Close the window at any time -training unaffected.

Rendered layers (back to front):

| Layer | Shape | Colour |
|-------|-------|--------|
| Lot boundary | Filled polygon | Light grey (#DDDDDD) |
| Perimeter cones | Small circles (r=0.3m) | Red |
| Bay outlines -perpendicular | Rectangle outline | Blue |
| Bay outlines -angled 45deg | Rectangle outline | Orange |
| Bay outlines -parallel | Rectangle outline | Green |
| Target bay | Rectangle outline + heading arrow | Bright green, thick |
| Static parked vehicles | Filled rectangle (2.0m x 4.5m) | Dark grey |
| Patrol NPC vehicles | Filled rectangle (2.0m x 4.5m) | Orange |
| Pedestrians | Filled circle (r=0.4m) | Magenta |
| Ego vehicle | Filled rectangle (2.0m x 4.5m) + heading arrow | Cyan |
| Ego trajectory (last 50 steps) | Faded line | Cyan, alpha=0.4 |

**`scripts/visualise_training.py`** -host-side standalone process:
- `--record` flag: accumulates frames, saves `outputs/recordings/<timestamp>.mp4` via
  `matplotlib.animation.FFMpegWriter` at 10 fps on window close
- Polls `outputs/vis_state.json` every 100 ms (mtime check, redraws on change)
- File missing: shows "Waiting for training..." text, retries
- Window close: `sys.exit(0)` -training unaffected

**`VisStateWriter`** -add to `uncertainty_rl/utils/visualisation.py` (training-side):
```python
class VisStateWriter:
    """
    @class VisStateWriter
    @brief Writes vis_state.json every step for the detachable 2D visualiser.
    Atomic write via tmp file + os.replace so visualiser never reads partial data.
    """
    def __init__(self, output_path: Path) -> None: ...
    def write(
        self,
        ego_transform: Dict,
        actor_transforms: List[Dict],
        target_bay: Dict,
        episode_info: Dict,
    ) -> None:
        # json.dumps -> write to .tmp -> os.replace
```

#### 3D Debug Draw Overlays (CARLA spectator, live during training)

CARLA's `world.debug` API renders server-side -no display required, visible whenever any spectator
connects (e.g. via `make docker-shell`). Drawn unconditionally in `step()` via
`_draw_debug_overlays()`. All calls use `life_time=0.05` (one frame).

| Element | API | Colour |
|---------|-----|--------|
| Bay outlines -all types | `draw_box` flat at z=0.05 | Blue/orange/green by type |
| Target bay | `draw_box` + `draw_string` "TARGET" | Bright green, thicker |
| Ego vehicle box | `draw_box` around ego bounding box | Cyan |
| Ego trajectory trail | `draw_point` for last 50 positions (ring buffer) | Cyan, fading |

No NPC/pedestrian overlays -their CARLA actors are already visible in spectator.

#### Demo Mode (3D Windowed CARLA, checkpoint playback)

`docker-compose.yml` `demo` Compose profile:
- `carla-server-demo`: identical to `carla-server` but without `-RenderOffScreen`; uses host
  `DISPLAY` + `/tmp/.X11-unix:/tmp/.X11-unix` bind-mount
- `training-demo`: runs `evaluate.py --model-path $MODEL` via `PPO.load()` against windowed CARLA
- Ports: 2100-2102 (separate from training stack 2000-2002 -both can run simultaneously)

`make docker-demo MODEL=...` expands to:
```bash
xhost +local:docker
DISPLAY=$(DISPLAY) MODEL=$(MODEL) docker compose --profile demo up --abort-on-container-exit
xhost -local:docker
```

No new Python code needed: reuses existing `evaluate.py` + `PPO.load()`.

### Sensor Suite Config
Add `sensor_suite: suite_a` key to `train_config.yaml`. Task 7 activates it; Task 5 just
establishes the structure to avoid a breaking config change later.

### Termination (priority order)
1. Clearance violation: < 0.8 m to any obstacle -> terminated, collision penalty
2. Success: pos error < 0.5 m, yaw error < 10 deg, vel < 0.1 m/s -> terminated, success bonus
3. Out-of-bounds: > 20 m from target -> terminated, no bonus
4. Time limit: `max_steps` steps -> truncate

---

## Config File Strategy

**`configs/layouts/`** -new directory, one YAML per floor plan, generated offline by
`scripts/generate_lot_layout.py`. Committed to repo. Contains pre-computed world-frame coordinates
(corners, bays, spawn, patrol waypoints). Env reads from these directly.

**`configs/train_config.yaml`** -add `parking_scenarios` top-level section:
```yaml
parking_scenarios:
  perimeter_cone_spacing: 2.0
  bay_occupancy_rate: 0.70
  num_patrol_vehicles_max: 3
  num_pedestrians_max: 4
  pedestrian_heading_resample_steps: 30
  floor_plans:
    rectangle:
      layout_file: configs/layouts/rectangle.yaml
      ood: false
    trapezoid:
      layout_file: configs/layouts/trapezoid.yaml
      ood: false
    irregular_a:
      layout_file: configs/layouts/irregular_a.yaml
      ood: true   # never sampled during training

sensor_suite: suite_a   # suite_a | suite_b | suite_c (Task 7 activates this)
```

**`configs/eval_config.yaml`** -add OOD presets + `worst_case` entry (fog 80-100, zero parked
cars, max patrol + pedestrians). No code changes -only new condition values.

**`outputs/layouts/`** -PNG bird's-eye plots (not committed, add to `.gitignore`).

---

## Critical Files to Modify

| File | What Changes |
|------|-------------|
| `uncertainty_rl/envs/ENVIRONMENT_DESIGN.md` | **Full update (Step 0):** Section 3 floor plans -> 3 total (2 training + 1 OOD), sizes -> 35m x 30m min; Section 5 bay dims -> German EAR 05, 1 config per floor plan, 5 bays per type, stratified sampling, add relative-obs overfitting argument; Section 9 eval strategy -> update floor plan names; Section 13 sim-to-real -> keep (already correct) |
| `uncertainty_rl/utils/constants.py` | `TOTAL_OBS_DIM` 15 -> 18; add `TARGET_POSE_DIM = 3` |
| `uncertainty_rl/envs/carla_parking.py` | Full env implementation (see Step 3); add `VisStateWriter` instantiation + `writer.write()` + `_draw_debug_overlays()` every step |
| `uncertainty_rl/utils/visualisation.py` | Add `VisStateWriter` class (atomic write via os.replace) |
| `scripts/visualise_training.py` | **New**: host-side 2D visualiser, polls `vis_state.json`, matplotlib window, `--record` flag saves MP4 |
| `scripts/generate_lot_layout.py` | **New**: offline layout generator -shape + dims + heading + origin -> layout YAML + 2D bird's-eye PNG |
| `scripts/explore_map.py` | Add `--mark` mode: fly spectator freely, ENTER to record origin + label, print YAML snippet at exit |
| `configs/train_config.yaml` | Add `parking_scenarios` section (layout file refs + actor params) + `sensor_suite` key |
| `configs/eval_config.yaml` | Add OOD preset entries + `worst_case` entry |
| `configs/layouts/rectangle.yaml` | **New**: pre-computed layout (generated by script) |
| `configs/layouts/trapezoid.yaml` | **New**: pre-computed layout (generated by script) |
| `configs/layouts/irregular_a.yaml` | **New**: pre-computed layout (generated by script) |
| `configs/baselines/vanilla_ppo.yaml` | Comment-only: "6-dim" -> "9-dim (6 EKF pose + 3 target)" |
| `configs/baselines/input_uncertainty.yaml` | Comment-only: "15-dim" -> "18-dim" |
| `configs/baselines/output_uncertainty.yaml` | Comment-only: "6-dim" -> "9-dim (6 EKF pose + 3 target)" |
| `configs/baselines/full_method.yaml` | Comment-only: "15-dim" -> "18-dim" |
| `uncertainty_rl/envs/CLAUDE.md` | Update state space table 15-dim -> 18-dim (add target pose rows 15-17); fix `include_covariance` description ("9-dim" not "6-dim" when false); fix NPC counts to match new `parking_scenarios` config (max 3 patrol + 4 pedestrians); update `When Modifying` note to reference 18-dim |
| `uncertainty_rl/networks/evidential_policy.py` | Verify input dim uses `TOTAL_OBS_DIM` from constants, not hardcoded |
| `uncertainty_rl/evaluation/evaluate.py` | Verify state index slicing uses constants |
| `uncertainty_rl/training/train_ppo.py` | No change expected (reads obs dim from env) |
| `tests/test_carla_parking.py` | Add/update unit tests (see Step 4) |
| `documentation/references/references.bib` | Add: `bengio2009curriculum`, `florensa2017reverse`, `cobbe2019quantifying`, `leurent2018highway` |
| `documentation/references/references.tex` | Add bullet: stratified bay sampling justification (Bengio/Florensa/Leurent/Cobbe) |
| `Makefile` | Add: `generate-layouts`, `visualise`, `visualise-record`, `docker-demo` targets |
| `docker-compose.yml` | Add `demo` Compose profile: `carla-server-demo` (no `-RenderOffScreen`, X11) + `training-demo` on ports 2100-2102 |
| `.gitignore` | Add: `outputs/layouts/`, `outputs/vis_state.json`, `outputs/recordings/` |
| `README.md` | Fix `15D state` -> `18D state` in architecture diagram; add "Parking Lot Layout Generation" section; add "Visualisation" section (2D view, 3D overlays, demo mode); add `make generate-layouts`, `make visualise`, `make visualise-record`, `make docker-demo` to Docker commands table |

---

## Implementation Order

### Pre-Step -Verify Town05_Opt Has Enough Flat Terrain (~10-20 minutes)

**Total terrain needed: 3 lots x 35m x 30m each, spaced ~60m apart.**

Arranged in a row that is roughly **220m x 35m**. Arranged in an L or triangle it fits in
roughly **160m x 100m**. Either way this is ~1.5-2 city blocks of flat uninterrupted ground.

```bash
make docker-shell
# Inside container:
python -c "
import carla
client = carla.Client('carla-server', 2000)
client.set_timeout(10.0)
maps = client.get_available_maps()
print([m for m in maps if 'Town05' in m])  # expect /Game/Carla/Maps/Town05_Opt
"
```

Fly spectator around Town05_Opt and confirm:
- [ ] `Town05_Opt` appears in the map list
- [ ] At least 3 flat patches (~40m x 35m each) exist with ~60m clearance between them
      (they do not need to be in a row -L-shape or triangle arrangement is fine)
- [ ] Ground is flat enough that spawned vehicles don't slide or clip terrain

If Town05_Opt cannot fit 3 lots, check other flat urban `_Opt` towns from the map list.
Note the chosen town and 3 approximate world positions before proceeding to Step 0.

### Step 0 -Fix ENVIRONMENT_DESIGN.md (must happen first -it is the reference for all steps)

Update the following sections:

**Section 3 (Floor Plans):** Replace 6-floor-plan table with:
- 3 floor plans: `rectangle` (training), `trapezoid` (training), `irregular_a` (OOD only)
- Sizes: 35m x 30m minimum (derivation as in plan above)
- Remove `irregular_b`, `ood_a`, `ood_b`
- Update "domain randomisation" justification: remove Cobbe minimum count claim, add relative-obs
  overfitting argument

**Section 5 (Bay Configurations):**
- Bay dims: replace UK BPA 2016 table with German EAR 05 (2.5m x 5.0m, 2.5m x 5.4m, 2.5m x 8.0m)
- Reference: "German EAR 05 / EU harmonised practice (FGSV 2005)" not "UK BPA 2016"
- 1 config per floor plan (not 3), 5 bays per type = 15 bays total
- Update configs table: remove config_b/config_c rows for all floor plans
- Stratified sampling description: 1/3 per type then uniform within type
- Add references: `bengio2009curriculum`, `florensa2017reverse`, `cobbe2019quantifying`, `leurent2018highway`

**Section 9 (Evaluation Strategy):** Update floor plan names in table to match new 3-plan set.

**Section 12 (Sensor Suite):** Remove stale GNSS mention from Suite A (GNSS excluded per
ENVIRONMENT_DESIGN.md Section 12 already). Keep as-is -already correct.

### Step 1 -Constants (`uncertainty_rl/utils/constants.py`)
- `TOTAL_OBS_DIM = VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM` -> `TOTAL_OBS_DIM = 18`
  (add `TARGET_POSE_DIM = 3`; the formula becomes `VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM + TARGET_POSE_DIM`)
- Grep for any hardcoded `15` or `6` (ablation dim) in non-constant files and replace with constants

### Step 2 -Layout Generation + Config

**Step 2a -Create `scripts/generate_lot_layout.py`** (no CARLA needed):
```bash
make generate-layouts
# Writes configs/layouts/{rectangle,trapezoid,irregular_a}.yaml  (origin=0,0,0 placeholder)
# Writes outputs/layouts/{rectangle,trapezoid,irregular_a}.png   (inspect these)
```
Script interface:
```bash
python scripts/generate_lot_layout.py \
  --shape rectangle --width 35 --depth 30 --heading 0 --origin 0 0 \
  --output configs/layouts/rectangle.yaml \
  --plot outputs/layouts/rectangle.png
```
Each layout YAML contains: corners, bays (x, y, yaw, width, depth, bay_type, always_empty),
spawn_transform, patrol_waypoints, pedestrian_zones.

Bird's-eye PNG: grey lot polygon, coloured bay rectangles, yaw arrows, spawn triangle, patrol path.

**Step 2b -Record real CARLA origins** (run once, requires CARLA):
- Add `--mark` mode to `scripts/explore_map.py`: loads Town05_Opt, prints spectator (x,y,z,yaw)
  live, ENTER records with label, YAML snippet at exit
```bash
make docker-shell
python scripts/explore_map.py --mark --town Town05_Opt
# Find 3 flat open areas (~40m x 35m each, ~60m apart), ENTER to record origins
# Paste 3 origins into configs/layouts/*.yaml, then re-run make generate-layouts
```

**Step 2c -Update configs:**
- `configs/train_config.yaml`: add `parking_scenarios` section + `sensor_suite` key (see above)
- `configs/eval_config.yaml`: add OOD entries + `worst_case` entry
- `configs/baselines/*.yaml`: comment-only dim updates

### Step 3 -Environment (`uncertainty_rl/envs/carla_parking.py`)

New/changed methods:

**`__init__()`**
- `observation_space = spaces.Box(shape=(TOTAL_OBS_DIM,), ...)` -now 18
- Instantiate `self._vis_writer = VisStateWriter(Path('outputs/vis_state.json'))`
- Initialise `self._trajectory_buffer: collections.deque` (maxlen=50) for debug overlays

**`reset()`**
1. `_cleanup_actors()`
2. Sample floor plan uniformly from `ood: false` plans (training) or `ood: true` (eval)
3. Load layout YAML for selected floor plan
4. Stratified-sample target bay (1/3 per type, then uniform within type)
5. Load Town05_Opt, unload all layers except `Ground`
6. Spawn ego at floor plan `spawn_transform`
7. `_spawn_perimeter_cones()` -interpolate along polygon edges
8. `_spawn_static_vehicles()` -70% fill, skip target bay + `always_empty` slots
9. `_spawn_npc_patrol()` -scripted patrol vehicles
10. `_spawn_pedestrians()` -random-walk walkers
11. Start `_CovarianceSubscriber` thread (existing)
12. Cache `self._target_bay` world-frame (x, y, yaw)

**`_get_state()`**
- Indices 0-14: unchanged (EKF pose + covariance)
- Indices 15-17: `(dx, dy, dyaw)` -recomputed every step (formula above)

**`step()`**
- After `_get_state()`: check clearance to all obstacle actors -> terminate if < 0.8 m
- Check success condition (CARLA ground truth)
- Check out-of-bounds (> 20 m from target ground truth position)
- At end: `self._vis_writer.write(...)` then `self._draw_debug_overlays()`

**`_compute_reward()`**
- Use CARLA ground truth `vehicle.get_transform()` for position/orientation error
- Keep existing formula -Task 9 rewrites it

**`_interpolate_cone_positions(corners, spacing)`** -pure geometry, unit-testable:
```python
def _interpolate_cone_positions(
    corners: List[Tuple[float, float]],
    spacing: float
) -> List[Tuple[float, float]]:
```

**`_spawn_npc_patrol()`**: proportional controller cycling through `patrol_waypoints`,
`steer = k_p * angle_to_next_waypoint`, `apply_control()` each step.

**`_spawn_pedestrians()`**: `WalkerControl(direction, speed=1.2)`, re-randomise heading every
`pedestrian_heading_resample_steps` steps.

**`_draw_debug_overlays()`** -new private method:
- Draw bay outlines: `world.debug.draw_box(carla.BoundingBox(...), ...)` flat at z=0.05,
  colour by type, `life_time=0.05`
- Target bay: same but bright green + `draw_string("TARGET")`
- Ego box: `draw_box` around ego bounding box, cyan
- Ego trail: `draw_point` for each position in `_trajectory_buffer`, cyan

**Note:** `_generate_floor_plan_corners()` and `_generate_bay_positions()` live in
`scripts/generate_lot_layout.py`, NOT in the env. The env reads pre-computed values from
`configs/layouts/<floor_plan>.yaml` directly.

### Step 3b -VisStateWriter (`uncertainty_rl/utils/visualisation.py`)

Add `VisStateWriter` alongside existing classes:

```python
class VisStateWriter:
    """
    @class VisStateWriter
    @brief Writes vis_state.json every step for the detachable 2D visualiser.
    Atomic write via tmp file + os.replace -- visualiser never reads partial data.
    """
    def __init__(self, output_path: Path) -> None: ...
    def write(
        self,
        ego_transform: Dict,
        actor_transforms: List[Dict],
        target_bay: Dict,
        episode_info: Dict,
    ) -> None:
        # 1. Build state dict
        # 2. json.dumps to string
        # 3. Write to output_path.with_suffix('.tmp')
        # 4. os.replace(tmp, output_path) -- atomic on POSIX
```

Add `scripts/visualise_training.py`:
- `--record` flag
- Polls `vis_state.json` every 100 ms (mtime check)
- Redraws all layers in order (lot polygon -> cones -> bay outlines -> target bay ->
  static vehicles -> patrol NPCs -> pedestrian dots -> ego rect + arrow + trail)
- `plt.pause(0.1)` between polls
- File missing: "Waiting for training..." text, retry
- Window close: `sys.exit(0)`; if `--record`: write MP4 first via `FFMpegWriter` at 10 fps

### Step 4 -Tests (`tests/test_carla_parking.py`)

Unit tests (all `not integration`, no CARLA connection):
- `test_interpolate_cone_positions_rectangle` -known corners, expected count
- `test_interpolate_cone_positions_spacing` -all adjacent cones within spacing + tolerance
- `test_relative_target_pose_identity` -ego at target -> (0, 0, 0)
- `test_relative_target_pose_ahead` -target directly ahead -> dx > 0, dy = 0
- `test_relative_target_pose_rotation` -verify angle_wrap
- `test_bay_sampling_respects_always_empty` -target never `always_empty: true`
- `test_parallel_neighbours_always_empty` -parallel bay's front/rear never occupied
- `test_obs_shape_18_dim` -mock env returns shape (18,)
- `test_obs_shape_9_dim_no_cov` -`include_covariance: false` -> shape (9,)
- `test_vis_state_writer_atomic` -`VisStateWriter.write()` produces valid JSON; tmp file cleaned up
- `test_vis_state_writer_no_display` -write + read cycle produces expected ego/actor keys

### Step 5 -References (`documentation/references/`)

Add to `references.bib`:
```bibtex
@inproceedings{bengio2009curriculum,
  title     = {Curriculum Learning},
  author    = {Bengio, Yoshua and Louradour, Jerome and Collobert, Ronan and Weston, Jason},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {41--48},
  year      = {2009},
  note      = {Justifies stratified bay type sampling: balanced difficulty distribution
               leads to faster convergence than natural skewed distribution.},
}

@inproceedings{florensa2017reverse,
  title     = {Reverse Curriculum Generation for Reinforcement Learning},
  author    = {Florensa, Carlos and Held, David and Wulfmeier, Markus and Abbeel, Pieter},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2017},
  note      = {Justifies stratified bay sampling: rare difficult scenarios (parallel bays)
               are never practiced enough without deliberate curriculum design.},
}

@inproceedings{cobbe2019quantifying,
  title     = {Quantifying Generalisation in Reinforcement Learning},
  author    = {Cobbe, Karl and Klimov, Oleg and Hesse, Christopher and Kim, Taehoon and
               Schulman, John},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2019},
  note      = {General instance diversity principle. Specific count thresholds (4000-16000
               CoinRun levels) do not transfer to vector-state parking -- cite only for
               the general principle.},
}

@misc{leurent2018highway,
  title  = {An Environment for Autonomous Driving Decision-Making},
  author = {Leurent, Edouard},
  year   = {2018},
  url    = {https://github.com/eleurent/highway-env},
  note   = {highway-env ParkingEnv: 28 same-type slots. Direct precedent for stratified
            bay type sampling across 3 manoeuvre types.},
}
```

Add bullet to `references.tex` under "Environment Design / Curriculum" section explaining
stratified sampling justification, relative-obs overfitting argument, and Cobbe scope limitation.

### Step 6 -README (`README.md`)

**Fix stale content:**
- Architecture diagram: `15D state` -> `18D state`

**Add to Docker commands table:**
```
| `make generate-layouts`   | Generate lot layout YAMLs + bird's-eye PNGs (no CARLA) |
| `make visualise`          | Open 2D bird's-eye view (detachable, training unaffected) |
| `make visualise-record`   | Same + saves MP4 on close |
| `make docker-demo MODEL=` | Windowed 3D CARLA demo with checkpoint (requires X11) |
```

**Add "Parking Lot Layout Generation" section:**
```markdown
## Parking Lot Layout Generation

Lot geometry (bay positions, perimeter corners, spawn transforms) is pre-computed offline
and stored in `configs/layouts/`. To regenerate or modify layouts:

### Step 1 -Generate from shape dimensions (no CARLA needed)
make generate-layouts
# Writes configs/layouts/{rectangle,trapezoid,irregular_a}.yaml
# Writes outputs/layouts/{rectangle,trapezoid,irregular_a}.png
# Inspect the PNGs to confirm bay placement and aisle clearances.

### Step 2 -Record CARLA world-frame origins (run once per floor plan)
make docker-shell
python scripts/explore_map.py --mark --town Town05_Opt
# Fly spectator to a flat open area, ENTER to record origin (x, y, z).
# Record 3 origins, paste into configs/layouts/*.yaml, re-run Step 1.
```

**Add "Visualisation" section:**
```markdown
## Visualisation

### 2D bird's-eye view (detachable, zero training overhead)
Training always runs headless. Attach the visualiser at any time from the host:

  make visualise          # live window -- close to detach, training unaffected
  make visualise-record   # live window + saves MP4 on close
  #   outputs/recordings/YYYY-MM-DD_HH-MM-SS.mp4

Shows: lot boundary, bay outlines (blue=perpendicular, orange=angled, green=parallel),
target bay (bright green), static vehicles (dark grey), patrol NPCs (orange),
pedestrians (magenta), ego vehicle (cyan) with heading arrow and 50-step trail.

### 3D overlays (CARLA spectator, live during training)
While training runs headless, connect the CARLA spectator (make docker-shell, fly camera)
to see real-time debug overlays drawn every step:
  - Bay outlines: colour-coded flat boxes (blue/orange/green by type)
  - Target bay: bright green box + "TARGET" label
  - Ego vehicle: cyan bounding box
  - Ego trajectory: last 50 positions as cyan fading dots

No configuration needed -- overlays are always active when the CARLA server is running.

### 3D demo mode (windowed CARLA, checkpoint playback)
For formal demonstrations. Requires X11 on host. Does not affect training.

  make docker-demo MODEL=checkpoints/final_model

Starts a fresh windowed CARLA server on a separate port (2100-2102), loads the
checkpoint via PPO.load(), and runs evaluation. All overlays are visible.
```

### Step 7 -Update envs/CLAUDE.md

Update the following sections:
- **State Space section header**: "15-dimensional" -> "18-dimensional"
- **State space table**: add rows 15-17 (`dx, dy, dyaw`, Target pose (relative))
- **`include_covariance` description**: "9-dim (pose only, no ROS 2 subscription)" not "6-dim"
- **NPC counts**: update from old "0-60 cars, 0-40 walkers" to new "0-3 patrol, 0-4 pedestrians"
  to match `parking_scenarios` config values
- **`When Modifying` note**: "15-dimensional" -> "18-dimensional"; update constants list to include
  `TARGET_POSE_DIM`

### Step 8 -Downstream consumers
- `uncertainty_rl/networks/evidential_policy.py`: verify input dim uses `TOTAL_OBS_DIM` import
- `uncertainty_rl/evaluation/evaluate.py`: verify state index slicing uses constants
- `configs/baselines/*.yaml`: comment-only dim updates (see table above)

---

## Coordinate Generation + Measurement Workflow

```bash
# Step 1: Generate layout YAMLs + bird's-eye PNGs offline (no CARLA)
make generate-layouts
# Writes configs/layouts/{rectangle,trapezoid,irregular_a}.yaml (origin=0,0,0 placeholder)
# Writes outputs/layouts/{rectangle,trapezoid,irregular_a}.png
# Inspect PNGs: confirm bay placement, aisle clearances, spawn position.

# Step 2: Record real CARLA origins (3 clicks)
make docker-shell
python scripts/explore_map.py --mark --town Town05_Opt
# Find 3 flat open areas (~40m x 35m each, ~60m apart), ENTER to record.
# Paste 3 origin values into configs/layouts/*.yaml (replace 0,0,0 placeholder).

# Step 3: Re-generate with real origins
make generate-layouts
# Inspect updated PNGs to confirm lot placement in world space.

# Step 4: Verify in CARLA spectator after first env reset()
```

---

## Verification

```bash
# CPU-only checks (no CARLA):
make verify

# Unit tests inside container:
make docker-test-unit

# Integration smoke test (requires CARLA running):
# Inside make docker-shell:
# python -c "
# from uncertainty_rl.envs.carla_parking import CARLAParkingEnv
# env = CARLAParkingEnv(...)
# obs, info = env.reset()
# assert obs.shape == (18,), f'Expected (18,), got {obs.shape}'
# obs, r, term, trunc, info = env.step(env.action_space.sample())
# assert obs.shape == (18,)
# print('Smoke test passed')
# "
```

**Task 5 acceptance criteria (from TODO.md):**
- [ ] Map choice verified via `client.get_available_maps()` + spectator flythrough; `<TOWN_OPT>` placeholder replaced with chosen town in all files
- [ ] Vehicle spawns at floor plan entrance; perimeter cones visible in spectator
- [ ] Static parked vehicles present in non-target bays at ~70% occupancy
- [ ] Target bay randomly selected each episode; relative target pose in observation [15-17]
- [ ] `TOTAL_OBS_DIM = 18` and all downstream consumers updated
- [ ] Collision termination fires correctly (< 0.8 m clearance)
- [ ] All 3 bay types reachable from spawn in their respective configs
- [ ] Floor plan and bay config selectable via YAML
- [ ] `make verify` passes (all CPU-only checks green)
- [ ] `ENVIRONMENT_DESIGN.md` reflects agreed design (German EAR 05, 3 floor plans, 1 config each)
- [ ] `envs/CLAUDE.md` reflects 18-dim state space
- [ ] `README.md` reflects 18D state, new Make targets, visualisation section
