# scripts/visualise/

Detachable 2D bird's-eye Pygame visualiser for CARLA parking training and evaluation. Runs on the host machine — no CARLA connection, no Docker, no GPU required.

## How it works

The environment writes one JSON line per step to `outputs/vis_history.jsonl` (atomic append via `_write_vis_state()` in `carla_parking.py`) whenever the signal file `outputs/.vis_active` exists. The visualiser creates that file on start and removes it on close, so the env only writes frames while the window is open.

The visualiser tails the JSONL file in real-time, groups lines into episodes, and renders each frame with Pygame. The static scene (lot boundary, bay outlines, parked vehicles) is pre-rendered to a surface once per episode; only dynamic actors (ego, patrol NPCs, pedestrians, trail) are redrawn each frame via blitting.

## Running

```bash
make visualise               # Live window during training
make eval-visualise-2d       # Load checkpoint + headless CARLA + live window
```

Both commands use the host's `python3` from `.venv-vis/` (Pygame + numpy only — no PyTorch, no gymnasium).

## File protocol

| File | Written by | Read by |
|------|-----------|---------|
| `outputs/vis_history.jsonl` | `carla_parking.py` (`_write_vis_state()`) | `LiveVisualiser._read_new_frames()` |
| `outputs/.vis_active` | `LiveVisualiser.__init__()` | `carla_parking.py` (guards writes) |

Each JSONL line is a complete frame dict with keys:

| Key | Type | Contents |
|-----|------|----------|
| `episode_id` | int | Monotonic episode counter |
| `episode_step` | int | Step index within episode |
| `sim_time` | float | Accumulated simulation time (s) |
| `floor_plan` | str | Layout name (rectangle, trapezoid, ...) |
| `ego` | dict | `{x, y, yaw}` in world frame (CARLA convention) |
| `actors` | list | NPC vehicles: `{x, y, yaw, type}` (`type="npc"` for patrol, `"static"` for parked) |
| `pedestrians` | list | `{x, y}` for each pedestrian |
| `target_bay` | dict | `{bay_id, x, y, yaw, bay_type}` |
| `bays` | list | All bays in the current layout |
| `corners` | list | Lot perimeter polygon `{x, y}` vertices |
| `trajectory` | list | `[x, y]` pairs for trail (last N steps) |
| `end_reason` | str | Present on the last frame only: `"success"`, `"collision"`, `"oob"`, `"timeout"` |
| `debug` | dict | Optional: `{pos_err, yaw_err_deg, speed, reward, cov_rms, ekf_drift, lidar_pts, obs_dist, steer, throttle, brake}` |

## Layers drawn (back to front)

1. Lot boundary polygon (light grey fill)
2. Bay outlines by type: perpendicular=blue, angled=yellow, parallel=violet
3. Target bay (bright green, thick outline + heading arrows for nose-in and nose-out)
4. Static parked vehicles (orange rectangles)
5. Patrol NPC vehicles (red rectangles)
6. Pedestrians (teal circles)
7. Ego trajectory trail (faded cyan, capped at 500 points)
8. Ego vehicle (cyan rectangle + heading arrow)
9. HUD bar (floor plan, episode, step, sim time, episode history position)
10. Debug HUD bar (position error, yaw error, speed, reward, covariance, actions) — only when `debug` key is present
11. Legend panel (right-hand side, static)

## Controls

| Key | Action |
|-----|--------|
| Right arrow | Next episode in history |
| Left arrow | Previous episode in history |
| F | Toggle fullscreen |
| ESC / Q | Exit (removes signal file, deletes vis_history.jsonl) |

## Package structure

| File | Purpose |
|------|---------|
| `visualiser.py` | `LiveVisualiser` class — JSONL tailing, Pygame rendering, viewport calculation |
| `__main__.py` | CLI entry point (`python -m scripts.visualise [--history-file PATH]`) |
| `__init__.py` | Package init — sets non-interactive Matplotlib backend (guard for any indirect import) |

## Dependencies

Pygame and numpy only. Both are installed in the host `.venv-vis/` virtual environment (separate from the training container). Colours are imported from `scripts/colours.py` (single source of truth for the full visualisation palette).

## Window dimensions

- Map viewport: 900 x 900 px
- Legend panel: 180 px wide (right-hand side)
- Total window: 1080 x 900 px
- FPS cap: 120 (well above CARLA sim rate of 20 Hz)
