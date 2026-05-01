# patrol_paths.md

Design rationale for NPC patrol vehicle routing in `uncertainty_rl/envs/sim/_npc_controller.py`.

## Why a proportional heading controller, not CARLA Traffic Manager

The parking lots are generated as flat-plane OpenDRIVE worlds with no road topology.
CARLA's Traffic Manager requires a road network to plan routes; it cannot operate
off-road. The proportional controller (`update_patrol()`) navigates via a list of
(x, y) waypoints stored in each layout YAML, using:

```
steer = clamp(k_p * heading_error, -1, 1)
```

where `heading_error = atan2(sin(target_yaw - npc_yaw), cos(target_yaw - npc_yaw))`
is the angular error to the next waypoint. `k_p` is `patrol_heading_gain` from
`configs/deployment/sim/env_config.yaml`.

## Waypoint loop direction

Each patrol NPC draws a random direction (+1 or -1) at spawn, which controls whether
it cycles the waypoint list forward or backward. This produces CW and CCW
simultaneously when multiple NPCs are active, increasing the variety of dynamic
obstacle patterns seen during training.

The closing duplicate waypoint (waypoint[-1] == waypoint[0]) is stripped at spawn so
the cyclic modulo wrap works correctly with a single `% len(waypoints)`.

## Ego and obstacle avoidance

Avoidance uses an angular forward cone (60-degree half-angle) rather than a
rectangular lateral cutoff:

```
fwd_proj = dot(to_target, forward_vec)
blocked if fwd_proj / dist > 0.5   # cos(60 deg) = 0.5
```

A rectangular lateral threshold misses oblique approaches; the cone check stops the
patrol for any ego within 60 degrees of the patrol vehicle's forward direction.

When blocked, the NPC is pinned via `enable_constant_velocity(Vector3D(0,0,0))`
rather than `set_target_velocity(0)`. The constant-velocity API bypasses the physics
engine and stops the vehicle instantly; `set_target_velocity` defers to the physics
engine and takes multiple ticks to converge to zero.

## Pedestrian avoidance

Pedestrian avoidance uses the same forward-cone geometry but with a separate
`patrol_pedestrian_distance` threshold (wider than the vehicle threshold to give
pedestrians more clearance).

## Patrol waypoints per layout

Waypoints are pre-computed in the layout scripts and stored in
`configs/layouts/<layout_name>.yaml` under the `patrol_waypoints` key. Each waypoint
is an (x, y) pair in CARLA's left-handed world frame.

See `scripts/layouts/rectangle.py`, `trapezoid.py`, `irregular_a.py` for derivations.
