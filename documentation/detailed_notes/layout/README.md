# layout

Geometry derivations and design rationale for the three parking-lot layouts used in simulation:
`rectangle`, `trapezoid`, and `irregular_a`.

## Why a subfolder?

Each layout type has its own coordinate-system derivation, patrol-path geometry, bay-sampling
logic, and cone-ring construction. Grouping them here keeps the top-level `detailed_notes/`
directory uncluttered while making it easy to find layout-specific explanations alongside each
other.

## Files (added as checkpoints progress)

| File | Topic |
|------|-------|
| `patrol_paths.md` | Waypoint derivation for NPC patrol routes in each layout |
| `bay_sampling.md` | Probability-weighted target-bay sampling across layout types |
| `cone_interpolation.md` | Perimeter cone spacing and interpolation along lot boundary edges |

## Source files

Content in this subfolder is extracted from:
- `scripts/layouts/rectangle.py`, `trapezoid.py`, `irregular_a.py`, `common.py`
- `uncertainty_rl/envs/sim/_npc_controller.py` (patrol path geometry)
- `uncertainty_rl/envs/sim/_lot_spawner.py` (bay sampling, cone placement)

## See also

- `documentation/detailed_notes/observation_space.md` - LiDAR sector derivation that depends
  on lot geometry.
- `scripts/layouts/CLAUDE.md` - authoring rules for layout scripts.
