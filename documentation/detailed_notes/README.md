# detailed_notes

Long-form explanations, derivations, and design rationale that have been moved out of source-code
docstrings to keep inline comments concise. Prose that belongs in the dissertation, a supervisor
discussion, or a design record lives here; code files contain only a short pointer back.

## Purpose

Every `.md` file here covers one topic that was previously embedded as a multi-paragraph comment
inside a `.py` or `.yaml` file. Keeping long explanations here lets the code stay navigable while
preserving the full reasoning for future reference.

## Contributing

- One topic per file. Use a clear, lowercase filename (e.g. `ekf_pipeline.md`).
- Begin each file with a one-line `# Title` and a short paragraph saying which source file(s)
  the content was extracted from and why it lives here rather than there.
- Cross-reference from code using: `# See documentation/detailed_notes/<filename>.md`
- British English, ASCII-only characters throughout.
- Files are created on demand as checkpoints reach them; not every topic listed in the project
  plan will necessarily exist.

## Layout subfolder

`layout/` contains geometry derivations specific to the three parking-lot layouts (rectangle,
trapezoid, irregular_a). It is the only nested subfolder; everything else is a flat `.md` file
at the top level of this directory.

## Cross-reference index

| File | Extracted from | Topic |
|------|---------------|-------|
| `evidential_nig_initialisation.md` | `networks/evidential_policy.py` `EvidentialLayer.__init__` | NIG hyperprior bias derivation and ortho_init interaction |
| `observation_space.md` | `envs/_parking_core.py`, `envs/CLAUDE.md` | 12-dim obs layout, LiDAR sector boundaries, covariance features |
| `ros2_architecture.md` | `envs/covariance_subscriber.py`, `ros2/` | DDS-bypass via shared JSON, atomicity, sequence-number guard |
| `layout/patrol_paths.md` | `envs/sim/_npc_controller.py` | Proportional heading controller, waypoint loop, avoidance cone geometry |
| `real_world_deployment.md` | `envs/real/deployment_utils.py`, `envs/real/inference_loop.py` | Sensor data flow, EKF frame calibration, surveyed datum, actuation calibration, GNSS/IMU driver notes |

## See also

- Root `CLAUDE.md` - project-wide coding and commenting standards.
- `documentation/CLAUDE.md` - documentation and writing conventions.
