# configs/deployment/real/

Real-vehicle deployment configs. These are site- and vehicle-specific: the values here
are placeholders, and each is set from the survey and calibration run for the site being
deployed to. The deployment path loads the `real_world_datum` and `actuation_calibration`
files referenced in `configs/deployment/agent_config.yaml` to calibrate the EKF frame and
actuators on the real vehicle.

**Status: never run on hardware.** The deployment path is implemented and unit-tested but
has never driven a physical vehicle, so none of it is empirically validated. Four things
are outstanding before a first trial: the LiDAR hardware callback, the vehicle command
interface, the surveyed lot datum, and the actuator calibration. The datum and calibration
ship as placeholders that reduce to identity transforms. See
[uncertainty_rl/envs/real/README.md](../../../uncertainty_rl/envs/real/README.md).

## Files

| File | Purpose | Status |
|------|---------|--------|
| `real_world_datum.yaml.example` | Surveyed lot datum for EKF frame calibration | Template - copy to `real_world_datum.yaml` and set from the site survey. The copy is gitignored and is not tracked |
| `actuation_calibration.yaml` | Per-actuator gain, deadband, deadband offset, bias and output clamp | Identity mapping; set from the vehicle's measured response curve |
| `mission.yaml` | Target bay and layout file for a deployment run | Placeholder `target_bay_id`; set before each run |

Only the `.example` is committed. `real_world_datum.yaml` itself is listed in
`.gitignore`, because the datum is per-site and must not be shared between deployments.

## Pre-deployment checklist

Before any real-vehicle deployment run:

- [ ] `cp real_world_datum.yaml.example real_world_datum.yaml` and fill in the reference marker's geodetic and UTM position, its `heading_deg`, and its `lot_x` / `lot_y` in the layout frame.
- [ ] Run actuation calibration and update `actuation_calibration.yaml` with measured gains and deadbands.
- [ ] Set `mission.yaml`: correct `target_bay_id` and `layout_file` for this run.
- [ ] Confirm `agent_config.yaml`'s `baseline:` names the baseline the deployed checkpoint was trained as (its `include_covariance` / `include_obstacle_obs` / `policy_type` must match the checkpoint).
- [ ] Confirm `agent_config.yaml`'s `real_world_datum` / `actuation_calibration` point at the filled-in files for this site.
- [ ] Set `model_path` in `agent_config.yaml` to the checkpoint to deploy. The key is absent from the shipped file, so it otherwise falls back to `checkpoints/final_model`.
- [ ] Confirm RTK-GNSS antenna has clear sky visibility from the deployment site.
- [ ] Wire the two hardware-specific stubs: `RealWorldInferenceLoop._get_lidar_scan()` and the vehicle command interface. Neither has been run against hardware.

`real_vehicle.launch.py` refuses to start while `datum_lat` and `datum_lon` are both
`0.0`, so an unfilled datum fails loudly rather than silently deploying an identity
transform. The actuation calibration has no such guard: an unfilled
`actuation_calibration.yaml` is a valid identity mapping and passes silently.

Appendix B of `docs/AntonioGaldes_Dissertation.pdf` is the canonical pre-deployment
sequence; this checklist is the config-side subset of it.

## `real_world_datum.yaml`

Surveyed reference marker used to align the EKF odometry frame with the lot layout frame.
`RealWorldDeployment.from_config()` (in `uncertainty_rl/envs/real/deployment_utils.py`)
loads the file and exposes the pose via `reference_pose()`;
`calibrate_ekf_frame_offset()` (in `uncertainty_rl/envs/_parking_core.py`) consumes that
pose to build the odom-to-world 2D rigid transform. `real_vehicle.launch.py` reads the
same file independently for the flat-earth projection datum.

The file `real_world_datum.yaml` does not exist until you create it from the template:

```bash
cp configs/deployment/real/real_world_datum.yaml.example \
   configs/deployment/real/real_world_datum.yaml
```

Then fill in the surveyed values for the site. `real_world_datum.yaml` is per-site and
is not committed.

All keys are nested under a top-level `datum:` mapping:

| Key | Default | Purpose |
|-----|---------|---------|
| `datum_lat`, `datum_lon` | `0.0` | Geodetic position of the reference marker. Passed by `real_vehicle.launch.py` to the `sensor_relay` node as the flat-earth projection origin. Both left at `0.0` aborts the launch |
| `utm_easting`, `utm_northing` | `0.0` | The same marker in UTM. Cross-check only; logged, never used in a computation |
| `utm_zone` | `""` | UTM zone string, e.g. `"30U"`. Cross-check only |
| `heading_deg` | `0.0` | Marker heading in degrees, ENU convention (0 = East, 90 = North). Rotates the odometry frame onto the lot frame |
| `lot_x`, `lot_y` | `0.0` | The marker's position in the lot layout frame, in metres |

`RealWorldDeployment` reads only `lot_x`, `lot_y` and `heading_deg` to build the
rigid-body transform; it converts `heading_deg` to radians on load. The flat-earth
projection is done by `GnssNoiseRelayNode`, which stands in for
`robot_localization`'s `navsat_transform_node` - the latter is not in the launch graph,
although a `navsat_transform` block survives in `configs/ros2_config.yaml`.

> **Note**: the template's procedure comments refer to `lot_origin_x` and `lot_origin_y`,
> whereas the keys it emits are `lot_x` and `lot_y`. The emitted names are the ones the
> loader expects.

## `actuation_calibration.yaml`

Maps policy normalised outputs to physical actuator commands. Consumed by
`ActuationCalibration.from_config()` in `uncertainty_rl/utils/actuation_calibration.py`.

`ActuationCalibration` wraps three `ActuatorMap` instances, one per action axis, under the
top-level `calibration:` key: `steering`, `throttle` and `brake`. This matches the 3-dim
action space `[steer, throttle, brake]`. There is no reverse gear.

Each `ActuatorMap` takes six parameters:

| Key | Steering | Throttle | Brake | Purpose |
|-----|----------|----------|-------|---------|
| `gain` | `1.0` | `1.0` | `1.0` | Command per unit policy output |
| `deadband` | `0.0` | `0.0` | `0.0` | Magnitude below which the output is held at `bias` |
| `deadband_offset` | `0.0` | `0.0` | `0.0` | Input shift applied before the deadband test, for an asymmetric deadband |
| `bias` | `0.0` | `0.0` | `0.0` | Neutral offset, e.g. straight-ahead trim or creep compensation |
| `min_output` | `-1.0` | `0.0` | `0.0` | Lower clamp. Steering is bipolar; throttle and brake are non-negative axes |
| `max_output` | `1.0` | `1.0` | `1.0` | Upper clamp |

The mapping applied per axis is `shifted = value - deadband_offset`, then `bias` if
`abs(shifted) < deadband` else `gain * shifted + bias`, finally clamped to
`[min_output, max_output]`.

An absent or unparseable file degrades to an identity calibration rather than raising.
The shipped values are already an identity mapping, so the file is inert until measured
values replace it. Set the per-axis terms from the vehicle's measured response curve,
since they are a property of its drive-by-wire hardware.

## `mission.yaml`

Per-run mission definition, read by `RealWorldDeployment.from_mission()`.

| Key | Shipped value | Purpose |
|-----|---------------|---------|
| `target_bay_id` | `"B01"` | Must match a bay `id` in the layout file |
| `layout_file` | `configs/layouts/rectangle.yaml` | Layout YAML defining lot geometry and bay poses. Must match the physical lot |

Bay identifiers follow the `<bay_type>_<index>` form the generator writes, for example
`perpendicular_12`. `rectangle.yaml` holds 49 bays: 47 `perpendicular_*` targets plus two
`motorcycle_*` bays flagged `always_empty`, which are rejected as targets.

> **Warning**: the shipped `target_bay_id: "B01"` is a placeholder and matches no bay in
> any generated layout, so `from_mission()` raises `ValueError` listing the available IDs.
> Set a real ID before each run.

Both keys are required: an empty `target_bay_id` or `layout_file` raises `ValueError`, and
a missing `mission.yaml` raises `FileNotFoundError`.

## See also

- [configs/deployment/README.md](../README.md) - shared sensor and agent configs
- [uncertainty_rl/envs/real/README.md](../../../uncertainty_rl/envs/real/README.md) - `RealWorldDeployment` and `RealWorldInferenceLoop`
- [docs/detailed_notes/deployment/real_world_deployment.md](../../../docs/detailed_notes/deployment/real_world_deployment.md) - deployment architecture and code map
- [docs/detailed_notes/deployment/sim_to_real_transfer.md](../../../docs/detailed_notes/deployment/sim_to_real_transfer.md) - sim-to-real gap analysis
- [configs/layouts/README.md](../../layouts/README.md) - layout files and bay identifiers
- `docs/AntonioGaldes_Dissertation.pdf`, Appendix B - canonical account of the deployment path (frame calibration, actuation model, pre-deployment sequence)
