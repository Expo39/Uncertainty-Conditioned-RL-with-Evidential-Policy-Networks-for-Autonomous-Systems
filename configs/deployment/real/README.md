# configs/deployment/real/

Real-vehicle deployment configs. These are site- and vehicle-specific: the values here
are the defaults, and each is set from the survey and calibration run for the site being
deployed to. The deployment path loads the `real_world_datum` and `actuation_calibration`
files referenced in `configs/deployment/agent_config.yaml` to calibrate the EKF frame and
actuators on the real vehicle.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `real_world_datum.yaml.example` | Surveyed lot datum for EKF frame calibration | Template - copy and set from the site survey |
| `actuation_calibration.yaml` | Per-actuator gain, deadband, bias mapping | Identity mapping; set from the vehicle's response curve |
| `mission.yaml` | Target bay and layout file for a deployment run | Set before each run |

## Pre-deployment checklist

Before any real-vehicle deployment run:

- [ ] `cp real_world_datum.yaml.example real_world_datum.yaml` and fill in the reference marker's geodetic and UTM position, its `heading_deg`, and its `lot_x` / `lot_y` in the layout frame.
- [ ] Run actuation calibration and update `actuation_calibration.yaml` with measured gains and deadbands.
- [ ] Set `mission.yaml`: correct `target_bay_id` and `layout_file` for this run.
- [ ] Confirm `agent_config.yaml`'s `baseline:` names the baseline the deployed checkpoint was trained as (its `include_covariance` / `include_obstacle_obs` / `policy_type` must match the checkpoint).
- [ ] Confirm `agent_config.yaml`'s `real_world_datum` / `actuation_calibration` point at the filled-in files for this site.
- [ ] Confirm RTK-GNSS antenna has clear sky visibility from the deployment site.

## `real_world_datum.yaml`

Surveyed reference marker used by `calibrate_ekf_frame_offset()` to align the EKF odometry
frame with the lot layout frame. Consumed by `RealWorldDeployment.from_config()`.

The file `real_world_datum.yaml` does not exist until you create it from the template:

```bash
cp configs/deployment/real/real_world_datum.yaml.example \
   configs/deployment/real/real_world_datum.yaml
```

Then fill in the surveyed values for the site. `real_world_datum.yaml` is per-site and
is not committed.

All keys are nested under a top-level `datum:` mapping:

| Key | Purpose |
|-----|---------|
| `datum_lat`, `datum_lon` | Geodetic position of the reference marker, consumed by `navsat_transform_node` |
| `utm_easting`, `utm_northing`, `utm_zone` | The same marker in UTM |
| `heading_deg` | Marker heading, used to rotate the odometry frame onto the lot frame |
| `lot_x`, `lot_y` | The marker's position in the lot layout frame |

`RealWorldDeployment` reads `lot_x`, `lot_y` and `heading_deg` to build the rigid-body
transform. Note that the template's procedure comments refer to `lot_origin_x` and
`lot_origin_y`, whereas the keys it emits are `lot_x` and `lot_y`. The emitted names are
the ones the loader expects.

## `actuation_calibration.yaml`

Maps policy normalised outputs to physical actuator commands. Consumed by
`ActuationCalibration.from_config()`. Each of the three actuators under `calibration:`
carries `gain`, `deadband`, `deadband_offset`, `bias`, `min_output` and `max_output`.
Note that `min_output` is `-1.0` for steering but
`0.0` for throttle and brake, which are non-negative axes. The shipped values are an
identity mapping; set the per-axis terms from the vehicle's measured response curve,
since they are a property of its drive-by-wire hardware.

## `mission.yaml`

Per-run mission definition. Set `target_bay_id` to the identifier of the target bay in
the layout YAML, and `layout_file` to the path of that layout.

Bay identifiers follow the `<bay_type>_<index>` form the generator writes, for example
`perpendicular_12`. Set `target_bay_id` for the site before each run.

## See also

- [configs/deployment/README.md](../README.md) - shared sensor and agent configs
- [uncertainty_rl/envs/real/README.md](../../../uncertainty_rl/envs/real/README.md) - `RealWorldDeployment` and `RealWorldInferenceLoop`
- [docs/detailed_notes/deployment/real_world_deployment.md](../../../docs/detailed_notes/deployment/real_world_deployment.md) - deployment architecture and sim-to-real transfer
