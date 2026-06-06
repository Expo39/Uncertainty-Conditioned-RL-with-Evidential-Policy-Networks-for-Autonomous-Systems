# configs/deployment/real/

Real-vehicle deployment configs. These files are placeholders until the test site is
instrumented and a calibration run is performed. The deployment path loads the
`real_world_datum` and `actuation_calibration` files referenced in
`configs/deployment/agent_config.yaml` to calibrate the EKF frame and actuators on
the real vehicle.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `real_world_datum.yaml.example` | Surveyed lot datum for EKF frame calibration | Template only - copy and fill in at test site |
| `actuation_calibration.yaml` | Per-actuator gain, deadband, bias mapping | Identity (uncalibrated) until calibration run |
| `mission.yaml` | Target bay and layout file for a deployment run | Set before each run |

## Pre-deployment checklist

Before any real-vehicle deployment run:

- [ ] `cp real_world_datum.yaml.example real_world_datum.yaml` and fill in the surveyed UTM easting/northing and heading of the reference marker.
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

Then fill in the surveyed values at the test site. Do not commit `real_world_datum.yaml`
with placeholder values.

Keys: `utm_easting`, `utm_northing`, `utm_heading_deg`, `lot_x`, `lot_y`, `lot_yaw_deg`.

## `actuation_calibration.yaml`

Maps policy normalised outputs `[-1, 1]` to physical actuator commands. Consumed by
`ActuationCalibration.from_config()`. All values are identity (gain=1.0, deadband=0.0,
bias=0.0) until a calibration run is performed.

## `mission.yaml`

Per-run mission definition. Set `target_bay_id` to the ID of the target bay in the
layout YAML, and `layout_file` to the path of the corresponding layout.

## See also

- [configs/deployment/README.md](../README.md) - shared sensor and agent configs
- [uncertainty_rl/envs/real/README.md](../../../uncertainty_rl/envs/real/README.md) - `RealWorldDeployment` and `RealWorldInferenceLoop`
- [docs/detailed_notes/real_world_deployment.md](../../../docs/detailed_notes/real_world_deployment.md) - deployment architecture and sim-to-real transfer
