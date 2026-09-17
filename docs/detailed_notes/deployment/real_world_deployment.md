# Real-World Deployment

Extracted from `uncertainty_rl/envs/real/`.

Appendix B of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the deployment path: code-path parity, the ROS interface and master-switch values, the
frame-calibration transform, the actuation model and the pre-deployment sequence. The
sim-to-real modelling gaps are in `sim_to_real_transfer.md`.

This note maps that material onto the code, so a reader can go from a step in
Appendix B to the function implementing it.

## Where each step lives

| Appendix B step | Code |
|-----------------|------|
| Sensor data flow, unchanged downstream of the two topics | `CovarianceExtractorNode` (ros2), `_CovarianceSubscriber` (`envs/covariance_subscriber.py`) - identical in sim and real |
| Sim-side GNSS driver, replaced on the vehicle | `GnssNoiseRelayNode` (`ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py`) |
| Lat/lon to local-frame projection | `navsat_transform_node`, configured in `configs/ros2_config.yaml` |
| EKF frame calibration | `calibrate_ekf_frame_offset()` in `envs/_parking_core.py`, called from `RealWorldInferenceLoop.prepare()` |
| Surveyed datum, reference pose | `RealWorldDeployment.reference_pose()` in `envs/real/deployment_utils.py`, reading `configs/deployment/real/real_world_datum.yaml` |
| Actuation calibration | `RealWorldDeployment.calibrate_action()`, reading `configs/deployment/real/actuation_calibration.yaml` |
| Mission definition | `configs/deployment/real/mission.yaml`, read only on the vehicle |

## Implementation notes not in Appendix B

`ekf_state.json` stores `y` in ROS REP-103 convention (y-left). Both the inference loop
and the simulation environment negate it before applying the frame transform, so the
pose matches the lot layout frame. The negation is easy to miss and easy to apply
twice.

The datum file is untracked: copy `real_world_datum.yaml.example` and fill in the
survey. A missing file degrades silently to an identity transform, as does an
uncalibrated actuation file, so both are pass-throughs rather than errors.

Any ROS 2 driver publishing `sensor_msgs/Imu` to `/imu/data` works. The EKF's
`imu0_config` in `configs/ros2_config.yaml` expects orientation, angular velocity and
linear acceleration.

## See also

| File | Role |
|------|------|
| `uncertainty_rl/envs/real/deployment_utils.py` | `RealWorldDeployment`: datum loading, target bay lookup, actuation calibration |
| `uncertainty_rl/envs/real/inference_loop.py` | `RealWorldInferenceLoop`: the deployment loop |
| `uncertainty_rl/envs/_parking_core.py` | `calibrate_ekf_frame_offset()`, `build_observation()`, `extract_obstacle_features()` |
| `uncertainty_rl/envs/covariance_subscriber.py` | `_CovarianceSubscriber`: file-based EKF state reader |
