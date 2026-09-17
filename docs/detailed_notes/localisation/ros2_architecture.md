# ros2_architecture

Extracted from `uncertainty_rl/envs/covariance_subscriber.py` and `uncertainty_rl/ros2/`.

Section 3.1 of the dissertation (`docs/AntonioGaldes_Dissertation.pdf`) is canonical for
the Jazzy-Humble file bridge: why DDS is bypassed at the training boundary, the three
JSON files and their roles, the atomic `tempfile` plus `os.replace` write, and the
sequence-number guard that keeps a stale record out of the first observation of an
episode. Section 3.4 covers the COG heading derivation and its publication gates, and
`uncertainty_rl/ros2/README.md` documents the nodes and topics.

This note records the file-path configuration, which is needed to run parallel workers
and is documented nowhere else.

## File paths

Single-worker defaults:

| File | Default path |
|------|--------------|
| `ekf_state.json` | `/workspace/outputs/ekf_state.json` |
| `initial_pose.json` | `/workspace/outputs/initial_pose.json` |
| `episode_config.json` | `/workspace/outputs/episode_config.json` |

All three live on a Docker shared volume mounted read-write in both containers. The
volume is a tmpfs-backed bind mount, so read latency is negligible against the EKF
publish rate.

## Per-worker paths for parallel training

Each worker needs its own set of files, or workers will overwrite one another's EKF
state. Paths are set either through `ros2_config["ekf_state_file"]` and its siblings in
`train_config.yaml`, or through the `EKF_STATE_FILE`, `INITIAL_POSE_FILE` and
`EPISODE_CONFIG_FILE` environment variables, set per container in
`docker-compose.env_workers.yml`.

## Sim-real parity switches

The sensor relays apply effects observed in real RTK and inertial hardware - GNSS
east/north anisotropy, per-callback dropout, tier-mapped `NavSatStatus`, and IMU
per-axis scale-factor error. All are gated behind the `enable_gnss_noise` and
`enable_imu_noise` master switches and are skipped on the real-vehicle launch
(`real_vehicle.launch.py`), because the physical sensors already exhibit them. Section
3.4 derives each effect; the parameters live in `configs/ros2_config.yaml` and
`configs/deployment/sim/gnss_noise_profiles.yaml`.

## See also

- `uncertainty_rl/envs/covariance_subscriber.py` - `_CovarianceSubscriber`
- `uncertainty_rl/ros2/uncertainty_rl_ros2/covariance_extractor.py` - `CovarianceExtractorNode`
- `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` - `GnssNoiseRelayNode`
- `gnss_markov_transitions.md` - the fix-state chain at runtime
