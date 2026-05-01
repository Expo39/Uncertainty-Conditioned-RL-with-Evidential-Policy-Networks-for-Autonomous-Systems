# ros2_architecture

Extracted from `uncertainty_rl/envs/covariance_subscriber.py` and `uncertainty_rl/ros2/`.

## DDS-bypass via shared JSON files

The training container uses ROS 2 Humble (Ubuntu 22.04 NGC base). The ros2-bridge
container uses ROS 2 Jazzy (Ubuntu 24.04). These are different DDS domains - direct
topic subscription across distros would require careful QoS negotiation and can silently
fail. The chosen design avoids DDS entirely at the Python training boundary:

```
ros2-bridge (Jazzy)                   training container (Humble)
-------------------------------        ------------------------------
CovarianceExtractorNode                _CovarianceSubscriber
  subscribes /odometry/filtered          polls ekf_state.json
  writes ekf_state.json  ------------>   reads ekf_state.json
  reads initial_pose.json <-----------   writes initial_pose.json
  reads gnss_noise_config.json <------   writes gnss_noise_config.json
  publishes /set_pose (DDS, Jazzy)       (no DDS in training container)
  publishes to GnssNoiseRelayNode
```

All three files live on a Docker shared volume (`outputs/`) mounted read-write in both
containers. The shared volume is a tmpfs-backed bind mount - read latency is negligible
compared to the 20 Hz EKF publish rate.

## Atomicity

All writes use `tempfile + os.replace`:
```python
with open(tmp_path, "w") as f:
    json.dump(data, f)
os.replace(str(tmp_path), str(target_path))
```
`os.replace` is atomic on POSIX (rename syscall). The reader never sees a partial write.

## Sequence-number guard (stale-data prevention)

Every JSON file includes a monotonically increasing `seq` field. `_CovarianceSubscriber`
records `_valid_after_seq` at each episode reset (`invalidate()`) and only accepts
files with `seq > _valid_after_seq`. This prevents the training loop from consuming
an EKF state written before the episode reset, even if the file has not yet been
overwritten when the loop checks it.

This is more reliable than `mtime`-based guards: Docker container clocks can differ
by up to a few milliseconds on some host configurations, causing `mtime` comparisons
to give false-positives (accepting a stale file as fresh).

## File paths (configurable)

Default paths (single-worker):
- `ekf_state.json` - `/workspace/outputs/ekf_state.json`
- `initial_pose.json` - `/workspace/outputs/initial_pose.json`
- `gnss_noise_config.json` - `/workspace/outputs/gnss_noise_config.json`

Per-worker paths for parallel training (multi-worker):
- Set via `ros2_config["ekf_state_file"]` in train_config.yaml, or
- `EKF_STATE_FILE` / `INITIAL_POSE_FILE` / `GNSS_NOISE_CONFIG_FILE` environment variables
  (set per container in `docker-compose.env_workers.yml`).

## Per-file semantics

| File | Written by | Read by | Purpose |
|------|-----------|---------|---------|
| `ekf_state.json` | CovarianceExtractorNode (~20 Hz) | _CovarianceSubscriber (per step) | EKF pose + covariance for observation |
| `initial_pose.json` | _CovarianceSubscriber (per episode reset) | CovarianceExtractorNode | Signal spawn pose for /set_pose |
| `gnss_noise_config.json` | _CovarianceSubscriber (per episode reset) | GnssNoiseRelayNode | Signal per-episode GNSS noise tier + datum |

## See also

- `uncertainty_rl/envs/covariance_subscriber.py` - `_CovarianceSubscriber`
- `uncertainty_rl/ros2/uncertainty_rl_ros2/covariance_extractor.py` - `CovarianceExtractorNode`
- `uncertainty_rl/ros2/uncertainty_rl_ros2/sensor_relay/gnss_noise_relay.py` - `GnssNoiseRelayNode`
