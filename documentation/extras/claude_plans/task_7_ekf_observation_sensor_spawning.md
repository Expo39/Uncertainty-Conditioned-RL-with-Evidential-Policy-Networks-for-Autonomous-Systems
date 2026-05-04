# Plan: Task 7  --  EKF Observation + All Suite Sensor Spawning

## Context

Currently `_get_state()` uses CARLA ground truth for observation indices 0-5 (pose + velocity).
In deployment there is no CARLA ground truth  --  everything comes from the EKF. Training on ground
truth means the policy will not transfer to the real world without retraining.

Additionally, `_spawn_sensors()` only spawns an IMU at `z=0.0` (ground level) and **no LiDAR at
all**, despite the config having Suite A LiDAR parameters. Without the 2D LiDAR the EKF has no
scan-matching input, so covariance variation is minimal and the uncertainty signal that drives
the thesis contribution is absent.

This task:
1. Extends `CovarianceEstimate.msg` to carry velocities so all 6 EKF state dims arrive on one topic
2. Updates the covariance extractor and subscriber to cache and expose the full 6-element pose+velocity
3. Updates `_get_state()` to read indices 0-5 from the EKF subscriber (with CARLA fallback for CI)
4. Extends observation to 21-dim: adds 3 obstacle awareness dims from LiDAR (gated by `include_obstacle_obs` config flag so they can be removed easily if supervisor advises against it)
5. Properly implements `_spawn_sensors()` for all three suites (A/B/C), selected by `sensor_suite` config
6. Adds sensor mount positions to `configs/train_config.yaml` for all sensors across all suites

The sensor inspection visualisation (`make docker-inspect-sensors`) is a **separate follow-up task**.

`_compute_reward()` uses CARLA ground truth throughout and is NOT changed.

**Observation space: 21-dim when `include_obstacle_obs: true` (new default), 18-dim when false.**
The 3 new dims are: nearest obstacle distance, bearing to nearest obstacle in ego frame, and
obstacle type (static=0.0 / dynamic=1.0). Extracted from the LiDAR point cloud in `_get_state()`.
All consumers (`TOTAL_OBS_DIM`, `observation_space`, networks, constants, tests) updated.

**Easily removable:** The entire obstacle feature block is gated behind `include_obstacle_obs`
config key. Setting it to `false` restores 18-dim obs with zero code changes. The baselines/
YAML files keep this flag consistent across all 4 ablation conditions.

---

## Files to Modify

| File | Change |
|------|--------|
| `uncertainty_rl/ros2/uncertainty_rl_msgs/msg/CovarianceEstimate.msg` | Add `vx`, `vy`, `vyaw` fields |
| `uncertainty_rl/ros2/uncertainty_rl_ros2/covariance_extractor.py` | Populate vx/vy/vyaw from `msg.twist.twist` |
| `uncertainty_rl/envs/covariance_subscriber.py` | Cache pose+velocity; add `get_latest_pose()` method |
| `uncertainty_rl/envs/carla_parking.py` | `_get_state()`: EKF pose + obstacle dims; `_spawn_sensors()`: all suites |
| `uncertainty_rl/utils/constants.py` | `TOTAL_OBS_DIM = 21`, add `OBSTACLE_FEATURES_DIM = 3` |
| `configs/train_config.yaml` | Add mount positions + Suite B/C params + `include_obstacle_obs: true` |
| `configs/ros2_config.yaml` | Add `lidar_filter.angle_min/max` params for 270 deg FOV filter |
| `uncertainty_rl/ros2/launch/carla_bridge.launch.py` | Add `laser_filters` node that clips rear 90 deg before Cartographer |
| `configs/baselines/*.yaml` | Add `include_obstacle_obs: true` to all 4 baseline overrides |
| `tests/test_carla_parking.py` | Update obs shape assertions from 18 to 21 |
| `tests/test_ros2_integration.py` | Add test verifying obs 0-5 differ from CARLA ground truth |

---

## Step 1  --  Extend `CovarianceEstimate.msg`

Add three velocity fields after `yaw`:

```
std_msgs/Header header
float64 x
float64 y
float64 yaw
float64 vx       # EKF-filtered longitudinal velocity (m/s)
float64 vy       # EKF-filtered lateral velocity (m/s)
float64 vyaw     # EKF-filtered yaw rate (rad/s)
float64[9] covariance
```

File: `uncertainty_rl/ros2/uncertainty_rl_msgs/msg/CovarianceEstimate.msg`

---

## Step 2  --  Update `CovarianceExtractorNode`

In `odom_callback()`, extract twist and extend the stored tuple:

```python
vx = msg.twist.twist.linear.x
vy = msg.twist.twist.linear.y
vyaw = msg.twist.twist.angular.z
self.latest_pose = (x, y, yaw, vx, vy, vyaw)
```

Update type annotation:
```python
self.latest_pose: Optional[Tuple[float, float, float, float, float, float]] = None
```

In `publish_covariance()`, populate the new fields:
```python
msg.vx = self.latest_pose[3]
msg.vy = self.latest_pose[4]
msg.vyaw = self.latest_pose[5]
```

File: `uncertainty_rl/ros2/uncertainty_rl_ros2/covariance_extractor.py`

---

## Step 3  --  Extend `_CovarianceSubscriber`

Add instance variable: `self._latest_pose: Optional[np.ndarray] = None`

In `_covariance_callback()`, cache pose inside the existing lock block:
```python
with self._lock:
    self._latest_uncertainty = features
    self._latest_pose = np.array(
        [msg.x, msg.y, msg.yaw, msg.vx, msg.vy, msg.vyaw], dtype=np.float64
    )
    self._message_count += 1
```

Add getter method:
```python
def get_latest_pose(self) -> Optional[np.ndarray]:
    """
    @brief Get the most recent EKF pose and velocity estimate.
    @return Array of shape (6,) = [x, y, yaw, vx, vy, vyaw] or None if no
            message has been received yet.
    """
    with self._lock:
        if self._latest_pose is not None:
            return cast(np.ndarray, self._latest_pose.copy())
        return None
```

File: `uncertainty_rl/envs/covariance_subscriber.py`

---

## Step 4  --  Update `_get_state()` to use EKF pose

Replace lines ~1202-1211 in `carla_parking.py`:

```python
ekf_pose: Optional[np.ndarray] = None
if self._cov_subscriber is not None:
    ekf_pose = self._cov_subscriber.get_latest_pose()

if ekf_pose is not None:
    x = float(ekf_pose[0])
    y = float(ekf_pose[1])
    yaw = float(ekf_pose[2])
    vx = float(ekf_pose[3])
    vy = float(ekf_pose[4])
    vyaw = float(ekf_pose[5])
else:
    # Fallback: EKF not yet initialised or ROS 2 unavailable (CI / unit tests)
    transform = self.vehicle.get_transform()
    velocity = self.vehicle.get_velocity()
    angular_vel = self.vehicle.get_angular_velocity()
    x = transform.location.x
    y = transform.location.y
    yaw = math.radians(transform.rotation.yaw)
    vx = velocity.x
    vy = velocity.y
    vyaw = math.radians(angular_vel.z)
```

The `include_covariance=False` branch (no subscriber) naturally uses CARLA fallback  --  no change.

File: `uncertainty_rl/envs/carla_parking.py`, lines ~1202-1211

---

## Step 4b  --  Add obstacle observation dims (21-dim, easily removable)

### New constants (`uncertainty_rl/utils/constants.py`)

```python
OBSTACLE_FEATURES_DIM: int = 3   # nearest_dist, nearest_bearing, obstacle_type
TOTAL_OBS_DIM: int = 21          # was 18: 6 pose + 9 covariance + 3 target + 3 obstacle
```

### Config key (`configs/train_config.yaml`)

```yaml
include_obstacle_obs: true  # Set false to restore 18-dim obs (remove obstacle dims)
```

Add the same key to all 4 baseline override files in `configs/baselines/`.

### `_get_state()` extension

After computing the relative target pose and (optionally) EKF covariance, compute obstacle
features from the LiDAR point cloud cache. Add to `_get_state()`:

```python
if self._include_obstacle_obs:
    obstacle_features = self._get_obstacle_features()
else:
    obstacle_features = np.zeros(OBSTACLE_FEATURES_DIM, dtype=np.float32)
```

The buffer fills: `[pose(6), covariance(9), target(3), obstacle(3)]` for 21-dim, or
`[pose(6), covariance(9), target(3)]` for 18-dim (when `include_obstacle_obs=False`).

### `_get_obstacle_features()` helper

```python
def _get_obstacle_features(self) -> np.ndarray:
    """
    @brief Extract nearest obstacle features from the cached LiDAR scan.
    @return Array [nearest_dist_m, bearing_rad, obstacle_type] or zeros if no scan.

    obstacle_type: 0.0 = static (parked car, cone, wall), 1.0 = dynamic (pedestrian, patrol).
    """
    # Read from self._latest_lidar_scan (populated by LiDAR sensor callback)
    # Find minimum-distance return
    # Compute bearing in ego frame
    # Classify as static/dynamic based on actor proximity check
    ...
```

The LiDAR sensor callback (in `_spawn_sensors()`) populates `self._latest_lidar_scan`
(a numpy array of [x, y, z] points in vehicle frame, or None if no scan yet).

To classify dynamic vs static: check if the nearest LiDAR return point is within 2m of
any known dynamic actor position (`self._patrol_npcs`, `self._pedestrian_actors`).
Simple Euclidean check  --  no semantic segmentation.

### `observation_space` update

When `include_obstacle_obs=True`, use `TOTAL_OBS_DIM = 21`. When False, use
`VEHICLE_STATE_DIM + COVARIANCE_FEATURES_DIM + TARGET_POSE_DIM = 18`.
Constructor reads the config flag and sets `self._include_obstacle_obs`.

### Removability

To remove the feature entirely:
1. Set `include_obstacle_obs: false` in `train_config.yaml`  --  reverts to 18-dim obs
2. Or delete Step 4b constants + config key + `_get_obstacle_features()` method

Files: `uncertainty_rl/utils/constants.py`, `uncertainty_rl/envs/carla_parking.py`,
       `configs/train_config.yaml`, `configs/baselines/*.yaml`, `tests/test_carla_parking.py`

---

## Step 5  --  Implement sensor spawning for all three suites

### Suite definitions (from TODO.md)

| Suite | Primary sensor | Additional |
|-------|---------------|------------|
| A | 2D LiDAR (`sensor.lidar.ray_cast`, 1 channel, bumper height) | IMU + wheel odometry + steering |
| B | 3D LiDAR (`sensor.lidar.ray_cast`, 16 channels, VLP-16 style, roof) | IMU + wheel odometry + steering |
| C | 3D LiDAR (same as B) + RGB camera (`sensor.camera.rgb`, front) | IMU + wheel odometry + steering |

Wheel odometry and steering angle come from the CARLA ROS bridge automatically  --  no sensor actor to spawn.

### Sensor mount positions (vehicle body frame, BMW Grand Tourer)

All positions configurable via YAML (see Step 6). Defaults:

| Sensor | x (m) | y (m) | z (m) | Notes |
|--------|--------|--------|--------|-------|
| IMU | 0.0 | 0.0 | 0.3 | Centre of mass height (all suites) |
| 2D LiDAR (Suite A) | 2.4 | 0.0 | 0.3 | Front bumper, horizontal 360 scan |
| 3D LiDAR (Suite B/C) | 0.0 | 0.0 | 1.5 | Roof centre, all-round coverage |
| RGB camera (Suite C) | 2.0 | 0.0 | 1.2 | Front windscreen, forward-facing, pitch -5 deg |

### `_spawn_sensors()` structure

Replace the current `_spawn_sensors()` with a dispatcher + private helpers:

```python
def _spawn_sensors(self) -> None:
    suite = self._sensors_config.get("sensor_suite", "suite_a")
    self._spawn_imu()
    if suite == "suite_a":
        self._spawn_lidar_2d()
    elif suite == "suite_b":
        self._spawn_lidar_3d()
    elif suite == "suite_c":
        self._spawn_lidar_3d()
        self._spawn_camera_rgb()

def _spawn_imu(self) -> None:
    # Read imu_config from self._sensors_config.get("imu", {})
    # Read mount from imu_config.get("mount", {})
    # Set blueprint attributes from config with defaults
    # Spawn at mount position, attach to self.vehicle
    # Append to self._spawned_sensors

def _spawn_lidar_2d(self) -> None:
    # sensor.lidar.ray_cast, channels=1, upper_fov=0, lower_fov=0
    # Read lidar config from self._sensors_config.get("lidar", {})
    # Mount at front bumper (default x=2.4, z=0.3)

def _spawn_lidar_3d(self) -> None:
    # sensor.lidar.ray_cast, channels=16
    # Read lidar_3d config from self._sensors_config.get("lidar_3d", {})
    # Mount at roof (default x=0.0, z=1.5)

def _spawn_camera_rgb(self) -> None:
    # sensor.camera.rgb
    # Read camera_rgb config from self._sensors_config.get("camera_rgb", {})
    # Mount at windscreen (default x=2.0, z=1.2, pitch=-5 deg)
```

Note: `self._sensors_config` is what is passed as `carla_sensors_config` to the constructor.
The `sensor_suite` key is at the top level of `train_config.yaml` (not nested under `carla_sensors`).
Check constructor: it is stored as `self._sensors_config = carla_sensors_config or {}`.
The `sensor_suite` key must therefore be added to `carla_sensors_config` when loading from YAML,
OR read from a separate top-level config. Check `train_ppo.py` to confirm how config dicts are
split and passed to the env. If `sensor_suite` is at root level it may need to be forwarded to the
sensor config dict. Resolve during implementation.

File: `uncertainty_rl/envs/carla_parking.py`

---

## Step 6  --  Update `configs/train_config.yaml`

Add mount positions to existing `carla_sensors.imu` and `carla_sensors.lidar`, and add new
`carla_sensors.lidar_3d` and `carla_sensors.camera_rgb` sections:

```yaml
carla_sensors:
  imu:
    # ...existing noise params unchanged...
    mount:
      x: 0.0    # Centre of mass (longitudinal)
      y: 0.0
      z: 0.3    # Centre-of-mass height (metres)

  lidar:        # Suite A: 2D LiDAR (SICK TiM 5xx / Hokuyo style)
    # ...existing params unchanged (channels: 1, range: 30.0, etc.)...
    # @note CARLA sensor.lidar.ray_cast always scans 360 deg (no FOV restriction).
    # Real deployment uses a 270 deg unit (SICK TiM 5xx). The extra 90 deg rear
    # scan in simulation improves Cartographer scan-matching quality; the covariance
    # variation pattern is unchanged. Add a ROS bridge point cloud filter later
    # (Task 15) if sim-to-real covariance calibration reveals a discrepancy.
    mount:
      x: 2.4    # Front bumper centre
      y: 0.0
      z: 0.3    # Bumper height for horizontal 2D scan

  lidar_3d:     # Suite B/C: 3D LiDAR (Velodyne VLP-16 style)
    channels: 16
    range: 100.0
    points_per_second: 300000
    rotation_frequency: 10.0
    upper_fov: 15.0
    lower_fov: -15.0
    sensor_tick: 0.05
    mount:
      x: 0.0    # Roof centre
      y: 0.0
      z: 1.5    # Roof height

  camera_rgb:   # Suite C: forward-facing RGB camera
    image_size_x: 640
    image_size_y: 480
    fov: 90.0
    sensor_tick: 0.05
    mount:
      x: 2.0    # Front windscreen
      y: 0.0
      z: 1.2    # Windscreen height
      pitch: -5.0   # Slight downward tilt for near-field coverage
```

Note: `sensor_suite: suite_a` key already exists at root level of `train_config.yaml`. During
implementation, ensure it is passed through to `_spawn_sensors()`  --  either via `carla_sensors_config`
or read from env config dict directly.

File: `configs/train_config.yaml`

---

## Step 7  --  Create next supervisor meeting agenda

Create directory `documentation/meeting_agendas/13-03-2026/` and write `agenda.tex`:

```latex
% ==========================================================================
% Supervisor Meeting Agenda - 13 March 2026 (30 minutes)
%
% Compile with: pdflatex agenda.tex
% ==========================================================================

\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=2.5cm]{geometry}
\usepackage{enumitem}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{xcolor}

\definecolor{done}{RGB}{34,139,34}
\definecolor{todo}{RGB}{200,50,50}
\definecolor{question}{RGB}{0,90,180}

\newcommand{\done}[1]{\textcolor{done}{\textbf{DONE:}} #1}
\newcommand{\todo}[1]{\textcolor{todo}{\textbf{TODO:}} #1}
\newcommand{\question}[1]{\textcolor{question}{\textbf{Q:}} #1}

\title{Supervisor Meeting Agenda\\
\large 13 March 2026 --- 30 minutes}
\author{Antonio Galdes}
\date{}

\begin{document}
\maketitle

% ------------------------------------------------------------------
\section*{1. Progress Update (5 min)}
% ------------------------------------------------------------------

\begin{itemize}[leftmargin=*]
  \item \done{Task 5 --- CARLA parking environment: Town05\_Opt, 3 floor plans
    (rectangle, trapezoid, irregular\_a), 15 bays/lot
    (perpendicular/angled/parallel), 18-dimensional observation space.
    163 unit tests passing.}
  \item \done{Task 6 --- Full Docker stack verified: CARLA 0.9.16, ros2-bridge
    (Jazzy), training container (Humble/NGC PyTorch). All 3 containers healthy
    on RTX 4070 Ti Super.}
  \item \todo{Task 7 --- EKF observation (in progress): wiring EKF filtered pose
    into observation indices 0--5, Suite A/B/C sensor spawning, 270\textdegree{}
    LiDAR FOV filter. 3 design decisions blocking sub-tasks (see below).}
\end{itemize}

% ------------------------------------------------------------------
\section*{2. Decisions Needed (15 min)}
% ------------------------------------------------------------------

\subsection*{2.1 Obstacle Perception in the Observation Space}

The current 18-dimensional observation contains EKF pose (0--5), EKF covariance
(6--14), and relative target bay pose (15--17). The agent has no direct awareness of
obstacles. Collision avoidance is learned implicitly through reward shaping (episode
termination when clearance \textless{} 0.8\,m).

The \texttt{irregular\_a} floor plan contains a 16\,m $\times$ 4\,m solid obstacle
between two rows of perpendicular bays (centre at world coordinates 131.25, 19.0).
An agent navigating to a bay on the far side must route around this obstacle without
direct obstacle knowledge.

\begin{table}[h]
\centering
\small
\begin{tabularx}{\textwidth}{@{}llXl@{}}
\toprule
\textbf{Option} & \textbf{Obs dims} & \textbf{Description} & \textbf{Ablation impact} \\
\midrule
A & 18 & No obstacle perception. Reward-shaped avoidance only.
  Clean contribution boundary. & None. \\
\addlinespace
B & 21 & Add 3 LiDAR dims: nearest obstacle distance (m), bearing
  in ego frame (rad), type (0 = static, 1 = dynamic). All 4 baselines
  use 21-dim obs. Experimental variable unchanged: EKF covariance
  (yes/no) $\times$ policy (standard/evidential).
  & All networks, constants, and tests updated. \\
\addlinespace
C & 21 & Same as B, plus a 5th ablation baseline: 21-dim obs,
  no EKF covariance, standard policy. Isolates perception contribution.
  & Adds 5th baseline, 10 more training runs. \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{Note:} Dynamic obstacles (pedestrians, patrol vehicles) degrade LiDAR
scan quality, causing EKF covariance spikes. Under Option B or C, this creates a
second uncertainty source that strengthens the thesis demonstration:
``dynamic obstacles increase localisation uncertainty $\to$ agent adapts behaviour.''

\question{Which option is appropriate for the dissertation scope? Is the central
obstacle in \texttt{irregular\_a} an intended navigation challenge, or should bays
behind it be excluded from target sampling until perception is resolved?}

\bigskip
\subsection*{2.2 LiDAR Scan-Matching Scope (Cartographer)}

Suite A uses 2D LiDAR for scan-matching via Google Cartographer. Without
Cartographer, the EKF fuses only IMU + wheel odometry, producing covariance
but with limited variation across lot conditions. Cartographer is required for
the primary thesis claim (lot occupancy drives covariance, which drives caution).

\begin{table}[h]
\centering
\small
\begin{tabularx}{\textwidth}{@{}lXl@{}}
\toprule
\textbf{Option} & \textbf{Description} & \textbf{Scope} \\
\midrule
IMU + odometry only & EKF without LiDAR. Covariance from IMU noise and motion
  dynamics only. Less variation across conditions.
  & Minimal (current state). \\
\addlinespace
Suite A + Cartographer & 2D LiDAR (270\textdegree{}, front bumper) feeds
  Cartographer $\to$ EKF. Covariance from scan-matching quality. Stronger
  thesis signal. Deployable in real-world (\pounds 500--1500).
  & +1--2 weeks. \\
\addlinespace
Suite B + Cartographer & 3D LiDAR (VLP-16 style, roof-mounted) instead of
  2D. Richer features, more uncertainty variation. Higher hardware cost
  (\pounds 3000--8000).
  & +1 additional week. \\
\bottomrule
\end{tabularx}
\end{table}

\question{Is Cartographer integration in scope? Should Suite A be the training
configuration, or start with IMU + odometry only and upgrade if covariance
variation proves insufficient after initial training runs?}

\bigskip
\subsection*{2.3 NPC Patrol Vehicle Obstacle Avoidance}

Patrol vehicles use a proportional steering controller with no obstacle awareness.
When the ego vehicle crosses the patrol path, the NPC drives into it. Three options:
(a) CARLA obstacle sensor on each NPC (event-driven callback when object within N metres
--- simplest, already supported), (b) bounding-box proximity polling each tick (no extra
sensor, polling overhead), (c) CARLA Traffic Manager (requires road network, may not work
in off-road parking lot geometry).

\question{Is NPC collision avoidance required for valid training, or can this be deferred?
If needed, is the obstacle sensor approach (option a) acceptable?}

% ------------------------------------------------------------------
\section*{3. Timeline and Next Steps (5 min)}
% ------------------------------------------------------------------

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Task} & \textbf{Est.\ Time} & \textbf{Blocked on} \\
\midrule
Task 7: EKF obs + sensor spawning & 3--5 days & In progress \\
Task 7.5: Cartographer config & 1--2 weeks & \textbf{Decision 2.2 today} \\
Task 7.6: Obstacle dims (21-dim) & 2--3 days & \textbf{Decision 2.1 today} \\
Task 8: End-to-end training verification & 2--3 days & Tasks 7, 7.5 \\
Task 9: Reward function tuning & 1--2 weeks & Task 8 \\
Training runs (4+ baselines) & 2--4 weeks & Task 9 \\
Tasks 10--14: Evaluation + plots & 2 weeks & Trained models \\
\bottomrule
\end{tabular}
\end{table}

\question{Does this timeline remain feasible given the September 2026 submission
deadline?}

% ------------------------------------------------------------------
\section*{4. Dissertation Writing (5 min)}
% ------------------------------------------------------------------

\begin{itemize}[leftmargin=*]
  \item Can begin drafting methodology chapter now: 2$\times$2 ablation design,
    evidential actor architecture, 21-dim observation space (18-dim pose + covariance
    + target, plus 3 obstacle proximity dims gated by \texttt{include\_obstacle\_obs}).
    Pending supervisor confirmation on ablation design (see decision 2.1).
  \item Experimental setup chapter: CARLA lot geometry, Docker stack, evaluation
    conditions, sensor suite.
  \item Background chapters (EKF localisation, evidential deep learning, PPO,
    uncertainty in RL) can be written independently of experimental results.
\end{itemize}

\question{Any guidance on chapter structure, expected length, or when to submit
draft chapters for feedback?}

\end{document}
```

File: `documentation/meeting_agendas/13-03-2026/agenda.tex`

---

## Step 8  --  Update integration tests

Add to `tests/test_ros2_integration.py`:

```python
def test_obs_pose_uses_ekf_not_carla_ground_truth(self) -> None:
    """
    @brief Verify obs indices 0-5 come from EKF estimate, not CARLA ground truth.

    After a few steps with a running EKF, the filtered pose should differ
    slightly from CARLA ground truth due to sensor noise and filter lag.
    """
    # Create env, reset, step a few times
    # Compare obs[0:3] with vehicle.get_transform() (ground truth)
    # Assert they differ by at least sensor noise magnitude (e.g. > 0.001 m)
```

File: `tests/test_ros2_integration.py`

---

## Step 8  --  Add 270 deg LiDAR FOV filter in ros2-bridge (Suite A)

CARLA `sensor.lidar.ray_cast` always publishes full 360 deg `sensor_msgs/PointCloud2`. The real
SICK TiM 5xx unit scans 270 deg. Pipeline to clip it before Cartographer:

```
/carla/ego_vehicle/lidar (PointCloud2, 360 deg)
    -> pointcloud_to_laserscan node -> /lidar/scan_raw (LaserScan, 360 deg)
    -> laser_filter_node (clips rear 90 deg) -> /lidar/scan (LaserScan, 270 deg)
    -> Cartographer (Task 7.5)
```

In `carla_bridge.launch.py`, add two nodes after the CARLA bridge:

```python
pc2_to_scan = Node(
    package="pointcloud_to_laserscan",
    executable="pointcloud_to_laserscan_node",
    name="pc2_to_scan",
    parameters=[{
        "target_frame": "ego_vehicle",
        "min_height": -0.1,
        "max_height": 0.1,
        "range_min": 0.1,
        "range_max": 30.0,
    }],
    remappings=[
        ("cloud_in", "/carla/ego_vehicle/lidar"),
        ("scan", "/lidar/scan_raw"),
    ],
)

lidar_filter_config = config.get("lidar_filter", {})
lidar_filter = Node(
    package="laser_filters",
    executable="scan_to_scan_filter_chain",
    name="lidar_angular_filter",
    parameters=[{
        "filter_chain_params_name": "laser_filters",
        "laser_filters": [{
            "name": "angular_bounds_filter",
            "type": "laser_filters/LaserScanAngularBoundsFilter",
            "params": {
                "lower_angle": lidar_filter_config.get("angle_min", -2.356),
                "upper_angle": lidar_filter_config.get("angle_max", 2.356),
            },
        }],
    }],
    remappings=[
        ("scan", "/lidar/scan_raw"),
        ("scan_filtered", "/lidar/scan"),
    ],
)
```

Add to `configs/ros2_config.yaml`:
```yaml
lidar_filter:
  angle_min: -2.356   # -135 deg = left limit of 270 deg FOV centred forward
  angle_max: 2.356    #  135 deg = right limit
```

Add to `uncertainty_rl/ros2/Dockerfile` (apt-get install):
```
ros-jazzy-pointcloud-to-laserscan
ros-jazzy-laser-filters
```

Note: Cartographer subscribes to `/lidar/scan` (filtered output). Cartographer configuration
is **Task 7.5**, not this task  --  these nodes are wired but Cartographer is not yet started.

Files: `uncertainty_rl/ros2/launch/carla_bridge.launch.py`, `configs/ros2_config.yaml`,
       `uncertainty_rl/ros2/Dockerfile`

---

## Acceptance Criteria (from TODO.md + scope additions)

**Status: ALL DONE (15-03-2026)**

- [x] `_CovarianceSubscriber` caches filtered pose+velocity (not just covariance)
- [x] `_get_state()` reads indices 0-5 from EKF subscriber; CARLA fallback for CI
- [x] `_compute_reward()` still uses CARLA ground truth (unchanged)
- [x] `TOTAL_OBS_DIM = 21` when `include_obstacle_obs: true` (default)
- [x] Obstacle dims (18-20): nearest distance, bearing, type computed from LiDAR scan
- [x] Setting `include_obstacle_obs: false` reverts to 18-dim obs with no other changes needed
- [x] IMU spawned at centre-of-mass height (x=0, y=0, z=0.3), mount configurable via YAML
- [x] Suite A: 2D LiDAR spawned at front bumper (x=2.4, y=0.0, z=0.5), configurable
- [x] Suite B: 3D LiDAR spawned at roof (x=-0.5, y=0.0, z=1.9), configurable
- [x] Suite C: 3D LiDAR + forward camera (x=0.2, y=0.0, z=1.4), configurable
- [x] `sensor_suite` config key selects which suite is spawned
- [x] 270 deg FOV filter pipeline wired (`pointcloud_to_laserscan` + `laser_filters`) for Suite A
- [x] Integration test added verifying obs 0-5 differ from CARLA ground truth
- [x] `make verify` passes (unit tests, black, isort, flake8, mypy -- no CARLA/ROS 2 needed)

## Verification

1. `make verify`  --  green locally (CARLA fallback path keeps unit tests passing)
2. Inside container (`make docker-shell`) with `sensor_suite: suite_a`:
   - `pytest tests/test_ros2_integration.py -m integration -v`
   - Confirm `obs.shape == (21,)` after reset
   - Confirm `obs[0:6]` non-zero and differ from CARLA API values after a few steps
   - Confirm `obs[6:15]` (covariance) non-zero and varies across steps
   - Confirm `obs[18:21]` (obstacle dims) non-zero when obstacles are present
3. Set `include_obstacle_obs: false`, re-run  --  confirm `obs.shape == (18,)`
4. `ros2 topic echo /lidar/scan`  --  confirm angular range is -135 to +135 deg (270 deg total)
5. In CARLA spectator (bird's-eye), confirm LiDAR point cloud appears at front bumper height
6. Switch to `sensor_suite: suite_b`  --  verify 3D LiDAR spawns without error
7. Switch to `sensor_suite: suite_c`  --  verify 3D LiDAR + camera both spawn

**Task 7.5: Cartographer configuration**  --  configure Google Cartographer in ros2-bridge to
consume `/lidar/scan` (filtered) and produce odometry for the EKF. Required for LiDAR
scan-matching to actually drive covariance variation.

**Task 7.6 (supervisor discussion required): Obstacle perception observation dims**  --  add
3 observation dimensions from the LiDAR point cloud to give the agent obstacle awareness:
  - `nearest_obstacle_distance` (metres, from min LiDAR return)
  - `nearest_obstacle_bearing` (radians in ego frame, angle to nearest return)
  - `nearest_obstacle_type` (0.0 = static, 1.0 = dynamic / moving object)
This changes `TOTAL_OBS_DIM` from 18 to 21 and affects all 4 ablation baselines (all baselines
would include these dims since they are safety-critical, not the experimental variable).
Dynamic obstacles (pedestrians, patrol cars) cause LiDAR scan degradation -> EKF covariance
spikes, which strengthens the thesis: "dynamic obstacles increase localisation uncertainty ->
agent drives more cautiously."
Discuss with supervisor before implementing: may require adding a 5th ablation baseline
(without obstacle dims) to isolate the contribution of perception vs uncertainty conditioning.
Flag this in the next meeting agenda.

**Task 7.7: `make docker-inspect-sensors`**  --  CARLA spectator visualisation:
- Bird's-eye and 1st-person views of ego vehicle
- Sensor attach points highlighted with labelled dots
- LiDAR 270 deg range overlay showing expected scan coverage
- 360 orbit around the car with translucent vehicle model
