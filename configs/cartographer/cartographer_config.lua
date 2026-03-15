-- ==========================================================================
-- cartographer_config.lua
-- @brief Google Cartographer 2D SLAM configuration for Suite A (2D LiDAR).
--
-- Cartographer performs scan-matching against a growing submap of the parking
-- lot. The quality of each scan match determines the covariance of the
-- published /scan_matched_odometry pose -- this covariance variation is the
-- primary uncertainty signal feeding the EKF and, ultimately, the RL policy.
--
-- Pipeline:
--   /lidar/scan (LaserScan, 270 deg filtered)
--       -> Cartographer -> /scan_matched_odometry (nav_msgs/Odometry with covariance)
--       -> robot_localisation EKF (odom0 correction step)
--
-- Tuning guidance:
--   - VOXEL_FILTER_SIZE: set to ~0.1 m for a 30 m range SICK TiM 5xx.
--   - LOOP_CLOSURE_SCORE_THRESHOLD: raise to 0.70 if false loop closures appear
--     in the OOD lot (irregular_a). Lower to 0.45 if the agent cannot close
--     the map on dense perpendicular-bay sections.
--   - TRANSLATION_WEIGHT / ROTATION_WEIGHT: balance how aggressively scan
--     matching constrains the pose. Higher values reduce covariance (less
--     uncertainty signal). Default values preserve meaningful covariance
--     variation across lot occupancy conditions.
--
-- @note cartographer_ros reads this file at startup via the --lua_config_file
--       argument in carla_bridge.launch.py.
-- ==========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- Topic names (must match the filtered LiDAR output and TF frame names)
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",

  -- Use the EKF odometry as a motion prior between scans.
  -- This improves scan-matching convergence at low speeds.
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  -- Use odometry from the EKF as a pose prior between scans.
  -- Set false if the EKF is not yet publishing (bootstrapping).
  use_odometry = false,

  use_nav_sat = false,

  -- No landmarks
  use_landmarks = false,

  -- 2D LiDAR (Suite A: CARLA sensor.lidar.ray_cast, 1 channel, horizontal)
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 0,

  -- Publish scan-matched pose to /scan_matched_odometry for the EKF odom0 input.
  -- robot_localisation reads this topic when odom0 is uncommented in ros2_config.yaml.
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,   -- 200 Hz (faster than EKF at 20 Hz)
  trajectory_publish_period_sec = 30e-3,

  -- Sampling ratio: fraction of range data used for scan-matching.
  -- 1.0 = use every scan. Lower if CPU is a bottleneck.
  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

-- 2D map builder (no 3D TSDF)
MAP_BUILDER.use_trajectory_builder_2d = true

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings
-- ---------------------------------------------------------------------------

-- Voxel filter: downsample the 270 deg scan to ~0.1 m resolution.
-- Reduces CPU load without losing wall/car-body scan features.
TRAJECTORY_BUILDER_2D.min_range = 0.1          -- metres (ignore returns < 10 cm)
TRAJECTORY_BUILDER_2D.max_range = 30.0         -- metres (SICK TiM 5xx max range)
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = false     -- IMU goes to EKF directly, not Cartographer

-- Adaptive voxel filter: target ~200 points per scan after downsampling
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

-- Scan matcher: Ceres-based optimisation (more robust than correlative matcher alone)
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)

-- Ceres scan matcher weights.
-- TRANSLATION_WEIGHT / ROTATION_WEIGHT: how strongly scan-matching constrains pose.
-- Higher = tighter constraint = lower covariance. Keep at defaults to preserve
-- meaningful uncertainty variation across lot occupancy conditions.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Submap size: 2 submaps of 90 scan ranges each.
-- Smaller submaps give faster loop closure in a compact parking lot.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05   -- 5 cm grid

-- ---------------------------------------------------------------------------
-- Pose graph (loop closure) settings
-- ---------------------------------------------------------------------------

POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60

-- Loop closure frequency: check every 35 nodes.
-- Frequent enough to catch drift in a small lot; not so frequent as to stall.
POSE_GRAPH.optimize_every_n_nodes = 35

-- Huber loss scale for outlier robustness in the pose graph.
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
