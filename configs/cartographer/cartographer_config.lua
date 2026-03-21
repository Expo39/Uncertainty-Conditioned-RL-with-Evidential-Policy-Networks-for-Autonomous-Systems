-- ==========================================================================
-- cartographer_config.lua
-- @brief Google Cartographer 2D SLAM configuration for Suite A (2D LiDAR).
--
-- Cartographer performs scan-matching against a growing submap of the parking
-- lot. The quality of each scan match determines the covariance of the
-- published /scan_matched_odometry pose -- this covariance variation is the
-- primary uncertainty signal feeding the EKF and, ultimately, the RL policy.
--
-- Pipeline (Suite A):
--   /carla/ego_vehicle/lidar (PointCloud2, 1-channel, 360 deg)
--       -> Cartographer (horizontal slice via num_point_clouds=1)
--       -> /scan_matched_odometry (Odometry + covariance)
--       -> robot_localisation EKF (odom0 correction step)
--
-- @note The CARLA ROS bridge in passive mode publishes sensor TF frames
--       directly under "map" (e.g. map -> ego_vehicle/lidar) without an
--       intermediate "ego_vehicle" frame. Cartographer therefore tracks
--       "ego_vehicle/lidar" and ingests PointCloud2 directly via
--       num_point_clouds=1 (no LaserScan conversion needed).
--
-- Tuning guidance:
--   - TRANSLATION_WEIGHT / ROTATION_WEIGHT: balance how aggressively scan
--     matching constrains the pose. Higher values reduce covariance (less
--     uncertainty signal). Default values preserve meaningful covariance
--     variation across lot occupancy conditions.
--
-- @note cartographer_ros reads this file at startup via
--       -configuration_basename in carla_bridge.launch.py.
-- ==========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- TF frame names: the CARLA ROS bridge in passive mode publishes sensor
  -- frames directly under "map" (map -> ego_vehicle/lidar). There is no
  -- intermediate "ego_vehicle" frame, so we track the LiDAR frame directly.
  map_frame = "map",
  tracking_frame = "ego_vehicle/lidar",
  published_frame = "ego_vehicle/lidar",
  odom_frame = "odom",

  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,

  -- PointCloud2 input: the CARLA bridge always publishes LiDAR data as
  -- PointCloud2 regardless of channel count. Cartographer extracts a
  -- horizontal slice for 2D scan-matching.
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,   -- 200 Hz (faster than EKF at 20 Hz)
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings
-- ---------------------------------------------------------------------------

TRAJECTORY_BUILDER_2D.min_range = 0.1          -- metres (ignore returns < 10 cm)
TRAJECTORY_BUILDER_2D.max_range = 30.0         -- metres (SICK TiM 5xx max range)
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = false     -- IMU goes to EKF directly

-- Adaptive voxel filter: target ~200 points per scan after downsampling
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)

-- Ceres scan matcher weights: preserve meaningful covariance variation
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Submap size: 90 scans per submap for compact parking lot
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05   -- 5 cm grid

-- ---------------------------------------------------------------------------
-- Pose graph (loop closure) settings
-- ---------------------------------------------------------------------------

POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
