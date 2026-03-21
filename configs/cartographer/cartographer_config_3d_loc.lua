-- ==========================================================================
-- cartographer_config_3d_loc.lua
-- @brief Google Cartographer 2D pure-localisation config for Suite B and C
--        (16-channel 3D LiDAR, Velodyne VLP-16 style, roof-mounted).
--
-- Pure localisation variant of cartographer_config_3d.lua. Matches against a
-- frozen .pbstream built during the one-time SLAM mapping run.
--
-- Compared to cartographer_config_3d.lua (SLAM mode):
--   - TRAJECTORY_BUILDER.pure_localization = true
--   - optimize_every_n_nodes = 0
--   - num_range_data = 10 (fast per-episode convergence)
--
-- Pipeline (Suite B/C, loc mode):
--   /carla/ego_vehicle/lidar_3d (PointCloud2, 16-channel, 360 deg)
--       -> Cartographer (match against frozen .pbstream)
--       -> /scan_matched_odometry (Odometry + covariance)
--       -> EKF odom0 correction -> /odometry/filtered
--
-- @note cartographer_ros reads this file at startup via --configuration_basename.
--       The .pbstream is loaded via --load_state_filename (separate argument).
-- ==========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- TF frame names: the CARLA ROS bridge in passive mode publishes sensor
  -- frames directly under "map" (map -> ego_vehicle/lidar_3d). There is no
  -- intermediate "ego_vehicle" frame, so we track the 3D LiDAR frame directly.
  map_frame = "map",
  tracking_frame = "ego_vehicle/lidar_3d",
  published_frame = "ego_vehicle/lidar_3d",
  odom_frame = "odom",

  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,

  -- 3D LiDAR as PointCloud2 (16-channel VLP-16 style)
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- Pure localisation: freeze the pose graph entirely. The loaded .pbstream
-- is the only reference for scan matching.
TRAJECTORY_BUILDER.pure_localization = true

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings (same as SLAM except num_range_data)
-- ---------------------------------------------------------------------------

TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 100.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = false

-- Denser point cloud from 16-layer fan: target 400 points after downsampling
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 400
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Small submap window for fast per-episode convergence in loc mode
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 10
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

-- ---------------------------------------------------------------------------
-- Pose graph: disabled for pure localisation (no new constraints added)
-- ---------------------------------------------------------------------------

POSE_GRAPH.optimize_every_n_nodes = 0
POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
