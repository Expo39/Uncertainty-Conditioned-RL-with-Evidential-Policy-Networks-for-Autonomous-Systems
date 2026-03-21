-- ==========================================================================
-- cartographer_config_3d.lua
-- @brief Google Cartographer 2D SLAM configuration for Suite B and C
--        (16-channel 3D LiDAR, Velodyne VLP-16 style, roof-mounted).
--
-- Suite B/C use a multi-channel 3D LiDAR (CARLA sensor.lidar.ray_cast,
-- 16 channels). Cartographer receives this as a PointCloud2 topic rather
-- than a LaserScan. The 2D map builder extracts a horizontal slice from
-- the point cloud for scan-matching -- functionally equivalent to Suite A
-- but with denser returns from the 16-layer fan.
--
-- Differences from cartographer_config.lua (Suite A):
--   - num_laser_scans = 0  (not a single-channel flat scan)
--   - num_point_clouds = 1 (multi-channel PointCloud2 from 3D LiDAR)
--   - min_range / max_range updated for VLP-16 specs (100 m)
--   - Voxel filter target increased to 400 points (denser 3D scan)
--   - TRANSLATION_WEIGHT / ROTATION_WEIGHT unchanged (same covariance
--     variation target)
--
-- Pipeline (Suite B/C):
--   /carla/ego_vehicle/lidar_3d (PointCloud2, 16-channel, 360 deg)
--       -> Cartographer (horizontal slice scan-matching)
--       -> /scan_matched_odometry (Odometry + covariance)
--       -> EKF odom0 correction -> /odometry/filtered
--
-- @note The launch file selects this config via CARTOGRAPHER_CONFIG_PATH
--       when sensor_suite is suite_b or suite_c.
-- @note Suite C also spawns an RGB camera (front windscreen). The camera
--       feed is available on /carla/ego_vehicle/rgb_front but is not
--       consumed by Cartographer. Visual odometry (ORB-SLAM3 or similar)
--       would be a separate pipeline -- out of scope for this dissertation.
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

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings (3D LiDAR input)
-- ---------------------------------------------------------------------------

-- VLP-16 range: 0.1 m to 100 m. Use only returns within bumper height
-- slice (z approximately -0.5 to +0.5 m relative to sensor at z=1.5 m)
-- for 2D map building. Cartographer handles this via the voxel filter.
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

-- Same weights as Suite A to preserve comparable covariance variation
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

TRAJECTORY_BUILDER_2D.submaps.num_range_data = 90
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

-- ---------------------------------------------------------------------------
-- Pose graph (loop closure) settings
-- ---------------------------------------------------------------------------

POSE_GRAPH.constraint_builder.min_score = 0.55
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.60
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
