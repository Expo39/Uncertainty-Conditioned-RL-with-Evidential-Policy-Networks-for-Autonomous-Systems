-- ==========================================================================
-- cartographer_config_3d.lua
-- @brief Google Cartographer 2D SLAM configuration for Suite B and C
--        (16-channel 3D LiDAR, Velodyne VLP-16 style, roof-mounted).
--
-- Suite B/C use a 16-channel 3D LiDAR. Cartographer runs the 2D trajectory
-- builder, projecting all channels onto the 2D plane via the min_z/max_z
-- horizontal slice. This gives denser 2D scans than Suite A's single-channel
-- LiDAR: each cone is hit by multiple vertical channels, improving scan
-- match robustness and producing more informative covariance variation.
--
-- 3D Cartographer (use_trajectory_builder_3d) was evaluated but fails on
-- sparse cone-perimeter environments: insufficient vertical structure for
-- 6DOF convergence. Reserved for real-world deployment where walls,
-- pillars, and ceilings provide rich 3D geometry.
--
-- Pipeline (Suite B/C):
--   /carla/ego_vehicle/lidar_3d (PointCloud2, 16-channel, 360 deg)
--       -> Cartographer 2D (project all channels via min_z/max_z slice)
--       -> TF: odom -> ego_vehicle/imu
--       -> tf_to_odom node -> /scan_matched_odometry (Odometry)
--       -> robot_localisation EKF (odom0 correction step)
--
-- @note IMU is mandatory (tracking_frame = ego_vehicle/imu).
-- @note Suite C also spawns an RGB camera -- not consumed by Cartographer.
-- ==========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- TF frame names: tracking ego_vehicle/imu satisfies Cartographer's IMU
  -- colocation requirement. 3D LiDAR data (ego_vehicle/lidar_3d) is
  -- transformed to the tracking frame via the static TF tree.
  map_frame = "map",
  tracking_frame = "ego_vehicle/imu",
  published_frame = "ego_vehicle/imu",
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

TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 50.0          -- Parking lot is ~65 x 45 m
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = true       -- IMU motion prior for sparse environment alignment

-- Z-range for horizontal slice extraction from 3D point cloud (sensor frame).
-- The 3D LiDAR is roof-mounted at z=1.9 in vehicle frame. Cones at ground
-- level are at z ~ -1.5 to -2.2 in sensor frame. Widen min_z to capture all
-- returns from ground level objects. max_z=0.5 excludes upward-looking rays.
TRAJECTORY_BUILDER_2D.min_z = -2.5
TRAJECTORY_BUILDER_2D.max_z = 0.5

-- Denser point cloud from 16-layer fan: target 400 points after downsampling
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 400
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.2
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)

-- Same weights as Suite A to preserve comparable covariance variation
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Submap size: 30 scans per submap for compact parking lot
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 30
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05   -- 5 cm grid

-- ---------------------------------------------------------------------------
-- Pose graph (loop closure) settings
-- ---------------------------------------------------------------------------

POSE_GRAPH.constraint_builder.min_score = 0.50
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.55
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
