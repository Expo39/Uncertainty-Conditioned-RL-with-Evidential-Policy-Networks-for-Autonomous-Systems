-- ==========================================================================
-- cartographer_config_loc.lua
-- @brief Google Cartographer 2D pure-localisation config for Suite A (2D LiDAR).
--
-- This config is used during training after the parking lot map has been
-- pre-built and serialised to a .pbstream file via the one-time SLAM mapping
-- workflow (make docker-map LAYOUT=<name>).
--
-- Cartographer runs in pure localisation mode: TRAJECTORY_BUILDER.pure_localization
-- freezes the pose graph completely. New scans are matched against the loaded
-- submap only -- no new nodes or submaps are ever created. This produces a
-- covariance signal that correctly correlates with driving difficulty:
-- empty lot (all cones visible) -> strong match -> low covariance (easy).
-- Crowded lot (cones occluded by NPCs) -> weak match -> high covariance (hard).
--
-- Compared to cartographer_config.lua (SLAM mode):
--   - TRAJECTORY_BUILDER.pure_localization_trimmer (max_submaps_to_keep = 3)
--   - optimize_every_n_nodes = 0  (belt-and-braces: also disables optimisation)
--   - num_range_data = 10  (fast convergence per episode; 30 not needed in loc mode)
--   - The .pbstream is loaded via --load_state_filename in carla_bridge.launch.py,
--     controlled by the CARTOGRAPHER_MAP env var.
--
-- @note cartographer_ros reads this file at startup via --configuration_basename.
--       The .pbstream is loaded via --load_state_filename (separate argument).
-- ==========================================================================

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  -- TF frame names: tracking ego_vehicle/lidar avoids a TF conflict where the
  -- CARLA bridge publishes map -> ego_vehicle/imu directly (37 Hz), which
  -- prevents Cartographer from publishing odom -> ego_vehicle/imu (two parents).
  -- ego_vehicle/lidar is the primary moving frame published by the bridge and
  -- has no Cartographer conflict. IMU data is still fused via the static TF
  -- ego_vehicle/lidar -> ego_vehicle -> ego_vehicle/imu in the launch file.
  map_frame = "map",
  tracking_frame = "ego_vehicle/lidar",
  published_frame = "ego_vehicle/lidar",
  odom_frame = "odom",

  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,

  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,

  -- Raw PointCloud2 input (360 deg), consistent with the SLAM mapping run.
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

-- 2D map builder in pure localisation mode.
-- The loaded .pbstream contains the frozen submap; no new submaps are created.
MAP_BUILDER.use_trajectory_builder_2d = true

-- Pure localisation: trim old submaps to keep only the frozen reference.
-- pure_localization_trimmer is required when using num_point_clouds=1
-- (TRAJECTORY_BUILDER.pure_localization only works with num_laser_scans).
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings (same as SLAM except num_range_data)
-- ---------------------------------------------------------------------------

TRAJECTORY_BUILDER_2D.min_range = 0.1
TRAJECTORY_BUILDER_2D.max_range = 25.0
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 5.0
TRAJECTORY_BUILDER_2D.use_imu_data = false  -- IMU disabled: tracking_frame=ego_vehicle/lidar has non-zero offset to IMU, violating Cartographer's colocation requirement

TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 50.0

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.2
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.0)

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Small submap window: in pure localisation, submaps are never finalised.
-- 10 scans (~0.5 s at 20 Hz) gives fast per-episode convergence.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 10
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

-- ---------------------------------------------------------------------------
-- Pose graph: disabled for pure localisation (no new constraints added)
-- ---------------------------------------------------------------------------

POSE_GRAPH.optimize_every_n_nodes = 0
POSE_GRAPH.constraint_builder.min_score = 0.50
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.55
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
