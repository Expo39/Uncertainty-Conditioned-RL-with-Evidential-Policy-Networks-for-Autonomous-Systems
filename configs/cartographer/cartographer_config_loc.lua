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

  -- TF frame names.
  -- tracking_frame = ego_vehicle/lidar: Cartographer tracks the LiDAR frame.
  -- published_frame = ego_vehicle/lidar: Cartographer publishes
  --   carto_map -> ego_vehicle/lidar (provide_odom_frame = false).
  --
  -- map_frame = "carto_map" (NOT "map"): avoids TF conflict with the CARLA
  -- bridge, which publishes map -> ego_vehicle/lidar at 37 Hz with ground-truth
  -- pose. Using a distinct frame name means Cartographer owns a separate TF edge
  -- (carto_map -> ego_vehicle/lidar) and the two never collide.
  --
  -- provide_odom_frame = false: Cartographer publishes the map-frame pose
  -- directly (globally anchored to the pbstream), not a drifting odom estimate.
  -- This gives consistent position accumulation across the episode.
  --
  -- tf_to_odom looks up carto_map -> ego_vehicle/lidar and publishes Odometry.
  -- The EKF uses world_frame = "carto_map" (set in ros2_config.yaml).
  map_frame = "carto_map",
  tracking_frame = "ego_vehicle/lidar",
  published_frame = "ego_vehicle/lidar",
  odom_frame = "odom",
  provide_odom_frame = false,
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

-- Pure localisation via trimmer: keeps 2 live submaps for initialisation
-- against the frozen .pbstream reference.  1 submap is insufficient for
-- Cartographer to bootstrap; 3 enables multi-hypothesis oscillation.
-- 2 is the minimum that allows initialisation while limiting drift.
-- TRAJECTORY_BUILDER_2D.pure_localization is NOT supported in the Jazzy
-- Cartographer build (crashes with "Key 'pure_localization' was used the wrong
-- number of times"). The trimmer is the correct API for this build.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 2,
}

-- ---------------------------------------------------------------------------
-- 2D trajectory builder settings (same as SLAM except num_range_data)
-- ---------------------------------------------------------------------------

TRAJECTORY_BUILDER_2D.min_range = 0.5
TRAJECTORY_BUILDER_2D.max_range = 25.0
-- Set missing_data_ray_length to max_range so phantom short rays do not
-- pollute the submap with fake obstacles.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 25.0
-- IMU disabled: tracking_frame=ego_vehicle/lidar has non-zero offset to IMU,
-- violating Cartographer's colocation requirement.
TRAJECTORY_BUILDER_2D.use_imu_data = false

-- Adaptive voxel filter: 100k raw rays -> ~64 pts at 0.5 m voxels.
-- min_num_points must be below the actual filtered count (64) so scans
-- are not silently discarded. Reduced max_length from 0.5 to 0.3 m to
-- retain more points (~150-200) for more reliable scan matching on the
-- sparse cone perimeter.
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.3
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 40
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 25.0

-- Search window: 0.5 m linear (matches SLAM config), 30 deg angular.
-- At 3 m/s max speed the vehicle moves 0.15 m per scan -- well within
-- the 0.5 m window. Wider windows (1.5 m) caused false matches at wrong
-- positions in the frozen pbstream. The trajectory restart per episode
-- (finish_trajectory + start_trajectory) prevents drift from accumulating
-- across episodes, so the window only needs to cover per-scan motion.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.5
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(30.0)

-- Increase rotation weight: yaw drift is the dominant error source without IMU.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 100.0

-- Submap window: 10 scans (0.5 s at 20 Hz). Fast turnover means live
-- submaps are discarded quickly (max_submaps_to_keep = 2), preventing
-- stale scan-match hypotheses from persisting. Combined with trajectory
-- restart at each episode reset, this gives ~0.5 s initial convergence.
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 10
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05

-- ---------------------------------------------------------------------------
-- Pose graph: disabled for pure localisation (no new constraints added)
-- ---------------------------------------------------------------------------

POSE_GRAPH.optimize_every_n_nodes = 0
-- min_score raised to 0.75: the sparse 64-point scans on symmetric cone
-- geometry produce borderline 67-81% matches at wrong positions. 0.75
-- rejects the ambiguous matches while still accepting strong ones (>80%).
POSE_GRAPH.constraint_builder.min_score = 0.75
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.80
-- Sampling ratio for loop closure: check every node against the pbstream.
POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
