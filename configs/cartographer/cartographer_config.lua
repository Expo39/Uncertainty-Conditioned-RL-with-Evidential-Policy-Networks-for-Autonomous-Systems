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
--   /carla/ego_vehicle/lidar (PointCloud2, 360 deg, single channel)
--       -> Cartographer (num_point_clouds=1, raw PointCloud2 scan matching)
--       -> TF: odom -> ego_vehicle/lidar
--       -> tf_to_odom node -> /scan_matched_odometry (Odometry, dynamic covariance)
--       -> robot_localisation EKF (odom0 correction step)
--
-- The tf_to_odom node publishes dynamic covariance: inflated when the TF is
-- stale (Cartographer lost scan-match lock) or jumps (relocalisation event).
-- This varying covariance is the primary uncertainty signal in the RL policy.
--
-- @note tracking_frame=ego_vehicle/lidar avoids a TF conflict: the CARLA bridge
--       publishes map -> ego_vehicle/imu directly, preventing Cartographer from
--       owning that frame. use_imu_data=false because Cartographer requires the
--       IMU frame to be colocated with the tracking frame (< 1e-5 m offset).
--       IMU yaw rate still feeds the EKF imu0 prediction step independently.
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

  -- Raw PointCloud2 input (360 deg). CARLA's ray_cast does not collide with the
  -- parent actor, so the full 360 deg is used for both mapping and localisation.
  -- On the real robot (TiM571, 270 deg), rebuild the pbstream with the physical
  -- sensor -- both phases will consistently use 270 deg.
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

TRAJECTORY_BUILDER_2D.min_range = 0.5          -- metres (ignore close returns from vehicle body)
TRAJECTORY_BUILDER_2D.max_range = 25.0         -- metres (SICK TiM571 max range)
-- Set to max_range so missing rays do not insert phantom short obstacles into
-- the submap, which would corrupt scan matching against the real environment.
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 25.0
-- IMU disabled: colocation requirement -- IMU frame is not at ego_vehicle/lidar.
TRAJECTORY_BUILDER_2D.use_imu_data = false

-- Adaptive voxel filter: min_num_points must be below actual scan size.
-- CARLA 2D LiDAR at 20 Hz / 2M pts_per_second produces ~64 pts per scan.
-- Setting 200 caused every scan to be discarded; 20 passes all scans through.
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_length = 0.5
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.min_num_points = 20
TRAJECTORY_BUILDER_2D.adaptive_voxel_filter.max_range = 25.0

-- Wider search window to compensate for IMU-free operation between scans.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.5
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(30.0)

-- Higher rotation weight: yaw drift is the dominant error without IMU.
-- Preserve meaningful covariance variation across lot occupancy conditions.
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 100.0

-- Submap size: 30 scans per submap for compact parking lot (~65 x 45 m).
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 30
TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05   -- 5 cm grid

-- ---------------------------------------------------------------------------
-- Pose graph (loop closure) settings
-- ---------------------------------------------------------------------------

-- Raised from 0.50: accepting low-score matches with sparse 64-pt scans
-- introduces false loop closures that corrupt the map.
POSE_GRAPH.constraint_builder.min_score = 0.60
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.65
POSE_GRAPH.optimize_every_n_nodes = 35
POSE_GRAPH.optimization_problem.huber_scale = 1e1

return options
