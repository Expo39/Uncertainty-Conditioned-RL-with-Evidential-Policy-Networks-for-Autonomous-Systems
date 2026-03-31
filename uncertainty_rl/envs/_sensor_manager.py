"""
@file _sensor_manager.py
@brief Sensor lifecycle manager for CARLAParkingEnv.

Owns all per-episode sensor state (IMU, LiDAR, collision sensor, camera) and
the spawn, callback, and cleanup logic for each. CARLAParkingEnv holds a
SensorManager instance and delegates sensor lifecycle calls to it.
"""

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import carla
except ImportError:
    carla = None  # Running without CARLA (CI or tests)

logger = logging.getLogger(__name__)


class SensorManager:
    """
    @class SensorManager
    @brief Manages sensor spawning, callbacks, and cleanup for one parking episode.

    Instantiated once by CARLAParkingEnv and reused across episodes. Call
    cleanup() at the start of each episode reset, then spawn() to attach
    sensors to the new ego vehicle. Sensor data is accessible via properties
    and consume methods.

    All CARLA world/vehicle handles are passed as parameters -- this class
    does not store world or vehicle references across calls to avoid holding
    stale references between episodes.
    """

    # Impulse threshold (N*s) below which a dynamic-actor collision is ignored.
    # A pedestrian walking into a stationary ego produces near-zero impulse;
    # this threshold filters those out so only ego-at-fault events are penalised.
    _DYNAMIC_COLLISION_IMPULSE_THRESHOLD: float = 500.0
    # Bytes per LiDAR point in the CARLA raw buffer: 4 float32 fields (x, y, z, intensity)
    _LIDAR_BYTES_PER_POINT: int = 16

    def __init__(
        self,
        sensors_config: Dict[str, Any],
        sensor_suite: str,
    ) -> None:
        """
        @brief Construct SensorManager with fixed config parameters.

        @param sensors_config: Sensor noise and mount config dict (carla_sensors
               section of train_config.yaml). Sub-keys: imu, lidar, lidar_3d,
               camera_rgb.
        @param sensor_suite: Which sensor suite to spawn:
               'suite_a' = 2D LiDAR + IMU,
               'suite_b' = 3D LiDAR + IMU,
               'suite_c' = 3D LiDAR + RGB camera + IMU.
        """
        self._sensors_config = sensors_config
        self._sensor_suite = sensor_suite

        # Per-episode sensor actor list
        self._spawned_sensors: List[Any] = []

        # Latest LiDAR point cloud in vehicle frame ([N, 3] float32 array).
        # Updated by _lidar_callback(). Used by CARLAParkingEnv._get_obstacle_features().
        self._latest_lidar_scan: Optional[np.ndarray] = None
        self._lidar_scan_lock = threading.Lock()

        # Collision state -- set by _on_collision(), consumed by CARLAParkingEnv.
        self._collision_detected: bool = False
        self._collision_impulse: float = 0.0

        # Reference to the NPC controller's patrol IDs set (shared by reference).
        # Populated by spawn() via the patrol_npc_ids argument.
        # Allows _on_collision to distinguish patrol vehicles from parked cars.
        self._patrol_npc_ids: Set[int] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def collision_detected(self) -> bool:
        """@brief True if a collision was detected since the last consume_collision()."""
        return self._collision_detected

    @property
    def collision_impulse(self) -> float:
        """@brief Impulse magnitude (N*s) of the last collision event."""
        return self._collision_impulse

    def consume_collision(self) -> bool:
        """
        @brief Read and clear the collision flag.
        @return True if a collision occurred since the last call.
        """
        detected = self._collision_detected
        self._collision_detected = False
        self._collision_impulse = 0.0
        return detected

    def get_latest_lidar_scan(self) -> Optional[np.ndarray]:
        """
        @brief Return a thread-safe snapshot of the latest LiDAR scan.
        @return (N, 3) float32 array in vehicle frame, or None if no scan yet.
        """
        with self._lidar_scan_lock:
            return (
                self._latest_lidar_scan.copy()
                if self._latest_lidar_scan is not None
                else None
            )

    def lidar_point_count(self) -> int:
        """
        @brief Return the number of points in the latest LiDAR scan (0 if none).
        @return Integer point count.
        """
        with self._lidar_scan_lock:
            return (
                len(self._latest_lidar_scan)
                if self._latest_lidar_scan is not None
                else 0
            )

    def spawn(
        self,
        world: Any,
        vehicle: Any,
        patrol_npc_ids: Set[int],
    ) -> None:
        """
        @brief Spawn all sensors for the configured suite attached to the ego vehicle.

        Dispatches to per-suite helpers based on self._sensor_suite. All suites
        include an IMU and a collision sensor.

        Suite A: 2D LiDAR (front bumper) + IMU.
        Suite B: 3D LiDAR (roof) + IMU.
        Suite C: 3D LiDAR (roof) + RGB camera (windscreen) + IMU.

        @param world: carla.World handle for the current episode.
        @param vehicle: Ego carla.Vehicle actor to attach sensors to.
        @param patrol_npc_ids: Mutable set of patrol vehicle actor IDs owned by
               NPCController. Shared by reference so _on_collision always sees
               the current contents even after NPCController mutates it.
        """
        if vehicle is None or world is None:
            return

        # Store the shared reference -- mutations from NPCController are visible
        self._patrol_npc_ids = patrol_npc_ids

        # IMU is present in all suites
        self._spawn_imu(world, vehicle)

        if self._sensor_suite == "suite_a":
            self._spawn_lidar_2d(world, vehicle)
        elif self._sensor_suite == "suite_b":
            self._spawn_lidar_3d(world, vehicle)
        elif self._sensor_suite == "suite_c":
            # @todo(AG) Camera is currently passive. Pending supervisor decision
            # on Suite C utility (visual odometry vs removal).
            self._spawn_lidar_3d(world, vehicle)
            self._spawn_camera_rgb(world, vehicle)
        else:
            logger.warning(
                "Unknown sensor_suite '%s'. Defaulting to suite_a (2D LiDAR + IMU).",
                self._sensor_suite,
            )
            self._spawn_lidar_2d(world, vehicle)

        # Collision sensor always spawned regardless of suite
        self._spawn_collision_sensor(world, vehicle)

    def cleanup(self) -> None:
        """
        @brief Stop and destroy all spawned sensors; reset collision and scan state.

        Called at the start of each episode reset before new sensors are spawned.

        Two-phase teardown:
        1. Stop all sensors (closes the underlying data stream).
        2. Brief sleep so the CARLA ROS bridge's _update_thread can process the
           stream closure before we call destroy().  Without this pause the bridge
           thread races to call sensor.stop() on a file descriptor the env has
           already closed, producing a "Bad file descriptor" crash that kills the
           bridge's update thread and hangs world.tick() in sync mode.
        3. Destroy all sensors.
        """
        alive = [s for s in self._spawned_sensors if s is not None and s.is_alive]
        for sensor in alive:
            try:
                sensor.stop()
            except Exception:
                pass
        # Give the bridge's _update_thread time to see the stream closure and
        # clean up its own wrapper before we call destroy().
        if alive:
            time.sleep(0.15)
        for sensor in alive:
            try:
                if sensor.is_alive:
                    sensor.destroy()
            except Exception:
                pass
        self._spawned_sensors.clear()

        self._collision_detected = False
        self._collision_impulse = 0.0

        # Clear stale scan so previous episode points are not used at the
        # start of the next episode before the first LiDAR tick arrives.
        with self._lidar_scan_lock:
            self._latest_lidar_scan = None

    # ------------------------------------------------------------------
    # Sensor spawn helpers
    # ------------------------------------------------------------------

    def _spawn_imu(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn IMU sensor at centre-of-mass height.

        Noise parameters and mount position come from sensors_config.imu.
        Mount defaults: x=0.0, y=0.0, z=0.3 (centre-of-mass height).

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        imu_config = self._sensors_config.get("imu", {})
        mount = imu_config.get("mount", {})

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        # role_name determines the ROS topic: /carla/ego_vehicle/<role_name>.
        # Each sensor needs a unique name to avoid topic collisions in the bridge.
        imu_bp.set_attribute("role_name", "imu")
        for attr, default in [
            ("noise_accel_stddev_x", 0.1),
            ("noise_accel_stddev_y", 0.1),
            ("noise_accel_stddev_z", 0.1),
            ("noise_gyro_stddev_x", 0.01),
            ("noise_gyro_stddev_y", 0.01),
            ("noise_gyro_stddev_z", 0.01),
            ("sensor_tick", 0.05),
        ]:
            imu_bp.set_attribute(attr, str(imu_config.get(attr, default)))

        imu_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 0.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 0.3)),
            )
        )
        imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        # Register a no-op listener so CARLA considers the stream open.
        # Without this, sensor.stop() during cleanup emits:
        # "attempting to unsubscribe from stream but sensor wasn't listening".
        imu_sensor.listen(lambda _: None)
        self._spawned_sensors.append(imu_sensor)

    def _spawn_lidar_2d(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn 2D LiDAR sensor at front bumper height (Suite A).

        Single-channel horizontal scan (SICK TiM 5xx / Hokuyo style). CARLA
        ray_cast scans 360 deg; Cartographer receives raw PointCloud2 directly.

        Config key: sensors_config.lidar. Mount defaults: x=2.4, z=0.5.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        lidar_config = self._sensors_config.get("lidar", {})
        mount = lidar_config.get("mount", {})

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("role_name", "lidar")
        for attr, default in [
            ("channels", 1),
            ("range", 30.0),
            ("points_per_second", 56000),
            ("rotation_frequency", 10.0),
            ("upper_fov", 0.0),
            ("lower_fov", 0.0),
            ("sensor_tick", 0.05),
        ]:
            lidar_bp.set_attribute(attr, str(lidar_config.get(attr, default)))

        lidar_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 2.4)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 0.5)),
            )
        )
        lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=vehicle
        )
        # Register callback to update the LiDAR scan cache for obstacle features
        lidar_sensor.listen(self._lidar_callback)
        self._spawned_sensors.append(lidar_sensor)

    def _spawn_lidar_3d(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn 3D LiDAR sensor at roof centre (Suite B and C).

        Multi-channel scan (Velodyne VLP-16 style). Provides richer point clouds
        for Cartographer and denser obstacle proximity information.

        Config key: sensors_config.lidar_3d. Mount defaults: x=0.0, z=1.5.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        lidar3d_config = self._sensors_config.get("lidar_3d", {})
        mount = lidar3d_config.get("mount", {})

        lidar_bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("role_name", "lidar_3d")
        for attr, default in [
            ("channels", 16),
            ("range", 100.0),
            ("points_per_second", 300000),
            ("rotation_frequency", 10.0),
            ("upper_fov", 15.0),
            ("lower_fov", -15.0),
            ("sensor_tick", 0.05),
        ]:
            lidar_bp.set_attribute(attr, str(lidar3d_config.get(attr, default)))

        lidar_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 0.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 1.5)),
            )
        )
        lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=vehicle
        )
        lidar_sensor.listen(self._lidar_callback)
        self._spawned_sensors.append(lidar_sensor)

    def _spawn_camera_rgb(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn forward-facing RGB camera at windscreen height (Suite C).

        @note The camera is currently passive: spawned and registered so it
              appears in CARLA diagnostics, but data is not consumed by the
              RL observation or EKF pipeline. The listener is a no-op.

        @todo(AG) Pending supervisor decision on Suite C utility.

        Config key: sensors_config.camera_rgb.
        Mount defaults: x=2.0, z=1.2, pitch=-5 deg.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        cam_config = self._sensors_config.get("camera_rgb", {})
        mount = cam_config.get("mount", {})

        cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        cam_bp.set_attribute("role_name", "rgb_front")
        for attr, default in [
            ("image_size_x", 640),
            ("image_size_y", 480),
            ("fov", 90.0),
            ("sensor_tick", 0.05),
        ]:
            cam_bp.set_attribute(attr, str(cam_config.get(attr, default)))

        cam_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 2.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 1.2)),
            ),
            carla.Rotation(pitch=float(mount.get("pitch", -5.0))),
        )
        cam_sensor = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
        # Camera data not consumed; listener is a no-op placeholder so the sensor
        # is registered and visible in CARLA diagnostics.
        cam_sensor.listen(lambda _: None)
        self._spawned_sensors.append(cam_sensor)

    def _spawn_collision_sensor(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn a CARLA collision sensor attached to the ego vehicle.

        The sensor fires _on_collision() on any physical contact. The callback
        sets _collision_detected and records the impulse magnitude so
        CARLAParkingEnv._compute_reward() can decide whether to penalise.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        bp = world.get_blueprint_library().find("sensor.other.collision")
        # Set role_name so the CARLA ROS bridge (register_all_sensors=True) can
        # namespace this sensor's topic distinctly from other ego_vehicle topics.
        # Without this, the bridge reuses "carla/ego_vehicle/front" and crashes
        # with a type-incompatible publisher error.
        bp.set_attribute("role_name", "collision")
        sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        sensor.listen(self._on_collision)
        self._spawned_sensors.append(sensor)

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _on_collision(self, event: Any) -> None:
        """
        @brief Collision sensor callback.

        Fires when the ego vehicle makes physical contact with any actor.
        Records the collision so CARLAParkingEnv._compute_reward() can apply
        the penalty.

        Dynamic actors (pedestrians, patrol NPCs) only set the flag when the
        ego vehicle was moving at the time (impulse > threshold), preventing
        a parked/slow ego from being penalised when a pedestrian walks into it.

        @param event: carla.CollisionEvent with other_actor and normal_impulse
                      fields.
        """
        other = event.other_actor
        impulse = event.normal_impulse
        impulse_magnitude = math.sqrt(
            impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2
        )

        is_pedestrian = other.type_id.startswith("walker.pedestrian")
        is_patrol = other.id in self._patrol_npc_ids
        is_dynamic = is_pedestrian or is_patrol

        if is_dynamic:
            # Only penalise dynamic actor collisions when ego was at fault
            # (impulse above threshold indicates ego was moving into them).
            # A pedestrian walking into a stationary ego produces near-zero impulse.
            if impulse_magnitude > self._DYNAMIC_COLLISION_IMPULSE_THRESHOLD:
                self._collision_detected = True
                self._collision_impulse = impulse_magnitude
                logger.debug(
                    "[collision] dynamic  actor=%s  impulse=%.1f N*s",
                    other.type_id,
                    impulse_magnitude,
                )
            else:
                logger.debug(
                    "[collision] dynamic IGNORED (low impulse)  actor=%s  "
                    "impulse=%.1f N*s",
                    other.type_id,
                    impulse_magnitude,
                )
        else:
            # Static objects (cones, parked cars, walls, perimeter): always penalise
            self._collision_detected = True
            self._collision_impulse = impulse_magnitude
            logger.debug(
                "[collision] static  actor=%s  impulse=%.1f N*s",
                other.type_id,
                impulse_magnitude,
            )

    def _lidar_callback(self, lidar_data: Any) -> None:
        """
        @brief CARLA LiDAR sensor callback -- caches point cloud for obstacle features.

        Converts the raw measurement to a (N, 3) float32 numpy array in vehicle
        frame (x-forward, y-left, z-up). Thread-safe via _lidar_scan_lock.

        @note CARLA ray_cast encodes each point as 4 float32 values (x, y, z,
              intensity) in a flat byte buffer. CARLA uses a left-handed coordinate
              system; y is negated to convert to the ROS right-handed convention
              (positive y = left).

        @param lidar_data: carla.LidarMeasurement from the ray_cast sensor.
        """
        raw = lidar_data.raw_data
        n_bytes = len(raw)
        n_points = n_bytes // self._LIDAR_BYTES_PER_POINT
        if n_points == 0:
            return

        arr = np.frombuffer(raw, dtype=np.float32).reshape(n_points, 4)
        # Negate y: CARLA left-handed -> ROS right-handed (positive y = left)
        points_xyz = arr[:, :3].copy()
        points_xyz[:, 1] *= -1.0

        with self._lidar_scan_lock:
            self._latest_lidar_scan = points_xyz
