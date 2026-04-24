"""
@file _sensor_manager.py
@brief Sensor lifecycle manager for CARLAParkingEnv.

Owns all per-episode sensor state (IMU, GNSS, 2D LiDAR, collision sensor)
and the spawn, callback, and cleanup logic for each. CARLAParkingEnv holds
a SensorManager instance and delegates sensor lifecycle calls to it.

Sensor roles:
  - RTK-GNSS + IMU: localisation via robot_localisation EKF.
  - 2D LiDAR: obstacle detection only (obs indices 15-19). NOT localisation.
  - Collision sensor: terminal reward signal.
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
    # A pedestrian walking into a stationary ego produces ~1-10 N*s impulse;
    # the ego driving into a pedestrian at parking speeds (~1-3 m/s) produces
    # ~50-200 N*s. Threshold set to 50.0 to catch ego-at-fault impacts while
    # filtering incidental contact from a pedestrian bumping a stopped vehicle.
    _DYNAMIC_COLLISION_IMPULSE_THRESHOLD: float = 50.0
    # Bytes per LiDAR point in CARLA raw buffer: 4 float32 fields
    # (x, y, z, intensity)
    _LIDAR_BYTES_PER_POINT: int = 16

    def __init__(
        self,
        sensors_config: Dict[str, Any],
    ) -> None:
        """
        @brief Construct SensorManager with fixed config parameters.

        @param sensors_config: Sensor noise and mount config dict (carla_sensors
               section of env_config.yaml). Sub-keys: imu, lidar, gnss.
        """
        self._sensors_config = sensors_config

        # Per-episode sensor actor list
        self._spawned_sensors: List[Any] = []

        # Latest LiDAR point cloud in vehicle frame ([N, 3] float32).
        # Updated by _lidar_callback(). Used by
        # CARLAParkingEnv._get_obstacle_features().
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
        """
        @brief True if a collision was detected since last
               consume_collision() call.
        """
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
        gnss_noise_multiplier: float = 1.0,
    ) -> None:
        """
        @brief Spawn all sensors attached to the ego vehicle.

        Always spawns: IMU + GNSS (front) + GNSS (rear, if gnss_rear in config)
        + 2D LiDAR + collision sensor. GNSS noise is scaled by
        gnss_noise_multiplier to simulate different RTK fix states
        (fixed, float, standalone, degraded). The rear antenna feeds
        GnssNoiseRelayNode's dual-antenna baseline heading on /gnss/heading.

        @param world: carla.World handle for the current episode.
        @param vehicle: Ego carla.Vehicle actor to attach sensors to.
        @param patrol_npc_ids: Mutable set of patrol vehicle actor IDs owned by
               NPCController. Shared by reference so _on_collision always sees
               the current contents even after NPCController mutates it.
        @param gnss_noise_multiplier: Scale factor applied to base GNSS noise
               stddevs. 1.0 = RTK fixed (~2 cm). Higher values simulate
               degraded fix states (e.g. 15.0 for RTK float, 100.0 for standalone).
        """
        if vehicle is None or world is None:
            return

        # Store the shared reference -- mutations from NPCController are visible
        self._patrol_npc_ids = patrol_npc_ids

        self._spawn_imu(world, vehicle)
        self._spawn_gnss(world, vehicle, gnss_noise_multiplier)
        self._spawn_gnss_rear(world, vehicle, gnss_noise_multiplier)
        self._spawn_lidar_2d(world, vehicle)
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

    def reset_state(self) -> None:
        """
        @brief Reset per-episode sensor state without touching CARLA actors.

        Used when the ego vehicle is reused across episodes (teleport path).
        Clears collision flags and stale LiDAR data so the new episode starts
        clean, without the destroy/respawn cycle that stresses the ROS bridge.
        """
        self._collision_detected = False
        self._collision_impulse = 0.0
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
        @brief Spawn 2D LiDAR sensor at front bumper height for obstacle detection.

        Single-channel horizontal scan (SICK TiM 5xx / Hokuyo style). Feeds
        obstacle clearance features (obs indices 15-19) only -- NOT used for
        localisation (that role belongs to RTK-GNSS + IMU).

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
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        # Register callback to update the LiDAR scan cache for obstacle features
        lidar_sensor.listen(self._lidar_callback)
        self._spawned_sensors.append(lidar_sensor)

    def _spawn_gnss(
        self,
        world: Any,
        vehicle: Any,
        noise_multiplier: float = 1.0,
    ) -> None:
        """
        @brief Spawn GNSS sensor at roof antenna position.

        CARLA's sensor.other.gnss outputs WGS84 lat/lon at the configured
        noise level. The base noise stddevs (from sensors_config.gnss) model
        RTK-fixed conditions (~2 cm). The noise_multiplier scales these to
        simulate degraded RTK states (float, standalone, degraded).

        The CARLA ROS bridge publishes this as sensor_msgs/NavSatFix on
        /carla/ego_vehicle/gnss. The GnssNoiseRelay node in the ros2-bridge
        adds additional noise, converts to local XY via flat-earth projection,
        and publishes Odometry on /odometry/gps for the EKF.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        @param noise_multiplier: Scale factor for base noise stddevs.
               1.0 = RTK fixed. Higher = degraded fix state.
        """
        gnss_config = self._sensors_config.get("gnss", {})
        mount = gnss_config.get("mount", {})

        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("role_name", "gnss")

        # Scale base noise by the per-episode multiplier
        base_lat = float(gnss_config.get("noise_lat_stddev", 0.0000002))
        base_lon = float(gnss_config.get("noise_lon_stddev", 0.0000002))
        base_alt = float(gnss_config.get("noise_alt_stddev", 0.05))

        for attr, value in [
            ("noise_alt_bias", gnss_config.get("noise_alt_bias", 0.0)),
            ("noise_alt_stddev", base_alt * noise_multiplier),
            ("noise_lat_bias", gnss_config.get("noise_lat_bias", 0.0)),
            ("noise_lat_stddev", base_lat * noise_multiplier),
            ("noise_lon_bias", gnss_config.get("noise_lon_bias", 0.0)),
            ("noise_lon_stddev", base_lon * noise_multiplier),
            ("sensor_tick", gnss_config.get("sensor_tick", 0.05)),
        ]:
            gnss_bp.set_attribute(attr, str(value))

        gnss_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 0.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 1.8)),
            )
        )
        gnss_sensor = world.spawn_actor(
            gnss_bp, gnss_transform, attach_to=vehicle
        )
        # Register a no-op listener so CARLA considers the stream open.
        # The ROS bridge publishes NavSatFix from the GNSS data automatically.
        gnss_sensor.listen(lambda _: None)
        self._spawned_sensors.append(gnss_sensor)
        logger.debug(
            "Spawned GNSS sensor (noise_multiplier=%.1f, lat_stddev=%.10f deg).",
            noise_multiplier,
            base_lat * noise_multiplier,
        )

    def _spawn_gnss_rear(
        self,
        world: Any,
        vehicle: Any,
        noise_multiplier: float = 1.0,
    ) -> None:
        """
        @brief Spawn the rear RTK-GNSS antenna for dual-antenna heading.

        The rear antenna, together with the front antenna spawned in _spawn_gnss(),
        forms a 1.5 m baseline along the vehicle longitudinal axis. GnssNoiseRelayNode
        derives heading from atan2(dy, dx) of the noisy baseline vector.

        Skipped silently when sensors_config has no gnss_rear section (backwards
        compatible with configs that predate the dual-antenna architecture).

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        @param noise_multiplier: Same tier multiplier as the front antenna.
        """
        gnss_rear_config = self._sensors_config.get("gnss_rear", {})
        if not gnss_rear_config:
            return

        mount = gnss_rear_config.get("mount", {})

        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("role_name", "gnss_rear")

        base_lat = float(gnss_rear_config.get("noise_lat_stddev", 0.0000002))
        base_lon = float(gnss_rear_config.get("noise_lon_stddev", 0.0000002))
        base_alt = float(gnss_rear_config.get("noise_alt_stddev", 0.05))

        for attr, value in [
            ("noise_alt_bias", gnss_rear_config.get("noise_alt_bias", 0.0)),
            ("noise_alt_stddev", base_alt * noise_multiplier),
            ("noise_lat_bias", gnss_rear_config.get("noise_lat_bias", 0.0)),
            ("noise_lat_stddev", base_lat * noise_multiplier),
            ("noise_lon_bias", gnss_rear_config.get("noise_lon_bias", 0.0)),
            ("noise_lon_stddev", base_lon * noise_multiplier),
            ("sensor_tick", gnss_rear_config.get("sensor_tick", 0.05)),
        ]:
            gnss_bp.set_attribute(attr, str(value))

        gnss_rear_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", -1.5)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 1.6)),
            )
        )
        gnss_rear_sensor = world.spawn_actor(
            gnss_bp, gnss_rear_transform, attach_to=vehicle
        )
        # No-op listener to keep the CARLA stream open.
        # The ROS bridge publishes NavSatFix on /carla/ego_vehicle/gnss_rear.
        gnss_rear_sensor.listen(lambda _: None)
        self._spawned_sensors.append(gnss_rear_sensor)
        logger.debug(
            "Spawned rear GNSS sensor at x=%.2f (noise_multiplier=%.1f).",
            float(mount.get("x", -1.5)),
            noise_multiplier,
        )

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
        impulse_magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

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
        # Negate y: CARLA left-handed (y rightward) -> vehicle frame (y leftward).
        points_xyz = arr[:, :3].copy()
        points_xyz[:, 1] *= -1.0

        with self._lidar_scan_lock:
            self._latest_lidar_scan = points_xyz
