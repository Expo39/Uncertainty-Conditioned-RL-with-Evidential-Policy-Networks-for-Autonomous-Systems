"""
@file _sensor_manager.py
@brief Sensor lifecycle manager for CARLAParkingEnv.

Owns all per-episode sensor state (IMU, GNSS, 2D LiDAR, collision sensor)
and the spawn, callback, and cleanup logic for each. CARLAParkingEnv holds
a SensorManager instance and delegates sensor lifecycle calls to it.

Sensor roles:
  - RTK-GNSS + IMU: localisation via robot_localisation EKF.
  - 2D LiDAR: obstacle detection only (obs indices 7-11). NOT localisation.
  - Collision sensor: terminal reward signal.
"""

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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

    All CARLA world/vehicle handles are passed as parameters - this class
    does not store world or vehicle references across calls to avoid holding
    stale references between episodes.
    """

    # Ego speed (m/s) below which the ego is considered stationary for fault
    # determination.
    _EGO_FAULT_SPEED_THRESHOLD_MS: float = 0.3
    # Bytes per LiDAR point in CARLA raw buffer: 4 float32 fields
    # (x, y, z, intensity)
    _LIDAR_BYTES_PER_POINT: int = 16

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        # Updated by _lidar_callback(); read via get_latest_lidar_scan().
        self._latest_lidar_scan: Optional[np.ndarray] = None
        self._lidar_scan_lock = threading.Lock()

        # Collision state - set by _on_collision(), consumed by CARLAParkingEnv.
        self._collision_detected: bool = False
        self._collision_ego_fault: bool = False

        # Reference to the NPC controller's patrol IDs set (shared by reference).
        # Populated by spawn() via the patrol_npc_ids argument.
        # Allows _on_collision to distinguish patrol vehicles from parked cars.
        self._patrol_npc_ids: Set[int] = set()

        # Ego vehicle actor - stored at spawn() so _on_collision can query
        # ego speed for pedestrian fault determination.
        self._ego_vehicle: Optional[Any] = None

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

    def consume_collision(self) -> Tuple[bool, bool]:
        """
        @brief Read and clear the collision flag.
        @return Tuple of (detected, ego_fault). detected is True if any
                collision occurred. ego_fault is True only when the ego
                was responsible (moving above threshold at moment of contact).
                Callers should terminate on detected but only penalise on ego_fault.
        """
        detected = self._collision_detected
        ego_fault = self._collision_ego_fault
        self._collision_detected = False
        self._collision_ego_fault = False
        return detected, ego_fault

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
        @brief Spawn all sensors attached to the ego vehicle.

        Always spawns: IMU + GNSS + 2D LiDAR + collision sensor.

        @param world: carla.World handle for the current episode.
        @param vehicle: Ego carla.Vehicle actor to attach sensors to.
        @param patrol_npc_ids: Mutable set of patrol vehicle actor IDs owned by
               NPCController. Shared by reference so _on_collision always sees
               the current contents even after NPCController mutates it.
        """
        if vehicle is None or world is None:
            return

        # Store the shared reference - mutations from NPCController are visible
        self._patrol_npc_ids = patrol_npc_ids
        self._ego_vehicle = vehicle

        self._spawn_imu(world, vehicle)
        self._spawn_gnss(world, vehicle)
        self._spawn_lidar_2d(world, vehicle)
        self._spawn_collision_sensor(world, vehicle)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        @brief Stop and destroy all spawned sensors; reset collision and scan state.

        Two-phase teardown: stop() closes the CARLA data stream, a short sleep
        lets the ROS bridge _update_thread see the closure, then destroy() removes
        the actor. Skipping the sleep races the bridge and can cause double-destroy
        exceptions on episode reset.

        Called at the start of each episode reset before new sensors are spawned.
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
        self._ego_vehicle = None

        self._collision_detected = False
        self._collision_ego_fault = False

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
        self._collision_ego_fault = False
        with self._lidar_scan_lock:
            self._latest_lidar_scan = None

    # ------------------------------------------------------------------
    # Sensor spawn helpers
    # ------------------------------------------------------------------

    def _spawn_imu(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn IMU sensor at centre-of-mass height.

        Noise parameters and mount position come from sensors_config.imu.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        imu_config = self._sensors_config.get("imu", {})
        mount = imu_config.get("mount", {})

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        # role_name determines the ROS topic: /carla/ego_vehicle/<role_name>.
        # Each sensor needs a unique name to avoid topic collisions in the bridge.
        imu_bp.set_attribute("role_name", "imu")
        # Noise is injected entirely by ImuNoiseRelayNode (ros2_config.yaml).
        # Zero all CARLA-side noise so it is never double-counted.
        for attr in [
            "noise_accel_stddev_x", "noise_accel_stddev_y", "noise_accel_stddev_z",
            "noise_gyro_stddev_x", "noise_gyro_stddev_y", "noise_gyro_stddev_z",
        ]:
            imu_bp.set_attribute(attr, "0.0")
        imu_bp.set_attribute("sensor_tick", str(imu_config.get("sensor_tick", 0.05)))

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

        Single-channel horizontal scan. Feeds obstacle clearance features
        (obs indices 7-11) only - NOT used for localisation.

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

    def _spawn_gnss(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn front RTK-GNSS antenna at roof position.

        All noise is injected by GnssNoiseRelayNode (ros2_config.yaml).
        CARLA-side noise attributes are zeroed so noise is never double-counted
        and can be toggled at runtime via enable_imu_noise / enable_markov_transitions
        without restarting CARLA.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        gnss_config = self._sensors_config.get("gnss", {})
        mount = gnss_config.get("mount", {})

        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("role_name", "gnss")
        # Zero all CARLA-side noise - GnssNoiseRelayNode owns all noise injection.
        for attr in [
            "noise_alt_bias", "noise_alt_stddev",
            "noise_lat_bias", "noise_lat_stddev",
            "noise_lon_bias", "noise_lon_stddev",
        ]:
            gnss_bp.set_attribute(attr, "0.0")
        gnss_bp.set_attribute("sensor_tick", str(gnss_config.get("sensor_tick", 0.05)))

        gnss_transform = carla.Transform(
            carla.Location(
                x=float(mount.get("x", 0.0)),
                y=float(mount.get("y", 0.0)),
                z=float(mount.get("z", 1.8)),
            )
        )
        gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        gnss_sensor.listen(lambda _: None)
        self._spawned_sensors.append(gnss_sensor)
        logger.debug("Spawned front GNSS sensor (CARLA noise zeroed; relay owns noise).")

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

        @param event: carla.CollisionEvent with other_actor and normal_impulse
                      fields.
        """
        other = event.other_actor
        impulse = event.normal_impulse
        impulse_magnitude = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

        is_dynamic = (
            other.type_id.startswith("walker.pedestrian")
            or other.id in self._patrol_npc_ids
        )

        self._collision_detected = True

        if is_dynamic:
            # Fault determined by ego speed, not impulse. Impulse reflects relative
            # velocity and misfires when both actors are moving.
            ego_speed = 0.0
            if self._ego_vehicle is not None:
                v = self._ego_vehicle.get_velocity()
                ego_speed = math.sqrt(v.x**2 + v.y**2)
            self._collision_ego_fault = ego_speed > self._EGO_FAULT_SPEED_THRESHOLD_MS
            logger.debug(
                "[collision] dynamic  actor=%s  ego_speed=%.2f m/s  ego_fault=%s",
                other.type_id,
                ego_speed,
                self._collision_ego_fault,
            )
        else:
            # Static objects (cones, parked cars, walls, perimeter): always penalise.
            self._collision_ego_fault = True
            logger.debug(
                "[collision] static  actor=%s  impulse=%.1f N*s",
                other.type_id,
                impulse_magnitude,
            )

    def _lidar_callback(self, lidar_data: Any) -> None:
        """
        @brief CARLA LiDAR sensor callback - caches point cloud for obstacle features.

        Converts the raw measurement to a (N, 3) float32 numpy array in vehicle
        frame (x-forward, y-left, z-up). Thread-safe via _lidar_scan_lock.

        @param lidar_data: carla.LidarMeasurement from the ray_cast sensor.
        """
        raw = lidar_data.raw_data
        n_bytes = len(raw)
        n_points = n_bytes // self._LIDAR_BYTES_PER_POINT
        if n_points == 0:
            return

        arr = np.frombuffer(raw, dtype=np.float32).reshape(n_points, 4)

        points_xyz = arr[:, :3].copy()
        points_xyz[:, 1] *= -1.0

        with self._lidar_scan_lock:
            self._latest_lidar_scan = points_xyz
