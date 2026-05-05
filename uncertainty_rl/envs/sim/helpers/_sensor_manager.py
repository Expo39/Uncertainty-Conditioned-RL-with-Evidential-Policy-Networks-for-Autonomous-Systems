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

# Applied once per LiDAR callback to flip the y-axis from CARLA frame to
# vehicle frame (x-forward, y-left, z-up).
_LIDAR_SIGN_FLIP = np.array([1.0, -1.0, 1.0], dtype=np.float32)


class SensorManager:
    """
    @class SensorManager
    @brief Manages sensor spawning, callbacks, and cleanup for one parking episode.

    Instantiated once by CARLAParkingEnv and reused across episodes. Call
    cleanup() at the start of each episode reset, then spawn() to attach
    sensors to the new ego vehicle. Sensor data is accessible via properties
    and consume methods.
    """

    _EGO_FAULT_SPEED_THRESHOLD_MS: float = 0.3
    _LIDAR_BYTES_PER_POINT: int = 16

    # Noise attribute names stored as class constants
    _IMU_NOISE_ATTRS: Tuple[str, ...] = (
        "noise_accel_stddev_x",
        "noise_accel_stddev_y",
        "noise_accel_stddev_z",
        "noise_gyro_stddev_x",
        "noise_gyro_stddev_y",
        "noise_gyro_stddev_z",
    )
    _GNSS_NOISE_ATTRS: Tuple[str, ...] = (
        "noise_alt_bias",
        "noise_alt_stddev",
        "noise_lat_bias",
        "noise_lat_stddev",
        "noise_lon_bias",
        "noise_lon_stddev",
    )
    # CARLA blueprint attribute zeroed for LiDAR; all noise applied in _apply_lidar_noise().
    _LIDAR_NOISE_ATTRS: Tuple[str, ...] = ("noise_stddev",)

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

        self._spawned_sensors: List[Any] = []

        # Latest LiDAR point cloud in vehicle frame ([N, 3] float32).
        self._latest_lidar_scan: Optional[np.ndarray] = None
        self._lidar_scan_lock = threading.Lock()

        self._collision_detected: bool = False
        self._collision_ego_fault: bool = False

        self._patrol_npc_ids: Set[int] = set()
        self._ego_vehicle: Optional[Any] = None

        # LiDAR noise model parameters parsed once from sensors_config.
        _lidar_noise_cfg: Dict[str, Any] = sensors_config.get("lidar", {}).get("noise", {})
        self._lidar_noise_enabled: bool = bool(_lidar_noise_cfg.get("enabled", False))
        self._lidar_range_random_stddev: float = float(
            _lidar_noise_cfg.get("range_random_stddev_m", 0.0)
        )
        self._lidar_range_bias_limit: float = float(
            _lidar_noise_cfg.get("range_bias_limit_m", 0.0)
        )
        self._lidar_dropout_rate: float = float(_lidar_noise_cfg.get("dropout_rate", 0.0))
        self._lidar_min_range: float = float(_lidar_noise_cfg.get("min_range_m", 0.05))
        # Per-episode systematic range bias (metres). Resampled each episode via
        # sample_lidar_noise_bias() called from CARLAParkingEnv.reset().
        self._lidar_range_bias_m: float = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def collision_detected(self) -> bool:
        """
        @brief True if a collision was detected since last consume_collision() call.
        """
        return self._collision_detected

    def consume_collision(self) -> Tuple[bool, bool]:
        """
        @brief Read and clear the collision flag.
        @return Tuple of (detected, ego_fault).
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
            return self._latest_lidar_scan

    def lidar_point_count(self) -> int:
        """
        @brief Return the number of points in the latest LiDAR scan (0 if none).
        @return Integer point count.
        """
        with self._lidar_scan_lock:
            return len(self._latest_lidar_scan) if self._latest_lidar_scan is not None else 0

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
        lets the ROS bridge _update_thread see the closure, then destroy()
        removes the actor.
        """
        alive = [s for s in self._spawned_sensors if s is not None and s.is_alive]
        for sensor in alive:
            try:
                sensor.stop()
            except Exception:
                pass
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
        self._lidar_range_bias_m = 0.0

        with self._lidar_scan_lock:
            self._latest_lidar_scan = None

    def reset_state(self) -> None:
        """
        @brief Reset per-episode sensor state without touching CARLA actors.

        Used when the ego vehicle is reused across episodes (teleport path).
        """
        self._collision_detected = False
        self._collision_ego_fault = False
        self._lidar_range_bias_m = 0.0
        with self._lidar_scan_lock:
            self._latest_lidar_scan = None

    # ------------------------------------------------------------------
    # Sensor spawn helpers
    # ------------------------------------------------------------------

    def _spawn_imu(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn IMU sensor at centre-of-mass height.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        imu_config = self._sensors_config.get("imu", {})
        mount = imu_config.get("mount", {})

        imu_bp = world.get_blueprint_library().find("sensor.other.imu")
        imu_bp.set_attribute("role_name", "imu")
        # Noise injected by ImuNoiseRelayNode - zero all CARLA-side noise.
        for attr in self._IMU_NOISE_ATTRS:
            imu_bp.set_attribute(attr, "0.0")
        imu_bp.set_attribute("sensor_tick", str(imu_config.get("sensor_tick", 0.05)))

        imu_sensor = world.spawn_actor(
            imu_bp,
            carla.Transform(
                carla.Location(
                    x=float(mount.get("x", 0.0)),
                    y=float(mount.get("y", 0.0)),
                    z=float(mount.get("z", 0.3)),
                )
            ),
            attach_to=vehicle,
        )
        imu_sensor.listen(lambda _: None)
        self._spawned_sensors.append(imu_sensor)

    def _spawn_lidar_2d(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn 2D LiDAR sensor at front bumper height for obstacle detection.

        Single-channel horizontal scan. Feeds obstacle clearance features
        (obs indices 7-11) only - NOT used for localisation.

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

        # Noise applied in _lidar_callback
        for attr in self._LIDAR_NOISE_ATTRS:
            if lidar_bp.has_attribute(attr):
                lidar_bp.set_attribute(attr, "0.0")

        lidar_sensor = world.spawn_actor(
            lidar_bp,
            carla.Transform(
                carla.Location(
                    x=float(mount.get("x", 2.4)),
                    y=float(mount.get("y", 0.0)),
                    z=float(mount.get("z", 0.5)),
                )
            ),
            attach_to=vehicle,
        )
        lidar_sensor.listen(self._lidar_callback)
        self._spawned_sensors.append(lidar_sensor)

    def _spawn_gnss(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn front RTK-GNSS antenna at roof position.

        All noise is injected by GnssNoiseRelayNode - CARLA-side noise zeroed.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        gnss_config = self._sensors_config.get("gnss", {})
        mount = gnss_config.get("mount", {})

        gnss_bp = world.get_blueprint_library().find("sensor.other.gnss")
        gnss_bp.set_attribute("role_name", "gnss")
        for attr in self._GNSS_NOISE_ATTRS:
            gnss_bp.set_attribute(attr, "0.0")
        gnss_bp.set_attribute("sensor_tick", str(gnss_config.get("sensor_tick", 0.05)))

        gnss_sensor = world.spawn_actor(
            gnss_bp,
            carla.Transform(
                carla.Location(
                    x=float(mount.get("x", 0.0)),
                    y=float(mount.get("y", 0.0)),
                    z=float(mount.get("z", 1.8)),
                )
            ),
            attach_to=vehicle,
        )
        gnss_sensor.listen(lambda _: None)
        self._spawned_sensors.append(gnss_sensor)
        logger.debug("Spawned front GNSS sensor (CARLA noise zeroed; relay owns noise).")

    def _spawn_collision_sensor(self, world: Any, vehicle: Any) -> None:
        """
        @brief Spawn a CARLA collision sensor attached to the ego vehicle.

        @param world: carla.World for the current episode.
        @param vehicle: Ego vehicle actor to attach to.
        """
        bp = world.get_blueprint_library().find("sensor.other.collision")
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

        Impulse magnitude is only computed when debug logging is active to
        avoid an unnecessary sqrt on every collision event.

        @param event: carla.CollisionEvent with other_actor and normal_impulse fields.
        """
        other = event.other_actor
        self._collision_detected = True

        is_dynamic = (
            other.type_id.startswith("walker.pedestrian")
            or other.id in self._patrol_npc_ids
        )

        if is_dynamic:
            ego_speed = 0.0
            if self._ego_vehicle is not None:
                v = self._ego_vehicle.get_velocity()
                ego_speed = math.hypot(v.x, v.y)
            self._collision_ego_fault = ego_speed > self._EGO_FAULT_SPEED_THRESHOLD_MS
            logger.debug(
                "[collision] dynamic  actor=%s  ego_speed=%.2f m/s  ego_fault=%s",
                other.type_id,
                ego_speed,
                self._collision_ego_fault,
            )
        else:
            self._collision_ego_fault = True
            if logger.isEnabledFor(logging.DEBUG):
                impulse = event.normal_impulse
                logger.debug(
                    "[collision] static  actor=%s  impulse=%.1f N*s",
                    other.type_id,
                    math.hypot(impulse.x, impulse.y, impulse.z),
                )

    def sample_lidar_noise_bias(self, rng: np.random.Generator) -> None:
        """
        @brief Sample a per-episode systematic range bias for the noise model.

        Draws one scalar from Uniform(-range_bias_limit_m, +range_bias_limit_m) and
        stores it in _lidar_range_bias_m. Called once per episode from
        CARLAParkingEnv.reset(). A no-op when noise is disabled or the limit is zero.

        @param rng: numpy Generator (np_random from CARLAParkingEnv) for reproducibility.
        @see documentation/detailed_notes/sensor_noise_models.md
        """
        if not self._lidar_noise_enabled or self._lidar_range_bias_limit == 0.0:
            self._lidar_range_bias_m = 0.0
        else:
            self._lidar_range_bias_m = float(
                rng.uniform(-self._lidar_range_bias_limit, self._lidar_range_bias_limit)
            )
            logger.debug(
                "LiDAR systematic bias sampled: %.4f m (limit=+-%.3f m)",
                self._lidar_range_bias_m,
                self._lidar_range_bias_limit,
            )

    def _apply_lidar_noise(self, points_xyz: np.ndarray) -> np.ndarray:
        """
        @brief Apply the sensor noise model to a raw LiDAR point cloud.

        Models two error sources from the TiM571 datasheet (P/N 1075091). No angular
        error is modelled - the datasheet gives no angular accuracy specification.

        @param points_xyz: (N, 3) float32 array in vehicle frame (x-forward, y-left, z-up)
                           after _LIDAR_SIGN_FLIP has been applied.
        @return (M, 3) float32 array with M <= N after noise and range filtering.
        @see documentation/detailed_notes/sensor_noise_models.md
        """
        if not self._lidar_noise_enabled or len(points_xyz) == 0:
            return points_xyz

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]

        r = np.hypot(x, y)
        theta = np.arctan2(y, x)

        rng = np.random.default_rng()

        # 1. Range noise: fixed per-episode bias + zero-mean Gaussian random per point.
        r_noisy = r + self._lidar_range_bias_m
        if self._lidar_range_random_stddev > 0.0:
            r_noisy = r_noisy + (
                rng.standard_normal(len(r)).astype(np.float32)
                * self._lidar_range_random_stddev
            )

        # 2. Discard returns that fell below the sensor minimum working range.
        valid = r_noisy >= self._lidar_min_range
        r_noisy = r_noisy[valid]
        theta = theta[valid]
        z = z[valid]

        if len(r_noisy) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 3. Optional point dropout (Bernoulli mask).
        if self._lidar_dropout_rate > 0.0:
            keep = rng.random(len(r_noisy)) >= self._lidar_dropout_rate
            r_noisy = r_noisy[keep]
            theta = theta[keep]
            z = z[keep]

        if len(r_noisy) == 0:
            return np.empty((0, 3), dtype=np.float32)

        # 4. Convert back to Cartesian; z is preserved unchanged.
        x_noisy = (r_noisy * np.cos(theta)).astype(np.float32)
        y_noisy = (r_noisy * np.sin(theta)).astype(np.float32)
        return np.column_stack([x_noisy, y_noisy, z.astype(np.float32)])

    def _lidar_callback(self, lidar_data: Any) -> None:
        """
        @brief CARLA LiDAR sensor callback - applies noise model then caches scan.

        @param lidar_data: carla.LidarMeasurement from the ray_cast sensor.
        """
        n_points = len(lidar_data.raw_data) // self._LIDAR_BYTES_PER_POINT
        if n_points == 0:
            return

        arr = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(n_points, 4)
        points_xyz = arr[:, :3] * _LIDAR_SIGN_FLIP
        points_xyz = self._apply_lidar_noise(points_xyz)

        with self._lidar_scan_lock:
            self._latest_lidar_scan = points_xyz
