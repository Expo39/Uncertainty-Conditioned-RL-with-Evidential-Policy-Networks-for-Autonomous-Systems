"""
@file tf_to_odom.py
@brief Bridge node that converts Cartographer TF to nav_msgs/Odometry with
       dynamic covariance reflecting scan-matching quality.

Cartographer publishes its scan-matched pose estimate purely via TF
(odom -> tracking_frame), not as an Odometry message. The robot_localisation
EKF requires an Odometry topic (odom0) for its correction step.

This node periodically looks up the TF from odom -> tracking_frame and
publishes it as nav_msgs/Odometry on /scan_matched_odometry. Crucially, the
pose covariance is **dynamic**: it inflates when TF updates become stale
(Cartographer has lost or degraded its scan match) and deflates when updates
are fresh and consistent. This covariance is the primary uncertainty signal
that propagates through the EKF and into the RL policy observation.

Covariance model:
  - Nominal (fresh TF, small position delta): base_xy_variance, base_yaw_variance
  - Stale TF (no update within stale_threshold_sec): inflated by staleness_scale
  - Jump (position delta > jump_threshold_m): inflated by jump_scale
  - Combined: max(staleness_inflation, jump_inflation) applied multiplicatively

@note The twist (velocity) is estimated via finite differences of successive TF
      lookups. When the TF jumps (Cartographer relocalises), the finite-diff
      velocity is unreliable. The twist covariance is inflated proportionally
      to the same staleness/jump signal to inform the EKF to discount it.

@author Antonio Galdes
"""

import math
from typing import List, Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener


def _yaw_from_quaternion(q_x: float, q_y: float, q_z: float, q_w: float) -> float:
    """
    @brief Extract yaw angle from a quaternion (2D mode), wrapped to [-pi, pi].
    @param q_x: Quaternion x component.
    @param q_y: Quaternion y component.
    @param q_z: Quaternion z component.
    @param q_w: Quaternion w component.
    @return Yaw angle in radians, wrapped to [-pi, pi].
    """
    siny_cosp = 2.0 * (q_w * q_z + q_x * q_y)
    cosy_cosp = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    return math.atan2(siny_cosp, cosy_cosp)


def _make_diagonal_covariance(diag: List[float]) -> List[float]:
    """
    @brief Build a flat 36-element covariance array from a 6-element diagonal.
    @param diag: Six diagonal variance values [x, y, z, roll, pitch, yaw].
    @return Flat list of 36 floats (row-major 6x6 matrix, zeros off-diagonal).
    """
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = v
    return cov


class TfToOdomNode(Node):
    """
    @class TfToOdomNode
    @brief Converts Cartographer TF to nav_msgs/Odometry with dynamic covariance.

    Publishes /scan_matched_odometry at a fixed rate. The pose covariance
    reflects Cartographer scan-matching quality via TF staleness and position
    jump detection. Stale or jumping TF -> high covariance -> EKF output
    covariance rises -> RL policy sees high localisation uncertainty.
    """

    # Nominal twist covariance values (scaled by the same inflation factor as pose).
    # vxy: 0.1 m^2/s^2 nominal; vyaw: 0.2 rad^2/s^2 nominal.
    # These are loose enough to allow the EKF to weight the scan-matched velocity
    # without dominating the IMU prediction between Cartographer updates.
    _NOMINAL_VXY_VAR: float = 0.1
    _NOMINAL_VYAW_VAR: float = 0.2

    # Unconstrained axes (z, roll, pitch) get a very large variance so the EKF
    # ignores them entirely (two_d_mode=true already disables them in the EKF,
    # but the message must still fill all 36 entries).
    _UNCONSTRAINED_VAR: float = 1e6

    def __init__(self, node_name: str = "tf_to_odom") -> None:
        """
        @brief Constructor for TfToOdomNode.
        @param node_name: Name of the ROS node.
        """
        super().__init__(node_name)

        # -- Parameters -------------------------------------------------------
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("tracking_frame", "ego_vehicle/lidar")
        self.declare_parameter("publish_topic", "/scan_matched_odometry")
        self.declare_parameter("publish_rate", 20.0)

        # Nominal (best-case) variances when TF is fresh and stable.
        # x/y: 0.05 m^2 (std ~0.22 m) reflects Cartographer's typical accuracy.
        # yaw: 0.05 rad^2 (std ~13 deg) is loose enough to let the EKF yaw
        # covariance vary meaningfully with scan quality.
        self.declare_parameter("base_xy_variance", 0.05)
        self.declare_parameter("base_yaw_variance", 0.05)

        # How long without a new TF before the covariance starts inflating (s).
        self.declare_parameter("stale_threshold_sec", 0.15)
        # Maximum covariance multiplier applied when TF is maximally stale.
        self.declare_parameter("staleness_scale", 100.0)
        # How long until maximum staleness inflation is reached (s).
        self.declare_parameter("stale_max_sec", 2.0)

        # Position delta (m) above which the TF is considered to have jumped.
        # A parking lot vehicle moves at most ~5 m/s; at 20 Hz that is 0.25 m
        # per step. Anything above 1 m between steps is a Cartographer jump.
        self.declare_parameter("jump_threshold_m", 1.0)
        # Covariance multiplier applied on a detected TF jump.
        self.declare_parameter("jump_scale", 50.0)
        # Number of steps over which the jump inflation decays back to 1.0.
        self.declare_parameter("jump_decay_steps", 10)

        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._tracking_frame = str(self.get_parameter("tracking_frame").value)
        publish_topic = str(self.get_parameter("publish_topic").value)
        publish_rate = float(self.get_parameter("publish_rate").value)

        self._base_xy_var: float = float(
            self.get_parameter("base_xy_variance").value
        )
        self._base_yaw_var: float = float(
            self.get_parameter("base_yaw_variance").value
        )
        self._stale_threshold: float = float(
            self.get_parameter("stale_threshold_sec").value
        )
        self._staleness_scale: float = float(
            self.get_parameter("staleness_scale").value
        )
        self._stale_max_sec: float = float(
            self.get_parameter("stale_max_sec").value
        )
        self._jump_threshold: float = float(
            self.get_parameter("jump_threshold_m").value
        )
        self._jump_scale: float = float(
            self.get_parameter("jump_scale").value
        )
        self._jump_decay_steps: int = int(
            self.get_parameter("jump_decay_steps").value
        )

        # Precompute the denominator for the staleness ramp to avoid the
        # max(..., 1e-6) guard on every timer tick. Validated once at init.
        ramp_range = self._stale_max_sec - self._stale_threshold
        if ramp_range <= 0.0:
            raise ValueError(
                f"stale_max_sec ({self._stale_max_sec}) must be greater than "
                f"stale_threshold_sec ({self._stale_threshold})."
            )
        self._stale_ramp_range: float = ramp_range

        # -- TF listener -------------------------------------------------------
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # -- Publisher ---------------------------------------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._odom_pub = self.create_publisher(Odometry, publish_topic, qos_profile)

        # -- State -------------------------------------------------------------
        self._prev_transform: Optional[TransformStamped] = None
        self._prev_time_sec: Optional[float] = None
        # Wall-clock time of last successful TF lookup (for staleness detection).
        self._last_tf_wall_sec: Optional[float] = None
        # Remaining steps of jump inflation.
        self._jump_inflation_steps: int = 0

        # -- Timer -------------------------------------------------------------
        period = 1.0 / publish_rate
        self._timer = self.create_timer(period, self._timer_callback)

        self.get_logger().info(
            f"TfToOdom: {self._odom_frame} -> {self._tracking_frame} "
            f"at {publish_rate} Hz on {publish_topic} "
            f"(dynamic covariance: base_xy={self._base_xy_var}, "
            f"base_yaw={self._base_yaw_var}, "
            f"stale_scale={self._staleness_scale}x, "
            f"jump_scale={self._jump_scale}x)"
        )

    def _compute_covariance_scale(
        self, position_delta_m: float, now_wall: float
    ) -> float:
        """
        @brief Compute the covariance inflation factor for this step.

        Combines staleness inflation (TF not updating -> Cartographer lost lock)
        and jump inflation (large position discontinuity -> relocalisation event).
        Returns the larger of the two, so either signal alone is sufficient to
        inflate the covariance and signal uncertainty to the EKF.

        @param position_delta_m: Distance moved since last TF lookup (metres).
        @param now_wall: Current wall-clock time in seconds (passed in to avoid
                         a redundant clock read; caller already holds this value).
        @return Multiplicative inflation factor >= 1.0.
        """
        # -- Staleness inflation -----------------------------------------------
        staleness_factor = 1.0
        if self._last_tf_wall_sec is not None:
            age = now_wall - self._last_tf_wall_sec
            if age > self._stale_threshold:
                # Linear ramp from 1.0 at stale_threshold to staleness_scale
                # at stale_max_sec, then clamped.
                t = min(
                    (age - self._stale_threshold) / self._stale_ramp_range,
                    1.0,
                )
                staleness_factor = 1.0 + t * (self._staleness_scale - 1.0)

        # -- Jump inflation ----------------------------------------------------
        jump_factor = 1.0
        if position_delta_m > self._jump_threshold:
            self._jump_inflation_steps = self._jump_decay_steps
            self.get_logger().debug(
                f"TF jump detected: delta={position_delta_m:.2f}m, "
                f"inflating covariance by {self._jump_scale}x "
                f"for {self._jump_decay_steps} steps"
            )

        if self._jump_inflation_steps > 0:
            # Linear decay from jump_scale back to 1.0 over jump_decay_steps.
            t = self._jump_inflation_steps / float(self._jump_decay_steps)
            jump_factor = 1.0 + t * (self._jump_scale - 1.0)
            self._jump_inflation_steps -= 1

        return max(staleness_factor, jump_factor)

    def _timer_callback(self) -> None:
        """
        @brief Timer callback: look up TF and publish Odometry with dynamic covariance.
        """
        # Single clock read for this tick; passed to _compute_covariance_scale
        # to avoid redundant clock calls.
        now_wall = self.get_clock().now().nanoseconds * 1e-9

        try:
            tf = self._tf_buffer.lookup_transform(
                self._odom_frame,
                self._tracking_frame,
                rclpy.time.Time(),
            )
        except Exception:
            # TF not yet available -- do not publish.
            # _last_tf_wall_sec remains None or stale, so staleness inflation
            # ramps up naturally on subsequent calls to _compute_covariance_scale.
            return

        # Record wall-clock time of successful lookup for staleness detection.
        self._last_tf_wall_sec = now_wall

        # -- Pose from TF -----------------------------------------------------
        curr_time_sec = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        curr_x = tf.transform.translation.x
        curr_y = tf.transform.translation.y
        curr_yaw = _yaw_from_quaternion(
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )

        # -- Previous pose (used for both jump detection and twist) -----------
        # Extract once here; reused in both the covariance-scale and twist
        # sections below without a second attribute lookup.
        position_delta = 0.0
        prev_x: Optional[float] = None
        prev_y: Optional[float] = None
        prev_yaw: Optional[float] = None
        if self._prev_transform is not None:
            prev_x = self._prev_transform.transform.translation.x
            prev_y = self._prev_transform.transform.translation.y
            prev_yaw = _yaw_from_quaternion(
                self._prev_transform.transform.rotation.x,
                self._prev_transform.transform.rotation.y,
                self._prev_transform.transform.rotation.z,
                self._prev_transform.transform.rotation.w,
            )
            position_delta = math.sqrt(
                (curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2
            )

        # -- Dynamic covariance -----------------------------------------------
        scale = self._compute_covariance_scale(position_delta, now_wall)
        xy_var = self._base_xy_var * scale
        yaw_var = self._base_yaw_var * scale
        # Twist covariance uses the same scale: jump/staleness equally affects
        # velocity reliability from finite differences.
        vxy_var = self._NOMINAL_VXY_VAR * scale
        vyaw_var = self._NOMINAL_VYAW_VAR * scale

        uc = self._UNCONSTRAINED_VAR
        pose_cov = _make_diagonal_covariance([xy_var, xy_var, uc, uc, uc, yaw_var])
        twist_cov = _make_diagonal_covariance([vxy_var, vxy_var, uc, uc, uc, vyaw_var])

        # -- Build Odometry message -------------------------------------------
        odom = Odometry()
        odom.header.stamp = tf.header.stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._tracking_frame
        odom.pose.pose.position.x = curr_x
        odom.pose.pose.position.y = curr_y
        odom.pose.pose.position.z = tf.transform.translation.z
        odom.pose.pose.orientation = tf.transform.rotation
        odom.pose.covariance = pose_cov

        # -- Twist via finite differences -------------------------------------
        if (
            prev_x is not None
            and prev_y is not None
            and prev_yaw is not None
            and self._prev_time_sec is not None
        ):
            dt = curr_time_sec - self._prev_time_sec
            if dt > 1e-6:
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                dyaw = math.atan2(
                    math.sin(curr_yaw - prev_yaw),
                    math.cos(curr_yaw - prev_yaw),
                )
                # Rotate odom-frame displacement into vehicle body frame.
                cos_yaw = math.cos(curr_yaw)
                sin_yaw = math.sin(curr_yaw)
                odom.twist.twist.linear.x = (cos_yaw * dx + sin_yaw * dy) / dt
                odom.twist.twist.linear.y = (-sin_yaw * dx + cos_yaw * dy) / dt
                odom.twist.twist.angular.z = dyaw / dt

        odom.twist.covariance = twist_cov

        self._odom_pub.publish(odom)

        self._prev_transform = tf
        self._prev_time_sec = curr_time_sec


def main() -> None:
    """
    @brief Entry point for the tf_to_odom node.
    """
    rclpy.init()
    node = TfToOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
