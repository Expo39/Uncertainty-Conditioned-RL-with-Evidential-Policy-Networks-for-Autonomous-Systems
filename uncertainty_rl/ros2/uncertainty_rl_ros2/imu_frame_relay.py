"""
@file imu_frame_relay.py
@brief Relay IMU messages with frame_id overridden to the Cartographer tracking frame.

Cartographer requires the IMU frame to be colocated (< 1e-5 m) with the tracking
frame. The CARLA bridge publishes IMU at ego_vehicle/imu, but the tracking frame
is ego_vehicle/lidar (to avoid a TF conflict with the bridge's map -> ego_vehicle/imu
transform). Since angular velocity is identical everywhere on a rigid body and
centripetal acceleration is negligible at parking speeds (< 3 m/s), relaying with
the tracking frame_id is physically correct.

On the real robot, mount the IMU at (or near) the LiDAR, or change the tracking
frame to the IMU location.

@author Antonio Galdes
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuFrameRelay(Node):
    """
    @class ImuFrameRelay
    @brief Subscribes to IMU data and republishes with a different frame_id.
    """

    def __init__(self) -> None:
        """
        @brief Initialise the IMU frame relay node.
        """
        super().__init__("imu_frame_relay")
        self.declare_parameter("target_frame", "ego_vehicle/lidar")
        self._target_frame: str = str(
            self.get_parameter("target_frame").value
        )

        self._sub = self.create_subscription(
            Imu, "imu_in", self._callback, 10
        )
        self._pub = self.create_publisher(Imu, "imu_out", 10)

        self.get_logger().info(
            f"Relaying IMU with frame_id -> '{self._target_frame}'"
        )

    def _callback(self, msg: Imu) -> None:
        """
        @brief Relay IMU message with overridden frame_id.
        @param msg: Incoming IMU message.
        """
        msg.header.frame_id = self._target_frame
        self._pub.publish(msg)


def main() -> None:
    """
    @brief Entry point for the IMU frame relay node.
    """
    rclpy.init()
    node = ImuFrameRelay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
