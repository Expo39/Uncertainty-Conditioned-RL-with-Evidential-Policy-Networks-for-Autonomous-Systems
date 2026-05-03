"""
@file sensor_relay.py
@brief Entry point that co-spins GnssNoiseRelayNode and ImuNoiseRelayNode.

Both nodes are independent (separate subscriptions and publishers) but are
spun together in a single process via MultiThreadedExecutor to avoid the
overhead of a second container process.
"""

from typing import Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor

from uncertainty_rl_ros2.sensor_relay.gnss_noise_relay import GnssNoiseRelayNode
from uncertainty_rl_ros2.sensor_relay.imu_noise_relay import ImuNoiseRelayNode


def main(args: Optional[object] = None) -> None:
    """@brief Entry point: co-spins GnssNoiseRelayNode and ImuNoiseRelayNode."""
    rclpy.init(args=args)
    gnss_node = GnssNoiseRelayNode()
    imu_node = ImuNoiseRelayNode()
    executor = MultiThreadedExecutor()
    executor.add_node(gnss_node)
    executor.add_node(imu_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        gnss_node.destroy_node()
        imu_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
