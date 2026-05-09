# 2D localisation covariance estimate from the robot_localisation EKF.
#
# Replaces std_msgs/Float64MultiArray:
# custom messages with semantic meaning instead of primitive arrays.
#
# Published by CovarianceExtractorNode, consumed by the training container's
# _CovarianceSubscriber for the RL agent's uncertainty-conditioned observation.

std_msgs/Header header

# EKF-filtered 2D pose estimate
float64 x        # Position X (metres, CARLA world frame)
float64 y        # Position Y (metres, CARLA world frame)
float64 yaw      # Heading (rad, wrapped to [-pi, pi])

# EKF yaw rate (from msg.twist.twist.angular.z)
float64 vyaw     # Yaw rate (rad/s), body frame

# 3x3 covariance submatrix for [x, y, yaw], row-major order.
# Extracted from the EKF's 6x6 pose covariance at indices [0, 1, 5].
float64[9] covariance
