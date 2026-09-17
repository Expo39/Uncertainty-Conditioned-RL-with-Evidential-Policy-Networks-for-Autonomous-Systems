# 2D localisation covariance estimate from the robot_localisation EKF.
# Published by CovarianceExtractorNode; consumed by the training container for
# the RL agent's uncertainty-conditioned observation.
# Field order is part of the serialisation contract - do not reorder.

std_msgs/Header header

float64 x        # Position X (metres, CARLA world frame)
float64 y        # Position Y (metres, CARLA world frame)
float64 yaw      # Heading (rad, wrapped to [-pi, pi])

float64 vyaw     # Yaw rate (rad/s), body frame

# 3x3 submatrix for [x, y, yaw], row-major, taken from the EKF's 6x6 pose
# covariance at indices [0, 1, 5].
float64[9] covariance
