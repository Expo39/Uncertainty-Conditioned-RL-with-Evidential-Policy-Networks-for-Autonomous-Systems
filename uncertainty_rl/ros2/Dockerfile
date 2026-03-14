# Dockerfile
# @file uncertainty_rl/ros2/Dockerfile
# @brief ROS 2 Jazzy container with CARLA bridge and robot_localisation for EKF.
#
# Components:
# - ROS 2 Jazzy base
# - CARLA ROS bridge
# - robot_localisation package for EKF covariance output
# - Python dependencies for covariance extraction
#
# @author Antonio Galdes

ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base-noble

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=${ROS_DISTRO}
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 packages:
# - robot-localization: EKF node (fuses IMU + odometry, outputs /odometry/filtered)
# - cv-bridge, derived-object-msgs: required by carla_ros_bridge internals
# - pointcloud-to-laserscan: converts LiDAR PointCloud2 to LaserScan for the filter pipeline
# - laser-filters: clips the 360 deg CARLA scan to 270 deg (rear 90 deg blocked by vehicle)
# - cartographer-ros: 2D SLAM / scan-matching; publishes /scan_matched_odometry for EKF correction
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-derived-object-msgs \
    ros-${ROS_DISTRO}-pointcloud-to-laserscan \
    ros-${ROS_DISTRO}-laser-filters \
    ros-${ROS_DISTRO}-cartographer-ros \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/src

# Clone CARLA ROS bridge (master branch has ROS 2 support for CARLA 0.9.13+)
# Patch CARLA_VERSION to 0.9.16: the bridge hardcodes 0.9.13 but is compatible with 0.9.16
RUN cd /workspace/src && \
    git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git && \
    echo "0.9.16" > /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION

# Install Python dependencies for CARLA
# --break-system-packages needed on Ubuntu 24.04 (PEP 668); safe in a container
RUN pip3 install --no-cache-dir --break-system-packages \
    carla==0.9.16 \
    numpy \
    transforms3d

# Install rosdep dependencies for the packages we need (skip GUI/rviz/PCL packages)
RUN rosdep update && \
    rosdep install --from-paths /workspace/src --ignore-src -r -y \
    --skip-keys="rviz2 rviz_common rqt_gui_py python_qt_binding pcl_conversions \
    qtbase5-dev libqt5core5t64 libqt5gui5t64 libqt5widgets5t64 \
    ackermann_msgs" || true

# Build only the CARLA bridge packages we need (no rviz, rqt, PCL, manual control)
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select carla_msgs carla_common ros_compatibility carla_spawn_objects carla_manual_control carla_ros_bridge"

# Copy custom message package (ament_cmake, builds first via dependency order)
COPY uncertainty_rl/ros2/uncertainty_rl_msgs /workspace/src/uncertainty_rl_msgs

# Copy custom ROS 2 nodes for covariance extraction (ament_python)
COPY uncertainty_rl/ros2/uncertainty_rl_ros2 /workspace/src/uncertainty_rl_ros2/uncertainty_rl_ros2
COPY uncertainty_rl/ros2/package.xml /workspace/src/uncertainty_rl_ros2/package.xml
COPY uncertainty_rl/ros2/setup.py /workspace/src/uncertainty_rl_ros2/setup.py
COPY uncertainty_rl/ros2/setup.cfg /workspace/src/uncertainty_rl_ros2/setup.cfg
COPY uncertainty_rl/ros2/resource /workspace/src/uncertainty_rl_ros2/resource
COPY uncertainty_rl/ros2/launch /workspace/src/uncertainty_rl_ros2/launch

# Copy ROS 2 config (loaded at launch time)
COPY configs/ros2_config.yaml /workspace/configs/ros2_config.yaml

# Copy Cartographer configs (Suite A: 2D LiDAR; Suite B/C: 3D LiDAR)
COPY configs/cartographer /workspace/configs/cartographer

# Build custom messages and nodes
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source /workspace/install/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select uncertainty_rl_msgs uncertainty_rl_ros2"

# Source ROS setup automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Health check for ROS 2 daemon
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ros2 topic list || exit 1

WORKDIR /workspace
CMD ["/bin/bash"]
