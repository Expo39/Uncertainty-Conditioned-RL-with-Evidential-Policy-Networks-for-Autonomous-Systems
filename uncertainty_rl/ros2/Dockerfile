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
# @author Antonio
# @date 2025-02-11

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

# Install robot_localisation for EKF
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/src

# Clone CARLA ROS bridge (compatible with CARLA 0.9.13+)
RUN cd /workspace/src && \
    git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git && \
    cd ros-bridge && \
    git checkout ros2

# Install Python dependencies for CARLA
RUN pip3 install --no-cache-dir \
    carla==0.9.15 \
    numpy \
    transforms3d

# Install rosdep dependencies
RUN rosdep update && \
    rosdep install --from-paths /workspace/src --ignore-src -r -y

# Build workspace
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

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

# Rebuild with custom messages and nodes
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Source ROS setup automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Health check for ROS 2 daemon
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD ros2 topic list || exit 1

WORKDIR /workspace
CMD ["/bin/bash"]
