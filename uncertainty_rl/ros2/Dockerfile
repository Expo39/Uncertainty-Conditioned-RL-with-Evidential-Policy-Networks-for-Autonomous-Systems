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
# - robot-localization: EKF node + navsat_transform_node (fuses GNSS+IMU, outputs /odometry/filtered)
# - cv-bridge, derived-object-msgs: required by carla_ros_bridge internals
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-tf2-geometry-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-derived-object-msgs \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Use CycloneDDS for cross-distro compatibility (Humble <-> Jazzy)
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Create workspace
WORKDIR /workspace
RUN mkdir -p /workspace/src

# Clone CARLA ROS bridge (master branch has ROS 2 support for CARLA 0.9.13+)
# Three patches applied to the cloned ros-bridge source:
#   1. CARLA_VERSION: bridge hardcodes 0.9.13 but is wire-compatible with 0.9.16.
#   2. generate_opendrive_world: add OpendriveGenerationParameters to disable nav
#      mesh (crashes on headless GPU) and match our FlatPlane world setup.
#   3. on_tick_id guard: AttributeError when shutdown() is called before on_tick_id
#      is assigned during rapid restart. getattr fallback makes it a no-op.
#   4. patch_bridge.py: catches RuntimeError on destroy() of already-gone actors
#      (sensor.py + actor_factory.py) to prevent hang in world.tick() sync mode.
COPY uncertainty_rl/ros2/patch_bridge.py /tmp/patch_bridge.py
RUN cd /workspace/src && \
    git clone --recurse-submodules https://github.com/carla-simulator/ros-bridge.git && \
    echo "0.9.16" > /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION && \
    sed -i 's|carla_world = carla_client.generate_opendrive_world(str(data))|carla_world = carla_client.generate_opendrive_world(str(data), carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=600.0, wall_height=0.0, additional_width=300.0, smooth_junctions=False, enable_mesh_visibility=True, enable_pedestrian_navigation=False))|' \
    /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/bridge.py && \
    sed -i 's|if self.on_tick_id:|if getattr(self, "on_tick_id", None):|' \
    /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/bridge.py && \
    python3 /tmp/patch_bridge.py

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

# Build only the CARLA bridge packages needed at runtime.
# carla_manual_control is included because carla_ros_bridge depends on it at
# build time (looks for carla_manual_control/package.sh during colcon build).
# It is never launched at runtime. rviz2/rqt packages are excluded (GUI only).
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select carla_msgs carla_common ros_compatibility carla_spawn_objects carla_manual_control carla_ros_bridge"

# Install the uncertainty_rl Python package (base deps only -- no torch/SB3).
# Covariance extraction nodes import uncertainty_rl.utils.covariance_utils.
# Placed outside /workspace/src so colcon does not find it as a duplicate package.
COPY pyproject.toml /opt/uncertainty_rl_pkg/pyproject.toml
COPY uncertainty_rl/__init__.py /opt/uncertainty_rl_pkg/uncertainty_rl/__init__.py
COPY uncertainty_rl/utils/ /opt/uncertainty_rl_pkg/uncertainty_rl/utils/
RUN pip install --no-cache-dir --break-system-packages -e "/opt/uncertainty_rl_pkg"

# Copy custom message package (ament_cmake, builds first via dependency order)
COPY uncertainty_rl/ros2/uncertainty_rl_msgs /workspace/src/uncertainty_rl_msgs

# Copy custom ROS 2 nodes for covariance extraction (ament_python)
COPY uncertainty_rl/ros2/uncertainty_rl_ros2 /workspace/src/uncertainty_rl_ros2/uncertainty_rl_ros2
COPY uncertainty_rl/ros2/package.xml /workspace/src/uncertainty_rl_ros2/package.xml
COPY uncertainty_rl/ros2/setup.py /workspace/src/uncertainty_rl_ros2/setup.py
COPY uncertainty_rl/ros2/setup.cfg /workspace/src/uncertainty_rl_ros2/setup.cfg
COPY uncertainty_rl/ros2/resource /workspace/src/uncertainty_rl_ros2/resource
COPY uncertainty_rl/ros2/launch /workspace/src/uncertainty_rl_ros2/launch

# All configs (ros2_config.yaml, env_config.yaml, gnss_noise_profiles.yaml) are
# provided at runtime via the ./configs:/workspace/configs:ro bind-mount.

# Build custom messages and nodes
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source /workspace/install/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select uncertainty_rl_msgs uncertainty_rl_ros2"

# Source ROS setup automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/install/setup.bash" >> ~/.bashrc

# Health check: verify the colcon install is intact and ROS 2 can be sourced.
# ros2 topic list hangs with CycloneDDS when no DDS participant is present,
# so check the install directory instead of querying the live graph.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD test -f /workspace/install/setup.bash || exit 1

WORKDIR /workspace
CMD ["/bin/bash"]
