# Dockerfile
# @file uncertainty_rl/ros2/Dockerfile
# @brief ROS 2 Jazzy container with CARLA bridge and robot_localisation for EKF.
#
# Components:
# - ROS 2 Jazzy base
# - CARLA ROS bridge
# - robot_localisation package for EKF covariance output
# - Python dependencies for covariance extraction

ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-base-noble

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=${ROS_DISTRO}
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# robot-localization supplies the EKF and navsat_transform nodes; cv-bridge and
# derived-object-msgs are carla_ros_bridge internals. Versions are deliberately
# unpinned: ROS repos are rolling and delete old versions, so a pin would
# eventually fail to fetch. Reproducibility comes from shipping the BUILT image.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    wget \
    curl \
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

# Clone CARLA ROS bridge (ROS 2 support for CARLA 0.9.13+). Pinned to an exact
# commit (not a moving branch) for reproducible builds across machines; this is
# the commit the working ros2-bridge image was built from. The repo's
# CARLA_VERSION is overwritten to 0.9.16 below.
ARG ROS_BRIDGE_COMMIT=e9063d97ff5a724f76adbb1b852dc71da1dcfeec
COPY uncertainty_rl/ros2/patch_bridge.py /tmp/patch_bridge.py
RUN cd /workspace/src && \
    git clone https://github.com/carla-simulator/ros-bridge.git && \
    cd ros-bridge && \
    git checkout ${ROS_BRIDGE_COMMIT} && \
    git submodule update --init --recursive && \
    cd /workspace/src && \
    echo "0.9.16" > /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/CARLA_VERSION && \
    sed -i 's|carla_world = carla_client.generate_opendrive_world(str(data))|carla_world = carla_client.generate_opendrive_world(str(data), carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=600.0, wall_height=0.0, additional_width=300.0, smooth_junctions=False, enable_mesh_visibility=True, enable_pedestrian_navigation=False))|' \
    /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/bridge.py && \
    sed -i 's|if self.on_tick_id:|if getattr(self, "on_tick_id", None):|' \
    /workspace/src/ros-bridge/carla_ros_bridge/src/carla_ros_bridge/bridge.py && \
    python3 /tmp/patch_bridge.py

# Install Python dependencies for CARLA (pinned for reproducible builds; versions
# captured from the working ros2-bridge image).
# --break-system-packages needed on Ubuntu 24.04 (PEP 668); safe in a container
RUN pip3 install --no-cache-dir --break-system-packages \
    carla==0.9.16 \
    numpy==1.26.4 \
    transforms3d==0.4.2

# Install rosdep dependencies for the packages we need (skip GUI/rviz/PCL packages)
RUN rosdep update && \
    rosdep install --from-paths /workspace/src --ignore-src -r -y \
    --skip-keys="rviz2 rviz_common rqt_gui_py python_qt_binding pcl_conversions \
    qtbase5-dev libqt5core5t64 libqt5gui5t64 libqt5widgets5t64 \
    ackermann_msgs" || true

# Build only the CARLA bridge packages needed at runtime.
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /workspace && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-select carla_msgs carla_common ros_compatibility carla_spawn_objects carla_manual_control carla_ros_bridge"

# --no-deps because the Jazzy base ships this image's only runtime deps (numpy,
# PyYAML) as apt packages with no pip RECORD: resolving the pyproject pins would
# try to upgrade them and fail, since pip cannot uninstall a Debian-managed
# package. The system packages already satisfy those pins here.
COPY pyproject.toml /opt/uncertainty_rl_pkg/pyproject.toml
COPY uncertainty_rl/__init__.py /opt/uncertainty_rl_pkg/uncertainty_rl/__init__.py
COPY uncertainty_rl/utils/ /opt/uncertainty_rl_pkg/uncertainty_rl/utils/
RUN pip install --no-cache-dir --break-system-packages --no-deps -e "/opt/uncertainty_rl_pkg"

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
