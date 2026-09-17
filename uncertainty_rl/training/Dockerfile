# @file uncertainty_rl/training/Dockerfile
# @brief Training container with PyTorch, Stable-Baselines3, evidential deep learning, and rclpy.

ARG PYTORCH_VERSION=24.10-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# apt versions are deliberately unpinned: ROS repos are rolling and delete old
# versions, so a pin would eventually fail to fetch. Reproducibility comes from
# distributing the BUILT image (docker save/load), not from rebuilding here.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    software-properties-common \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble (minimal) for rclpy - subscribes to EKF covariance via DDS.
# NGC PyTorch 24.10 is Ubuntu 22.04 (Jammy) -> ROS 2 Humble.
# Only rclpy + message types needed; no full ROS 2 desktop install.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    python3-colcon-common-extensions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

# Build uncertainty_rl_msgs (CovarianceEstimate) into the ROS 2 install prefix
# so it is importable alongside rclpy without extra PYTHONPATH entries.
COPY uncertainty_rl/ros2/uncertainty_rl_msgs /tmp/msgs_ws/src/uncertainty_rl_msgs
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /tmp/msgs_ws && \
    colcon build --install-base /opt/ros/humble" && \
    rm -rf /tmp/msgs_ws

# Set ROS 2 environment variables so rclpy is importable without sourcing setup.bash.
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu
ENV ROS_DISTRO=humble
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Upgrade pip and install build tools (pinned for reproducible builds).
RUN pip install --no-cache-dir pip==26.1.2 setuptools==82.0.1 wheel==0.47.0

# Create workspace directory
WORKDIR /workspace

# Copy pyproject.toml for pip install -e
COPY pyproject.toml /workspace/pyproject.toml
# Minimal stub so pip install -e resolves the package at build time.
# The real source is provided at runtime via the bind-mount in docker-compose.yml,
# which overlays /workspace/uncertainty_rl with the live host files.
RUN mkdir -p /workspace/uncertainty_rl && \
    echo '__version__ = "0.1.0"\n__author__ = "Antonio Galdes"' \
    > /workspace/uncertainty_rl/__init__.py

# Install CARLA Python API (system-level dependency, not on PyPI reliably)
RUN pip install --no-cache-dir carla==0.9.16

# Install all dependencies from pyproject.toml.
# [training] brings in gymnasium, stable-baselines3, tensorboard (and their torch/CUDA chain).
# [dev] brings in pytest, black, flake8, mypy etc. for testing/linting inside the container.
RUN pip install --no-cache-dir -e "/workspace[training,dev]"


# Create anchor directories so bind-mounts have a target even on first run
RUN mkdir -p /workspace/logs /workspace/checkpoints

# Default command: interactive bash
CMD ["/bin/bash"]
