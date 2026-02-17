# @file uncertainty_rl/training/Dockerfile
# @brief Training container with PyTorch, Stable-Baselines3, evidential deep learning, and rclpy.
#
# Based on NVIDIA NGC PyTorch container for optimised CUDA kernels.
# All Python dependencies are defined in pyproject.toml (single source of truth).
# ROS 2 Humble (matching Ubuntu 22.04 base) is installed for rclpy so the
# training container can subscribe to EKF covariance topics via DDS.
# DDS wire protocol is distro-agnostic, so Humble here communicates with
# Jazzy in the ros2-bridge container without issues.
#
# @author Antonio Galdes

ARG PYTORCH_VERSION=24.01-py3
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    software-properties-common \
    gnupg \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------
# ROS 2 Humble (minimal) for rclpy - subscribes to EKF covariance via DDS
# NGC PyTorch 24.01 is Ubuntu 22.04 (Jammy) -> ROS 2 Humble
# Only rclpy + message types needed; no full ROS 2 desktop install
# -----------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    python3-colcon-common-extensions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*

# Build uncertainty_rl_msgs (CovarianceEstimate) into the ROS 2 install prefix
# so it is importable alongside rclpy without extra PYTHONPATH entries
COPY uncertainty_rl/ros2/uncertainty_rl_msgs /tmp/msgs_ws/src/uncertainty_rl_msgs
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    cd /tmp/msgs_ws && \
    colcon build --install-base /opt/ros/humble" && \
    rm -rf /tmp/msgs_ws

# Set ROS 2 environment variables so rclpy is importable without sourcing setup.bash
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib:${LD_LIBRARY_PATH}
ENV ROS_DISTRO=humble

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspace

# Copy project config first (better layer caching - only rebuilds deps when pyproject.toml changes)
COPY pyproject.toml /workspace/pyproject.toml
COPY uncertainty_rl /workspace/uncertainty_rl

# Install CARLA Python API (system-level dependency, not on PyPI reliably)
RUN pip install --no-cache-dir carla==0.9.15

# Install dependencies from pyproject.toml
# Set DEV_INSTALL=false for lighter production builds without linting/testing tools
ARG DEV_INSTALL=true
RUN if [ "$DEV_INSTALL" = "true" ]; then \
        pip install --no-cache-dir -e "/workspace[dev]"; \
    else \
        pip install --no-cache-dir -e "/workspace"; \
    fi

# Copy remaining project files
COPY configs /workspace/configs
COPY tests /workspace/tests

# Create directories for outputs
RUN mkdir -p /workspace/logs /workspace/checkpoints /workspace/results

# Default command: interactive bash
CMD ["/bin/bash"]
