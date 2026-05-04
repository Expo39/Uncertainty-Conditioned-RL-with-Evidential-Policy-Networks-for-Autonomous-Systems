#!/usr/bin/env bash
# @file scripts/training/tune.sh
# @brief Wrapper for Optuna hyperparameter tuning inside Docker container.
#
# Filters ROS 2 noise (debug logs, QoS warnings) and forwards all args
# to uncertainty_rl.training.tune_hyperparams:main via the console script.

set -euo pipefail

# Filter ROS 2 noise
export RCUTILS_LOGGING_USE_STDOUT=false
export ROS_LOG_DIR=/tmp/ros_logs

# Run tuning script
python -m uncertainty_rl.training.tune_hyperparams "$@"
