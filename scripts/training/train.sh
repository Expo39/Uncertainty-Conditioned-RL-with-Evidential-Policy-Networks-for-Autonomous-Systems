#!/usr/bin/env bash
# @file train.sh
# @brief Run train_ppo.py inside the training container, filtering known
#        benign ROS 2 DDS noise from stderr.
#
# Called by `make docker-train-loc` and `make docker-train-loc-short`.
# Runs inside the training container (not on the host).
#
# Usage:
#   scripts/training/train.sh [--total-timesteps N] [extra train_ppo.py args...]

set -euo pipefail

# Filter patterns: ROS 2 / CycloneDDS internal chatter that is not actionable.
ROS_NOISE='^(>>>|<<<$|$|This error state|with this new error|rcutils_reset_error'
ROS_NOISE+='|rcutils_set_error_state|error_handling\.c|serdata\.cpp'
ROS_NOISE+='|should be called after|.*serdata.*)'

python uncertainty_rl/training/train_ppo.py \
    --config configs/train_config.yaml \
    --log-dir logs \
    --checkpoint-dir checkpoints \
    "$@" \
    2> >(grep -Ev "${ROS_NOISE}" >&2)
