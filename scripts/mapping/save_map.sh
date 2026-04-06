#!/usr/bin/env bash
# @file save_map.sh
# @brief Serialise the running Cartographer session to a .pbstream file and
#        optionally convert the occupancy grid to a PNG.
#
# Called by `make docker-map` after mapping_drive.py completes.
# Runs on the host (invokes `docker exec` on the worker ros2-bridge container).
#
# Usage:
#   scripts/mapping/save_map.sh <layout> <map_dim> [worker_rank]
#   e.g.  scripts/mapping/save_map.sh rectangle 2d 0

set -euo pipefail

LAYOUT="${1:?Usage: save_map.sh <layout> <map_dim>}"
MAP_DIM="${2:?Usage: save_map.sh <layout> <map_dim>}"
PBSTREAM="configs/maps/${MAP_DIM}/${LAYOUT}.pbstream"

WORKER_RANK="${3:-0}"
ROS2_CONTAINER="uncertainty-rl-ros2-${WORKER_RANK}"

echo "Serialising Cartographer state to ${PBSTREAM} (container: ${ROS2_CONTAINER})..."

# Do NOT call finish_trajectory before write_state. Finishing triggers a
# final global pose graph optimisation that shuffles submap poses based on
# low-confidence loop closure constraints (70% scores on sparse cone scans),
# corrupting the map. write_state with include_unfinished_submaps=true
# serialises the submaps at their local scan-matcher poses, which are correct.
docker exec "${ROS2_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=\$((42 + ${WORKER_RANK}))

    echo 'Writing map state...'
    ros2 service call /write_state cartographer_ros_msgs/srv/WriteState \
        '{filename: \"/workspace/${PBSTREAM}\", include_unfinished_submaps: true}'
"

echo "Saved ${PBSTREAM}"

# Occupancy grid PNG -- optional, Cairo can fail on small maps so errors are ignored.
docker exec "${ROS2_CONTAINER}" bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export ROS_DOMAIN_ID=\$((42 + ${WORKER_RANK}))
    ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
        --pbstream_filename /workspace/${PBSTREAM} \
        --map_filestem /workspace/configs/maps/${MAP_DIM}/${LAYOUT}_grid \
        --resolution 0.05
" 2>/dev/null || true

if [ -f "configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm" ]; then
    docker exec uncertainty-rl-training bash -c "
python -c \"
from PIL import Image
Image.open('/workspace/configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm') \
    .convert('RGB') \
    .save('/workspace/outputs/maps/${MAP_DIM}/${LAYOUT}.png')
\"
" 2>/dev/null || true
    rm -f "configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm" \
          "configs/maps/${MAP_DIM}/${LAYOUT}_grid.yaml"
    echo "Saved outputs/maps/${MAP_DIM}/${LAYOUT}.png"
fi
