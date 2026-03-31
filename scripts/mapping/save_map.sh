#!/usr/bin/env bash
# @file save_map.sh
# @brief Serialise the running Cartographer session to a .pbstream file and
#        optionally convert the occupancy grid to a PNG.
#
# Called by `make docker-map` after mapping_drive.py completes.
# Runs on the host (invokes `docker compose exec` internally).
#
# Usage:
#   scripts/mapping/save_map.sh <layout> <map_dim>
#   e.g.  scripts/mapping/save_map.sh rectangle 2d

set -euo pipefail

LAYOUT="${1:?Usage: save_map.sh <layout> <map_dim>}"
MAP_DIM="${2:?Usage: save_map.sh <layout> <map_dim>}"
PBSTREAM="configs/maps/${MAP_DIM}/${LAYOUT}.pbstream"

echo "Serialising Cartographer state to ${PBSTREAM}..."

docker compose exec ros2-bridge bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    ros2 service call /write_state cartographer_ros_msgs/srv/WriteState \
        '{filename: \"/workspace/${PBSTREAM}\", include_unfinished_submaps: true}'
"

echo "Saved ${PBSTREAM}"

# Occupancy grid PNG -- optional, Cairo can fail on small maps so errors are ignored.
docker compose exec ros2-bridge bash -c "
    source /opt/ros/jazzy/setup.bash
    source /workspace/install/setup.bash
    ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
        --pbstream_filename /workspace/${PBSTREAM} \
        --map_filestem /workspace/configs/maps/${MAP_DIM}/${LAYOUT}_grid \
        --resolution 0.05
" 2>/dev/null || true

if [ -f "configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm" ]; then
    docker compose exec training python -c "
from PIL import Image
Image.open('/workspace/configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm') \
    .convert('RGB') \
    .save('/workspace/outputs/maps/${MAP_DIM}/${LAYOUT}.png')
"
    rm -f "configs/maps/${MAP_DIM}/${LAYOUT}_grid.pgm" \
          "configs/maps/${MAP_DIM}/${LAYOUT}_grid.yaml"
    echo "Saved outputs/maps/${MAP_DIM}/${LAYOUT}.png"
fi
