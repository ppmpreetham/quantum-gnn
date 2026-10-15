#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

available_kb=$(df -Pk . | awk 'NR==2 {print $4}')
if (( available_kb < 8 * 1024 * 1024 )); then
  echo "At least 8 GB free is recommended before building the ROS/Gazebo image." >&2
  echo "Available: $((available_kb / 1024 / 1024)) GB" >&2
  exit 2
fi

docker compose -f docker-compose.yml run --rm quantum-gnn-gazebo "$@"
