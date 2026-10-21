#!/usr/bin/env bash
set -Eeuo pipefail

# ROS setup scripts read unset variables, so relax -u while sourcing them
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="/workspace:${PYTHONPATH:-}"
export GZ_SIM_RESOURCE_PATH="/workspace/gazebo_sim/models:${GZ_SIM_RESOURCE_PATH:-}"

steps="${STEPS:-12}"
headless=1
if [[ "${1:-}" == "--gui" ]]; then
  headless=0
  shift
fi

cleanup() {
  set +e
  [[ -n "${BRIDGE_PID:-}" ]] && kill "${BRIDGE_PID}" 2>/dev/null || true
  [[ -n "${GZ_PID:-}" ]] && kill "${GZ_PID}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bridge_topics=(
  "/model/rover_0/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
  "/model/rover_1/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
  "/model/rover_2/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
  "/model/rover_3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"
)

ros2 run ros_gz_bridge parameter_bridge "${bridge_topics[@]}" \
  > /tmp/quantum_gnn_bridge.log 2>&1 &
BRIDGE_PID=$!

gz_args="-r"
if [[ "${headless}" == "1" ]]; then
  gz_args="-r -s"
fi
gz sim ${gz_args} /workspace/gazebo_sim/worlds/rover_world.sdf \
  > /tmp/quantum_gnn_gazebo.log 2>&1 &
GZ_PID=$!

sleep 3
if ! kill -0 "${GZ_PID}" 2>/dev/null; then
  echo "Gazebo failed to start. Server log:" >&2
  cat /tmp/quantum_gnn_gazebo.log >&2 || true
  exit 1
fi

exec python3 /workspace/gazebo_sim/planner_node.py --steps "${steps}" "$@"
