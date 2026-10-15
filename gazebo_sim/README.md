# Gazebo integration

This is a headless ROS 2 Jazzy + Gazebo Harmonic demo of the project. It
spawns four differential-drive rovers, bridges their Gazebo odometry into ROS
2, and runs the existing QUBO pruner and TrustGNN planner every cycle.

From the repository root:

```bash
STEPS=12 ./gazebo_sim/run.sh
```

The planner prints task assignments and publishes JSON messages on:

- `/quantum_gnn/status`
- `/quantum_gnn/assignments`

The container is headless so it works without an X server. To request the
Gazebo GUI, pass `--gui` (host display forwarding may still be required):

```bash
./gazebo_sim/run.sh --gui
```

The first build downloads several gigabytes of ROS/Gazebo packages. Keep at
least 8 GB free before building; Docker may retain additional build cache.
