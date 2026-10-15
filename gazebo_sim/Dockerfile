FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-numpy \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-nav-msgs \
        ros-jazzy-geometry-msgs \
        ros-jazzy-std-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /workspace/gazebo_sim/entrypoint.sh

ENTRYPOINT ["/workspace/gazebo_sim/entrypoint.sh"]
