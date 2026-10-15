#!/usr/bin/env python3
"""ROS 2 planner node for the Gazebo version of the quantum-GNN demo.

The QUBO and GNN remain ordinary Python modules from the repository.  This
node replaces the synthetic simulator's world state with odometry received
from Gazebo and publishes a JSON assignment/status stream for easy inspection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from main import Edge, EdgeFeatures, EdgeSelectionQUBO, ProjectedEdgeAnnealing
from trust_gnn import NodeFeatures, TaskRequirements, TrustGNN, TrustModel, assign_tasks


class GazeboQuantumGNN(Node):
    def __init__(self, steps: int, cycle_period: float, comm_range: float) -> None:
        super().__init__("quantum_gnn_planner")
        self.steps = steps
        self.cycle_period = cycle_period
        self.comm_range = comm_range
        self.nodes = list(range(4))
        self.root = 0
        self.cycle = 0
        self.rng = np.random.default_rng(17)

        # These are also the initial poses in rover_world.sdf.  They make the
        # demo useful even if odometry bridge startup takes a few seconds.
        self.positions: Dict[int, np.ndarray] = {
            0: np.array([-3.0, 0.0]),
            1: np.array([-1.0, 0.4]),
            2: np.array([1.0, 0.1]),
            3: np.array([3.0, 0.5]),
        }
        self.have_odom = {n: False for n in self.nodes}
        self.battery = {n: 0.95 - 0.04 * n for n in self.nodes}
        self.load = {n: 0.05 * n for n in self.nodes}
        self.capability = {n: 0.65 + 0.08 * n for n in self.nodes}
        self.reliability = {0: 0.92, 1: 0.82, 2: 0.75, 3: 0.66}
        self.trust = TrustModel(alpha=0.55, initial=0.5, decay=0.08)
        self.gnn = TrustGNN()
        self.previous: Dict[Edge, int] = {}

        self.status_pub = self.create_publisher(String, "/quantum_gnn/status", 10)
        self.assignment_pub = self.create_publisher(
            String, "/quantum_gnn/assignments", 10
        )
        for node in self.nodes:
            self.create_subscription(
                Odometry,
                f"/model/rover_{node}/odometry",
                lambda msg, n=node: self._odom_callback(n, msg),
                10,
            )

        self.timer = self.create_timer(self.cycle_period, self._run_cycle)
        self.get_logger().info(
            "Quantum-GNN planner started: %d rovers, %.1fm communication range, "
            "%d cycles",
            len(self.nodes),
            self.comm_range,
            self.steps,
        )

    def _odom_callback(self, node: int, msg: Odometry) -> None:
        self.positions[node] = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )
        self.have_odom[node] = True

    def _build_graph(self) -> tuple[List[Edge], Dict[Edge, EdgeFeatures]]:
        edges: List[Edge] = []
        features: Dict[Edge, EdgeFeatures] = {}
        for i, u in enumerate(self.nodes):
            for v in self.nodes[i + 1 :]:
                distance = float(np.linalg.norm(self.positions[u] - self.positions[v]))
                if distance > self.comm_range:
                    continue
                proximity = float(np.clip(1.0 - distance / self.comm_range, 0.0, 1.0))
                # A deterministic link model makes repeated Docker runs
                # comparable while leaving Gazebo responsible for geometry.
                link_quality = float(np.clip(0.85 * proximity + 0.15, 0.0, 1.0))
                cost = float(np.clip(0.10 + 0.50 * (1.0 - proximity), 0.0, 1.0))
                edge = (u, v)
                edges.append(edge)
                features[edge] = EdgeFeatures(
                    trust=min(self.trust.get(u), self.trust.get(v)),
                    link_quality=link_quality,
                    battery=min(self.battery[u], self.battery[v]),
                    proximity=proximity,
                    freshness=1.0,
                    task_criticality=0.5,
                    processing_cost=cost,
                )
        return edges, features

    def _path_factor(self, node: int, kept: List[Edge], features: Dict[Edge, EdgeFeatures]) -> float:
        adjacency = {n: [] for n in self.nodes}
        for u, v in kept:
            w = features[(u, v)].link_quality
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
        factors = {n: 0.0 for n in self.nodes}
        factors[self.root] = 1.0
        stack = [self.root]
        while stack:
            u = stack.pop()
            for v, quality in adjacency[u]:
                candidate = min(factors[u], quality)
                if candidate > factors[v]:
                    factors[v] = candidate
                    stack.append(v)
        return factors[node]

    def _prune(self, edges: List[Edge], features: Dict[Edge, EdgeFeatures]) -> tuple[List[Edge], float, str]:
        if not edges:
            return [], 0.0, "empty"
        started = time.perf_counter()
        config_kwargs = dict(
            nodes=self.nodes,
            edges=edges,
            features=features,
            previous_decisions=self.previous,
            required_nodes=self.nodes,
            root=self.root,
            degree_limits={n: 3 for n in self.nodes},
            global_budget=25,
        )
        try:
            builder = EdgeSelectionQUBO(**config_kwargs)
            result = ProjectedEdgeAnnealing(
                builder,
                initial_edge_sets=[set(self.previous)] if self.previous else [],
                initial_temperature=0.5,
                final_temperature=0.05,
                cooling_rate=0.80,
                steps_per_temperature=2,
                swap_probability=0.5,
                restarts=2,
                seed=100 + self.cycle,
            ).run()
            mode = "qubo"
            kept = result.kept_edges
        except (RuntimeError, ValueError) as exc:
            # A disconnected or budget-infeasible physical graph should keep
            # the demo alive and be visible in the status stream.
            self.get_logger().warning("QUBO fallback: %s", exc)
            kept = list(edges)
            mode = "keep-all-fallback"
        return kept, (time.perf_counter() - started) * 1e3, mode

    def _run_cycle(self) -> None:
        if self.cycle >= self.steps:
            self.timer.cancel()
            self.get_logger().info("Simulation complete after %d cycles", self.steps)
            rclpy.shutdown()
            return

        edges, features = self._build_graph()
        kept, solve_ms, mode = self._prune(edges, features)
        self.previous = {edge: 1 for edge in kept}

        node_features = {
            n: NodeFeatures(
                trust=self.trust.get(n),
                battery=self.battery[n],
                load=self.load[n],
                capability=self.capability[n],
            )
            for n in self.nodes
        }
        tasks = {
            f"task-{self.cycle}-0": TaskRequirements(criticality=0.7),
            f"task-{self.cycle}-1": TaskRequirements(criticality=0.4),
        }
        hidden = self.gnn.hidden_states(self.nodes, node_features, kept, features)
        scores = {
            node: {
                task_id: self.gnn.score_from_hidden(node, hidden[node], node_features[node], task)
                * (0.5 + 0.5 * self._path_factor(node, kept, features))
                for task_id, task in tasks.items()
            }
            for node in self.nodes
        }
        assignment = assign_tasks(scores)

        for task_id, node in assignment.items():
            execution_ok = self.rng.random() < self.reliability[node]
            delivery_ok = self.rng.random() < max(0.01, self._path_factor(node, kept, features))
            mission_ok = execution_ok and delivery_ok
            self.trust.update(node, 1.0 if execution_ok else 0.0)
            self.load[node] = min(0.95, self.load[node] + 0.12)
            self.get_logger().info(
                "%s -> rover_%d | execute=%s deliver=%s mission=%s",
                task_id,
                node,
                execution_ok,
                delivery_ok,
                mission_ok,
            )
        for node in set(self.nodes) - set(assignment.values()):
            self.trust.relax(node)
        for node in self.nodes:
            self.load[node] *= 0.60
            self.battery[node] = max(0.1, self.battery[node] - 0.01 * len(assignment))

        payload = {
            "cycle": self.cycle,
            "odom_received": sum(self.have_odom.values()),
            "candidate_edges": len(edges),
            "kept_edges": [list(e) for e in kept],
            "pruner": mode,
            "solve_ms": round(solve_ms, 3),
            "assignment": assignment,
            "trust": {str(n): round(self.trust.get(n), 4) for n in self.nodes},
        }
        status = String()
        status.data = json.dumps(payload, sort_keys=True)
        self.status_pub.publish(status)
        assignment_msg = String()
        assignment_msg.data = json.dumps({"cycle": self.cycle, "assignment": assignment})
        self.assignment_pub.publish(assignment_msg)
        self.get_logger().info(
            "cycle=%d odom=%d/4 candidates=%d kept=%d mode=%s solve=%.1fms",
            self.cycle,
            sum(self.have_odom.values()),
            len(edges),
            len(kept),
            mode,
            solve_ms,
        )
        self.cycle += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cycle-period", type=float, default=1.0)
    parser.add_argument("--comm-range", type=float, default=5.0)
    args = parser.parse_args()
    rclpy.init()
    node = GazeboQuantumGNN(args.steps, args.cycle_period, args.comm_range)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
