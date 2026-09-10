"""Scaling runner: the QUBO's evidence base.

Range-limited sparse graphs (k-nearest neighbors, as with ESP-NOW rovers) at
increasing fleet sizes. Per size: edge/var counts, Q memory, build and solve
time, quality vs the greedy top-K baseline (same edge budget), and GNN
inference latency on the full vs pruned graph.

Note: the flow-based connectivity encoding scales O(m log n) variables with a
dense Q, so connectivity is only enabled for the smaller sizes; the dense-Q
memory bound is itself a scope limit worth reporting.

Run:  python3 scale_runner.py
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from main import EdgeFeatures, EdgeSelectionQUBO, ProjectedEdgeAnnealing, QUBOConfig
from simulation import edge_utility
from trust_gnn import NodeFeatures, TaskRequirements, TrustGNN


def make_sparse_instance(rng: np.random.Generator, n: int, k: int = 4):
    """Range-limited graph: each node links to its k nearest neighbors."""
    pos = rng.uniform(0, 100, size=(n, 2))
    edges = set()
    # random spanning tree first: k-NN graphs can be disconnected, and the
    # connectivity constraint must be exercisable
    perm = rng.permutation(n).tolist()
    for i in range(1, n):
        j = int(rng.integers(0, i))
        edges.add((min(perm[i], perm[j]), max(perm[i], perm[j])))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        for j in np.argsort(d)[1:k + 1]:
            edges.add((min(i, j), max(i, int(j))))
    edges = sorted(edges)
    features = {
        e: EdgeFeatures(*[float(rng.random()) for _ in range(7)])
        for e in edges
    }
    previous = {e: int(rng.integers(0, 2)) for e in edges}
    degree_limits = {i: 4 for i in range(n)}
    budget = int(0.6 * len(edges) * 5)
    return pos, edges, features, previous, degree_limits, budget


def repair_to_feasible(builder: EdgeSelectionQUBO, selected, features) -> tuple:
    """
    Drop lowest-utility edges until degree/budget constraints hold.
    Connectivity cannot be repaired by removal; report it as a flag.
    Returns (edge_set, fully_feasible).
    """
    sel = set(selected)
    for _ in range(len(sel) + 1):
        z = np.zeros(len(builder.variables))
        for e in builder.optional_edges:
            if e in sel:
                z[builder.edge_var[e]] = 1
        issues = builder.check_constraints(z)
        if not issues:
            return sel, True
        if any(i.startswith("connectivity") for i in issues):
            return sel, False
        over_nodes = set()
        budget_over = False
        for i in issues:
            if i.startswith("degree"):
                over_nodes.add(int(i.split("[")[1].split("]")[0]))
            elif i.startswith("budget"):
                budget_over = True
        drop_candidates = [
            e for e in sel
            if e not in builder.hard_critical
            and (e[0] in over_nodes or e[1] in over_nodes or budget_over)
        ]
        if not drop_candidates:
            return sel, False
        sel.remove(min(drop_candidates, key=lambda e: edge_utility(features[e])))
    return sel, False


def main() -> None:
    rng = np.random.default_rng(21)
    gnn = TrustGNN()
    print(f"{'n':>4s} {'edges':>6s} {'vars':>6s} {'Q MB':>7s} {'build s':>8s} "
          f"{'solve s':>8s} {'kept':>5s} {'E_proj':>9s} {'E_greedy':>9s} "
          f"{'gnn full ms':>11s} {'gnn pruned ms':>13s}")
    for n in (10, 20, 30, 50):
        pos, edges, features, previous, degree_limits, budget = \
            make_sparse_instance(rng, n)
        conn = n <= 20
        kwargs = dict(
            nodes=list(range(n)), edges=edges, features=features,
            previous_decisions=previous, degree_limits=degree_limits,
            global_budget=budget,
        )
        if conn:
            kwargs.update(required_nodes=list(range(n)), root=0)

        t0 = time.perf_counter()
        try:
            builder = EdgeSelectionQUBO(**kwargs)
        except ValueError:
            kwargs.pop("required_nodes", None)
            kwargs.pop("root", None)
            builder = EdgeSelectionQUBO(**kwargs)
            conn = False
        t_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            result = ProjectedEdgeAnnealing(
                builder, initial_edge_sets=[set(edges)], restarts=3,
                seed=n,
            ).run()
            kept = result.kept_edges
            e_proj = result.energy
        except ValueError:
            # infeasible joint constraints: retry without connectivity
            kwargs.pop("required_nodes", None)
            kwargs.pop("root", None)
            builder = EdgeSelectionQUBO(**kwargs)
            result = ProjectedEdgeAnnealing(
                builder, initial_edge_sets=[set(edges)], restarts=3,
                seed=n,
            ).run()
            kept = result.kept_edges
            e_proj = result.energy
        t_solve = time.perf_counter() - t0

        # greedy top-K baseline with the SAME edge budget, repaired to
        # feasibility (raw greedy violates degree/budget caps)
        k = len(kept)
        ranked = sorted(edges, key=lambda e: -edge_utility(features[e]))
        greedy, greedy_ok = repair_to_feasible(builder, set(ranked[:k]), features)
        e_greedy = builder.soft_energy(greedy) if greedy_ok else float("nan")

        # GNN latency: full vs pruned graph
        node_features = {
            i: NodeFeatures(trust=float(rng.random()), battery=float(rng.random()),
                            load=float(rng.random() * 0.5),
                            capability=float(rng.random()))
            for i in range(n)
        }
        task = TaskRequirements()
        t0 = time.perf_counter()
        gnn.scores(list(range(n)), node_features, edges, task, features)
        t_full = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        gnn.scores(list(range(n)), node_features, kept, task, features)
        t_pruned = (time.perf_counter() - t0) * 1e3

        q_mb = builder.Q.nbytes / 1e6
        print(f"{n:>4d} {len(edges):>6d} {len(builder.variables):>6d} "
              f"{q_mb:7.2f} {t_build:8.2f} {t_solve:8.2f} {k:>5d} "
              f"{e_proj:9.3f} {e_greedy:9.3f} "
              f"{t_full:11.2f} {t_pruned:13.2f}")


if __name__ == "__main__":
    main()
