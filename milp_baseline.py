"""Exact MILP baseline for the constrained edge-selection problem.

This is the strong classical reference the QUBO formulation must be compared
against: the same constrained graph-selection problem solved directly as a
mixed-integer linear program (CBC via PuLP), with no penalty encoding and no
annealing. For single-commodity flow with integer demands the flow relaxation
is integral, so continuous flow variables are exact here.

Objective and constraints are identical in meaning to the QUBO's soft terms:

    min  sum_e (-U_e + lambda) x_e + eta * sum_e |x_e - x_prev_e|
    s.t. per-node degree caps, global cost budget,
         required-set connectivity via single-commodity flow.

Requires: pulp (bundles the CBC solver).
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

from main import Edge, EdgeFeatures, Node, QUBOConfig, canonical_edge


def solve_qubo_exact(Q, constant: float = 0.0, time_limit: int = 300) -> float:
    """
    Exact minimum of E(z) = z^T Q z + constant over binary z, by standard
    linearization (one auxiliary variable per nonzero quadratic term),
    solved with CBC. This is the exact QUBO solver reference: it works on
    the actual Q matrix, not on the original constrained problem.
    """
    import pulp
    import numpy as np

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    prob = pulp.LpProblem("qubo_exact", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

    obj = [float(constant)]
    for i in range(n):
        if Q[i, i] != 0.0:
            obj.append(float(Q[i, i]) * x[i])
    for i in range(n):
        for j in range(i + 1, n):
            qij = 2.0 * float(Q[i, j])  # z^T Q z counts both (i,j) and (j,i)
            if qij == 0.0:
                continue
            z = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
            # z = x_i AND x_j
            prob += z <= x[i]
            prob += z <= x[j]
            prob += z >= x[i] + x[j] - 1
            obj.append(qij * z)
    prob += pulp.lpSum(obj)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"exact QUBO solve did not prove optimality: {pulp.LpStatus[status]}"
        )
    return float(pulp.value(prob.objective))


def solve_milp(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    features: Dict[Edge, EdgeFeatures],
    previous_decisions: Optional[Dict[Edge, int]] = None,
    hard_critical_edges: Iterable[Edge] = (),
    required_nodes: Optional[Sequence[Node]] = None,
    root: Optional[Node] = None,
    degree_limits: Optional[Dict[Node, int]] = None,
    global_budget: Optional[int] = None,
    config: Optional[QUBOConfig] = None,
    time_limit: int = 60,
) -> Tuple[float, list[Edge]]:
    """Return (soft energy of the optimum, kept edges)."""
    import pulp

    cfg = config or QUBOConfig()
    nodes = sorted(set(nodes))
    edges = sorted({canonical_edge(e) for e in edges})
    hard = {canonical_edge(e) for e in hard_critical_edges}
    optional = [e for e in edges if e not in hard]
    previous = {canonical_edge(e): int(v)
                for e, v in (previous_decisions or {}).items()}
    cost_units = {
        e: max(1, int(round(features[e].processing_cost * cfg.cost_scale)))
        for e in edges
    }
    weights = dict(
        trust=cfg.w_trust, link_quality=cfg.w_link, battery=cfg.w_battery,
        proximity=cfg.w_proximity, freshness=cfg.w_freshness,
        task_criticality=cfg.w_task,
    )

    def utility(e: Edge) -> float:
        f = features[e]
        return sum(getattr(f, k) * w for k, w in weights.items())

    prob = pulp.LpProblem("edge_selection", pulp.LpMinimize)
    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat="Binary") for e in optional}

    # objective: utility + sparsity + stability (linear for binary x)
    obj = []
    for e in optional:
        prev = previous.get(e, 0)
        obj.append((-utility(e) + cfg.edge_penalty) * x[e])
        if prev == 0:
            obj.append(cfg.stability_penalty * x[e])
        else:
            obj.append(cfg.stability_penalty * (1 - x[e]))
    prob += pulp.lpSum(obj)

    # degree caps
    fixed_deg = {n: 0 for n in nodes}
    for u, v in hard:
        fixed_deg[u] += 1
        fixed_deg[v] += 1
    for n, limit in (degree_limits or {}).items():
        prob += (
            pulp.lpSum(x[e] for e in optional if n in e)
            <= limit - fixed_deg[n]
        ), f"deg_{n}"

    # budget
    if global_budget is not None:
        fixed_cost = sum(cost_units[e] for e in hard)
        prob += (
            pulp.lpSum(cost_units[e] * x[e] for e in optional)
            <= global_budget - fixed_cost
        ), "budget"

    # connectivity via single-commodity flow (continuous; integral here)
    if required_nodes:
        units = len(required_nodes) - 1
        f = {}
        for u, v in edges:
            for a, b in ((u, v), (v, u)):
                f[(a, b)] = pulp.LpVariable(f"f_{a}_{b}", lowBound=0,
                                            upBound=units)
        required_set = set(required_nodes)
        for n in nodes:
            inc = pulp.lpSum(f[(a, b)] for (a, b) in f if b == n)
            out = pulp.lpSum(f[(a, b)] for (a, b) in f if a == n)
            if n == root:
                prob += (inc - out == -units), f"flow_{n}"
            elif n in required_set:
                prob += (inc - out == 1), f"flow_{n}"
            else:
                prob += (inc - out == 0), f"flow_{n}"
        for e in edges:
            u, v = e
            cap = pulp.lpSum([f[(u, v)], f[(v, u)]])
            if e in hard:
                prob += (cap <= units), f"cap_{u}_{v}"
            else:
                prob += (cap <= units * x[e]), f"cap_{u}_{v}"

    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        raise ValueError(f"MILP infeasible: {pulp.LpStatus[status]}")

    kept = set(hard)
    kept.update(e for e in optional if x[e].value() > 0.5)
    energy = float(pulp.value(prob.objective))
    return energy, sorted(kept)
