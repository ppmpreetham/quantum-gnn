"""Worked example for the presentation document.

Builds a small, fully inspectable instance, then:
  1. enumerates ALL 2^6 = 64 optional-edge subsets (battle_test's
     exact_edge_enumeration procedure: closed-form slack assignment +
     connectivity check), so every number below is exactly verifiable;
  2. cross-checks the enumerated optimum against the Q matrix itself
     (builder.energy on the feasible solution vector);
  3. runs the production projected annealer cold (no warm start);
  4. runs the exact MILP baseline (CBC) on the linearized Q matrix;
  5. prints a JSON blob with every number used in the document.

Run:  .venv/Scripts/python.exe worked_example.py
"""

from __future__ import annotations

import json

from main import (
    EdgeFeatures,
    EdgeSelectionQUBO,
    ProjectedEdgeAnnealing,
    QUBOConfig,
)
from milp_baseline import solve_milp

# ----------------------------------------------------------------------
# A 5-node scenario: root 0, required {0, 2, 3, 4}; node 1 is optional relay.
# ----------------------------------------------------------------------
nodes = [0, 1, 2, 3, 4]
edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (2, 4), (3, 4)]

#                     trust link bat  prox fresh task  cost
features = {
    (0, 1): EdgeFeatures(0.90, 0.90, 0.80, 0.70, 0.90, 0.50, 0.20),
    (0, 2): EdgeFeatures(0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.10),
    (0, 3): EdgeFeatures(0.40, 0.50, 0.50, 0.40, 0.40, 0.20, 0.80),
    (1, 2): EdgeFeatures(0.80, 0.70, 0.90, 0.60, 0.80, 0.40, 0.30),
    (2, 3): EdgeFeatures(0.70, 0.80, 0.80, 0.70, 0.90, 0.60, 0.40),
    (2, 4): EdgeFeatures(0.85, 0.85, 0.75, 0.80, 0.85, 0.55, 0.20),
    (3, 4): EdgeFeatures(0.60, 0.60, 0.70, 0.50, 0.60, 0.30, 0.60),
}

previous = {(0, 1): 1, (0, 2): 1, (0, 3): 0, (1, 2): 1,
            (2, 3): 1, (2, 4): 0, (3, 4): 1}

cfg = QUBOConfig()  # default weights/penalties, seed 42

builder = EdgeSelectionQUBO(
    nodes=nodes,
    edges=edges,
    features=features,
    previous_decisions=previous,
    hard_critical_edges=[(0, 1)],          # fixed keep: only link to node 1
    required_nodes=[0, 2, 3, 4],
    root=0,
    degree_limits={0: 3, 1: 2, 2: 3, 3: 2, 4: 2},
    global_budget=12,                       # cost units after cost_scale=10
    config=cfg,
)

Q, const, names = builder.get_qubo()
optional = builder.optional_edges
n_opt = len(optional)


def utility_of(edge) -> float:
    f = features[edge]
    return (cfg.w_trust * f.trust + cfg.w_link * f.link_quality
            + cfg.w_battery * f.battery + cfg.w_proximity * f.proximity
            + cfg.w_freshness * f.freshness + cfg.w_task * f.task_criticality)


# ----------------------------------------------------------------------
# 1) Exhaustive enumeration over all 2^6 = 64 subsets, battle_test-style:
#    closed-form best slack assignment per subset + connectivity check.
# ----------------------------------------------------------------------
records = []
for mask in range(1 << n_opt):
    sel = set(builder.hard_critical)
    chosen = []
    E = 0.0
    for k, e in enumerate(optional):
        x = (mask >> k) & 1
        if x:
            sel.add(e)
            chosen.append(e)
            E += -utility_of(e) + cfg.edge_penalty
        prev = builder.previous.get(e, 0)
        E += cfg.stability_penalty * (x - prev) ** 2

    deg_viol, bud_viol = 0.0, 0.0
    for node, limit in builder.degree_limits.items():
        d = sum(node in e for e in sel)
        if d > limit:
            deg_viol = max(deg_viol, d - limit)
            E += cfg.degree_penalty * (d - limit) ** 2
    if builder.global_budget is not None:
        cost = sum(builder.edge_cost_units[e] for e in sel)
        if cost > builder.global_budget:
            bud_viol = cost - builder.global_budget
            E += cfg.budget_penalty * (cost - builder.global_budget) ** 2

    adjacency = {n: [] for n in builder.nodes}
    for u, v in sel:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen, stack = {builder.root}, [builder.root]
    while stack:
        for v in adjacency[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    conn_ok = set(builder.required_nodes).issubset(seen)

    records.append({
        "kept": sorted(f"{a}-{b}" for a, b in sel),
        "n_kept": len(sel),
        "energy": E,
        "feasible": conn_ok and deg_viol == 0 and bud_viol == 0,
        "degree_over": deg_viol,
        "budget_over": bud_viol,
        "disconnected": sorted(set(builder.required_nodes) - seen) if not conn_ok else [],
        "total_cost": sum(builder.edge_cost_units[e] for e in sel),
    })

feasible = [r for r in records if r["feasible"]]
infeasible = [r for r in records if not r["feasible"]]
# connectivity traps: no closed-form penalty exists (enumeration skips them,
# as in battle_test); degree/budget violators carry explicit penalties.
conn_traps = [r for r in infeasible if r["disconnected"]]
deg_bud_violators = [r for r in infeasible if not r["disconnected"]]
best_feas = min(feasible, key=lambda r: r["energy"])
best_inf = min(deg_bud_violators, key=lambda r: r["energy"]) if deg_bud_violators else None
best_trap = min(conn_traps, key=lambda r: r["energy"])
feas_energies = sorted(r["energy"] for r in feasible)
best_set = {tuple(map(int, e.split("-"))) for e in best_feas["kept"]}

# ----------------------------------------------------------------------
# 2) Cross-check: enumerated optimum must equal the Q-matrix energy of the
#    same subset (mirrors battle_test B.cross).
# ----------------------------------------------------------------------
z_best = builder.build_feasible_solution({tuple(map(int, e.split("-")))
                                          for e in best_feas["kept"]})
qmatrix_energy = float(builder.energy(z_best))
cross_check_ok = abs(qmatrix_energy - best_feas["energy"]) < 1e-9

# ----------------------------------------------------------------------
# 3) Production projected annealer, cold start (no warm start), default seed.
# ----------------------------------------------------------------------
annealer = ProjectedEdgeAnnealing(builder)
res = annealer.run()
anneal = {
    "kept": sorted(f"{a}-{b}" for a, b in res.kept_edges),
    "energy": float(builder.energy(res.solution)),
    "violations": builder.check_constraints(res.solution),
    "is_optimal": {tuple(e) for e in res.kept_edges} == best_set,
}

# ----------------------------------------------------------------------
# 4) Exact MILP baseline (CBC), direct constrained formulation (no
#    penalties). For feasible subsets, enumeration energy == soft energy,
#    so the two must agree exactly on the optimum.
# ----------------------------------------------------------------------
milp_energy, milp_edges = solve_milp(
    nodes=nodes, edges=edges, features=features,
    previous_decisions=previous, hard_critical_edges=[(0, 1)],
    required_nodes=[0, 2, 3, 4], root=0,
    degree_limits={0: 3, 1: 2, 2: 3, 3: 2, 4: 2},
    global_budget=12, config=cfg,
)
milp_out = {
    "soft_energy": float(milp_energy),
    "kept": sorted(f"{a}-{b}" for a, b in milp_edges),
    "matches_bruteforce": abs(float(milp_energy) - best_feas["energy"]) < 1e-6,
}

# top-5 feasible ladder and the two cheapest infeasible traps, for the doc
top5 = sorted(feasible, key=lambda r: r["energy"])[:5]
traps = sorted(infeasible, key=lambda r: r["energy"])[:2]

out = {
    "instance": {
        "nodes": nodes, "edges": [list(e) for e in edges],
        "hard_critical": [0, 1], "required": [0, 2, 3, 4], "root": 0,
        "degree_limits": {0: 3, 1: 2, 2: 3, 3: 2, 4: 2}, "budget_units": 12,
        "edge_costs": {f"{a}-{b}": builder.edge_cost_units.get((a, b))
                       for a, b in edges},
        "utilities": {f"{a}-{b}": round(utility_of((a, b)), 4)
                      for a, b in optional},
        "previous": {f"{a}-{b}": previous[(a, b)] for a, b in edges},
        "soft_terms": {"edge_penalty": cfg.edge_penalty,
                       "stability_penalty": cfg.stability_penalty,
                       "w_trust": cfg.w_trust, "w_link": cfg.w_link,
                       "w_battery": cfg.w_battery, "w_proximity": cfg.w_proximity,
                       "w_freshness": cfg.w_freshness, "w_task": cfg.w_task,
                       "degree_penalty": cfg.degree_penalty,
                       "budget_penalty": cfg.budget_penalty},
    },
    "qubo": {"n_variables": len(names),
             "n_edge_vars": n_opt,
             "n_slack_flow_vars": len(names) - n_opt},
    "brute_force": {
        "n_subsets": len(records),
        "n_feasible": len(feasible),
        "n_infeasible": len(infeasible),
        "n_conn_traps": len(conn_traps),
        "best_feasible": best_feas,
        "best_degree_budget_violator": best_inf,
        "best_connectivity_trap": best_trap,
        "feasible_energy_band": [feas_energies[0], feas_energies[-1]],
        "degree_budget_barrier": (best_inf["energy"] - best_feas["energy"])
                                 if best_inf else None,
        "trap_lure": best_feas["energy"] - best_trap["energy"],
        "top5_feasible": top5,
        "cheapest_infeasible_traps": traps,
    },
    "cross_check": {"qmatrix_energy": qmatrix_energy,
                    "matches_enumeration": cross_check_ok},
    "annealer": anneal,
    "milp": milp_out,
}
print(json.dumps(out, indent=1, default=str))
