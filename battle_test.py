"""Battle-test suite for the QUBO edge-selection module (main.py).

Covers:
  A. QUBO construction invariants (symmetry, independent energy recomputation)
  B. Brute-force ground truth vs simulated annealing on small instances
  C. Analytic optimum checks (unconstrained / stability-only cases)
  D. Randomized scenario matrix (degree/budget/connectivity/hard-edge mixes)
  E. Input-validation and edge-case behaviour
  F. Scaling / performance sanity

Run:  python3 battle_test.py
"""

from __future__ import annotations

import itertools
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import neal  # optional: real QUBO backend (pip install dwave-neal)
    HAVE_NEAL = True
except ImportError:
    HAVE_NEAL = False

from main import (
    EdgeFeatures,
    EdgeSelectionQUBO,
    ProjectedEdgeAnnealing,
    QUBOConfig,
    SimulatedAnnealing,
    canonical_edge,
)

PASS = 0
FAIL = 0
FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}: {detail}")


def expect_raises(name: str, exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        check(name, True)
    except Exception as e:  # noqa: BLE001
        check(name, False, f"expected {exc_type.__name__}, got {type(e).__name__}: {e}")
    else:
        check(name, False, "no exception raised")


# ----------------------------------------------------------------------
# Independent energy evaluator (recomputes the spec, ignores Q)
# ----------------------------------------------------------------------

WEIGHT_ATTRS = [
    ("w_trust", "trust"),
    ("w_link", "link_quality"),
    ("w_battery", "battery"),
    ("w_proximity", "proximity"),
    ("w_freshness", "freshness"),
    ("w_task", "task_criticality"),
]


def utility_of(cfg: QUBOConfig, f: EdgeFeatures) -> float:
    return sum(getattr(cfg, w) * getattr(f, a) for w, a in WEIGHT_ATTRS)


def independent_energy(b: EdgeSelectionQUBO, z: np.ndarray) -> float:
    cfg = b.cfg
    z = np.rint(z).astype(int)
    E = 0.0

    # soft objective
    for e in b.optional_edges:
        x = int(z[b.edge_var[e]])
        E += x * (-utility_of(cfg, b.features[e]) + cfg.edge_penalty)
        prev = b.previous.get(e, 0)
        E += cfg.stability_penalty * (x - prev) ** 2

    # degree constraints (only for nodes with a limit)
    fixed_deg = {n: 0 for n in b.nodes}
    for u, v in b.hard_critical:
        fixed_deg[u] += 1
        fixed_deg[v] += 1
    for n in b.nodes:
        if n not in b.degree_limits:
            continue
        rhs = b.degree_limits[n] - fixed_deg[n]
        opt_deg = sum(n in e for e in b.optional_edges if z[b.edge_var[e]] == 1)
        slack = sum(int(z[i]) * (2**k) for k, i in enumerate(b.degree_slack_bits[n]))
        E += cfg.degree_penalty * (opt_deg + slack - rhs) ** 2

    # budget constraint
    if b.global_budget is not None:
        fixed_cost = sum(b.edge_cost_units[e] for e in b.hard_critical)
        rhs = b.global_budget - fixed_cost
        opt_cost = sum(
            b.edge_cost_units[e] for e in b.optional_edges if z[b.edge_var[e]] == 1
        )
        slack = sum(int(z[i]) * (2**k) for k, i in enumerate(b.budget_slack_bits))
        E += cfg.budget_penalty * (opt_cost + slack - rhs) ** 2

    # connectivity flow
    if b.required_nodes:
        units = len(b.required_nodes) - 1

        def flow(u, v):
            return sum(int(z[i]) * (2**k) for k, i in enumerate(b.flow_bits[(u, v)]))

        for n in b.nodes:
            inc = out = 0
            for u, v in b.edges:
                if n == u:
                    inc += flow(v, u)
                    out += flow(u, v)
                elif n == v:
                    inc += flow(u, v)
                    out += flow(v, u)
            if n == b.root:
                target = -units
            elif n in set(b.required_nodes):
                target = 1
            else:
                target = 0
            E += cfg.flow_penalty * (inc - out - target) ** 2

        for e in b.edges:
            u, v = e
            q = sum(int(z[i]) * (2**k) for k, i in enumerate(b.flow_slack_bits[e]))
            x = 1 if e in b.hard_critical else int(z[b.edge_var[e]])
            E += cfg.flow_capacity_penalty * (flow(u, v) + flow(v, u) + q - units * x) ** 2

    return E


def brute_force(Q: np.ndarray, c: float) -> Tuple[np.ndarray, float]:
    n = Q.shape[0]
    assert n <= 20, f"too many vars for brute force: {n}"
    best_e = float("inf")
    best_z = None
    for chunk in itertools.batched(range(2**n), 65536):
        states = np.array(
            [[(s >> k) & 1 for k in range(n)] for s in chunk], dtype=float
        )
        energies = np.einsum("bi,ij,bj->b", states, Q, states) + c
        i = int(np.argmin(energies))
        if energies[i] < best_e:
            best_e = float(energies[i])
            best_z = states[i].copy()
    return best_z, best_e


# ----------------------------------------------------------------------
# Random instance generation
# ----------------------------------------------------------------------


def random_connected_instance(
    rng: np.random.Generator,
    n: int,
    extra_edge_p: float = 0.4,
    with_budget: bool = True,
    with_required: bool = True,
    tight: bool = False,
) -> dict:
    # random spanning tree guarantees connectivity
    perm = rng.permutation(n).tolist()
    tree = []
    for i in range(1, n):
        j = int(rng.integers(0, i))
        tree.append(canonical_edge((perm[i], perm[j])))
    edges = set(tree)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < extra_edge_p:
                edges.add((i, j))
    edges = sorted(edges)

    features = {
        e: EdgeFeatures(*[float(rng.random()) for _ in range(7)]) for e in edges
    }
    previous = {e: int(rng.integers(0, 2)) for e in edges}

    tree_deg = {i: 0 for i in range(n)}
    for u, v in tree:
        tree_deg[u] += 1
        tree_deg[v] += 1

    slack_deg = 1 if tight else 3
    degree_limits = {i: int(tree_deg[i] + rng.integers(0, slack_deg + 1)) for i in range(n)}

    cost = {e: max(1, round(features[e].processing_cost * 10)) for e in edges}
    tree_cost = sum(cost[e] for e in tree)
    budget = None
    if with_budget:
        budget = int(tree_cost + (rng.integers(0, 3) if tight else rng.integers(2, 20)))

    required: List[int] = []
    root = None
    if with_required:
        k = int(rng.integers(2, n + 1))
        required = sorted(rng.choice(n, size=k, replace=False).tolist())
        root = required[0]

    hard = [e for e in tree if rng.random() < 0.25]

    return dict(
        nodes=list(range(n)),
        edges=edges,
        features=features,
        previous_decisions=previous,
        hard_critical_edges=hard,
        required_nodes=required,
        root=root,
        degree_limits=degree_limits,
        global_budget=budget,
        warm_edges=set(tree) | set(hard),
    )


def fast_config(seed: int, restarts: int = 6) -> QUBOConfig:
    return QUBOConfig(
        initial_temperature=5.0,
        cooling_rate=0.9,
        sweeps_per_temperature=2,
        restarts=restarts,
        seed=seed,
    )


def ctor_args(inst: dict) -> dict:
    return {k: v for k, v in inst.items() if k != "warm_edges"}


def solve(builder: EdgeSelectionQUBO, warm_edges, cfg: QUBOConfig):
    Q, c, _ = builder.get_qubo()
    warm = None
    try:
        warm = builder.build_feasible_solution(warm_edges)
    except Exception:  # noqa: BLE001
        pass
    sa = SimulatedAnnealing(
        Q,
        c,
        initial_temperature=cfg.initial_temperature,
        final_temperature=cfg.final_temperature,
        cooling_rate=cfg.cooling_rate,
        sweeps_per_temperature=cfg.sweeps_per_temperature,
        restarts=cfg.restarts,
        seed=cfg.seed,
        initial_states=[warm] if warm is not None else None,
    )
    z, e = sa.run()
    return z, e, warm


# ----------------------------------------------------------------------
# A. Construction invariants
# ----------------------------------------------------------------------


def test_invariants() -> None:
    print("A. QUBO construction invariants")
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = int(rng.integers(3, 8))
        inst = random_connected_instance(rng, n)
        cfg = fast_config(trial)
        b = EdgeSelectionQUBO(config=cfg, **ctor_args(inst))
        Q, c, names = b.get_qubo()
        check(f"A.sym t{trial}", np.allclose(Q, Q.T), "Q not symmetric")
        check(f"A.fin t{trial}", np.all(np.isfinite(Q)), "Q has non-finite entries")
        check(f"A.uniq t{trial}", len(names) == len(set(names)), "duplicate var names")
        for _ in range(5):
            z = rng.integers(0, 2, size=len(names)).astype(float)
            got = b.energy(z)
            want = independent_energy(b, z)
            check(
                f"A.energy t{trial}",
                abs(got - want) < 1e-6 * max(1.0, abs(want)),
                f"energy {got} != independent {want}",
            )


def exact_edge_enumeration(b: EdgeSelectionQUBO) -> float:
    """Exact optimum over optional-edge subsets.

    Valid because, for any fixed edge subset, the slack/flow variables have a
    closed-form best assignment: degree/budget slacks cancel the residual when
    the subset is within limits, and connectivity feasibility is equivalent to
    graph connectivity (tree flow always fits capacity).
    """
    opts = b.optional_edges
    assert len(opts) <= 20, "too many optional edges to enumerate"
    cfg = b.cfg
    req = set(b.required_nodes)
    best = float("inf")

    for mask in range(1 << len(opts)):
        sel = set(b.hard_critical)
        E = 0.0
        for k, e in enumerate(opts):
            x = (mask >> k) & 1
            if x:
                sel.add(e)
                E += -utility_of(cfg, b.features[e]) + cfg.edge_penalty
            prev = b.previous.get(e, 0)
            E += cfg.stability_penalty * (x - prev) ** 2

        for node, limit in b.degree_limits.items():
            d = sum(node in e for e in sel)
            if d > limit:
                E += cfg.degree_penalty * (d - limit) ** 2

        if b.global_budget is not None:
            cost = sum(b.edge_cost_units[e] for e in sel)
            if cost > b.global_budget:
                E += cfg.budget_penalty * (cost - b.global_budget) ** 2

        if req:
            adjacency = {n: [] for n in b.nodes}
            for u, v in sel:
                adjacency[u].append(v)
                adjacency[v].append(u)
            seen = {b.root}
            stack = [b.root]
            while stack:
                for v in adjacency[stack.pop()]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            if not req.issubset(seen):
                continue  # infeasible: no flow can fix it

        best = min(best, E)
    return best


# ----------------------------------------------------------------------
# B. Brute-force ground truth vs SA
# ----------------------------------------------------------------------


def test_brute_force() -> None:
    print("B. SA vs brute-force optimum (small instances)")
    rng = np.random.default_rng(1)
    gaps = []
    raw_infeasible = 0
    repair_fail = 0
    total = 0
    for trial in range(60):
        n = int(rng.integers(3, 6))
        inst = random_connected_instance(
            rng,
            n,
            with_budget=bool(rng.random() < 0.5),
            with_required=bool(rng.random() < 0.5),
            tight=bool(rng.random() < 0.5),
        )
        cfg = fast_config(1000 + trial, restarts=8)
        b = EdgeSelectionQUBO(config=cfg, **ctor_args(inst))
        Q, c, names = b.get_qubo()
        if len(names) > 20:
            continue
        total += 1
        z_opt, e_opt = brute_force(Q, c)
        e_edge = exact_edge_enumeration(b)
        check(f"B.cross t{trial}", abs(e_edge - e_opt) < 1e-9,
              f"edge-enumeration {e_edge} != bit brute-force {e_opt}")
        check(f"B.bf-feas t{trial}", b._constraint_energy(z_opt) < 1e-9,
              f"brute-force optimum violates constraints (penalty leakage), e={e_opt}")

        z_sa, e_sa, warm = solve(b, inst["warm_edges"], cfg)
        if b.check_constraints(z_sa):
            raw_infeasible += 1
        # repair path used by example.py
        selected = set(b.selected_edges(z_sa))
        try:
            repaired = b.build_feasible_solution(selected)
            e_rep = b.energy(repaired)
        except Exception:  # noqa: BLE001
            repair_fail += 1
            e_rep = warm is not None and b.energy(warm) or float("inf")
        best_e = min(e_sa if not b.check_constraints(z_sa) else float("inf"), e_rep)
        if np.isfinite(best_e):
            gaps.append(best_e - e_opt)
    print(f"  brute-forced {total} instances; raw SA infeasible: {raw_infeasible}, "
          f"repair failures: {repair_fail}")
    if gaps:
        gaps = np.array(gaps)
        print(f"  optimality gap: mean={gaps.mean():.4f} max={gaps.max():.4f} "
              f"exact={np.mean(gaps < 1e-9)*100:.0f}%")
        check("B.gap", np.percentile(gaps, 90) < 0.5,
              f"90th percentile gap too large: {np.percentile(gaps, 90)}")
    check("B.repair", repair_fail == 0, f"{repair_fail} repair failures")


# ----------------------------------------------------------------------
# C. Analytic optima
# ----------------------------------------------------------------------


def test_analytic() -> None:
    print("C. Analytic optimum checks")
    rng = np.random.default_rng(2)
    for trial in range(20):
        n = int(rng.integers(3, 7))
        inst = random_connected_instance(rng, n, with_budget=False, with_required=False)
        cfg = QUBOConfig(stability_penalty=0.0)
        # drop all degree limits -> fully unconstrained
        inst["degree_limits"] = {}
        b = EdgeSelectionQUBO(config=cfg, **{k: v for k, v in inst.items() if k != "warm_edges"})
        Q, c, names = b.get_qubo()
        if len(names) > 20:
            continue
        _, e_opt = brute_force(Q, c)
        # analytic: keep edge iff utility > edge_penalty
        expected = sum(
            min(0.0, -utility_of(cfg, b.features[e]) + cfg.edge_penalty)
            for e in b.optional_edges
        ) + sum(cfg.stability_penalty for e in b.hard_critical if False)  # hard edges: no vars
        # hard edges contribute no soft terms (they are fixed, not variables)
        check(f"C.uncon t{trial}", abs(e_opt - expected) < 1e-9,
              f"bf {e_opt} != analytic {expected}")

    # stability only: keep iff utility - edge_penalty + stability*(2*prev-1) > 0
    for trial in range(20):
        n = int(rng.integers(3, 7))
        inst = random_connected_instance(rng, n, with_budget=False, with_required=False)
        cfg = QUBOConfig(stability_penalty=0.3, edge_penalty=0.1)
        inst["degree_limits"] = {}
        b = EdgeSelectionQUBO(config=cfg, **{k: v for k, v in inst.items() if k != "warm_edges"})
        Q, c, names = b.get_qubo()
        if len(names) > 20:
            continue
        _, e_opt = brute_force(Q, c)
        expected = 0.0
        for e in b.optional_edges:
            prev = b.previous.get(e, 0)
            a0 = cfg.stability_penalty * (0 - prev) ** 2
            a1 = -utility_of(cfg, b.features[e]) + cfg.edge_penalty + cfg.stability_penalty * (1 - prev) ** 2
            expected += min(a0, a1)
        check(f"C.stab t{trial}", abs(e_opt - expected) < 1e-9,
              f"bf {e_opt} != analytic {expected}")


# ----------------------------------------------------------------------
# D. Scenario matrix stats
# ----------------------------------------------------------------------


def test_scenario_matrix() -> None:
    print("D. Randomized scenario matrix (larger, no brute force)")
    rng = np.random.default_rng(3)
    raw_infeasible = 0
    final_infeasible = 0
    total = 0
    for trial in range(80):
        n = int(rng.integers(4, 11))
        inst = random_connected_instance(
            rng,
            n,
            extra_edge_p=float(rng.uniform(0.2, 0.7)),
            tight=bool(rng.random() < 0.5),
        )
        cfg = fast_config(2000 + trial)
        b = EdgeSelectionQUBO(config=cfg, **ctor_args(inst))
        z, _, warm = solve(b, inst["warm_edges"], cfg)
        total += 1
        if b.check_constraints(z):
            raw_infeasible += 1
        try:
            repaired = b.build_feasible_solution(set(b.selected_edges(z)))
            final_z = repaired
        except Exception:  # noqa: BLE001
            final_z = warm if warm is not None else z
        issues = b.check_constraints(final_z)
        if issues:
            final_infeasible += 1
            check(f"D.feas t{trial}", False, f"post-repair infeasible: {issues}")
    print(f"  {total} instances; raw SA infeasible {raw_infeasible} "
          f"({raw_infeasible/total*100:.0f}%), post-repair infeasible {final_infeasible}")
    check("D.final", final_infeasible == 0, f"{final_infeasible} final infeasible")


# ----------------------------------------------------------------------
# E. Validation / edge cases
# ----------------------------------------------------------------------


def F(**kw) -> EdgeFeatures:
    base = dict(trust=.5, link_quality=.5, battery=.5, proximity=.5,
                freshness=.5, task_criticality=.5, processing_cost=.5)
    base.update(kw)
    return EdgeFeatures(**base)


def test_edge_cases() -> None:
    print("E. Validation and edge cases")

    expect_raises("E.selfloop", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0], edges=[(0, 0)], features={}))
    expect_raises("E.unknown-node", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0], edges=[(0, 1)], features={}))
    expect_raises("E.feat-range", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F(trust=1.5)}))
    expect_raises("E.feat-nan", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F(trust=float("nan"))}))
    expect_raises("E.missing-features", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
                                            features={(0, 1): F()}))
    expect_raises("E.weights-sum", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F()},
                                            config=QUBOConfig(w_trust=0.9)))
    expect_raises("E.bad-previous", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F()},
                                            previous_decisions={(0, 1): 2}))
    expect_raises("E.infeasible-conn", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1)],
                                            features={(0, 1): F()},
                                            required_nodes=[0, 2], root=0))
    expect_raises("E.neg-budget", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F()}, global_budget=-1))
    expect_raises("E.root-not-required", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                                            features={(0, 1): F()},
                                            required_nodes=[0, 1], root=2))

    # --- bug probes: these must raise *clear* ValueErrors ---
    expect_raises("E.hard-not-edge", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1)],
                                            features={(0, 1): F()},
                                            hard_critical_edges=[(1, 2)]))
    expect_raises("E.hard-missing-features", ValueError,
                  lambda: EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
                                            features={(0, 1): F()},
                                            hard_critical_edges=[(1, 2)]))

    # nodes omitted from degree_limits must be UNCONSTRAINED, not capped at 0
    b = EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
                          features={(0, 1): F(trust=1.0), (1, 2): F(trust=1.0)},
                          degree_limits={0: 2})
    z = np.zeros(len(b.variables))
    z[b.edge_var[(0, 1)]] = 1
    z[b.edge_var[(1, 2)]] = 1
    check("E.degree-default", not b.check_constraints(z),
          f"omitted degree_limits treated as 0: {b.check_constraints(z)}")

    # budget smaller than cheapest optional edge -> edge must be pruned
    b = EdgeSelectionQUBO(nodes=[0, 1], edges=[(0, 1)],
                          features={(0, 1): F(processing_cost=1.0)},
                          global_budget=0)
    check("E.budget-zero", True)  # construction must not crash
    Q, c, names = b.get_qubo()
    z_opt, _ = brute_force(Q, c) if len(names) <= 20 else (None, None)
    if z_opt is not None:
        check("E.budget-zero-prune", b.selected_edges(z_opt) == [],
              f"edge kept despite zero budget: {b.selected_edges(z_opt)}")

    # empty variable space must not crash SA
    b = EdgeSelectionQUBO(nodes=[0], edges=[], features={})
    Q, c, names = b.get_qubo()
    sa = SimulatedAnnealing(Q, c, restarts=2, seed=1)
    z, e = sa.run()
    check("E.empty", z.shape == (0,), f"empty problem: shape {z.shape}")

    # determinism
    b = EdgeSelectionQUBO(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)],
                          features={(0, 1): F(), (1, 2): F()},
                          degree_limits={0: 2, 1: 2, 2: 2})
    Q, c, _ = b.get_qubo()
    e1 = SimulatedAnnealing(Q, c, restarts=3, seed=7).run()[1]
    e2 = SimulatedAnnealing(Q, c, restarts=3, seed=7).run()[1]
    check("E.determinism", e1 == e2, f"same seed, different energies {e1} vs {e2}")


# ----------------------------------------------------------------------
# F. Scaling sanity
# ----------------------------------------------------------------------


def test_scaling() -> None:
    print("F. Scaling sanity")
    for n in (8, 12, 16):
        rng = np.random.default_rng(10 + n)
        inst = random_connected_instance(rng, n, extra_edge_p=0.5)
        cfg = fast_config(99, restarts=3)
        t0 = time.time()
        b = EdgeSelectionQUBO(config=cfg, **ctor_args(inst))
        z, _, warm = solve(b, inst["warm_edges"], cfg)
        dt = time.time() - t0
        try:
            final = b.build_feasible_solution(set(b.selected_edges(z)))
        except Exception:  # noqa: BLE001
            final = warm if warm is not None else z
        issues = b.check_constraints(final)
        mem_mb = b.Q.nbytes / 1e6
        print(f"  n={n:2d} edges={len(inst['edges']):3d} vars={len(b.variables):4d} "
              f"Q={mem_mb:6.1f}MB time={dt:5.1f}s feasible={not issues}")
        check(f"F.feas n{n}", not issues, f"infeasible: {issues}")
        check(f"F.time n{n}", dt < 60, f"too slow: {dt:.1f}s")


# ----------------------------------------------------------------------
# G. Targeted scenario tests (domain-meaningful situations)
# ----------------------------------------------------------------------


def solve_default(builder: EdgeSelectionQUBO, warm_edges):
    Q, c, _ = builder.get_qubo()
    warm = builder.build_feasible_solution(warm_edges)
    sa = SimulatedAnnealing(Q, c, restarts=10, seed=5, cooling_rate=0.95,
                            sweeps_per_temperature=3,
                            initial_states=[warm])
    z, _ = sa.run()
    try:
        z = builder.build_feasible_solution(set(builder.selected_edges(z)))
    except Exception:  # noqa: BLE001
        z = warm
    return builder.selected_edges(z)


def test_scenarios() -> None:
    print("G. Targeted domain scenarios")

    # G1: low-trust bridge is the ONLY path to a required node -> must be kept
    b = EdgeSelectionQUBO(
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2)],
        features={(0, 1): F(trust=0.95), (1, 2): F(trust=0.05)},
        required_nodes=[0, 1, 2], root=0,
        degree_limits={0: 2, 1: 2, 2: 2},
    )
    kept = solve_default(b, {(0, 1), (1, 2)})
    check("G1.bridge-kept", kept == [(0, 1), (1, 2)],
          f"connectivity lost to please utility: {kept}")

    # G2: redundant parallel paths -> the weak one is pruned
    weak = F(trust=0.05, link_quality=0.05, battery=0.05, proximity=0.05,
             freshness=0.05, task_criticality=0.05, processing_cost=0.5)
    b = EdgeSelectionQUBO(
        nodes=[0, 1, 2],
        edges=[(0, 1), (1, 2), (0, 2)],
        features={(0, 1): F(trust=0.9, link_quality=0.9),
                  (1, 2): F(trust=0.9, link_quality=0.9),
                  (0, 2): weak},
        required_nodes=[0, 1, 2], root=0,
        degree_limits={0: 2, 1: 2, 2: 2},
    )
    kept = solve_default(b, {(0, 1), (1, 2)})
    check("G2.weak-pruned", (0, 2) not in kept and set(kept) == {(0, 1), (1, 2)},
          f"kept: {kept}")

    # G3: stability hysteresis - borderline previously-kept edge survives,
    # borderline previously-absent edge does not (identical features)
    # utility 0.07 sits just below edge_penalty 0.08: only stability decides
    feat = F(trust=0.07, link_quality=0.07, battery=0.07, proximity=0.07,
             freshness=0.07, task_criticality=0.07, processing_cost=0.2)
    b = EdgeSelectionQUBO(
        nodes=[0, 1, 2, 3], edges=[(0, 1), (2, 3)],
        features={(0, 1): feat, (2, 3): feat},
        previous_decisions={(0, 1): 1, (2, 3): 0},
        degree_limits={i: 1 for i in range(4)},
    )
    kept = solve_default(b, {(0, 1)})
    check("G3.hysteresis", kept == [(0, 1)], f"kept: {kept}")

    # G4: star overload - center degree cap forces dropping the worst leaf
    edges = [(0, 1), (0, 2), (0, 3)]
    feats = {(0, 1): F(trust=0.9), (0, 2): F(trust=0.8), (0, 3): F(trust=0.1)}
    b = EdgeSelectionQUBO(nodes=[0, 1, 2, 3], edges=edges, features=feats,
                          degree_limits={0: 2, 1: 1, 2: 1, 3: 1})
    kept = solve_default(b, {(0, 1), (0, 2)})
    check("G4.star-overload", set(kept) == {(0, 1), (0, 2)}, f"kept: {kept}")

    # G5: all-identical features -> still feasible, deterministic
    n = 5
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    feats = {e: F() for e in edges}
    b = EdgeSelectionQUBO(nodes=list(range(n)), edges=edges, features=feats,
                          required_nodes=list(range(n)), root=0,
                          degree_limits={i: 4 for i in range(n)})
    kept1 = solve_default(b, {(0, 1), (1, 2), (2, 3), (3, 4)})
    kept2 = solve_default(b, {(0, 1), (1, 2), (2, 3), (3, 4)})
    check("G5.identical", kept1 == kept2 and len(kept1) >= 4,
          f"kept1={kept1} kept2={kept2}")

    # G6: tight budget picks cheap good edges over expensive good edges
    edges = [(0, 1), (0, 2), (1, 2)]
    feats = {(0, 1): F(trust=0.9, processing_cost=0.1),   # cost 1
             (0, 2): F(trust=0.9, processing_cost=0.1),   # cost 1
             (1, 2): F(trust=0.9, processing_cost=1.0)}   # cost 10
    b = EdgeSelectionQUBO(nodes=[0, 1, 2], edges=edges, features=feats,
                          global_budget=2,
                          degree_limits={0: 2, 1: 2, 2: 2})
    kept = solve_default(b, {(0, 1), (0, 2)})
    check("G6.budget", set(kept) == {(0, 1), (0, 2)}, f"kept: {kept}")

    # G7: chain under full connectivity requirement keeps everything
    edges = [(0, 1), (1, 2), (2, 3)]
    feats = {e: F(trust=0.01) for e in edges}
    b = EdgeSelectionQUBO(nodes=[0, 1, 2, 3], edges=edges, features=feats,
                          required_nodes=[0, 1, 2, 3], root=0,
                          degree_limits={i: 2 for i in range(4)})
    kept = solve_default(b, set(edges))
    check("G7.chain", set(kept) == set(edges), f"kept: {kept}")


# ----------------------------------------------------------------------
# H. SA quality vs settings (diagnostic, reported not asserted)
# ----------------------------------------------------------------------


def _dimod_qubo(Q: np.ndarray) -> dict:
    """dimod convention: E = sum_i Qii xi + sum_{i<j} Qij xi xj."""
    n = Q.shape[0]
    Qd = {}
    for a in range(n):
        if Q[a, a] != 0.0:
            Qd[(a, a)] = float(Q[a, a])
        for bb in range(a + 1, n):
            if Q[a, bb] != 0.0:
                Qd[(a, bb)] = float(2.0 * Q[a, bb])
    return Qd


def _energy_of_sample(b: EdgeSelectionQUBO, z: np.ndarray, warm: np.ndarray) -> float:
    try:
        z = b.build_feasible_solution(set(b.selected_edges(z)))
        return b.energy(z)
    except Exception:  # noqa: BLE001
        return b.energy(warm)


def test_qubo_solvers() -> None:
    """The main solver benchmark: the actual Q matrix against serious QUBO
    solvers, with exact ground truth from both enumeration and an exact
    linearized-QUBO solve (CBC)."""
    print("H. QUBO solver benchmark (on the actual Q matrix)")
    rng = np.random.default_rng(4)
    instances = []
    trial = 0
    while len(instances) < 25 and trial < 500:
        n = int(rng.integers(4, 8))
        inst = random_connected_instance(rng, n, tight=True)
        b = EdgeSelectionQUBO(config=fast_config(trial), **ctor_args(inst))
        trial += 1
        if len(b.optional_edges) > 16:
            continue
        e_opt = exact_edge_enumeration(b)
        instances.append((b, e_opt, inst["warm_edges"]))
    print(f"  {len(instances)} exactly-solved instances")

    from milp_baseline import solve_qubo_exact

    def run_local(b, warm_edges, seed, **kw):
        Q, c, _ = b.get_qubo()
        warm = b.build_feasible_solution(warm_edges)
        sa = SimulatedAnnealing(Q, c, seed=seed, sweeps_per_temperature=2,
                                initial_states=[warm], **kw)
        z, _ = sa.run()
        return _energy_of_sample(b, z, warm)

    def run_dimod(sampler, b, warm_edges, s, extra_kw=None):
        Q, c, _ = b.get_qubo()
        warm = b.build_feasible_solution(warm_edges)
        kw = {"num_reads": 100}
        kw.update(extra_kw or {})
        sample = sampler.sample_qubo(_dimod_qubo(Q), **kw).first.sample
        z = np.array([sample[k] for k in range(Q.shape[0])], dtype=float)
        return _energy_of_sample(b, z, warm)

    solvers = [
        ("bit-SA (local)", lambda b, w, s: run_local(
            b, w, s, restarts=20, cooling_rate=0.98)),
    ]
    if HAVE_NEAL:
        solvers.append(("neal SA", lambda b, w, s: run_dimod(
            neal.SimulatedAnnealingSampler(), b, w, s, {"seed": s})))
    try:
        import tabu
        solvers.append(("tabu search", lambda b, w, s: run_dimod(
            tabu.TabuSampler(), b, w, s,
            {"num_reads": 10, "timeout": 200, "seed": s})))
    except ImportError:
        pass
    try:
        import openjij as oj
        solvers.append(("SQA (openjij)", lambda b, w, s: run_dimod(
            oj.SQASampler(), b, w, s, {"num_reads": 25})))
    except ImportError:
        pass

    # exact ground truth: enumeration AND exact linearized QUBO must agree
    # (small Q only; the linearization grows quadratically with var count)
    checked = 0
    for i, (b, e_opt, _) in enumerate(instances):
        if checked >= 5:
            break
        Q, c, _ = b.get_qubo()
        if Q.shape[0] > 60:
            continue
        try:
            e_exact = solve_qubo_exact(Q, c, time_limit=120)
        except ValueError:
            continue
        checked += 1
        check(f"H.exact-agree {i}", abs(e_exact - e_opt) < 1e-7,
              f"linearized exact {e_exact} != enumeration {e_opt}")

    for label, fn in solvers:
        gaps, times = [], []
        for i, (b, e_opt, warm_edges) in enumerate(instances):
            t0 = time.time()
            e = fn(b, warm_edges, i)
            times.append(time.time() - t0)
            gaps.append(norm_gap(e, e_opt))
        g = np.array(gaps)
        print(f"  {label:16s}: exact={np.mean(g < 1e-9)*100:4.0f}%  "
              f"mean_gap={g.mean():.4f}  p90={np.percentile(g, 90):.4f}  "
              f"time={np.mean(times)*1e3:7.1f} ms")
    print(f"  {'exact (CBC)':16s}: exact=100%  (linearized QUBO, ground truth)")

    # projected annealer: production solver, kept OUT of the scientific
    # comparison; reported only as an engineering note with an exactness floor
    gaps = []
    for i, (b, e_opt, warm_edges) in enumerate(instances):
        sa = ProjectedEdgeAnnealing(b, initial_edge_sets=[warm_edges],
                                    restarts=3, seed=i)
        gaps.append(norm_gap(sa.run().energy, e_opt))
    gaps = np.array(gaps)
    print(f"  [production] projected annealing: exact="
          f"{np.mean(gaps < 1e-9)*100:.0f}% (not part of the comparison)")
    check("H.projected", np.mean(gaps < 1e-9) >= 0.9,
          f"projected SA exact rate too low: {np.mean(gaps < 1e-9)*100:.0f}%")

    # do random restarts alone (no warm start) reach feasibility?
    infeas = 0
    for i, (b, _, _) in enumerate(instances):
        Q, c, _ = b.get_qubo()
        sa = SimulatedAnnealing(Q, c, seed=100 + i, restarts=10,
                                cooling_rate=0.95, sweeps_per_temperature=2)
        z, _ = sa.run()
        if b.check_constraints(z):
            infeas += 1
    print(f"  random-start-only raw SA infeasible: {infeas}/{len(instances)}")


# ----------------------------------------------------------------------
# I. Trust-GNN layer
# ----------------------------------------------------------------------


def test_gnn() -> None:
    print("I. Trust-GNN layer")
    from simulation import RoverSimulation
    from trust_gnn import (
        GNNWeights,
        NodeFeatures,
        TaskRequirements,
        TrustGNN,
        TrustModel,
        assign_tasks,
    )

    gnn = TrustGNN()
    NF = lambda **kw: NodeFeatures(
        **{**dict(trust=0.5, battery=0.5, load=0.0, capability=0.5), **kw}
    )
    task = TaskRequirements()

    # I1: permutation equivariance - relabelling nodes permutes scores
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    feats = {0: NF(trust=0.9, battery=0.9), 1: NF(), 2: NF(trust=0.2, battery=0.2)}
    s1 = gnn.scores(nodes, feats, edges, task)
    mapping = {0: 10, 1: 11, 2: 12}
    nodes2 = [mapping[n] for n in nodes]
    edges2 = [(mapping[u], mapping[v]) for u, v in edges]
    feats2 = {mapping[n]: f for n, f in feats.items()}
    s2 = gnn.scores(nodes2, feats2, edges2, task)
    check("I1.equivariance",
          all(abs(s1[n] - s2[mapping[n]]) < 1e-12 for n in nodes),
          f"{s1} vs {s2}")

    # I2: ordering - better node scores higher
    check("I2.ordering", s1[0] > s1[1] > s1[2], f"scores {s1}")

    # I3: eligibility gates
    feats3 = {0: NF(trust=0.1), 1: NF(battery=0.05), 2: NF(load=0.95)}
    s3 = gnn.scores([0, 1, 2], feats3, [(0, 1)], task)
    check("I3.gates", s3[0] == 0.0 and s3[1] == 0.0 and s3[2] == 0.0,
          f"ineligible nodes scored: {s3}")

    # I4: neighborhood effect - node next to strong neighbors scores higher
    # than an identical isolated node
    feats4 = {0: NF(), 1: NF(trust=0.95, battery=0.95, capability=0.95), 2: NF()}
    s4 = gnn.scores([0, 1, 2], feats4, [(0, 1)], task)
    check("I4.neighborhood", s4[0] > s4[2],
          f"connected {s4[0]} not > isolated {s4[2]}")

    # I5: invalid inputs
    expect_raises("I5.missing-node", ValueError,
                  lambda: gnn.scores([0, 1], {0: NF()}, [], task))
    expect_raises("I5.bad-range", ValueError,
                  lambda: gnn.scores([0], {0: NF(trust=2.0)}, [], task))
    expect_raises("I5.bad-weights", ValueError,
                  lambda: TrustGNN(GNNWeights(out=np.zeros(3))))

    # I6: assignment respects capacity and never assigns zero-score nodes
    scores = {
        0: {"a": 0.9, "b": 0.8},
        1: {"a": 0.7, "b": 0.6},
        2: {"a": 0.0, "b": 0.0},
    }
    a = assign_tasks(scores)
    check("I6.capacity", len(set(a.values())) == len(a) and len(a) == 2,
          f"assignment {a}")
    check("I6.no-zero", 2 not in a.values(), f"zero-score node assigned: {a}")
    a2 = assign_tasks(scores, capacity={0: 2, 1: 0, 2: 1})
    check("I6.custom-cap", a2.get("a") == 0 and a2.get("b") == 0,
          f"capacity ignored: {a2}")

    # I7: trust model dynamics
    tm = TrustModel(alpha=0.5, initial=0.5)
    t0 = tm.update(0, 1.0)
    t1 = tm.update(0, 1.0)
    t2 = tm.update(0, 0.0)
    check("I7.monotone", 0.5 < t0 < t1 and t2 < t1,
          f"trust not monotone: {t0}, {t1}, {t2}")
    for _ in range(50):
        tf = tm.update(1, 1.0)
    check("I7.converge", abs(tf - 1.0) < 1e-3, f"no convergence: {tf}")
    for _ in range(20):
        tm.update(2, float(np.random.random()), float(np.random.random()))
    check("I7.bounds", 0.0 <= tm.get(2) <= 1.0, f"out of bounds: {tm.get(2)}")
    check("I7.initial", tm.get(99) == 0.5, "unknown node should get initial trust")
    expect_raises("I7.bad-outcome", ValueError, lambda: tm.update(0, 1.5))

    # I8: integration - trust-aware GNN assignment beats random assignment
    gnn_rates, rand_rates = [], []
    for seed in range(8):
        gnn_rates.append(
            RoverSimulation(steps=20, seed=100 + seed, faults=[],
                            n_rovers=10, comm_range=6.0, tasks_per_cycle=3,
                            reliability_drift=0.02).run(verbose=False)["success_rate"])
        rand_rates.append(
            RoverSimulation(steps=20, seed=100 + seed, faults=[],
                            n_rovers=10, comm_range=6.0, tasks_per_cycle=3,
                            random_assignment=True,
                            reliability_drift=0.02).run(verbose=False)["success_rate"])
    gnn_mean = float(np.mean(gnn_rates))
    rand_mean = float(np.mean(rand_rates))
    margin = gnn_mean - rand_mean
    print(f"  success rate: GNN {gnn_mean:.2%} vs random {rand_mean:.2%} "
          f"(margin {margin:+.2%}) "
          f"(per-seed GNN: {[f'{r:.0%}' for r in gnn_rates]}, "
          f"rand: {[f'{r:.0%}' for r in rand_rates]})")
    check("I8.beats-random", margin > 0.03,
          f"GNN {gnn_mean:.2%} vs random {rand_mean:.2%}: margin too small")

    # I9: non-canonical edge in `edges` must still find its features
    # (regression test for the canonical-lookup bug)
    adj = gnn._adjacency_weights(
        [0, 1], [(1, 0)],  # non-canonical order
        {(0, 1): EdgeFeatures(0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)},
    )
    check("I9.canonical", abs(adj[0][0][1] - 0.4 * 0.5) < 1e-12,
          f"edge weight fell back to 1.0: {adj}")

    # I10: batch scoring == per-task scoring
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    feats = {n: NF(trust=0.4 + 0.1 * n, battery=0.8, load=0.1,
                   capability=0.5) for n in nodes}
    efeats = {e: EdgeFeatures(0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.3) for e in edges}
    tasks = {"a": TaskRequirements(), "b": TaskRequirements(criticality=0.9)}
    batch = gnn.scores_for_tasks(nodes, feats, edges, tasks, efeats)
    for tid, t in tasks.items():
        single = gnn.scores(nodes, feats, edges, t, efeats)
        check(f"I10.batch[{tid}]",
              all(abs(batch[tid][n] - single[n]) < 1e-12 for n in nodes),
              "batch scoring diverges from per-task scoring")

    # I11: readout training learns the trust signal
    from trust_gnn import train_out_weights
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(400):
        h = rng.random(4)
        y = 1.0 if h[0] + 0.1 * rng.normal() > 0.5 else 0.0
        samples.append((h, y))
    trained = train_out_weights(samples)
    check("I11.learned-trust",
          trained.out[0] == trained.out.max() and trained.out[0] > 1.0,
          f"trust weight not dominant after training: {trained.out}")
    p = [float(1 / (1 + np.exp(-(trained.out @ h + trained.out_bias))))
         for h, _ in samples]
    ys = [y for _, y in samples]
    auc_pairs = sum(
        (pi > pj) + 0.5 * (pi == pj)
        for pi, yi in zip(p, ys) if yi == 1.0
        for pj, yj in zip(p, ys) if yj == 0.0
    ) / (sum(ys) * (len(ys) - sum(ys)))
    check("I11.auc", auc_pairs > 0.9, f"AUC {auc_pairs:.3f}")
    expect_raises("I11.one-class", ValueError,
                  lambda: train_out_weights([(np.zeros(4), 1.0)] * 5))


# ----------------------------------------------------------------------
# K. Simulation harness (baselines, faults, freshness, lifecycle)
# ----------------------------------------------------------------------


def test_simulation_harness() -> None:
    print("K. Simulation harness")
    from simulation import (EdgeLifecycleTracker, FaultEvent, RoverSimulation,
                            spearman)

    # K1: all pruners + random baseline run end-to-end
    for pruner in ("full", "threshold", "topk", "qubo"):
        s = RoverSimulation(steps=5, seed=3, pruner=pruner).run(verbose=False)
        check(f"K1.{pruner}",
              0.0 <= s["success_rate"] <= 1.0 and "timings_ms" in s,
              f"bad summary: {s}")
    s = RoverSimulation(steps=5, seed=3, random_assignment=True).run(verbose=False)
    check("K1.random", 0.0 <= s["success_rate"] <= 1.0, "random arm failed")

    # K2: fault injection produces recovery measurements
    s = RoverSimulation(steps=15, seed=5).run(verbose=False)
    check("K2.recovery", len(s["recovery"]) > 0,
          f"no recovery data: {s['recovery']}")

    # K3: freshness decays with age under total packet loss
    sim = RoverSimulation(steps=8, seed=1, packet_loss=1.0)
    sim._move(0)
    _, f0 = sim._graph(0)
    sim._move(5)
    _, f5 = sim._graph(5)
    common = set(f0) & set(f5)
    if common:
        e = sorted(common)[0]
        check("K3.freshness", f5[e].freshness < f0[e].freshness,
              f"freshness did not decay: {f0[e].freshness} -> {f5[e].freshness}")

    # K4: spearman correctness
    check("K4.spearman-pos",
          abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9, "")
    check("K4.spearman-neg",
          abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9, "")
    check("K4.spearman-ties",
          -1.0 <= spearman([1, 1, 2, 2], [1, 2, 3, 4]) <= 1.0, "ties broke")

    # K5: edge lifecycle transitions
    t = EdgeLifecycleTracker()
    e = (0, 1)
    t.update([e], [])          # pruned once -> SUSPECT
    s1 = t.states[e]
    t.update([e], [])          # -> PROBATION
    s2 = t.states[e]
    t.update([e], [])
    t.update([e], [])          # -> REMOVED
    s3 = t.states[e]
    t.update([e], [e])         # kept -> ACTIVE
    s4 = t.states[e]
    check("K5.lifecycle",
          (s1, s2, s3, s4) == ("SUSPECT", "PROBATION", "REMOVED", "ACTIVE"),
          f"{s1},{s2},{s3},{s4}")

    # K6: scripted custom faults apply without crashing
    s = RoverSimulation(
        steps=8, seed=2,
        faults=[FaultEvent(cycle=2, kind="battery_drain", node=None,
                           magnitude=0.5, duration=2)],
    ).run(verbose=False)
    check("K6.custom-fault", 0.0 <= s["success_rate"] <= 1.0, "")


# ----------------------------------------------------------------------
# J. Optimality-gap benchmark
# ----------------------------------------------------------------------


def soft_lower_bound(b: EdgeSelectionQUBO) -> float:
    """
    Certified lower bound on H*: the per-edge unconstrained soft minimum.
    Valid because penalties are >= 0, so min over ALL subsets of the soft
    objective lower-bounds the constrained optimum.
    """
    cfg = b.cfg
    lb = 0.0
    for e in b.optional_edges:
        u = utility_of(cfg, b.features[e])
        prev = b.previous.get(e, 0)
        a0 = cfg.stability_penalty * (0 - prev) ** 2
        a1 = -u + cfg.edge_penalty + cfg.stability_penalty * (1 - prev) ** 2
        lb += min(a0, a1)
    return lb


def norm_gap(h_sa: float, h_star: float) -> float:
    """(H_SA - H*) / max(1, |H*|)"""
    return (h_sa - h_star) / max(1.0, abs(h_star))


def _anneal_energy(b: EdgeSelectionQUBO, warm, seed: int) -> float:
    # fixed production settings: ProjectedEdgeAnnealing defaults
    return ProjectedEdgeAnnealing(
        b, initial_edge_sets=[warm], seed=seed
    ).run().energy


def test_benchmark_gap() -> None:
    print("J. Optimality-gap benchmark")
    rng = np.random.default_rng(11)

    # --- small instances: exact H* via enumeration ---
    small_gaps: List[float] = []
    trial = 0
    while len(small_gaps) < 25 and trial < 400:
        n = int(rng.integers(4, 8))
        inst = random_connected_instance(rng, n, tight=bool(rng.random() < 0.5))
        b = EdgeSelectionQUBO(config=QUBOConfig(), **ctor_args(inst))
        trial += 1
        if not 4 <= len(b.optional_edges) <= 16:
            continue
        h_star = exact_edge_enumeration(b)
        res = ProjectedEdgeAnnealing(
            b, initial_edge_sets=[inst["warm_edges"]], seed=5000 + trial
        ).run()
        check(f"J.soft-eq s{trial}",
              abs(res.energy - b.soft_energy(res.kept_edges)) < 1e-9,
              "soft_energy != QUBO energy on feasible result")
        small_gaps.append(norm_gap(res.energy, h_star))
    small_gaps_arr = np.array(small_gaps)
    print(f"  small (exact H*, n={len(small_gaps)}): "
          f"exact={np.mean(small_gaps_arr < 1e-9)*100:.0f}% "
          f"mean_gap={small_gaps_arr.mean():.5f} max={small_gaps_arr.max():.5f}")
    check("J.small-exact", np.mean(small_gaps_arr < 1e-9) >= 0.9,
          f"exact rate {np.mean(small_gaps_arr < 1e-9)*100:.0f}%")

    # --- larger instances: enumeration impossible ---
    # H* estimated by best-known solution (BKS) over multiple annealer seeds;
    # additionally a certified upper bound on the true gap is reported using
    # the soft lower bound: (H_SA - LB)/max(1,|LB|) >= true gap >= BKS gap.
    bks_gaps: List[float] = []
    cert_bounds: List[float] = []
    max_vars = 0
    trial = 0
    while len(bks_gaps) < 20 and trial < 400:
        n = int(rng.integers(8, 15))
        inst = random_connected_instance(
            rng, n, extra_edge_p=float(rng.uniform(0.4, 0.6)), tight=True
        )
        b = EdgeSelectionQUBO(config=QUBOConfig(), **ctor_args(inst))
        trial += 1
        if len(b.optional_edges) <= 20:
            continue
        max_vars = max(max_vars, len(b.variables))
        h_sa = _anneal_energy(b, inst["warm_edges"], seed=9000 + trial)
        bks = h_sa
        for s in range(3):
            bks = min(bks, _anneal_energy(b, inst["warm_edges"],
                                          seed=9000 + trial + 997 * (s + 1)))
        lb = soft_lower_bound(b)
        bks_gaps.append(norm_gap(h_sa, bks))
        cert_bounds.append(norm_gap(h_sa, lb))
    bks_arr = np.array(bks_gaps)
    cert_arr = np.array(cert_bounds)
    print(f"  large (BKS proxy, n={len(bks_gaps)}, up to {max_vars} vars): "
          f"mean_gap={bks_arr.mean():.5f} p90={np.percentile(bks_arr, 90):.5f} "
          f"max={bks_arr.max():.5f}")
    print(f"  large certified bound (vs lower bound): mean={cert_arr.mean():.4f} "
          f"p90={np.percentile(cert_arr, 90):.4f} max={cert_arr.max():.4f}")
    check("J.large-mean", bks_arr.mean() < 0.10,
          f"mean BKS gap {bks_arr.mean():.4f}")
    check("J.large-p90", np.percentile(bks_arr, 90) < 0.25,
          f"p90 BKS gap {np.percentile(bks_arr, 90):.4f}")


# ----------------------------------------------------------------------
# L. Review-2 regression tests
# ----------------------------------------------------------------------


def test_review2_regressions() -> None:
    print("L. Review-2 regressions")

    # L1 (E1): connectivity-constrained instance with NO warm start must
    # still solve — spanning-tree-seeded random starts
    f05 = F(processing_cost=0.2)
    b = EdgeSelectionQUBO(
        nodes=[0, 1, 2, 3], edges=[(0, 1), (1, 2), (2, 3)],
        features={(0, 1): f05, (1, 2): f05, (2, 3): f05},
        required_nodes=[0, 1, 2, 3], root=0,
        degree_limits={i: 2 for i in range(4)},
    )
    res = ProjectedEdgeAnnealing(b, restarts=4, seed=0).run()
    check("L1.chain-bootstrap",
          set(res.kept_edges) == {(0, 1), (1, 2), (2, 3)}
          and not res.violated_constraints,
          f"kept={res.kept_edges} violations={res.violated_constraints}")

    # L2 (E2): greedy repair restores feasibility
    from scale_runner import make_sparse_instance, repair_to_feasible
    from simulation import edge_utility
    rng = np.random.default_rng(21)
    _, edges, feats, prev, dlim, budget = make_sparse_instance(rng, 10)
    b = EdgeSelectionQUBO(nodes=list(range(10)), edges=edges, features=feats,
                          previous_decisions=prev, degree_limits=dlim,
                          global_budget=budget)
    greedy = set(sorted(edges, key=lambda e: -edge_utility(feats[e]))[:17])
    z = np.zeros(len(b.variables))
    for e in b.optional_edges:
        if e in greedy:
            z[b.edge_var[e]] = 1
    check("L2.greedy-was-infeasible", bool(b.check_constraints(z)),
          "test premise broken: greedy should violate constraints here")
    repaired, ok = repair_to_feasible(b, greedy, feats)
    z = np.zeros(len(b.variables))
    for e in b.optional_edges:
        if e in repaired:
            z[b.edge_var[e]] = 1
    check("L2.repair-feasible", ok and not b.check_constraints(z),
          f"repair failed: {b.check_constraints(z)}")

    # L3: criticality scales the probability, not the logit — direction must
    # be sign-independent (higher criticality never INCREASES a score)
    from trust_gnn import NodeFeatures, TaskRequirements, TrustGNN
    gnn = TrustGNN()
    feats = {
        0: NodeFeatures(trust=0.2, battery=0.9, load=0.0, capability=0.5),
        1: NodeFeatures(trust=0.9, battery=0.9, load=0.0, capability=0.5),
    }
    hi = TaskRequirements(criticality=0.9)
    lo = TaskRequirements(criticality=0.2)
    for n in (0, 1):
        s_hi = gnn.scores([0, 1], feats, [], hi)[n]
        s_lo = gnn.scores([0, 1], feats, [], lo)[n]
        check(f"L3.crit-sign n{n}", s_hi <= s_lo + 1e-12,
              f"critical task raised score: {s_lo} -> {s_hi}")

    # L4: global reliability_drop (node=None) actually applies
    from simulation import FaultEvent, RoverSimulation
    sim = RoverSimulation(steps=4, seed=2, faults=[
        FaultEvent(cycle=1, kind="reliability_drop", node=None,
                   magnitude=0.9, duration=2)])
    sim._move(1)
    base = sim.base_reliability
    dropped = all(sim.state[n].reliability <= max(0.0, base[n] - 0.9) + 1e-9
                  for n in sim.nodes)
    check("L4.global-fault", dropped, "node=None fault did not apply globally")

    # L5: timing sanity — gnn timing is a non-negative median, gnn_full
    # skippable via flag
    s = RoverSimulation(steps=3, seed=2, measure_full_gnn=False).run(verbose=False)
    check("L5.gnn-full-gated",
          np.isnan(s["timings_ms"]["gnn_full"]) or s["timings_ms"]["gnn_full"] == 0.0,
          f"gnn_full measured despite flag: {s['timings_ms']}")


# ----------------------------------------------------------------------
# M. MILP ground truth (strong classical baseline, true H*)
# ----------------------------------------------------------------------


def test_milp_ground_truth() -> None:
    print("M. MILP ground truth (CBC) vs projected annealing")
    try:
        from milp_baseline import solve_milp  # noqa: F401
    except ImportError:
        print("  pulp not installed; skipping MILP comparison")
        return
    from milp_baseline import solve_milp

    rng = np.random.default_rng(17)
    gaps, t_proj, t_milp, evals = [], [], [], []
    trial = 0
    while len(gaps) < 20 and trial < 300:
        n = int(rng.integers(4, 10))
        inst = random_connected_instance(
            rng, n,
            with_budget=True, with_required=True, tight=True,
        )
        b = EdgeSelectionQUBO(config=QUBOConfig(), **ctor_args(inst))
        trial += 1
        if not 4 <= len(b.optional_edges) <= 40:
            continue
        t0 = time.time()
        try:
            e_star, kept_star = solve_milp(
                nodes=inst["nodes"], edges=inst["edges"],
                features=inst["features"],
                previous_decisions=inst["previous_decisions"],
                hard_critical_edges=inst["hard_critical_edges"],
                required_nodes=inst["required_nodes"] or None,
                root=inst["root"],
                degree_limits=inst["degree_limits"],
                global_budget=inst["global_budget"],
            )
        except Exception:  # noqa: BLE001
            continue
        t_milp.append(time.time() - t0)
        # MILP objective == soft energy (same linearization); verify
        check(f"M.obj-eq t{trial}",
              abs(e_star - b.soft_energy(kept_star)) < 1e-6,
              f"MILP objective {e_star} != soft energy {b.soft_energy(kept_star)}")
        t0 = time.time()
        ann = ProjectedEdgeAnnealing(b, initial_edge_sets=[inst["warm_edges"]],
                                     seed=trial)
        res = ann.run()
        t_proj.append(time.time() - t0)
        evals.append(ann.energy_evals)
        gaps.append(norm_gap(res.energy, e_star))
    g = np.array(gaps)
    print(f"  true optimality gap vs MILP: exact={np.mean(g < 1e-9)*100:.0f}% "
          f"mean={g.mean():.5f} p90={np.percentile(g, 90):.5f} "
          f"max={g.max():.5f}")
    print(f"  time: projected {np.mean(t_proj)*1e3:.0f} ms, "
          f"MILP {np.mean(t_milp)*1e3:.0f} ms; "
          f"annealer energy evals mean {np.mean(evals):.0f}")
    check("M.exact-rate", np.mean(g < 1e-9) >= 0.8,
          f"projected annealer exact rate vs MILP {np.mean(g < 1e-9)*100:.0f}%")


# ----------------------------------------------------------------------
# N. Sensitivity diagnostics (weights, penalties, annealer params)
# ----------------------------------------------------------------------


def test_sensitivity() -> None:
    print("N. Sensitivity diagnostics")
    rng = np.random.default_rng(23)
    base_instances = []
    trial = 0
    while len(base_instances) < 15 and trial < 300:
        n = int(rng.integers(4, 7))
        inst = random_connected_instance(rng, n, tight=True)
        b = EdgeSelectionQUBO(config=QUBOConfig(), **ctor_args(inst))
        trial += 1
        if 4 <= len(b.optional_edges) <= 16:
            base_instances.append(inst)

    def exact_rate(config_fn) -> float:
        hits = 0
        for inst in base_instances:
            b = EdgeSelectionQUBO(config=config_fn(), **ctor_args(inst))
            e_opt = exact_edge_enumeration(b)  # same-config ground truth
            res = ProjectedEdgeAnnealing(
                b, initial_edge_sets=[inst["warm_edges"]], restarts=2,
                seed=7, cooling_rate=0.995, steps_per_temperature=2).run()
            if res.energy < e_opt + 1e-9:
                hits += 1
        return hits / len(base_instances)

    r_base = exact_rate(QUBOConfig)
    print(f"  baseline exact rate (fixed settings): {r_base:.0%}")
    for label, cfg in [
        ("weights x1.5 trust", lambda: QUBOConfig(
            w_trust=0.375, w_link=0.125, w_battery=0.10, w_proximity=0.10,
            w_freshness=0.15, w_task=0.15)),
        ("weights uniform", lambda: QUBOConfig(
            w_trust=1/6, w_link=1/6, w_battery=1/6, w_proximity=1/6,
            w_freshness=1/6, w_task=1/6)),
        ("penalties 20/100", lambda: QUBOConfig(
            degree_penalty=20.0, budget_penalty=20.0,
            flow_penalty=100.0, flow_capacity_penalty=100.0)),
        ("penalties 500/2000", lambda: QUBOConfig(
            degree_penalty=500.0, budget_penalty=500.0,
            flow_penalty=2000.0, flow_capacity_penalty=2000.0)),
        ("edge_penalty 0.3", lambda: QUBOConfig(edge_penalty=0.3)),
        ("stability 0.6", lambda: QUBOConfig(stability_penalty=0.6)),
    ]:
        r = exact_rate(cfg)
        print(f"  {label:22s}: exact {r:.0%}")
        check(f"N.{label}", r >= 0.7,
              f"solver fragile to {label}: exact {r:.0%}")


if __name__ == "__main__":
    test_invariants()
    test_brute_force()
    test_analytic()
    test_scenario_matrix()
    test_edge_cases()
    test_scaling()
    test_scenarios()
    test_qubo_solvers()
    test_gnn()
    test_simulation_harness()
    test_review2_regressions()
    test_benchmark_gap()
    test_milp_ground_truth()
    test_sensitivity()
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(" -", f)
    raise SystemExit(1 if FAIL else 0)
