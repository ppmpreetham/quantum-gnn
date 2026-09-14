"""End-to-end dynamic simulation and benchmark harness for the rover pipeline.

Each cycle:
    1. Rovers move (proximity changes), links are observed subject to packet
       loss (freshness decays for unobserved links), batteries drain.
    2. The dynamic candidate graph is built with per-edge features.
    3. A pruner selects the working graph: QUBO (production), or one of the
       baselines (full / threshold / top-K) for comparison.
    4. TrustGNN scores nodes on the working graph (single pass per cycle);
       tasks are assigned; scripted faults may be active.
    5. Task outcomes are sampled from ground-truth reliability; TrustModel
       updates trust, which feeds the next cycle's features.

Metrics: task success rate, per-stage wall-clock latency, trust-vs-ground-truth
Spearman correlation, topology churn, edge lifecycle states, and fault
recovery time.

Run:  python3 simulation.py            (single run, per-cycle log)
      python3 simulation.py benchmark  (pruner comparison table)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from math import exp
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from main import Edge, EdgeFeatures, EdgeSelectionQUBO, ProjectedEdgeAnnealing
from trust_gnn import (
    Node,
    NodeFeatures,
    TaskRequirements,
    TrustGNN,
    TrustModel,
    assign_tasks,
    train_out_weights,
)

EDGE_WEIGHTS = dict(w_trust=0.25, w_link=0.25, w_battery=0.10,
                    w_proximity=0.10, w_freshness=0.15, w_task=0.15)


def edge_utility(f: EdgeFeatures) -> float:
    """Same utility the QUBO uses (shared so baselines are comparable)."""
    return (
        EDGE_WEIGHTS["w_trust"] * f.trust
        + EDGE_WEIGHTS["w_link"] * f.link_quality
        + EDGE_WEIGHTS["w_battery"] * f.battery
        + EDGE_WEIGHTS["w_proximity"] * f.proximity
        + EDGE_WEIGHTS["w_freshness"] * f.freshness
        + EDGE_WEIGHTS["w_task"] * f.task_criticality
    )


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan")

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        r = np.empty(n, dtype=float)
        r[order] = np.arange(n, dtype=float)
        # average ranks for ties
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((dx @ dx) * (dy @ dy)))
    return float(dx @ dy / denom) if denom > 0 else float("nan")


def _timed_median_ms(fn, repeats: int = 25) -> float:
    """Median wall-clock of fn() in ms; robust at sub-ms scale."""
    fn()  # warmup
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1e3


def auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Rank AUC of scores against binary labels."""
    pos = [s for s, y in zip(scores, labels) if y == 1.0]
    neg = [s for s, y in zip(scores, labels) if y == 0.0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def paired_bootstrap_ci(diffs: Sequence[float], n_boot: int = 10000,
                        alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean of paired differences."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def permutation_pvalue(diffs: Sequence[float], n_perm: int = 20000,
                       seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for paired differences."""
    d = np.asarray(diffs, dtype=float)
    if len(d) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    obs = d.mean()
    return float((np.abs(null) >= abs(obs)).mean())


@dataclass
class RoverState:
    position: np.ndarray
    battery: float          # 0..1
    load: float             # 0..1
    capability: float       # 0..1
    reliability: float      # hidden ground truth: P(task success)


@dataclass(frozen=True)
class FaultEvent:
    """Scripted disturbance, active during cycles [cycle, cycle+duration)."""

    cycle: int
    kind: str          # reliability_drop | battery_drain | packet_loss_spike | position_jump
    node: Optional[int] = None   # None = global
    magnitude: float = 0.5
    duration: int = 4


class EdgeLifecycleTracker:
    """
    ACTIVE -> SUSPECT -> PROBATION -> REMOVED based on consecutive prunes.
    Candidate edges are rebuilt from telemetry every cycle, so pruning can
    never permanently blind the trust model; this tracker makes the edge
    health observable and gives PROBATION edges a freshness floor so they get
    a fair re-evaluation instead of rotting at the bottom of the utility rank.
    """

    PRUNE_TO_PROBATION = 2
    PRUNE_TO_REMOVED = 4

    def __init__(self) -> None:
        self.misses: Dict[Edge, int] = {}
        self.states: Dict[Edge, str] = {}

    def update(self, candidates: Iterable[Edge], kept: Iterable[Edge]) -> None:
        kept_set = set(kept)
        for e in candidates:
            if e in kept_set:
                self.misses[e] = 0
                self.states[e] = "ACTIVE"
            else:
                self.misses[e] = self.misses.get(e, 0) + 1
                m = self.misses[e]
                self.states[e] = (
                    "SUSPECT" if m < self.PRUNE_TO_PROBATION
                    else "PROBATION" if m < self.PRUNE_TO_REMOVED
                    else "REMOVED"
                )

    def probation_boost(self, edge: Edge) -> float:
        """Freshness floor for PROBATION edges (re-evaluation chance)."""
        return 0.4 if self.states.get(edge) == "PROBATION" else 0.0

    def counts(self) -> Dict[str, int]:
        out = {"ACTIVE": 0, "SUSPECT": 0, "PROBATION": 0, "REMOVED": 0}
        for s in self.states.values():
            out[s] += 1
        return out


class RoverSimulation:
    def __init__(
        self,
        n_rovers: int = 6,
        tasks_per_cycle: int = 2,
        comm_range: float = 4.0,
        steps: int = 10,
        seed: int = 7,
        random_assignment: bool = False,
        assigner: str = "gnn",   # gnn | trust_only | nograph | random
        pruner: str = "qubo",   # qubo | full | threshold | topk
        threshold_tau: float = 0.4,
        topk_fraction: float = 0.5,
        load_increment: float = 0.15,
        load_decay: float = 0.5,
        min_trust: float = 0.25,
        compute_need: float = 0.3,
        exploration: float = 0.15,
        freshness_tau: float = 3.0,
        packet_loss: float = 0.05,
        faults: Optional[List[FaultEvent]] = None,
        gnn_weights=None,
        collect_training: bool = False,
        measure_full_gnn: bool = True,
        solve_interval: int = 1,   # solve QUBO every k cycles (cloud amortization)
        n_mission_critical: int = 3,
        reliability_drift: float = 0.0,   # per-cycle std of hidden reliability walk
        qubo_budget: int = 15,   # global edge-cost budget for the QUBO pruner
    ) -> None:
        if pruner not in ("qubo", "full", "threshold", "topk"):
            raise ValueError(f"unknown pruner {pruner}")
        if assigner not in ("gnn", "trust_only", "nograph", "random"):
            raise ValueError(f"unknown assigner {assigner}")
        if solve_interval <= 0:
            raise ValueError("solve_interval must be positive")
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = list(range(n_rovers))
        self.tasks_per_cycle = tasks_per_cycle
        self.comm_range = comm_range
        self.steps = steps
        self.random_assignment = random_assignment or assigner == "random"
        self.assigner = assigner
        self.pruner = pruner
        self.threshold_tau = threshold_tau
        self.topk_fraction = topk_fraction
        self.load_increment = load_increment
        self.load_decay = load_decay
        self.min_trust = min_trust
        self.compute_need = compute_need
        self.exploration = exploration
        self.freshness_tau = freshness_tau
        self.base_packet_loss = packet_loss
        self.collect_training = collect_training
        self.measure_full_gnn = measure_full_gnn
        self.solve_interval = solve_interval
        self.reliability_drift = reliability_drift
        self.qubo_budget = qubo_budget

        self.state: Dict[Node, RoverState] = {}
        # spatially correlated reliability field: nearby rovers share
        # terrain/interference conditions, so neighbor observations carry
        # information about unobserved nodes (this is what a GNN exploits)
        field_centers = self.rng.uniform(0, 10, size=(3, 2))
        field_amps = self.rng.uniform(-0.3, 0.3, size=3)
        for n in self.nodes:
            pos = self.rng.uniform(0, 10, size=2)
            field = float(sum(
                field_amps[k] * np.exp(-np.sum((pos - field_centers[k]) ** 2)
                                       / (2 * 2.5 ** 2))
                for k in range(3)))
            self.state[n] = RoverState(
                position=pos,
                battery=float(self.rng.uniform(0.5, 1.0)),
                load=float(self.rng.uniform(0.0, 0.5)),
                capability=float(self.rng.uniform(0.4, 1.0)),
                reliability=float(np.clip(
                    self.rng.uniform(0.35, 0.75) + field, 0.05, 0.98)),
            )
        self.base_reliability = {n: self.state[n].reliability for n in self.nodes}
        self.trust = TrustModel(alpha=0.5, initial=0.5)
        self.gnn = TrustGNN(weights=gnn_weights)
        self.previous_decisions: Dict[Edge, int] = {}
        self.lifecycle = EdgeLifecycleTracker()
        self.faults = faults if faults is not None else self._default_faults()

        # fixed mission-critical set (root + random others), chosen once:
        # NOT "whatever happens to be reachable this cycle"
        self.root = self.nodes[0]
        n_crit = min(n_mission_critical, n_rovers)
        self.mission_critical = sorted(
            {self.root}
            | set(self.rng.choice(self.nodes[1:], size=n_crit - 1,
                                  replace=False).tolist())
        )
        self.last_kept: List[Edge] = []

        # link observation memory for real freshness
        self.link_quality: Dict[Edge, float] = {}
        self.last_observed: Dict[Edge, int] = {}
        # criticality memory from last cycle's assignments
        self.node_criticality: Dict[Node, float] = {n: 0.3 for n in self.nodes}

        # metrics
        self.timings: Dict[str, List[float]] = {
            k: [] for k in ("build", "solve", "gnn", "assign", "gnn_full")
        }
        self.success_history: List[float] = []
        self.trust_spearman: List[float] = []
        self.training_samples: List[Tuple[np.ndarray, float, bool]] = []
        self.prune_mode_counts: Dict[str, int] = {}
        self.solves_performed = 0
        self.edges_history: List[Tuple[int, int]] = []  # (candidates, kept)

    def _default_faults(self) -> List[FaultEvent]:
        c = max(2, self.steps // 3)
        # worst case for the system: the most reliable rover degrades
        best = max(self.nodes, key=lambda n: self.base_reliability[n])
        return [
            FaultEvent(cycle=c, kind="reliability_drop", node=best,
                       magnitude=0.6, duration=4),
            FaultEvent(cycle=c + 1, kind="packet_loss_spike", node=None,
                       magnitude=0.4, duration=3),
        ]

    def _active_faults(self, cycle: int) -> List[FaultEvent]:
        return [f for f in self.faults if f.cycle <= cycle < f.cycle + f.duration]

    # --------------------------------------------------------------
    # Environment dynamics
    # --------------------------------------------------------------

    def _move(self, cycle: int) -> None:
        packet_loss = self.base_packet_loss
        for f in self._active_faults(cycle):
            if f.kind == "packet_loss_spike":
                packet_loss = min(1.0, packet_loss + f.magnitude)
        for n in self.nodes:
            s = self.state[n]
            if self.reliability_drift > 0:
                # hidden reliability itself wanders: the trust model must
                # track a moving target, not a stationary Bernoulli mean
                self.base_reliability[n] = float(np.clip(
                    self.base_reliability[n]
                    + self.rng.normal(0, self.reliability_drift), 0.05, 0.98))
            s.reliability = self.base_reliability[n]
            for f in self._active_faults(cycle):
                if f.kind == "reliability_drop" and (f.node is None or f.node == n):
                    s.reliability = max(
                        0.0, self.base_reliability[n] - f.magnitude
                    )
                elif (f.kind == "intermittent"
                      and (f.node is None or f.node == n)
                      and (cycle - f.cycle) % 2 == 0):
                    # toggles every other cycle while active
                    s.reliability = max(
                        0.0, self.base_reliability[n] - f.magnitude
                    )
                elif f.kind == "battery_drain" and (f.node is None or f.node == n):
                    s.battery = max(0.0, s.battery - f.magnitude)
                elif f.kind == "position_jump" and (f.node is None or f.node == n):
                    s.position = np.clip(
                        s.position + self.rng.normal(0, f.magnitude * 5, size=2),
                        0, 10,
                    )
            s.position = np.clip(s.position + self.rng.normal(0, 0.8, size=2), 0, 10)
            # power: assigned rovers drain net; idle rovers recharge (solar/dock)
            s.battery = float(np.clip(
                s.battery - self.rng.uniform(0.01, 0.05)
                + (0.0 if s.load > 0.4 else 0.03), 0, 1))
            # tasks complete over time: load partially frees each cycle
            s.load = float(np.clip(s.load * self.load_decay
                                   + self.rng.normal(0, 0.05), 0, 0.95))
        self._packet_loss = packet_loss

    def _graph(self, cycle: int) -> Tuple[List[Edge], Dict[Edge, EdgeFeatures]]:
        edges: List[Edge] = []
        features: Dict[Edge, EdgeFeatures] = {}
        for i in self.nodes:
            for j in self.nodes:
                if j <= i:
                    continue
                si, sj = self.state[i], self.state[j]
                dist = float(np.linalg.norm(si.position - sj.position))
                if dist > self.comm_range:
                    continue
                edge = (i, j)
                observed = self.rng.random() >= self._packet_loss
                if observed or edge not in self.link_quality:
                    self.link_quality[edge] = float(np.clip(
                        1.0 - dist / self.comm_range
                        + self.rng.normal(0, 0.05), 0, 1))
                    self.last_observed[edge] = cycle
                freshness = exp(-(cycle - self.last_observed[edge])
                                / self.freshness_tau)
                freshness = max(freshness, self.lifecycle.probation_boost(edge))
                proximity = 1.0 - dist / self.comm_range
                # edge cost has physical meaning: transmission time grows
                # with distance, endpoint compute queues add processing time
                cost = 0.1 + 0.5 * (1.0 - proximity) \
                    + 0.3 * (si.load + sj.load) / 2.0
                edges.append(edge)
                features[edge] = EdgeFeatures(
                    trust=min(self.trust.get(i), self.trust.get(j)),
                    link_quality=self.link_quality[edge],
                    battery=min(si.battery, sj.battery),
                    proximity=proximity,
                    freshness=freshness,
                    task_criticality=max(self.node_criticality[i],
                                         self.node_criticality[j]),
                    processing_cost=float(np.clip(cost, 0.0, 1.0)),
                )
        return edges, features

    # --------------------------------------------------------------
    # Pruners (QUBO production path + baselines)
    # --------------------------------------------------------------

    def _prune(self, edges: List[Edge], features: Dict[Edge, EdgeFeatures],
               cycle: int) -> Tuple[List[Edge], float, float]:
        """Return (kept_edges, build_seconds, solve_seconds)."""
        if self.pruner == "full":
            return list(edges), 0.0, 0.0
        if self.pruner == "threshold":
            kept = [e for e in edges
                    if edge_utility(features[e]) >= self.threshold_tau]
            return kept, 0.0, 0.0
        if self.pruner == "topk":
            k = max(1, int(round(self.topk_fraction * len(edges))))
            ranked = sorted(edges, key=lambda e: -edge_utility(features[e]))
            return sorted(ranked[:k]), 0.0, 0.0

        # cloud amortization: solve every solve_interval cycles, reuse the
        # stale decision (intersected with the live graph) in between
        if cycle % self.solve_interval != 0 and self.last_kept:
            self.prune_mode_counts["stale-reuse"] = \
                self.prune_mode_counts.get("stale-reuse", 0) + 1
            return [e for e in self.last_kept if e in set(edges)], 0.0, 0.0

        # qubo
        t0 = time.perf_counter()
        # required = the FIXED mission-critical set that is currently
        # reachable from the root; unreachable mission nodes are counted,
        # not silently redefined away
        adjacency = {n: [] for n in self.nodes}
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        seen = {self.root}
        stack = [self.root]
        while stack:
            for v in adjacency[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        required = sorted(n for n in self.mission_critical if n in seen)
        lost = len(self.mission_critical) - len(required)
        if lost:
            self.prune_mode_counts["mission-unreachable"] = \
                self.prune_mode_counts.get("mission-unreachable", 0) + 1

        current_edges = set(edges)
        warm = {e for e, v in self.previous_decisions.items()
                if v == 1} & current_edges
        if not warm:
            warm = current_edges

        # Progressive relaxation: connectivity + budget can be jointly
        # infeasible under tight degree caps; drop budget, then connectivity.
        # Every fallback is counted so the guarantee's coverage is measurable.
        attempts = [
            ("full", dict(required_nodes=required or None,
                          root=self.root if required else None,
                          global_budget=self.qubo_budget)),
            ("no-budget", dict(required_nodes=required or None,
                               root=self.root if required else None,
                               global_budget=None)),
            ("budget-only", dict(global_budget=self.qubo_budget)),
            ("unconstrained", dict()),
        ]
        result = None
        mode = None
        for mode, extra in attempts:
            try:
                builder = EdgeSelectionQUBO(
                    nodes=self.nodes,
                    edges=edges,
                    features=features,
                    previous_decisions=self.previous_decisions,
                    degree_limits={n: 3 for n in self.nodes},
                    **extra,
                )
            except ValueError:
                continue
            t1 = time.perf_counter()
            try:
                result = ProjectedEdgeAnnealing(
                    builder, initial_edge_sets=[warm], restarts=2, seed=cycle,
                    steps_per_temperature=2, cooling_rate=0.995,
                ).run()
            except ValueError:
                continue
            t2 = time.perf_counter()
            break
        if result is None:  # degree caps alone are infeasible: keep everything
            self.prune_mode_counts["keep-all-fallback"] = \
                self.prune_mode_counts.get("keep-all-fallback", 0) + 1
            return list(edges), 0.0, 0.0
        self.solves_performed += 1
        self.prune_mode_counts[mode] = self.prune_mode_counts.get(mode, 0) + 1
        self.last_kept = result.kept_edges
        return result.kept_edges, t1 - t0, t2 - t1

    # --------------------------------------------------------------
    # One pipeline cycle
    # --------------------------------------------------------------

    def _score_nodes(self, scores, tasks, node_features, path_factor):
        """Ablation arms for assignment: what does the GNN actually add?"""
        if self.assigner == "gnn":
            raw = {n: {tid: s[n] for tid, s in scores.items()}
                   for n in self.nodes}
        else:
            raw = {}
            for n in self.nodes:
                f = node_features[n]
                for tid, t in tasks.items():
                    if (f.trust < t.min_trust or f.battery < t.min_battery
                            or (1.0 - f.load) < t.compute_need):
                        s = 0.0
                    elif self.assigner == "trust_only":
                        # the brutal baseline: pick the most trusted rover
                        s = f.trust
                    else:  # nograph: logistic readout on raw features, no graph
                        z = (float(self.gnn.w.out @ f.vector())
                             + self.gnn.w.out_bias)
                        s = float(1.0 / (1.0 + np.exp(-z)))
                    raw.setdefault(n, {})[tid] = s
        # path quality matters but must not dominate: partial multiplier
        # keeps the root from absorbing every task merely by being the root
        raw = {n: {tid: (s * (0.5 + 0.5 * path_factor[n])
                         if path_factor[n] > 0.0 else 0.0)
                   for tid, s in per.items()}
               for n, per in raw.items()}
        # deadlock fallback: if every candidate is trust-gated, the mission
        # still runs; relax the trust gate, keep the resource gates
        for tid, t in tasks.items():
            if all(raw[n][tid] == 0.0 for n in self.nodes):
                for n in self.nodes:
                    f = node_features[n]
                    if (f.battery >= t.min_battery
                            and (1.0 - f.load) >= t.compute_need
                            and path_factor[n] > 0.0):
                        raw[n][tid] = f.trust * (0.5 + 0.5 * path_factor[n])
        return raw

    def _path_factors(self, kept, features) -> Dict[Node, float]:
        """Weakest link quality on the path from each node to the root in
        the kept graph; 0.0 if unreachable. Task data must cross this path."""
        adjacency = {n: [] for n in self.nodes}
        for e in kept:
            u, v = e
            w = features[e].link_quality
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))
        factor = {n: 0.0 for n in self.nodes}
        factor[self.root] = 1.0
        # BFS maximizing the bottleneck link (widest path)
        stack = [self.root]
        while stack:
            u = stack.pop()
            for v, w in adjacency[u]:
                cand = min(factor[u], w)
                if cand > factor[v]:
                    factor[v] = cand
                    stack.append(v)
        return factor

    def step(self, cycle: int) -> dict:
        self._move(cycle)
        edges, features = self._graph(cycle)
        if not edges:
            return {"cycle": cycle, "edges": 0, "note": "graph empty"}

        kept, t_build, t_solve = self._prune(edges, features, cycle)
        self.previous_decisions = {e: 1 for e in kept}
        self.lifecycle.update(edges, kept)

        node_features = {
            n: NodeFeatures(
                trust=self.trust.get(n),
                battery=self.state[n].battery,
                load=self.state[n].load,
                capability=self.state[n].capability,
            )
            for n in self.nodes
        }
        tasks = {
            f"task-{cycle}-{k}": TaskRequirements(
                min_trust=self.min_trust,
                compute_need=self.compute_need,
                criticality=float(self.rng.uniform(0.2, 0.9)),
            )
            for k in range(self.tasks_per_cycle)
        }

        # --- scoring and assignment ---
        # a node whose data cannot reach the root over the kept graph is not
        # a valid assignment target, regardless of trust
        path_factor = self._path_factors(kept, features)
        hidden = self.gnn.hidden_states(self.nodes, node_features, kept, features)
        scores = {
            tid: {n: self.gnn.score_from_hidden(n, hidden[n], node_features[n], t)
                  for n in self.nodes}
            for tid, t in tasks.items()
        }
        t_gnn = _timed_median_ms(
            lambda: self.gnn.hidden_states(self.nodes, node_features, kept,
                                           features))
        if self.measure_full_gnn:
            t_gnn_full = _timed_median_ms(
                lambda: self.gnn.hidden_states(self.nodes, node_features,
                                               edges, features))
        else:
            t_gnn_full = float("nan")

        if self.random_assignment:
            eligible = [
                n for n in self.nodes
                if self.state[n].battery >= 0.15
                and (1.0 - self.state[n].load) >= self.compute_need
                and path_factor[n] > 0.0
            ]
            chosen = self.rng.choice(
                eligible, size=min(len(tasks), len(eligible)), replace=False
            )
            assignment = {tid: int(n) for tid, n in zip(tasks, chosen)}
        else:
            node_scores = self._score_nodes(scores, tasks, node_features,
                                            path_factor)
            assignment = assign_tasks(node_scores)
            t_assign = _timed_median_ms(lambda: assign_tasks(node_scores))
            # epsilon-exploration: occasionally probe another eligible node;
            # exploration assignments are tagged because they form the
            # random-policy (unbiased) subset of the training data
            explored = set()
            for tid in list(assignment):
                if self.rng.random() >= self.exploration:
                    continue
                used = set(assignment.values()) - {assignment[tid]}
                candidates = [
                    n for n in self.nodes
                    if n not in used
                    and self.state[n].battery >= 0.15
                    and (1.0 - self.state[n].load) >= self.compute_need
                    and self.trust.get(n) >= self.min_trust
                    and path_factor[n] > 0.0
                ]
                if candidates:
                    assignment[tid] = int(
                        candidates[self.rng.integers(len(candidates))]
                    )
                    explored.add(tid)
        if self.random_assignment:
            t_assign = float("nan")
            explored = set()

        # --- outcomes: causality runs THROUGH the kept graph ---
        # execution success (rover did the work) drives trust; mission
        # success additionally requires the result to cross the kept
        # topology to the root (delivery over the weakest path link)
        successes = 0
        assigned_nodes = set(assignment.values())
        self.node_criticality = {n: 0.3 for n in self.nodes}
        for tid, n in assignment.items():
            exec_ok = (
                self.rng.random() < self.state[n].reliability
                and self.state[n].battery >= 0.15
            )
            delivered = n == self.root or self.rng.random() < path_factor[n]
            ok = exec_ok and delivered
            successes += int(ok)
            self.trust.update(n, 1.0 if exec_ok else 0.0)
            self.state[n].load = min(0.95, self.state[n].load
                                     + self.load_increment)
            self.node_criticality[n] = tasks[tid].criticality
            if self.collect_training:
                self.training_samples.append(
                    (hidden[n].copy(), float(ok), tid in explored))
        for n in set(self.nodes) - assigned_nodes:
            # no observation: trust ages toward the prior (slow rehabilitation)
            self.trust.relax(n)

        rate = successes / max(1, len(tasks))
        self.success_history.append(rate)
        self.edges_history.append((len(edges), len(kept)))
        self.trust_spearman.append(spearman(
            [self.trust.get(n) for n in self.nodes],
            [self.state[n].reliability for n in self.nodes],
        ))
        for k, v in (("build", t_build * 1e3), ("solve", t_solve * 1e3),
                     ("gnn", t_gnn), ("assign", t_assign),
                     ("gnn_full", t_gnn_full)):
            self.timings[k].append(v)

        return {
            "cycle": cycle,
            "edges": len(edges),
            "kept": len(kept),
            "assignment": assignment,
            "success": successes,
            "tasks": len(tasks),
            "mean_trust": float(np.mean([self.trust.get(n) for n in self.nodes])),
            "lifecycle": self.lifecycle.counts(),
            "faults": [f.kind for f in self._active_faults(cycle)],
        }

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    def recovery_times(self) -> Dict[str, Optional[int]]:
        """Cycles until per-cycle success returns to the pre-fault baseline."""
        out: Dict[str, Optional[int]] = {}
        for f in self.faults:
            pre = self.success_history[max(0, f.cycle - 3):f.cycle]
            if not pre or f.cycle >= len(self.success_history):
                continue
            baseline = float(np.mean(pre))
            recovery = None
            for c in range(f.cycle + f.duration, len(self.success_history)):
                if self.success_history[c] >= baseline:
                    recovery = c - f.cycle
                    break
            out[f"{f.kind}@{f.cycle}"] = recovery
        return out

    def summary(self) -> dict:
        total_ok = sum(int(r * self.tasks_per_cycle + 1e-9)
                       for r in self.success_history)
        total = len(self.success_history) * self.tasks_per_cycle
        timings = {}
        for k, v in self.timings.items():
            finite = [x for x in v if np.isfinite(x)]
            timings[k] = float(np.mean(finite)) if finite else float("nan")
        # trust quality as a classifier of "above-median reliability"
        rel = [self.state[n].reliability for n in self.nodes]
        med = float(np.median(rel))
        trust_auc = auc([self.trust.get(n) for n in self.nodes],
                        [1.0 if r > med else 0.0 for r in rel])
        kept_ratio = (float(np.mean([k / c for c, k in self.edges_history
                                     if c > 0]))
                      if self.edges_history else float("nan"))
        return {
            "success_rate": total_ok / max(1, total),
            "timings_ms": timings,
            "trust_spearman": (float(np.nanmean(self.trust_spearman))
                               if self.trust_spearman else float("nan")),
            "trust_auc": trust_auc,
            "kept_ratio": kept_ratio,
            "prune_modes": dict(self.prune_mode_counts),
            "solves_per_cycle": (self.solves_performed
                                 / max(1, len(self.success_history))),
            "recovery": self.recovery_times(),
            "lifecycle": self.lifecycle.counts(),
            "training_samples": len(self.training_samples),
        }

    def run(self, verbose: bool = True) -> dict:
        prev_kept: Optional[set] = None
        for cycle in range(self.steps):
            stats = self.step(cycle)
            kept_set = set(self.previous_decisions)
            churn = (
                0.0
                if prev_kept is None
                else len(kept_set ^ prev_kept) / max(1, len(kept_set | prev_kept))
            )
            prev_kept = kept_set
            if verbose:
                fault = f" FAULT:{','.join(stats['faults'])}" if stats.get("faults") else ""
                print(
                    f"cycle {stats['cycle']:2d} | edges {stats.get('edges', 0):2d} "
                    f"kept {stats.get('kept', 0):2d} churn {churn:4.2f} | "
                    f"assigned {stats.get('assignment', {})} | "
                    f"trust {stats.get('mean_trust', float('nan')):.3f}"
                    f"{fault}"
                )
        s = self.summary()
        if verbose:
            print(f"\ntask success rate : {s['success_rate']:.2%}")
            print(f"trust Spearman    : {s['trust_spearman']:+.3f}")
            print(f"recovery (cycles) : {s['recovery']}")
            print(f"edge lifecycle    : {s['lifecycle']}")
            print("latency (ms/cycle): "
                  + "  ".join(f"{k}={v:.2f}" for k, v in s["timings_ms"].items()))
        return s


def run_benchmark(steps: int = 20, seeds: int = 10,
                  reliability_drift: float = 0.02) -> None:
    """
    Identical scenario through each pruner; the paper's headline table.
    Uses a denser fleet (10 rovers) — at 6 rovers the graph is so sparse
    that pruning rarely binds, which the table should honestly show.
    Reports paired statistics (bootstrap CI + permutation p) for QUBO vs the
    best baseline, replacing bare "indistinguishable" claims.
    """
    configs = dict(n_rovers=10, comm_range=6.0, tasks_per_cycle=3,
                   reliability_drift=reliability_drift)
    pruners = ("full", "threshold", "topk", "qubo")
    rows: Dict[str, List[dict]] = {p: [] for p in pruners}
    for s in range(seeds):
        for pruner in pruners:
            sim = RoverSimulation(steps=steps, seed=s, pruner=pruner, **configs)
            rows[pruner].append(sim.run(verbose=False))
    print(f"pruner comparison: {configs['n_rovers']} rovers, "
          f"{steps} cycles x {seeds} seeds, drift={reliability_drift}")
    print(f"{'pruner':10s} {'success':>8s} {'solve ms':>9s} {'gnn ms':>8s} "
          f"{'kept%':>7s} {'spearman':>9s} {'auc':>6s}")
    for pruner, runs in rows.items():
        succ = float(np.mean([r["success_rate"] for r in runs]))
        solve = float(np.mean([r["timings_ms"]["solve"] for r in runs]))
        gnn = float(np.mean([r["timings_ms"]["gnn"] for r in runs]))
        kept = float(np.nanmean([r["kept_ratio"] for r in runs]))
        sp = float(np.nanmean([r["trust_spearman"] for r in runs]))
        a = float(np.nanmean([r["trust_auc"] for r in runs]))
        print(f"{pruner:10s} {succ:8.2%} {solve:9.2f} {gnn:8.2f} "
              f"{kept*100:6.0f}% {sp:9.3f} {a:6.3f}")
    # paired statistics: QUBO vs best baseline on per-seed success
    best_baseline = max((p for p in pruners if p != "qubo"),
                        key=lambda p: np.mean([r["success_rate"]
                                               for r in rows[p]]))
    diffs = [q["success_rate"] - b["success_rate"]
             for q, b in zip(rows["qubo"], rows[best_baseline])]
    mean, lo, hi = paired_bootstrap_ci(diffs)
    p = permutation_pvalue(diffs)
    print(f"qubo vs {best_baseline} (paired): mean diff {mean:+.2%}, "
          f"95% CI [{lo:+.2%}, {hi:+.2%}], permutation p = {p:.3f}")


def run_ablation(steps: int = 20, seeds: int = 8,
                 reliability_drift: float = 0.02) -> None:
    """Assignment ablations: isolating what the GNN contributes."""
    arms = ("random", "trust_only", "nograph", "gnn")
    rows: Dict[str, List[float]] = {a: [] for a in arms}
    for s in range(seeds):
        for arm in arms:
            sim = RoverSimulation(steps=steps, seed=100 + s, faults=[],
                                  assigner=arm, n_rovers=10, comm_range=6.0,
                                  tasks_per_cycle=3,
                                  reliability_drift=reliability_drift)
            rows[arm].append(sim.run(verbose=False)["success_rate"])
    print(f"assignment ablation: {steps} cycles x {seeds} seeds, "
          f"drift={reliability_drift}")
    for arm in arms:
        rates = rows[arm]
        print(f"  {arm:10s} {np.mean(rates):.2%} "
              f"(per-seed {[f'{r:.0%}' for r in rates]})")
    for arm in ("trust_only", "nograph"):
        diffs = [g - o for g, o in zip(rows["gnn"], rows[arm])]
        mean, lo, hi = paired_bootstrap_ci(diffs)
        p = permutation_pvalue(diffs)
        print(f"  gnn vs {arm}: mean {mean:+.2%}, 95% CI [{lo:+.2%}, {hi:+.2%}], "
              f"p = {p:.3f}")


def run_staleness(steps: int = 20, seeds: int = 6) -> None:
    """Cloud amortization: solve every k cycles, reuse stale decisions."""
    print(f"staleness experiment: {steps} cycles x {seeds} seeds, 10 rovers")
    print(f"{'interval':>9s} {'success':>9s} {'solves/cyc':>11s} {'solve ms':>9s}")
    for interval in (1, 2, 4, 8):
        rates, spc, solve = [], [], []
        for s in range(seeds):
            sim = RoverSimulation(steps=steps, seed=s, n_rovers=10,
                                  comm_range=6.0, tasks_per_cycle=3,
                                  solve_interval=interval,
                                  reliability_drift=0.02)
            summ = sim.run(verbose=False)
            rates.append(summ["success_rate"])
            spc.append(summ["solves_per_cycle"])
            solve.append(summ["timings_ms"]["solve"])
        print(f"{interval:>9d} {np.mean(rates):>9.2%} {np.mean(spc):>11.2f} "
              f"{np.mean(solve):>9.1f}")


def train_and_evaluate(steps: int = 20, train_seeds: int = 5,
                       eval_seeds: int = 3) -> None:
    """
    Cloud-tier GNN training: collect outcome samples across several training
    scenarios, fit the readout, evaluate on HELD-OUT scenarios.
    Training uses the exploration subset (assignments made by the random
    exploration policy), which is free of the model's own selection bias.
    """
    on_policy: List[Tuple[np.ndarray, float]] = []
    off_policy: List[Tuple[np.ndarray, float]] = []
    for s in range(train_seeds):
        # high exploration during collection builds a random-policy sample
        # set (free of the model's own selection bias)
        collector = RoverSimulation(steps=steps, seed=s, collect_training=True,
                                    exploration=0.5)
        collector.run(verbose=False)
        for h, y, was_exploration in collector.training_samples:
            (off_policy if was_exploration else on_policy).append((h, y))
    train_set = off_policy if len(off_policy) >= 30 else on_policy
    used = "exploration (random-policy, unbiased)" \
        if train_set is off_policy else "all samples (on-policy, biased)"
    weights = train_out_weights(train_set)
    print(f"trained readout on {len(train_set)} samples [{used}]: "
          f"out={np.round(weights.out, 3)} bias={weights.out_bias:+.3f}")
    fixed_rates, trained_rates = [], []
    eval_cfg = dict(n_rovers=10, comm_range=6.0, tasks_per_cycle=3,
                    reliability_drift=0.02)
    for s in range(eval_seeds):
        eval_seed = 1000 + s
        fixed_rates.append(
            RoverSimulation(steps=steps, seed=eval_seed, **eval_cfg)
            .run(verbose=False)["success_rate"])
        trained_rates.append(
            RoverSimulation(steps=steps, seed=eval_seed,
                            gnn_weights=weights, **eval_cfg)
            .run(verbose=False)["success_rate"])
    diffs = [t - f for t, f in zip(trained_rates, fixed_rates)]
    mean, lo, hi = paired_bootstrap_ci(diffs)
    p = permutation_pvalue(diffs)
    print(f"held-out success: fixed {np.mean(fixed_rates):.2%} "
          f"-> trained {np.mean(trained_rates):.2%} "
          f"(diff {mean:+.2%}, 95% CI [{lo:+.2%}, {hi:+.2%}], p = {p:.3f})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == "ablate":
        run_ablation()
    elif len(sys.argv) > 1 and sys.argv[1] == "staleness":
        run_staleness()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_evaluate()
    else:
        RoverSimulation().run()
