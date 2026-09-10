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

    PRUNE_TO_SUSPECT = 1
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
    ) -> None:
        if pruner not in ("qubo", "full", "threshold", "topk"):
            raise ValueError(f"unknown pruner {pruner}")
        self.rng = np.random.default_rng(seed)
        self.nodes: List[Node] = list(range(n_rovers))
        self.tasks_per_cycle = tasks_per_cycle
        self.comm_range = comm_range
        self.steps = steps
        self.random_assignment = random_assignment
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

        self.state: Dict[Node, RoverState] = {}
        for n in self.nodes:
            self.state[n] = RoverState(
                position=self.rng.uniform(0, 10, size=2),
                battery=float(self.rng.uniform(0.5, 1.0)),
                load=float(self.rng.uniform(0.0, 0.5)),
                capability=float(self.rng.uniform(0.4, 1.0)),
                reliability=float(self.rng.uniform(0.2, 0.95)),
            )
        self.base_reliability = {n: self.state[n].reliability for n in self.nodes}
        self.trust = TrustModel(alpha=0.5, initial=0.5)
        self.gnn = TrustGNN(weights=gnn_weights)
        self.previous_decisions: Dict[Edge, int] = {}
        self.lifecycle = EdgeLifecycleTracker()
        self.faults = faults if faults is not None else self._default_faults()

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
        self.training_samples: List[Tuple[np.ndarray, float]] = []

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
            for f in self._active_faults(cycle):
                if f.kind == "reliability_drop" and f.node == n:
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
            if cycle >= max((f.cycle + f.duration for f in self.faults
                             if f.kind == "reliability_drop" and f.node == n),
                            default=-1):
                s.reliability = self.base_reliability[n]
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
                edges.append(edge)
                features[edge] = EdgeFeatures(
                    trust=min(self.trust.get(i), self.trust.get(j)),
                    link_quality=self.link_quality[edge],
                    battery=min(si.battery, sj.battery),
                    proximity=proximity,
                    freshness=freshness,
                    task_criticality=max(self.node_criticality[i],
                                         self.node_criticality[j]),
                    processing_cost=float(self.rng.uniform(0.1, 0.6)),
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

        # qubo
        t0 = time.perf_counter()
        # connectivity: keep the root's connected component connected
        root = self.nodes[0]
        adjacency = {n: [] for n in self.nodes}
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)
        seen = {root}
        stack = [root]
        while stack:
            for v in adjacency[stack.pop()]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        required = sorted(seen) if len(seen) >= 2 else []

        current_edges = set(edges)
        warm = {e for e, v in self.previous_decisions.items()
                if v == 1} & current_edges
        if not warm:
            warm = current_edges

        # Progressive relaxation: connectivity + budget can be jointly
        # infeasible under tight degree caps; drop budget, then connectivity.
        attempts = [
            dict(required_nodes=required or None, root=root if required else None,
                 global_budget=15),
            dict(required_nodes=required or None, root=root if required else None,
                 global_budget=None),
            dict(global_budget=15),
            dict(),
        ]
        result = None
        for extra in attempts:
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
            return list(edges), 0.0, 0.0
        return result.kept_edges, t1 - t0, t2 - t1

    # --------------------------------------------------------------
    # One pipeline cycle
    # --------------------------------------------------------------

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

        # --- GNN inference: ONE pass per cycle on the working graph ---
        t0 = time.perf_counter()
        hidden = self.gnn.hidden_states(self.nodes, node_features, kept, features)
        scores = {
            tid: {n: self.gnn.score_from_hidden(n, hidden[n], node_features[n], t)
                  for n in self.nodes}
            for tid, t in tasks.items()
        }
        t_gnn = time.perf_counter() - t0
        # comparison: what GNN inference would cost on the FULL graph
        t0 = time.perf_counter()
        self.gnn.hidden_states(self.nodes, node_features, edges, features)
        t_gnn_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.random_assignment:
            eligible = [
                n for n in self.nodes
                if self.state[n].battery >= 0.15
                and (1.0 - self.state[n].load) >= self.compute_need
            ]
            chosen = self.rng.choice(
                eligible, size=min(len(tasks), len(eligible)), replace=False
            )
            assignment = {tid: int(n) for tid, n in zip(tasks, chosen)}
        else:
            # assign_tasks expects {node: {task: score}}
            assignment = assign_tasks(
                {n: {tid: s[n] for tid, s in scores.items()}
                 for n in self.nodes}
            )
            # epsilon-exploration: occasionally probe another eligible node
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
                ]
                if candidates:
                    assignment[tid] = int(
                        candidates[self.rng.integers(len(candidates))]
                    )
        t_assign = time.perf_counter() - t0

        # --- outcomes, trust update, training data ---
        successes = 0
        assigned_nodes = set(assignment.values())
        self.node_criticality = {n: 0.3 for n in self.nodes}
        for tid, n in assignment.items():
            ok = (
                self.rng.random() < self.state[n].reliability
                and self.state[n].battery >= 0.15
            )
            successes += int(ok)
            self.trust.update(n, 1.0 if ok else 0.0)
            self.state[n].load = min(0.95, self.state[n].load
                                     + self.load_increment)
            self.node_criticality[n] = tasks[tid].criticality
            if self.collect_training:
                self.training_samples.append((hidden[n].copy(), float(ok)))
        for n in set(self.nodes) - assigned_nodes:
            # no observation: trust ages toward the prior (slow rehabilitation)
            self.trust.relax(n)

        rate = successes / max(1, len(tasks))
        self.success_history.append(rate)
        self.trust_spearman.append(spearman(
            [self.trust.get(n) for n in self.nodes],
            [self.state[n].reliability for n in self.nodes],
        ))
        for k, v in (("build", t_build), ("solve", t_solve),
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
        timings = {k: float(np.mean(v)) * 1e3 if v else 0.0
                   for k, v in self.timings.items()}  # ms
        return {
            "success_rate": total_ok / max(1, total),
            "timings_ms": timings,
            "trust_spearman": (float(np.nanmean(self.trust_spearman))
                               if self.trust_spearman else float("nan")),
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


def run_benchmark(steps: int = 20, seeds: int = 3) -> None:
    """
    Identical scenario through each pruner; the paper's headline table.
    Uses a denser fleet (10 rovers) — at 6 rovers the graph is so sparse
    that pruning rarely binds, which the table should honestly show.
    """
    configs = dict(n_rovers=10, comm_range=6.0, tasks_per_cycle=3)
    rows: Dict[str, List[dict]] = {p: [] for p in ("full", "threshold", "topk", "qubo")}
    for s in range(seeds):
        for pruner in rows:
            sim = RoverSimulation(steps=steps, seed=s, pruner=pruner, **configs)
            rows[pruner].append(sim.run(verbose=False))
    print(f"pruner comparison: {configs['n_rovers']} rovers, "
          f"{steps} cycles x {seeds} seeds")
    print(f"{'pruner':10s} {'success':>8s} {'solve ms':>9s} {'gnn ms':>8s} "
          f"{'gnn/full ms':>11s} {'spearman':>9s}")
    for pruner, runs in rows.items():
        succ = float(np.mean([r["success_rate"] for r in runs]))
        solve = float(np.mean([r["timings_ms"]["solve"] for r in runs]))
        gnn = float(np.mean([r["timings_ms"]["gnn"] for r in runs]))
        gnn_full = float(np.mean([r["timings_ms"]["gnn_full"] for r in runs]))
        sp = float(np.nanmean([r["trust_spearman"] for r in runs]))
        print(f"{pruner:10s} {succ:8.2%} {solve:9.2f} {gnn:8.2f} "
              f"{gnn_full:11.2f} {sp:9.3f}")


def train_and_evaluate(steps: int = 20, train_seeds: int = 5,
                       eval_seeds: int = 3) -> None:
    """
    Cloud-tier GNN training: collect outcome samples across several training
    scenarios, fit the readout, evaluate on HELD-OUT scenarios.
    """
    samples: List[Tuple[np.ndarray, float]] = []
    for s in range(train_seeds):
        collector = RoverSimulation(steps=steps, seed=s, collect_training=True)
        collector.run(verbose=False)
        samples.extend(collector.training_samples)
    weights = train_out_weights(samples)
    print(f"trained readout on {len(samples)} samples: "
          f"out={np.round(weights.out, 3)} bias={weights.out_bias:+.3f}")
    fixed_rates, trained_rates = [], []
    for s in range(eval_seeds):
        eval_seed = 1000 + s
        fixed_rates.append(
            RoverSimulation(steps=steps, seed=eval_seed).run(verbose=False)["success_rate"])
        trained_rates.append(
            RoverSimulation(steps=steps, seed=eval_seed,
                            gnn_weights=weights).run(verbose=False)["success_rate"])
    print(f"held-out success: fixed {np.mean(fixed_rates):.2%} "
          f"-> trained {np.mean(trained_rates):.2%} "
          f"(per-seed fixed {[f'{r:.0%}' for r in fixed_rates]}, "
          f"trained {[f'{r:.0%}' for r in trained_rates]})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_evaluate()
    else:
        RoverSimulation().run()
