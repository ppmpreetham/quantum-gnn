"""Planner sessions: the repository's pipeline driven by live rover positions.

The browser owns kinematics only. Everything the console reports about
pruning, assignment, outcomes and trust comes from RoverSimulation, with the
live rover positions substituted for its random walk.
"""
from __future__ import annotations

import json
import math
import threading
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from main import Edge, EdgeSelectionQUBO, QUBOConfig, SimulatedAnnealing
from simulation import FaultEvent, RoverSimulation, edge_utility

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "shared" / "fleet-config.json").read_text()
)
PARAM_SPECS = {spec["key"]: spec for spec in CONFIG["params"]}
SESSION_KEYS = [k for k, s in PARAM_SPECS.items() if s["scope"] == "session"]
PLANNER_KEYS = [k for k, s in PARAM_SPECS.items() if s["scope"] != "client"]
ARENA_RADIUS = CONFIG["arena"]["radius"]
ASSIGNMENT_BASELINES = ("trust_only", "random", "nograph")
# kind -> (magnitude, duration in cycles, needs a target rover)
FAULTS = {
    "reliability_drop": (0.6, 4, True),
    "battery_drain": (0.25, 2, True),
    "packet_loss_spike": (0.4, 3, False),
}
OPTIONAL_SOLVERS = {"neal": "neal", "tabu": "tabu", "sqa": "openjij"}


def available_solvers() -> Dict[str, bool]:
    return {"bitsa": True,
            **{name: find_spec(module) is not None
               for name, module in OPTIONAL_SOLVERS.items()}}


def _number(value, spec) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{spec['key']} must be a finite number")
    if not spec["min"] <= value <= spec["max"]:
        raise ValueError(f"{spec['key']} must be in [{spec['min']}, {spec['max']}]")
    if spec.get("integer") and int(value) != value:
        raise ValueError(f"{spec['key']} must be an integer")
    return int(value) if spec.get("integer") else float(value)


def validate_params(raw) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    return {key: _number(raw.get(key), PARAM_SPECS[key]) for key in PLANNER_KEYS}


def validate_positions(raw, count: int) -> List[Tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError("Position count must equal fleet size")
    limit = ARENA_RADIUS + 1e-6
    points = []
    for point in raw:
        if (not isinstance(point, list) or len(point) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in point)
                or math.hypot(*point) > limit):
            raise ValueError("Positions must be finite [x, z] points inside the arena")
        points.append((float(point[0]), float(point[1])))
    return points


def validate_faults(raw, count: int) -> List[Tuple[str, Optional[int]]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 8:
        raise ValueError("faults must be a list of at most 8 entries")
    faults = []
    for fault in raw:
        kind = fault.get("kind") if isinstance(fault, dict) else None
        if kind not in FAULTS:
            raise ValueError("Unknown fault kind")
        rover = fault.get("rover")
        if FAULTS[kind][2]:
            if isinstance(rover, bool) or not isinstance(rover, int) or not 0 <= rover < count:
                raise ValueError("Fault target must be a rover index")
        else:
            rover = None
        faults.append((kind, rover))
    return faults


def validate_comparison(raw) -> str:
    known = {c["id"] for c in CONFIG["comparisons"]}
    if raw not in known:
        raise ValueError("Unknown comparison")
    return raw


def min_spanning_cost(nodes, edges: List[Edge], features, root) -> Optional[int]:
    """Cheapest spanning tree of the root's component in QUBO budget units.

    A lower bound on any budget that also demands connectivity: degree caps
    can only raise it. None when the component is a single rover.
    """
    scale = QUBOConfig().cost_scale
    weighted = sorted(edges, key=lambda e: features[e].processing_cost)
    parent = {n: n for n in nodes}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    used = []
    for edge in weighted:
        a, b = find(edge[0]), find(edge[1])
        if a != b:
            parent[a] = b
            used.append(edge)
    tree = [e for e in used if find(e[0]) == find(root)]
    return sum(int(round(features[e].processing_cost * scale)) for e in tree) if tree else None


class LiveSimulation(RoverSimulation):
    """RoverSimulation whose rovers sit where the browser says they are.

    A follower copies its leader's pruned topology so that a paired
    assignment comparison differs in assignment policy only.
    """

    def __init__(self, params: dict, *, assigner: str = "gnn",
                 leader: Optional["LiveSimulation"] = None) -> None:
        n = params["roverCount"]
        super().__init__(
            n_rovers=n, seed=params["seed"], assigner=assigner, pruner="qubo",
            tasks_per_cycle=min(params["tasksPerCycle"], n),
            n_mission_critical=n, faults=[], measure_full_gnn=False,
        )
        self.leader = leader
        self.cycle = 0
        self.live_positions: List[Tuple[float, float]] = []
        self.edges: List[Edge] = []
        self.features = {}
        self.kept: List[Edge] = []
        self.build_ms = self.solve_ms = 0.0
        self.successes = self.tasks_done = 0
        self.configure(params)

    def configure(self, params: dict) -> None:
        """Live tuning: every planner parameter except fleet size and seed."""
        self.comm_range = params["communicationRange"]
        self.base_packet_loss = params["packetLoss"]
        self.degree_cap = params["maxDegree"]
        self.qubo_budget = params["budget"]
        self.annealer_options = {"restarts": params["restarts"],
                                 "cooling_rate": params["coolingRate"]}
        self.tasks_per_cycle = min(params["tasksPerCycle"], len(self.nodes))
        self.exploration = params["exploration"]
        self.trust.alpha = params["trustMemory"]
        self.reliability_drift = params["reliabilityDrift"]

    def inject(self, kind: str, rover: Optional[int]) -> None:
        magnitude, duration, _ = FAULTS[kind]
        self.faults.append(FaultEvent(cycle=self.cycle, kind=kind, node=rover,
                                      magnitude=magnitude, duration=duration))

    def active_faults(self) -> List[dict]:
        return [{"kind": f.kind, "rover": f.node,
                 "remaining": f.cycle + f.duration - self.cycle}
                for f in self._active_faults(self.cycle)]

    def _move(self, cycle: int) -> None:
        super()._move(cycle)
        for node, position in enumerate(self.live_positions):
            self.state[node].position = np.array(position, dtype=float)

    def _graph(self, cycle: int):
        self.edges, self.features = super()._graph(cycle)
        return self.edges, self.features

    def _prune(self, edges, features, cycle):
        if self.leader is not None:
            live = set(edges)
            self.last_prune = ("paired", None, None)
            self.kept = [e for e in self.leader.kept if e in live]
            return self.kept, 0.0, 0.0
        start = time.perf_counter()
        self.kept, build, solve = super()._prune(edges, features, cycle)
        self.build_ms = build * 1e3
        self.solve_ms = max(solve, time.perf_counter() - start - build) * 1e3
        return self.kept, build, solve

    def advance(self, positions: List[Tuple[float, float]]) -> dict:
        self.live_positions = positions
        self.edges, self.features, self.kept = [], {}, []
        self.build_ms = self.solve_ms = 0.0
        self.last_prune = ("none", None, None)
        self.last_outcomes = []
        self.last_path_factor = {n: float(n == self.root) for n in self.nodes}
        result = self.step(self.cycle)
        self.cycle += 1
        self.successes += result.get("success", 0)
        self.tasks_done += result.get("tasks", 0)
        return result


def _dimod_qubo(Q: np.ndarray) -> dict:
    """dimod convention: E = sum_i Qii xi + sum_{i<j} Qij xi xj."""
    rows, cols = np.nonzero(np.triu(Q))
    return {(int(i), int(j)): float(Q[i, j] if i == j else 2.0 * Q[i, j])
            for i, j in zip(rows, cols)}


def _sample(name: str, Q: np.ndarray, constant: float, warm: Optional[np.ndarray],
            seed: int) -> np.ndarray:
    if name == "bitsa":
        z, _ = SimulatedAnnealing(
            Q, constant, restarts=2, cooling_rate=0.97, sweeps_per_temperature=2,
            seed=seed, initial_states=[warm] if warm is not None else None,
        ).run()
        return z
    if name == "neal":
        import neal
        sampler, kw = neal.SimulatedAnnealingSampler(), {"num_reads": 50, "seed": seed}
    elif name == "tabu":
        import tabu
        sampler, kw = tabu.TabuSampler(), {"num_reads": 5, "timeout": 100, "seed": seed}
    else:
        import openjij
        sampler, kw = openjij.SQASampler(), {"num_reads": 10}
    sample = sampler.sample_qubo(_dimod_qubo(Q), **kw).first.sample
    return np.array([sample.get(k, 0) for k in range(Q.shape[0])], dtype=float)


def run_solver_baseline(name: str, builder: EdgeSelectionQUBO, warm_edges,
                        ours_energy: float, seed: int) -> dict:
    """Run an off-the-shelf sampler on the exact Q matrix our annealer solved."""
    Q, constant, _ = builder.get_qubo()
    try:
        warm = builder.build_feasible_solution(warm_edges)
    except ValueError:
        warm = None
    start = time.perf_counter()
    z = _sample(name, Q, constant, warm, seed)
    elapsed = (time.perf_counter() - start) * 1e3
    energy = builder.energy(z)
    feasible = not builder.check_constraints(np.rint(z))
    scale = max(1.0, abs(ours_energy))
    return {
        "feasible": feasible, "energy": energy, "oursEnergy": ours_energy,
        "gap": (energy - ours_energy) / scale, "ms": elapsed,
        "matched": feasible and energy <= ours_energy + 1e-9 * scale,
    }


class PlannerSession:
    """One browser mission: the primary pipeline plus paired baselines."""

    def __init__(self, params: dict) -> None:
        self.params = params
        self.primary = LiveSimulation(params)
        # assignment baselines run from cycle 0 so a comparison chosen
        # mid-mission is still paired from the same starting state
        self.followers = {name: LiveSimulation(params, assigner=name, leader=self.primary)
                          for name in ASSIGNMENT_BASELINES}
        self.solver_name: Optional[str] = None
        self.solver_totals = self._fresh_solver_totals()
        self.lock = threading.Lock()
        self.last_used = time.monotonic()

    @staticmethod
    def _fresh_solver_totals() -> dict:
        return {"runs": 0, "feasible": 0, "matched": 0, "gapSum": 0.0, "msSum": 0.0}

    def matches(self, params: dict) -> bool:
        return all(params[k] == self.params[k] for k in SESSION_KEYS)

    def configure(self, params: dict) -> None:
        self.params = params
        for sim in (self.primary, *self.followers.values()):
            sim.configure(params)

    def step(self, positions, faults, comparison: str) -> dict:
        self.last_used = time.monotonic()
        for kind, rover in faults:
            for sim in (self.primary, *self.followers.values()):
                sim.inject(kind, rover)
        warm_edges = set(self.primary.last_kept)
        result = self.primary.advance(positions)
        for follower in self.followers.values():
            follower.advance(positions)
        return self._payload(result, comparison, warm_edges)

    def _solver_comparison(self, comparison: str, warm_edges) -> Optional[dict]:
        if not comparison.startswith("solver:"):
            return None
        name = comparison.split(":", 1)[1]
        if name != self.solver_name:
            self.solver_name, self.solver_totals = name, self._fresh_solver_totals()
        mode, builder, result = self.primary.last_prune
        latest = None
        if not available_solvers().get(name):
            latest = {"error": f"{name} is not installed in the planner's Python environment"}
        elif builder is not None and result is not None:
            live = set(builder.edges)
            latest = run_solver_baseline(name, builder, warm_edges & live,
                                         result.energy, seed=self.primary.cycle)
            totals = self.solver_totals
            totals["runs"] += 1
            totals["feasible"] += int(latest["feasible"])
            totals["matched"] += int(latest["matched"])
            totals["msSum"] += latest["ms"]
            if latest["feasible"]:
                totals["gapSum"] += latest["gap"]
        return {"baseline": name, "latest": latest, "totals": dict(self.solver_totals),
                "oursMs": self.primary.solve_ms}

    def _payload(self, result: dict, comparison: str, warm_edges) -> dict:
        sim = self.primary
        assignment = result.get("assignment", {})
        kept = set(sim.kept)
        mode, builder, solved = sim.last_prune
        reachable = {n for n, f in sim.last_path_factor.items() if f > 0.0}
        return {
            "cycle": sim.cycle,
            "rovers": [{
                "index": n, "trust": sim.trust.get(n),
                "battery": sim.state[n].battery, "compute": 1.0 - sim.state[n].load,
                "capability": sim.state[n].capability,
                "reliability": sim.state[n].reliability,
                "pathFactor": sim.last_path_factor.get(n, 0.0),
                "task": next((t for t, a in assignment.items() if a == n), None),
            } for n in sim.nodes],
            "edges": [{
                "from": a, "to": b, "active": (a, b) in kept,
                "quality": sim.features[(a, b)].link_quality,
                "utility": edge_utility(sim.features[(a, b)]),
            } for a, b in sim.edges],
            "tasks": [{"id": tid, "rover": n, "executed": ex, "delivered": dl,
                       "success": ex and dl, "explored": explored}
                      for tid, n, ex, dl, explored in sim.last_outcomes],
            "faults": sim.active_faults(),
            "metrics": {
                "successes": sim.successes, "tasks": sim.tasks_done,
                "cycleSuccesses": result.get("success", 0),
                "cycleTasks": result.get("tasks", 0),
                "buildMs": sim.build_ms, "solveMs": sim.solve_ms,
                "gnnMs": sim.timings["gnn"][-1] if result.get("tasks") else 0.0,
                "pruneMode": mode,
                "variables": len(builder.variables) if builder is not None else 0,
                "energy": solved.energy if solved is not None else None,
                "keptCount": len(kept), "candidateCount": len(sim.edges),
                "reachable": len(reachable), "fleet": len(sim.nodes),
                "minSpanningCost": min_spanning_cost(sim.nodes, sim.edges, sim.features, sim.root)
                if sim.edges else None,
                "budget": sim.qubo_budget, "degreeCap": sim.degree_cap,
            },
            "assignmentBaselines": {
                name: {"successes": f.successes, "tasks": f.tasks_done,
                       "cycleSuccesses": sum(ok and dl for _, _, ok, dl, _ in f.last_outcomes),
                       "cycleTasks": len(f.last_outcomes)}
                for name, f in self.followers.items()
            },
            "solver": self._solver_comparison(comparison, warm_edges),
        }


class SessionStore:
    """Bounded, expiring session registry shared by the HTTP threads.

    When full, the longest-idle session is evicted (a closed tab or a page
    that reloaded); only sessions active within ``busy_window`` seconds are
    protected, so stale sessions can never lock out a live page.
    """

    def __init__(self, capacity: int = 16, ttl: float = 600.0, busy_window: float = 15.0) -> None:
        self.capacity, self.ttl, self.busy_window = capacity, ttl, busy_window
        self.sessions: Dict[str, PlannerSession] = {}
        self.lock = threading.Lock()

    def acquire(self, session_id: Optional[str], params: dict, new_id) -> Tuple[str, PlannerSession]:
        with self.lock:
            now = time.monotonic()
            for key in [k for k, s in self.sessions.items() if now - s.last_used > self.ttl]:
                del self.sessions[key]
            session = self.sessions.get(session_id) if session_id else None
            if session is not None and session.matches(params):
                session.configure(params)
                return session_id, session
            if session is not None:
                del self.sessions[session_id]
            if len(self.sessions) >= self.capacity:
                idle_id = min(self.sessions, key=lambda k: self.sessions[k].last_used)
                if now - self.sessions[idle_id].last_used < self.busy_window:
                    raise OverflowError("Every planner session is in use; retry later")
                del self.sessions[idle_id]
            session_id = new_id()
            self.sessions[session_id] = PlannerSession(params)
            return session_id, self.sessions[session_id]
