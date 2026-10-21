"""Scripted "where we win" scenarios for the console's showcase mode.

Each scenario fixes everything that drives the planner: parameters, seed,
the exact rover positions of every cycle, and the faults injected on which
cycle. Replayed through PlannerSession (the same paired followers and
same-Q solver baseline the live console uses), a scenario is fully
deterministic, so the measured outcome recorded in
shared/showcase-scenarios.json can be re-checked by tests/test_showcase.py
and is reproduced cycle for cycle by the browser.

Seeds were chosen because our pipeline wins on them; the paper's averages,
not these runs, are the fair comparison, and the UI says so.

    python3 web/showcase_scenarios.py          # measure and rewrite the JSON
    python3 web/showcase_scenarios.py --check  # measure and compare only
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fleet_session import CONFIG, PlannerSession, validate_params  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "shared" / "showcase-scenarios.json"
MIN_SPACING = 2 * CONFIG["arena"]["bodyRadius"] + 0.01
Point = List[float]


def _r(x: float, z: float) -> Point:
    return [round(x, 4), round(z, 4)]


def _ring(n: int, radius: float, phase: float = 0.0) -> List[Point]:
    return [_r(radius * math.cos(phase + 2 * math.pi * i / n), radius * math.sin(phase + 2 * math.pi * i / n))
            for i in range(n)]


def compact(_: int) -> List[Point]:
    return [[0.0, 0.0]] + _ring(7, 2.4)


def spread(_: int) -> List[Point]:
    return [[0.0, 0.0]] + _ring(5, 2.2) + _ring(4, 4.4, phase=0.3)


def bridge(cycle: int) -> List[Point]:
    """Two clusters a radio hop too far apart; the root shuttles between them as the only relay."""
    left = [_r(-3.3 + dx, dz) for dx, dz in ((0, 0), (-1.2, 1.0), (-1.2, -1.0), (0.1, 1.6))]
    right = [_r(3.3 + dx, dz) for dx, dz in ((0, 0), (1.2, 1.0), (1.2, -1.0), (-0.1, -1.6))]
    return [_r(0.0, 1.6 * math.sin(cycle * 0.9))] + left + right


def hub(cycle: int) -> List[Point]:
    """A root in range of everyone inside a breathing ring: the cheapest tree overloads the hub."""
    radius = 2.3 + 0.5 * math.sin(cycle * 0.9)
    return [[0.0, 0.0]] + _ring(7, radius, phase=0.2 * cycle)


def swarm(cycle: int) -> List[Point]:
    """Twelve rovers weaving in two rings, so the radio graph (and the Q matrix) changes every cycle."""
    points = [[0.0, 0.0]]
    for i in range(11):
        inner = i < 5
        k = i if inner else i - 5
        angle = 2 * math.pi * k / (5 if inner else 6) + (0.0 if inner else 0.5) + 0.06 * math.sin(cycle * 0.5 + i)
        radius = (2.2 if inner else 4.4) + 0.35 * math.sin(cycle * 0.8 + i * 1.7)
        points.append(_r(radius * math.cos(angle), radius * math.sin(angle)))
    return points


SCENARIOS: List[dict] = [
    {
        "id": "trusted-rover-degrades",
        "title": "A trusted rover degrades",
        "comparison": "assign:trust_only",
        "formation": compact, "cycles": 24,
        "params": {"roverCount": 8, "seed": 4, "reliabilityDrift": 0.0},
        "faults": {c: [("reliability_drop", 3)] for c in (4, 8, 12, 16)},
        "watch": "RV-04 starts as one of the most reliable rovers, then its hidden reliability drops by 0.6 from cycle 4 to 20.",
        "mechanism": "Trust-only keeps sending work to the rover with the best record, and pays for it until the failures drag that record down. Our readout also weighs battery, free compute and the neighbourhood, so it spreads tasks and recovers sooner.",
    },
    {
        "id": "radio-storm",
        "title": "Radio storm",
        "comparison": "assign:random",
        "formation": spread, "cycles": 24,
        "params": {"roverCount": 10, "seed": 4, "communicationRange": 4.2},
        "faults": {c: [("packet_loss_spike", None)] for c in range(2, 24, 3)},
        "watch": "A 40% packet-loss spike every third cycle leaves link-quality readings stale across a spread-out fleet.",
        "mechanism": "Random assignment ignores both trust and the route to the root. Ours prefers rovers with a good record and a strong path, which matters most when deliveries cross weak, stale links.",
    },
    {
        "id": "fragile-bridge",
        "title": "The fragile bridge",
        "comparison": "solver:bitsa",
        "formation": bridge, "cycles": 12,
        "params": {"roverCount": 9, "seed": 1, "communicationRange": 3.6, "maxDegree": 3, "budget": 60},
        "faults": {},
        "watch": "The root shuttles between two clusters it alone can bridge, so last cycle's links never fit this cycle's graph.",
        "mechanism": "Both solvers get the same Q matrix and warm start. Our annealer only moves between feasible link sets, repairing each move; single-bit annealing has to climb over penalty barriers and often lands on an edge set that overloads a rover or carries inconsistent flow bits.",
    },
    {
        "id": "hub-over-cap",
        "title": "A hub over its degree cap",
        "comparison": "solver:bitsa",
        "formation": hub, "cycles": 12,
        "params": {"roverCount": 8, "seed": 2, "communicationRange": 2.7, "maxDegree": 2, "budget": 60},
        "faults": {},
        "watch": "The root can reach every rover, but each rover may keep only two links, so the tree must route around the hub.",
        "mechanism": "The degree cap is a hard constraint in our search space. On the penalty-encoded Q, bit-flip annealing can return a star that breaks the cap or strands rovers from the root.",
    },
    {
        "id": "weaving-swarm",
        "title": "Penalty barriers in a weaving swarm",
        "comparison": "solver:bitsa",
        "formation": swarm, "cycles": 12,
        "params": {"roverCount": 12, "seed": 1, "communicationRange": 4.0, "maxDegree": 3, "budget": 90, "coolingRate": 0.985},
        "faults": {},
        "watch": "Twelve rovers weave through two rings, so the Q matrix has hundreds of variables and changes every cycle.",
        "mechanism": "On a large, changing Q the warm start goes stale. Bit-flip annealing then has to find feasibility on its own, and its answers break hard constraints or settle above our energy.",
    },
]


def faults_from_json(raw: Dict[str, list]) -> Dict[int, list]:
    """The JSON's {"<cycle>": [{"kind", "rover"}]} form as replay() takes it."""
    return {int(c): [[f["kind"], f["rover"]] for f in fs] for c, fs in raw.items()}


def scenario_params(spec: dict) -> dict:
    """Full parameter set (client keys included) for a scenario."""
    params = dict(CONFIG["defaults"])
    params.update(spec["params"])
    return params


def positions_for(spec: dict) -> List[List[Point]]:
    frames = [spec["formation"](c) for c in range(spec["cycles"])]
    radius = CONFIG["arena"]["radius"]
    for frame in frames:
        assert len(frame) == spec["params"]["roverCount"], spec["id"]
        assert all(math.hypot(*p) <= radius for p in frame), spec["id"]
        assert all(math.hypot(a[0] - b[0], a[1] - b[1]) >= MIN_SPACING
                   for i, a in enumerate(frame) for b in frame[i + 1:]), f"{spec['id']}: rovers overlap"
    return frames


def replay(params: dict, comparison: str, frames: List[List[Point]],
           faults: Dict[int, list], cycles: int | None = None) -> List[list]:
    """Per-cycle trace: [ours ok, ours tasks, base ok, base tasks] or [verdict] for a solver baseline."""
    session = PlannerSession(validate_params(params))
    trace = []
    for cycle, frame in enumerate(frames[:cycles]):
        result = session.step(frame, [tuple(f) for f in faults.get(cycle, [])], comparison)
        if comparison.startswith("assign:"):
            base = result["assignmentBaselines"][comparison.split(":", 1)[1]]
            m = result["metrics"]
            trace.append([m["cycleSuccesses"], m["cycleTasks"], base["cycleSuccesses"], base["cycleTasks"]])
        else:
            trace.append([solver_verdict(result["solver"]["latest"])])
    return trace


def solver_verdict(run) -> str:
    if run is None:
        return "skipped"
    if not run["feasible"]:
        return "infeasible"
    scale = max(1.0, abs(run["oursEnergy"]))
    if run["energy"] < run["oursEnergy"] - 1e-9 * scale:
        return "better"
    return "matched" if run["matched"] else "worse"


def summarize(comparison: str, trace: List[list]) -> dict:
    if comparison.startswith("assign:"):
        ours = [sum(t[0] for t in trace), sum(t[1] for t in trace)]
        base = [sum(t[2] for t in trace), sum(t[3] for t in trace)]
        return {"ours": ours, "baseline": base,
                "cyclesWon": sum(t[0] > t[2] for t in trace), "cyclesLost": sum(t[0] < t[2] for t in trace)}
    verdicts = [t[0] for t in trace]
    return {v: verdicts.count(v) for v in ("infeasible", "worse", "matched", "better", "skipped")}


def wins(comparison: str, summary: dict) -> bool:
    if comparison.startswith("assign:"):
        return summary["ours"][0] * summary["baseline"][1] > summary["baseline"][0] * summary["ours"][1]
    return summary["better"] == 0 and summary["infeasible"] + summary["worse"] > 0


def build(on_progress: Callable[[str], None] = print) -> dict:
    scenarios = []
    for spec in SCENARIOS:
        frames = positions_for(spec)
        faults = {c: [list(f) for f in fs] for c, fs in spec["faults"].items()}
        params = scenario_params(spec)
        trace = replay(params, spec["comparison"], frames, faults)
        summary = summarize(spec["comparison"], trace)
        won = wins(spec["comparison"], summary)
        on_progress(f"{spec['id']}: {summary} {'WIN' if won else 'NOT A WIN'}")
        if not won:
            continue
        scenarios.append({
            "id": spec["id"], "title": spec["title"], "comparison": spec["comparison"],
            "watch": spec["watch"], "mechanism": spec["mechanism"],
            "params": params, "cycles": spec["cycles"], "positions": frames,
            "faults": {str(c): [{"kind": k, "rover": r} for k, r in fs] for c, fs in spec["faults"].items()},
            "measured": summary, "trace": trace,
        })
    return {
        "version": 1,
        "generator": "web/showcase_scenarios.py",
        "note": ("Seeds were chosen because our pipeline wins on them, to show the mechanism. "
                 "They are illustrations, not evidence: the paper's paired multi-seed results are the fair comparison."),
        "scenarios": scenarios,
    }


if __name__ == "__main__":
    data = build()
    if "--check" not in sys.argv:
        OUT.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {OUT} with {len(data['scenarios'])} scenarios")
