"""Constraint-pressure experiment: find the regime where constraint-carrying
pruning beats constraint-blind pruning.

The paper's honest question: the QUBO pruner loses to threshold/top-K on raw
success when constraints barely bind (10 rovers, 84-100% of edges kept). So we
tighten the constraints and measure success AND constraint violations:

  - budget pressure: K = budget_factor x the total cost of the cycle's
    candidate graph (0.35/0.5/0.75/1.0)
  - degree pressure: per-node degree cap of 2, 3, or 4
  - roster pressure: |R| mission-critical nodes out of 10 (3/5/8)

Arms: full graph (no pruning, reference only - it does not honor the
constraints), threshold, top-K (constraint-blind), threshold+repair,
top-K+repair (constraint-aware: same caps/budget/connectivity as the QUBO,
repaired via a minimum-cost spanning tree plus utility-ordered deletion),
and the QUBO.

Cells are appended to sweep_<kind>.jsonl as they complete and skipped on
restart, so the sweep is resumable. Solve times measured under parallel load
are inflated; latency claims come from the solo benchmark, not this script.

Run:  python constraint_pressure.py [budget|degree|roster] [quick]
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple

import numpy as np

from simulation import RoverSimulation, paired_bootstrap_ci, permutation_pvalue

ARMS = ("full", "threshold", "topk", "threshold_repair", "topk_repair", "qubo")
LEVELS = {
    "budget": (0.35, 0.5, 0.75, 1.0),
    "degree": (2, 3, 4),
    "roster": (3, 5, 8),
}


def run_cell(seed: int, arm: str, pressure: Tuple[str, float],
             steps: int) -> dict:
    kind, level = pressure
    kw: Dict = dict(n_rovers=10, comm_range=6.0, tasks_per_cycle=3,
                    steps=steps, seed=seed, reliability_drift=0.02,
                    pruner=arm)
    if kind == "budget":
        kw["budget_factor"] = level
    elif kind == "degree":
        kw["degree_cap"] = int(level)
    elif kind == "roster":
        kw["n_mission_critical"] = int(level)
    s = RoverSimulation(**kw).run(verbose=False)
    return {
        "seed": seed,
        "arm": arm,
        "level": level,
        "steps": steps,
        "success_rate": s["success_rate"],
        "delivery_fail_rate": s["delivery_fail_rate"],
        "offload_rate": s["offload_rate"],
        "solve_ms": s["timings_ms"]["solve"],
        "kept_ratio": s["kept_ratio"],
        "violations": s["violations"],
    }


def _worker(args):
    seed, arm, kind, level, steps = args
    return run_cell(seed, arm, (kind, level), steps)


def load_done(path: str, steps: int) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if int(r.get("steps", -1)) == steps:
                        done.add((r["seed"], r["arm"], float(r["level"])))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def sweep(kind: str, seeds: int, steps: int, workers: int) -> List[dict]:
    path = f"sweep_{kind}.jsonl"
    done = load_done(path, steps)
    jobs = []
    for level in LEVELS[kind]:
        for arm in ARMS:
            for seed in range(seeds):
                if (seed, arm, float(level)) not in done:
                    jobs.append((seed, arm, kind, float(level), steps))
    print(f"[{kind}] {len(done)} cells done, {len(jobs)} to run "
          f"({seeds} seeds x {len(LEVELS[kind])} levels x {len(ARMS)} arms)",
          flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(_worker, jobs, chunksize=1)):
                with open(path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(res) + "\n")
                if (i + 1) % 48 == 0:
                    print(f"[{kind}] {i + 1}/{len(jobs)} new cells", flush=True)
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fmt_level(kind: str, level: float) -> str:
    if kind == "budget":
        return f"K={level:.2f}"
    if kind == "degree":
        return f"D={int(level)}"
    return f"|R|={int(level)}"


def cells_for(cells, kind: str, level: float, arm: str) -> List[dict]:
    return [c for c in cells if c["arm"] == arm
            and float(c["level"]) == float(level)]


def report(kind: str, cells: List[dict], seeds: int, steps: int) -> None:
    print(f"\n=== {kind} pressure ({steps} cycles x {seeds} seeds) ===")
    print(f"{'level':>8s} {'arm':>17s} {'success':>8s} {'delfail':>8s} "
          f"{'offload':>8s} {'viol d/b/c':>11s}")
    for level in LEVELS[kind]:
        for arm in ARMS:
            sel = cells_for(cells, kind, level, arm)
            if not sel:
                continue
            v = {k: int(np.mean([c["violations"][k] for c in sel]))
                 for k in ("degree", "budget", "connectivity")}
            succ = float(np.mean([c["success_rate"] for c in sel]))
            dfail = float(np.mean([c["delivery_fail_rate"] for c in sel]))
            off = float(np.mean([c["offload_rate"] for c in sel]))
            print(f"{fmt_level(kind, level):>8s} {arm:>17s} {succ:8.2%} "
                  f"{dfail:8.2%} {off:8.2%} "
                  f"{v['degree']:>3d}/{v['budget']:>3d}/{v['connectivity']:>3d}")

    # paired statistics: QUBO vs best legitimate opponent per level.
    # The full-graph arm is excluded from the comparison (it does not honor
    # the constraints being pressured) and is printed as a reference only.
    print(f"--- paired stats: QUBO vs best constrained arm ({kind}) ---")
    for level in LEVELS[kind]:
        per_arm = {arm: [c["success_rate"]
                         for c in cells_for(cells, kind, level, arm)]
                   for arm in ARMS}
        full = per_arm["full"]
        others = {a: v for a, v in per_arm.items()
                  if a not in ("qubo", "full") and v}
        best = max(others, key=lambda a: float(np.mean(others[a])))
        q = per_arm["qubo"]
        diffs = [a - b for a, b in zip(q, others[best])]
        mean, lo, hi = paired_bootstrap_ci(diffs)
        p = permutation_pvalue(diffs)
        print(f"  {fmt_level(kind, level):>8s}: qubo {np.mean(q):6.2%} "
              f"vs best({best:>16s}) {np.mean(others[best]):6.2%}  "
              f"diff {mean:+6.2%} CI [{lo:+.2%},{hi:+.2%}] p={p:.3f}   "
              f"[full-graph ref {np.mean(full):6.2%}]")


def main() -> None:
    quick = "quick" in sys.argv
    kinds = [a for a in sys.argv[1:] if a in LEVELS] or list(LEVELS)
    seeds = 4 if quick else 16
    steps = 10 if quick else 20
    workers = min(14, os.cpu_count() or 8)
    for kind in kinds:
        cells = sweep(kind, seeds, steps, workers)
        report(kind, cells, seeds, steps)


if __name__ == "__main__":
    main()
