"""Extended pruner benchmark: 6 arms x seeds, sequential so wall-clock
timings are clean (the constraint sweep runs in parallel and must not be
used for latency). Resumable via benchmark.jsonl.

Arms: full, threshold, topk (constraint-blind), threshold_repair,
topk_repair (constraint-aware: same degree caps, budget, mission-critical
connectivity as the QUBO, repaired via minimum-cost spanning tree plus
utility-ordered deletion), qubo.

Run:  python benchmark_full.py [start_seed] [end_seed]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from simulation import (RoverSimulation, paired_bootstrap_ci,
                        permutation_pvalue)

ARMS = ("full", "threshold", "topk", "threshold_repair", "topk_repair", "qubo")


def run_one(seed: int, budget_factor=None) -> dict:
    row = {"seed": seed, "budget_factor": budget_factor}
    for arm in ARMS:
        sim = RoverSimulation(steps=20, seed=seed, pruner=arm, n_rovers=10,
                              comm_range=6.0, tasks_per_cycle=3,
                              reliability_drift=0.02,
                              budget_factor=budget_factor)
        s = sim.run(verbose=False)
        row[arm] = {
            "success": s["success_rate"],
            "offload": s["offload_rate"],
            "delfail": s["delivery_fail_rate"],
            "solve_ms": s["timings_ms"]["solve"],
            "gnn_ms": s["timings_ms"]["gnn"],
            "kept": s["kept_ratio"],
            "spearman": s["trust_spearman"],
            "auc": s["trust_auc"],
            "viol": s["violations"],
            "modes": s["prune_modes"],
            "precision": s["pruning_precision"],
            "recall": s["pruning_recall"],
        }
    return row


def load_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def report(rows) -> None:
    bf = rows[0].get("budget_factor") if rows else None
    label = f"K={bf:.2f}x candidate cost" if bf else "K=15 (paper nominal)"
    print(f"\n=== extended pruner benchmark: 20 cycles x {len(rows)} seeds, "
          f"10 rovers, drift=0.02, budget {label} (sequential timings) ===")
    print(f"{'arm':>17s} {'success':>8s} {'offload':>8s} {'delfail':>8s} "
          f"{'solve ms':>9s} {'kept%':>6s} {'prec':>6s} {'recall':>6s} "
          f"{'viol d/b/c':>11s} {'spearman':>9s} {'auc':>6s}")
    for arm in ARMS:
        cells = [r[arm] for r in rows]

        def mean(key):
            vals = [c[key] for c in cells
                    if isinstance(c[key], float) and np.isfinite(c[key])]
            return float(np.mean(vals)) if vals else float("nan")

        v = {k: int(np.mean([c["viol"][k] for c in cells]))
             for k in ("degree", "budget", "connectivity")}
        print(f"{arm:>17s} {mean('success'):8.2%} {mean('offload'):8.2%} "
              f"{mean('delfail'):8.2%} {mean('solve_ms'):9.1f} "
              f"{mean('kept')*100:5.0f}% {mean('precision'):6.3f} "
              f"{mean('recall'):6.3f} "
              f"{v['degree']:>3d}/{v['budget']:>3d}/{v['connectivity']:>3d} "
              f"{mean('spearman'):9.3f} {mean('auc'):6.3f}")
    qmodes: dict = {}
    for r in rows:
        for k, v in r["qubo"]["modes"].items():
            qmodes[k] = qmodes.get(k, 0) + v
    print(f"qubo relaxation modes over {len(rows) * 20} cycles: {qmodes}")
    for other in ("threshold", "threshold_repair", "topk_repair"):
        diffs = [r["qubo"]["success"] - r[other]["success"] for r in rows]
        m, lo, hi = paired_bootstrap_ci(diffs)
        p = permutation_pvalue(diffs)
        print(f"qubo vs {other:>16s}: diff {m:+.2%}, "
              f"95% CI [{lo:+.2%},{hi:+.2%}], permutation p={p:.3f}")


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    bf = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    tag = f"_bf{bf:.2f}" if bf else ""
    path = f"benchmark{tag}.jsonl"
    done = {r["seed"] for r in load_rows(path)}
    for seed in range(start, end):
        if seed in done:
            continue
        row = run_one(seed, bf)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"seed {seed} done", flush=True)
    report(load_rows(path))


if __name__ == "__main__":
    main()
