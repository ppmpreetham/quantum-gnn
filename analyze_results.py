"""Extract paper-ready numbers from sweep_*.jsonl and benchmark*.jsonl.

Prints, for each sweep level: mean success/violations per arm and paired
QUBO-vs-repaired statistics (per-seed bootstrap CI + permutation p).
"""

from __future__ import annotations

import json
from typing import List

import numpy as np

from simulation import paired_bootstrap_ci, permutation_pvalue

LEVELS = {
    "budget": (0.35, 0.5, 0.75, 1.0),
    "degree": (2, 3, 4),
    "roster": (3, 5, 8),
}
ARMS = ("full", "threshold", "topk", "threshold_repair", "topk_repair", "qubo")


def load(path) -> List[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    except FileNotFoundError:
        return []


def cells(rows, kind, level, arm):
    return [r for r in rows if r["arm"] == arm
            and float(r["level"]) == float(level)]


def paired(rows, kind, level, a, b):
    ca, cb = cells(rows, kind, level, a), cells(rows, kind, level, b)
    amap = {c["seed"]: c["success_rate"] for c in ca}
    bmap = {c["seed"]: c["success_rate"] for c in cb}
    common = sorted(set(amap) & set(bmap))
    diffs = [amap[s] - bmap[s] for s in common]
    m, lo, hi = paired_bootstrap_ci(diffs)
    p = permutation_pvalue(diffs)
    return m, lo, hi, p


def main() -> None:
    for kind, levels in LEVELS.items():
        rows = [r for r in load(f"sweep_{kind}.jsonl")
                if r.get("steps", 20) == 20]
        print(f"\n## {kind} sweep ({len(rows)} cells)")
        for level in levels:
            line = [f"{kind}={level}:"]
            for arm in ARMS:
                sel = cells(rows, kind, level, arm)
                if not sel:
                    continue
                v = {k: np.mean([c["violations"][k] for c in sel])
                     for k in ("degree", "budget", "connectivity")}
                s = np.mean([c["success_rate"] for c in sel])
                line.append(f"{arm}={s:.4f}(d{v['degree']:.1f}/b{v['budget']:.1f}"
                            f"/c{v['connectivity']:.1f})")
            print("  " + " ".join(line))
            for other in ("threshold_repair", "topk_repair"):
                m, lo, hi, p = paired(rows, kind, level, "qubo", other)
                print(f"    qubo - {other}: {m:+.4f} CI [{lo:+.4f},{hi:+.4f}] "
                      f"p={p:.4f}")

    for path, label in (("benchmark.jsonl", "K=15 nominal"),
                        ("benchmark_bf0.75.jsonl", "K=0.75 binding")):
        rows = load(path)
        if not rows:
            continue
        print(f"\n## benchmark {label} ({len(rows)} seeds)")
        for arm in ARMS:
            s = np.mean([r[arm]["success"] for r in rows])
            o = np.mean([r[arm]["offload"] for r in rows])
            d = np.mean([r[arm]["delfail"] for r in rows])
            ms = np.mean([r[arm]["solve_ms"] for r in rows])
            k = np.mean([r[arm]["kept"] for r in rows])
            pr = np.mean([r[arm]["precision"] for r in rows])
            rc = np.mean([r[arm]["recall"] for r in rows])
            v = {kk: np.mean([r[arm]["viol"][kk] for r in rows])
                 for kk in ("degree", "budget", "connectivity")}
            sp = np.mean([r[arm]["spearman"] for r in rows])
            print(f"  {arm:>17s}: succ {s:.4f} offload {o:.4f} delfail {d:.4f} "
                  f"solve {ms:.1f}ms kept {k:.3f} prec {pr:.4f} rec {rc:.4f} "
                  f"viol {v['degree']:.2f}/{v['budget']:.2f}/{v['connectivity']:.2f} "
                  f"spearman {sp:.4f}")
        qm: dict = {}
        for r in rows:
            for kk, vv in r["qubo"]["modes"].items():
                qm[kk] = qm.get(kk, 0) + vv
        print(f"  qubo modes: {qm} (of {len(rows) * 20} cycles)")
        for other in ("threshold", "threshold_repair", "topk_repair"):
            diffs = [r["qubo"]["success"] - r[other]["success"] for r in rows]
            m, lo, hi = paired_bootstrap_ci(diffs)
            p = permutation_pvalue(diffs)
            print(f"  qubo - {other}: {m:+.4f} CI [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
        # delivery-fail paired (qubo vs thr_repair) for the metrics section
        for key, other in (("delfail", "threshold_repair"),
                           ("offload", "threshold_repair"),
                           ("delfail", "topk"),
                           ("offload", "topk")):
            diffs = [r["qubo"][key] - r[other][key] for r in rows]
            m, lo, hi = paired_bootstrap_ci(diffs)
            print(f"  qubo - {other} [{key}]: {m:+.4f} CI [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
