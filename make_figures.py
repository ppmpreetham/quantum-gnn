"""Generate all data figures for the presentation document from the real
experiment artifacts (benchmark.jsonl, benchmark_bf0.75.jsonl, sweep_*.jsonl,
worked_example.json).

Outputs PDFs into paper/figs_doc/ and a doc_data.json with every aggregated
number the document text quotes, so text and figures can never disagree.

Run:  .venv/Scripts/python.exe make_figures.py
"""

from __future__ import annotations

import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join("paper", "figs_doc")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

ARMS = ["full", "threshold", "topk", "threshold_repair", "topk_repair", "qubo"]
LABELS = {
    "full": "full graph", "threshold": "threshold", "topk": "top-K",
    "threshold_repair": "threshold+repair", "topk_repair": "top-K+repair",
    "qubo": "QUBO (ours)",
}
COLORS = {
    "full": "#888888", "threshold": "#1f77b4", "topk": "#aec7e8",
    "threshold_repair": "#2ca02c", "topk_repair": "#98df8a",
    "qubo": "#d62728",
}


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def arm_summary(rows):
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        for arm in ARMS:
            s = row.get(arm)
            if not s:
                continue
            for key in ("success", "offload", "delfail", "solve_ms", "kept",
                        "spearman", "auc", "precision", "recall"):
                if key in s:
                    acc[arm][key].append(s[key])
            v = s.get("viol") or {}
            for k in ("degree", "budget", "connectivity"):
                if k in v:
                    acc[arm][f"viol_{k}"].append(v[k])
    return {arm: {k: float(np.mean(v)) for k, v in d.items()}
            for arm, d in acc.items()}


def paired_diff(rows, arm_a, arm_b, key="success"):
    """Per-seed paired difference a-b."""
    diffs = []
    for row in rows:
        if arm_a in row and arm_b in row:
            diffs.append(row[arm_a][key] - row[arm_b][key])
    diffs = np.array(diffs)
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(diffs, len(diffs)))) for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = float(np.mean(diffs))
    # two-sided sign-flip permutation p
    n = len(diffs)
    count = 0
    for _ in range(10000):
        signs = rng.choice([-1, 1], n)
        if abs(float(np.mean(diffs * signs))) >= abs(obs):
            count += 1
    return obs, lo, hi, count / 10000


# ----------------------------------------------------------------------
# Aggregate the two benchmarks and the sweeps
# ----------------------------------------------------------------------
nominal = arm_summary(load_jsonl("benchmark.jsonl"))
binding = arm_summary(load_jsonl("benchmark_bf0.75.jsonl"))

sweep = {}
for kind, fname in (("budget", "sweep_budget.jsonl"),
                    ("degree", "sweep_degree.jsonl"),
                    ("roster", "sweep_roster.jsonl")):
    rows = load_jsonl(fname)
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:                      # flat schema: one row per seed x arm x level
        level, arm = row["level"], row["arm"]
        cells[level][f"succ_{arm}"].append(row["success_rate"])
        cells[level][f"offload_{arm}"].append(row["offload_rate"])
        for k in ("degree", "budget", "connectivity"):
            cells[level][f"viol_{k}_{arm}"].append(row["violations"].get(k, 0))
    sweep[kind] = {lvl: {k: float(np.mean(v)) for k, v in d.items()}
                   for lvl, d in cells.items()}

# ----------------------------------------------------------------------
# Figure 1: the pipeline pipeline — left as TikZ in the doc. Figures here:
# Fig A: pruner benchmark success + violations (nominal K=15)
# Fig B: binding-budget benchmark (the honest table figure)
# Fig C: budget pressure sweep success curves
# Fig D: degree + roster sweeps
# Fig E: delivery metrics (offload vs delivery-fail vs recall)
# Fig F: worked example energy landscape
# ----------------------------------------------------------------------

# --- Fig A/B: benchmark bars ------------------------------------------
def benchmark_figure(summary, title, fname, show_budget_viol=True):
    arms = [a for a in ARMS]
    x = np.arange(len(arms))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.7))
    succ = [100 * summary[a]["success"] for a in arms]
    bars = ax1.bar(x, succ, color=[COLORS[a] for a in arms])
    ax1.set_xticks(x, [LABELS[a] for a in arms], rotation=30, ha="right")
    ax1.set_ylabel("task success (%)")
    ax1.set_title(title)
    ax1.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax1.set_ylim(0, 45)

    # violations: degree / budget / connectivity as grouped bars
    w = 0.27
    deg = [summary[a].get("viol_degree", 0) for a in arms]
    bud = [summary[a].get("viol_budget", 0) for a in arms]
    con = [summary[a].get("viol_connectivity", 0) for a in arms]
    ax2.bar(x - w, deg, w, label="degree", color="#1f77b4")
    ax2.bar(x, bud, w, label="budget", color="#ff7f0e")
    ax2.bar(x + w, con, w, label="connectivity", color="#2ca02c")
    ax2.set_xticks(x, [LABELS[a] for a in arms], rotation=30, ha="right")
    ax2.set_ylabel("violations per 20-cycle run")
    ax2.set_title("constraint violations (mean count)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)


benchmark_figure(nominal, "Nominal fixed budget K=15 (budget infeasible: ladder relaxes)",
                 "fig_bench_nominal.pdf")
benchmark_figure(binding, "Binding budget (0.75 x full-graph cost, fully constrained)",
                 "fig_bench_binding.pdf")

# --- Fig C: budget pressure sweep --------------------------------------
levels = sorted(sweep["budget"].keys())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
show = ["qubo", "threshold", "topk", "threshold_repair", "topk_repair"]
for arm in show:
    ys = [100 * sweep["budget"][l][f"succ_{arm}"] for l in levels]
    style = dict(color=COLORS[arm], marker="o", lw=2 if arm == "qubo" else 1.2)
    if arm in ("threshold", "topk"):
        style["ls"] = "--"
    ax1.plot(levels, ys, label=LABELS[arm], **style)
ax1.set_xlabel("budget factor  (fraction of full-graph cost allowed)")
ax1.set_ylabel("task success (%)")
ax1.set_title("Budget pressure sweep")
ax1.legend(fontsize=7, loc="lower left")
ax1.set_xticks(levels)

# budget violations per arm at each level (per-level aggregated means)
for arm in show:
    ys = [sweep["budget"][l].get(f"viol_budget_{arm}", [0]) and
          float(np.mean(sweep["budget"][l][f"viol_budget_{arm}"]))
          for l in levels]
    style = dict(color=COLORS[arm], marker="o", lw=2 if arm == "qubo" else 1.2)
    if arm in ("threshold", "topk"):
        style["ls"] = "--"
    ax2.plot(levels, ys, label=LABELS[arm], **style)
ax2.set_xlabel("budget factor")
ax2.set_ylabel("budget violations / run (mean)")
ax2.set_title("Did the arm actually honor the budget?")
ax2.set_xticks(levels)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_budget_sweep.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig D: degree + roster sweeps -------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
for ax, kind, xl in ((axes[0], "degree", "degree cap D"),
                     (axes[1], "roster", "mission-critical roster |R|")):
    levels_k = sorted(sweep[kind].keys())
    for arm in show:
        ys = [100 * sweep[kind][l][f"succ_{arm}"] for l in levels_k]
        style = dict(color=COLORS[arm], marker="o", lw=2 if arm == "qubo" else 1.2)
        if arm in ("threshold", "topk"):
            style["ls"] = "--"
        ax.plot(levels_k, ys, label=LABELS[arm], **style)
    ax.set_xlabel(xl)
    ax.set_ylabel("task success (%)")
    ax.set_xticks(levels_k)
axes[0].set_title("Degree pressure sweep")
axes[1].set_title("Roster pressure sweep")
axes[1].legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_degree_roster.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig E: delivery metrics (binding budget) --------------------------
fig, ax = plt.subplots(figsize=(4.6, 2.9))
x = np.arange(len(ARMS))
w = 0.27
ax.bar(x - w, [100 * binding[a]["offload"] for a in ARMS], w,
       label="offload rate (%)", color="#1f77b4")
ax.bar(x, [100 * binding[a]["delfail"] for a in ARMS], w,
       label="delivery failure (%)", color="#d62728")
ax.bar(x + w, [100 * binding[a]["recall"] for a in ARMS], w,
       label="delivery-critical recall (%)", color="#2ca02c")
ax.set_xticks(x, [LABELS[a] for a in ARMS], rotation=30, ha="right")
ax.set_ylim(0, 105)
ax.legend(fontsize=7)
ax.set_title("Offload vs delivery quality (binding budget)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_delivery.pdf"), bbox_inches="tight")
plt.close(fig)

# --- Fig F: worked example landscape -----------------------------------
we = json.load(open("worked_example.json"))
bf = we["brute_force"]
fig, ax = plt.subplots(figsize=(6.4, 2.9))
band = bf["feasible_energy_band"]
ax.axhspan(band[0], band[1], color="#2ca02c", alpha=0.12)
ax.axhline(band[0], color="#2ca02c", lw=1.5)
ax.annotate(f"best feasible  {band[0]:.3f}\n({', '.join(bf['best_feasible']['kept'])})",
            xy=(0.02, band[0]), xycoords=("axes fraction", "data"),
            fontsize=8, color="#2ca02c", va="top")
trap = bf["best_connectivity_trap"]
ax.axhline(trap["energy"], color="#d62728", lw=1.5, ls="--")
ax.annotate(f"infeasible trap  {trap['energy']:.3f}\ndisconnects rover "
            f"{trap['disconnected'][0]} — looks BETTER than optimum",
            xy=(0.98, trap["energy"]), xycoords=("axes fraction", "data"),
            fontsize=8, color="#d62728", ha="right", va="bottom")
lo = min(band[0], trap["energy"]) - 0.25
hi = 0.4
ax.set_ylim(lo, hi)
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_ylabel("QUBO energy (lower = better)")
ax.set_title("Worked example: 64 subsets, only 3 feasible — the trap a naive pruner takes")
# feasible-band members as dots
for i, r in enumerate(bf["top5_feasible"]):
    ax.plot(0.5 + 0.02 * (i - 2), r["energy"], "o", color="#2ca02c", ms=5)
    if i < 3:
        ax.annotate(f"{r['energy']:.3f}", xy=(0.5 + 0.02 * (i - 2), r["energy"]),
                    xytext=(0, -12), textcoords="offset points",
                    fontsize=7, ha="center", color="#2ca02c")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_worked_landscape.pdf"), bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# doc_data.json — every number the document will quote
# ----------------------------------------------------------------------
def pstats(rows, a, b, key="success"):
    d, lo, hi, p = paired_diff(rows, a, b, key)
    return {"diff": d, "lo": lo, "hi": hi, "p": p}


doc = {
    "worked": {
        "instance": we["instance"], "qubo": we["qubo"],
        "bf": {k: we["brute_force"][k] for k in
               ("n_subsets", "n_feasible", "n_infeasible", "n_conn_traps",
                "best_feasible", "best_degree_budget_violator",
                "best_connectivity_trap", "trap_lure", "degree_budget_barrier",
                "feasible_energy_band")},
        "annealer": we["annealer"], "milp": we["milp"],
        "cross_check": we["cross_check"],
    },
    "nominal": nominal,
    "binding": binding,
    "sweep": sweep,
    "paired": {
        "nominal_qubo_vs_threshold": pstats(load_jsonl("benchmark.jsonl"),
                                            "qubo", "threshold"),
        "nominal_qubo_vs_threshold_repair": pstats(load_jsonl("benchmark.jsonl"),
                                                   "qubo", "threshold_repair"),
        "binding_qubo_vs_threshold_repair": pstats(load_jsonl("benchmark_bf0.75.jsonl"),
                                                   "qubo", "threshold_repair"),
        "binding_qubo_vs_topk": pstats(load_jsonl("benchmark_bf0.75.jsonl"),
                                       "qubo", "topk"),
        "binding_qubo_vs_topk_repair": pstats(load_jsonl("benchmark_bf0.75.jsonl"),
                                              "qubo", "topk_repair"),
        "binding_topk_vs_threshold_repair_delfail": pstats(
            load_jsonl("benchmark_bf0.75.jsonl"), "topk", "threshold_repair",
            "delfail"),
    },
}
with open("doc_data.json", "w") as fh:
    json.dump(doc, fh, indent=1)

print("figures written to", OUT)
print("doc_data.json written")
for k in ("nominal", "binding"):
    print(k, {a: round(doc[k][a]["success"], 4) for a in ARMS})
