# Project Review & Recommended Changes

**Date:** 2026-09-11
**Sources:** design conversation (ChatGPT "Explain Rover GNNs"), full code audit of
`main.py`, `trust_gnn.py`, `simulation.py`, `battle_test.py`, `example.py`, README.
**Verified by running:** `example.py` (feasible, energy −2.115), `simulation.py`
(75% task success), `battle_test.py` (**434 passed, 0 failed**).

---

## Verdict

The QUBO layer is in genuinely good shape — arguably past research-code grade. The
project's remaining risk is **not** the optimizer; it is that the evidence needed to
defend the research claim (pruning baselines, latency, trust accuracy, scale) does not
exist yet. The repo currently proves *the formulation is correct*; it does not yet
prove *the formulation earns its place*.

## What is already solved — do not re-litigate

| Concern from the design discussion | Status |
|---|---|
| Trust circularity (QUBO using same-cycle GNN trust) | Solved — sim uses `trust.get()` before updates, i.e. `T^(t-1)` |
| Slack variables not QUBO-binary | Solved — binary-encoded integer bits (`add_integer_bits`) |
| Critical edges removable by optimizer | Solved — hard-fixed outside the variable set |
| Connectivity as hidden heuristic | Solved — single-commodity flow constraints inside the QUBO |
| Penalty hierarchy (hard ≫ soft) | Solved — 500/100 vs ~0.15 soft terms, verified by brute force |
| Graph oscillation | Partially solved — stability term + hysteresis (battle test G3) |
| Infeasible connectivity crashing the solver | Solved — constructor raises early, checked by tests |
| Deterministic tie-breaking in assignment | Solved — `assign_tasks` sorts on (score, task) |
| SA returning infeasible solutions | Solved — `ProjectedEdgeAnnealing` projects to feasible states; bit-level SA kept only as diagnostic (100% vs 52–56% exact rate, correctly documented) |
| QUBO energy correctness | Proven — independent re-derivation, brute-force cross-check, analytic optima (tests A/B/C) |

## Changes

### P0 — Research-critical (the paper stands or falls on these)

1. **Add the pruning baselines — the "killer comparison."**
   The design conversation settled this repeatedly: without it there is no evidence
   QUBO beats simpler pruning. Add to `simulation.py` three alternative pruners and
   run identical scenarios through each:
   - full graph (no pruning) → GNN
   - threshold pruning (keep edge iff `utility > τ`) → GNN
   - greedy top-K by utility → GNN
   Report the same metrics as the QUBO path. QUBO must win on the
   quality-vs-cost tradeoff (or the paper must honestly say it doesn't).

2. **Measure inference latency — the central claim metric.**
   The whole point of pruning is cheaper GNN inference, and nothing times anything.
   Add per-cycle timing of: QUBO build time, QUBO solve time, GNN inference time on
   pruned vs full graph, assignment time. A wall-clock table
   (full-graph vs threshold vs greedy vs QUBO) is the paper's headline result.

3. **Make the "solved as QUBO" claim real.**
   The production solver (`ProjectedEdgeAnnealing`) never touches the `Q` matrix —
   it anneals over feasible edge sets using `soft_energy`. The QUBO is currently a
   *specification*, not a *method*. Wire in a real QUBO backend
   (`dwave-neal` was already suggested; D-Wave Leap for real annealing later), report
   its optimality gap vs `ProjectedEdgeAnnealing` on the same instances, and keep the
   projected annealer as the strong classical baseline. Without this, a reviewer's
   "what is quantum-inspired here?" has no good answer.

4. **Measure trust quality against ground truth.**
   `RoverState.reliability` already encodes true per-rover reliability and is never
   compared to learned trust. Trivial to add and high-value: Spearman/Kendall
   correlation of `trust.get(n)` vs `reliability` per cycle, or AUC for
   "reliability > 0.5" classification. This is the trust-F1/AUC metric from the
   proposal, nearly free.

### P1 — Fidelity to the designed architecture

5. **Exercise the connectivity constraint in the end-to-end sim.**
   `simulation.py` never passes `required_nodes`/`root`, so the formulation's
   flagship guarantee (pruning can never disconnect the mission) is untested in the
   loop where it matters. Pass the mission-critical subset (or all rovers) and show
   a fault scenario where a low-quality bridge edge is *retained*.

6. **Add fault-injection scenarios and recovery-time metric.**
   The proposal defines the sequence: normal motion → range loss → packet-loss
   spike → low battery → sensor fault → recovery. None exist. Implement as scripted
   per-cycle disturbances on `RoverState`; measure recovery time = cycles until
   task-success rate / trust returns to pre-fault baseline. This is also the demo
   story for the physical testbed.

7. **Fix the dead and noisy features.**
   - `task_criticality` is constant `0.5` for every edge (dead feature — weight
     `w_task=0.15` multiplies a constant). Derive it from active task assignments,
     e.g. edges on the routing path of a critical task.
   - `freshness` is random `U(0.7, 1.0)` per cycle, not actual link age. Track
     last-observation timestamps and use `A_e = exp(-Δt/τ)` as designed.

8. **Model the Cloud/Fog split — at least measure it.**
   The architecture (and the chat's feasibility verdict) depends on expensive QUBO
   solving living in the Cloud while the Fog runs lightweight decisions. Currently
   the QUBO is solved synchronously every cycle inline. Minimum viable version:
   report QUBO solve time vs GNN inference time per cycle and discuss. Real version:
   amortize — solve offline/periodically ("cloud"), let the fog apply a cheap policy
   between solves.

9. **The "GNN" is currently a fixed-weight heuristic — decide which story to tell.**
   `TrustGNN` uses hand-set identity-scaled weights, one relation type, no training
   loop. Options:
   - **Train it:** even a small supervised step (fit `GNNWeights.out` / matrices
     against task outcomes in the "cloud" tier) makes "GNN" defensible; or
   - implement the planned PyTorch Geometric HGNN with heterogeneous relations
     (communication / proximity / task) and attention, cloud-trained, fog-evaluated; or
   - rename honestly (trust-weighted message-passing scorer) and keep the HGNN as
     future work.
   The current middle ground — calling a fixed-weight scorer a GNN — is the most
   attackable spot in the writeup.

### P2 — Scale & robustness experiments

10. **Large-graph experiments.**
    The standing conclusion from the design work: 3–4 rovers (≤6 edges) cannot
    justify QUBO; the contribution must be shown at 20/50/100/500 nodes. The battle
    suite reaches 16 nodes / 817 vars; add a scaling runner (edge-count, solve time,
    optimality gap vs greedy, GNN latency) as the QUBO's actual evidence base. The
    physical 3–4 rover demo then validates the *system*, not the *optimizer*.

11. **Edge lifecycle states (nice-to-have from the design discussion).**
    ACTIVE → SUSPECT → PROBATION → REMOVED instead of hard keep/prune, so pruning
    can't permanently blind the trust model to a temporarily-degraded rover.
    The `relax()` rehabilitation in `TrustModel` covers nodes; edges have no
    equivalent.

### P3 — Hygiene

12. **README run instructions are wrong.** `uv pip sync requirements.txt` — no
    `requirements.txt` exists. `uv sync` (or `uv run …`) alone is correct since
    `pyproject.toml` + `uv.lock` are present.
13. **Commit the three core modules.** `trust_gnn.py`, `simulation.py`,
    `battle_test.py` are untracked; the committed README references files that don't
    exist in the repo. Remove the committed `__pycache__/` and add a `.gitignore`.
14. **Latent bug — `trust_gnn._adjacency_weights`:** `edge_features.get(edge)`
    should look up `canonical_edge(edge)`; a non-canonical key silently falls back
    to weight 1.0 instead of using link evidence. (Current callers happen to pass
    canonical edges, so it's latent.)
15. **GNN scoring cost note:** `simulation.step` runs the full GNN once per
    (task, node) pair — O(tasks × nodes) passes. Fine at 6 rovers, but flatten to
    one pass per task before the latency benchmark, or the timing will measure the
    harness, not the model.

---

## Suggested order of attack

1. P0-2 (latency timing) and P0-1 (baselines) together — one harness, the headline table.
2. P0-4 (trust vs ground truth) — small, high value.
3. P1-5/6/7 (connectivity in sim, faults, real features) — makes the demo real.
4. P0-3 (dwave-neal backend) and P1-9 (train or rename the GNN) — the two claim-shaping decisions.
5. P2-10 (scale runner) — the QUBO's evidence base.
6. P3 in one cleanup commit.
