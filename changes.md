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

---
---

# Review 2 — Mathematical Audit (2026-09-11)

After the review-1 changes were implemented (baselines ✓, latency timing ✓, trust
Spearman ✓, faults + recovery ✓, live features ✓, readout training ✓, scale runner ✓,
hygiene ✓ — `.gitignore`, files committed, README fixed). Suite: **452 passed, 0 failed**.
This pass derives the formulation from the spec and verifies the implementation
mathematically, term by term.

## The formulation under test

For optional edges $O = E \setminus H$, hard-critical $H$, required set $R$ with root
$r$, $m = |R|-1$, quantized integer costs $c_e \ge 1$, slack/flow integers binary-encoded
($s = \sum_k 2^k b_k$):

$$
H(z) = \underbrace{\sum_{e \in O} (-U_e + \lambda)\, x_e}_{\text{utility + sparsity}}
+ \underbrace{\eta \sum_{e \in O} (x_e - x_e^{t-1})^2}_{\text{stability}}
+ \underbrace{\rho_d \sum_{i \in V} \Big( \textstyle\sum_{e \in \delta_O(i)} x_e + s_i - (D_i - \deg_H(i)) \Big)^2}_{\text{degree}}
+ \underbrace{\rho_C \Big( \textstyle\sum_{e \in O} c_e x_e + s_C - (K - c_H) \Big)^2}_{\text{budget}}
+ \underbrace{\rho_F \sum_{v \in V} (\text{in}_v - \text{out}_v - b_v)^2}_{\text{flow conservation}}
+ \underbrace{\rho_{FC} \sum_{e \in E} (f_{uv} + f_{vu} + q_e - m\, x_e)^2}_{\text{flow capacity}}
$$

with $b_r = -m$, $b_v = 1$ for $v \in R \setminus \{r\}$, $b_v = 0$ otherwise; $x_e \coloneqq 1$ for $e \in H$.

## Mathematical verification — what is CORRECT

| Claim | Method | Result |
|---|---|---|
| Q matrix ≡ spec $H(z)$ | Independent probe: 40 random instances × 8 states, variables decoded **by name**, energy re-derived from the formula above | max diff **2.3e-10** (float noise) — construction exact |
| Square-penalty expansion | Algebra: linear $2ca_i + a_i^2$, off-diagonal $2a_ia_j$ halved into symmetric Q | Correct |
| SA flip increment $\Delta E = 2\delta(Qz)_i + Q_{ii}$, $\delta = 1-2z_i$ | Derived from $E = z^TQz$ | Correct |
| Penalty dominance at **all** sizes | Per-unit argument: one unit of degree/budget violation costs $\rho k^2 \ge 100$ but unlocks ≤ 1 edge of gain ≤ 1 (since $c_e \ge 1$); one flow violation costs 500 vs gain ≤ $\deg \cdot \lambda \approx 0.5$ | Holds for any $\lvert O \rvert$ (quadratic penalty vs linear gain) |
| Bit-width sufficiency | $\lceil\log_2(\max+1)\rceil$ covers every equality-feasible value; feasible defaults exist ($x{=}0, s{=}D$) | Correct |
| `soft_energy` ≡ QUBO energy on feasible states | Test J.soft-eq + derivation | Correct |
| Spearman implementation | Average ranks (ties) + Pearson on ranks | Correct |
| Logistic-regression readout gradients | $\nabla_w = X^T(p{-}y)/n + \ell_2 w$ | Correct |
| Brute-force optimum never penalty-leaking | B.bf-feas over 38 instances | Correct |

**Verdict: the QUBO formulation and its matrix construction are mathematically sound.
No errors in the math of `main.py`'s encoding.**

## Errors found — by mathematical analysis, verified empirically

### E1 (HIGH) — `ProjectedEdgeAnnealing` cannot bootstrap under connectivity constraints

`_random_feasible_start` grows a set edge-by-edge, keeping only candidates that are
already feasible. That assumes feasibility is **prefix-monotone** — true for
degree/budget (empty set is feasible) but **false for connectivity** (no single edge
connects a chain). Verified on a 4-node chain, all nodes required:

- `_random_feasible_start()` returns `[]` → `connectivity_missing=[1, 2, 3]`
- `run()` with no warm start raises *"No feasible edge subset exists"* **even though
  the full chain is feasible**
- with a warm start, 4/4 random restarts attempted, 0 usable — **restarts are dead
  weight** whenever connectivity is on; every result rides on the single warm start

Consequences (each verified):

1. **`simulation._prune` silently drops connectivity at the worst moment.** When the
   warm set (previous kept ∩ current edges) no longer connects the root's component —
   i.e. right after a topology change, the paper's core scenario — attempts 0–1
   (both with `required_nodes`) always raise and the code falls through to
   budget-only pruning. The "pruning never disconnects the mission" guarantee lapses
   exactly when the graph is dynamic.
2. **`scale_runner` never enforces connectivity at any size.** Its warm start
   (`set(edges)`) is infeasible under degree caps (k-NN graphs give every node
   degree ≥ 4, cap = 4), so the connectivity attempt always raises and silently falls
   back. Proof from the printed table's own variable counts: n=10 → 24 edge vars +
   30 degree bits + 7 budget bits = 61 (exact, **zero flow variables**); n=20 →
   54+60+8 = 122 (exact). The docstring's "connectivity enabled for n ≤ 20" is
   false in practice.

**Fix:** seed random starts with a random spanning tree of the required component
(hard edges ∪ random BFS/DFS tree from the root, rejected if degree/budget caps are
violated, a few retries). $O(V+E)$ per attempt. This restores restart diversity,
field degradation, and the scale runner's stated design.

### E2 (MEDIUM) — `scale_runner` greedy baseline is infeasible at every size

Greedy top-K ignores degree caps and budget. Measured (scale_runner's own instances):
degree violations at 2–10 nodes per size, budget overruns 91>72, 155>144, 244>215,
450>386. So `E_greedy` is the soft energy of an **unachievable** edge set compared
against the QUBO's feasible one. (The QUBO wins anyway — fixing this only
strengthens the result.) **Fix:** repair greedy (drop worst-violating edges until
feasible) or report an infeasibility flag in the table.

### E3 (MEDIUM) — "neal-equivalent backend" is a label, not an integration

Battle test H now says *“QUBO bit-sampler on Q (neal-equivalent backend)”* but
`dwave-neal` is not integrated; the sampler is still the local bit-flip SA, which the
same test shows is 52–56% exact vs the projected annealer's 100%. Review-1 P0-3
(“make the solved-as-QUBO claim real”) remains open. Either wire in `dwave-neal`
(one pip dependency, drop-in for `SimulatedAnnealing`) or revert the label.

### E4 (MEDIUM) — latency numbers at small n are measurement noise

The benchmark's own output exposes it: for the `full` pruner, `gnn` (0.31 ms) and
`gnn_full` (0.22 ms) are **identical work**, ±30% apart from single-shot
`perf_counter` calls at sub-millisecond scale. All gnn-column differences between
pruners (0.28–0.31) are within noise. And the honest headline at 10 rovers:
**QUBO spends ~134 ms solving to save ~0.05 ms of GNN inference** — the end-to-end
latency claim is not supported at this scale.
**Fix:** report median/min of ~50 repeats (or an op-count proxy: edges × layers),
and move the latency claim to where it can be true — larger fleets, or wall-clock on
the actual Pi. The cloud-amortization story (solve offline, apply lightweight policy)
is the other honest route and is still unimplemented (review-1 P1-8).

### Minor notes

- `score_from_hidden` multiplies the **logit** by $(0.5 + 0.5\,\text{trust})$ for
  critical tasks; for negative logits this *raises* the score — sign interaction
  contrary to intent (gated nodes mostly have positive logits, so impact is small).
- Readout training is off-policy: samples only from assigned nodes → selection bias;
  the learned compute weight (−0.33, "more available compute → less success") is a
  confounding artifact worth a footnote.
- `FaultEvent(kind="reliability_drop", node=None)` is accepted but silently does
  nothing (`f.node == n` never true); `PRUNE_TO_SUSPECT` is a dead constant.
- `recovery_times` reports "recovered after `duration`" even for faults that never
  affected success (baseline met immediately after expiry).
- `gnn_full` timing runs inside every production cycle — measurement overhead in the
  loop it measures; fine for now, worth gating behind a debug flag later.
- Benchmark arms at 3 seeds × 20 cycles are within noise of each other
  (65.0–67.8%); the paper table needs ≥10 seeds or paired-per-cycle statistics.

## Review-2 bottom line

The **mathematics of the formulation is correct and now independently proven** —
that box is closed. What the audit exposed is one real algorithmic bug (E1) whose
effect is that the flagship connectivity guarantee silently deactivates in exactly
the dynamic scenarios the research targets, plus a baseline-fairness bug (E2) in the
new evidence harness. Both are small, localized fixes (a spanning-tree start; a
feasibility repair). E3/E4 are honesty items about claims, not correctness.

---
---

# Review 2 — Resolution (2026-09-11)

All items addressed; suite **459 passed, 0 failed** (incl. real `dwave-neal` backend).

- **E1 (fixed, verified):** `ProjectedEdgeAnnealing` random starts are now seeded with
  a degree-cap-aware random spanning tree (Prim-style, residual caps). Chain graph with
  no warm start now solves (regression L1). Consequences visible and intended: the
  simulation's connectivity constraint is now actually active (solve cost rose
  0.13s → 1.76s/cycle — the flagship guarantee's real price), and `scale_runner`
  enforces connectivity at n=10/20 (427/1076 vars, 12.5s/41s).
- **E2 (fixed):** greedy baseline repaired to feasibility (`repair_to_feasible`);
  connectivity-infeasible cases are flagged as `nan`. As predicted, fixing this
  strengthens the QUBO result: E_proj vs E_greedy is now −36.7 vs −24.3 at n=50
  (was −31.4 vs −30.2 against the infeasible greedy).
- **E3 (fixed):** real `dwave-neal` integrated (optional import) and benchmarked on
  the same Q matrices: neal 60% exact / mean gap 0.083 vs projected annealer 100% /
  0.000 (local bit-SA: 52–56% / 0.145). The ordering predicted in review-1 holds.
- **E4 (fixed):** GNN timings are now medians of 25 repeats; `gnn_full` measurement
  gated behind `measure_full_gnn`; benchmark runs 10 seeds. Benchmark table (10 rovers,
  10 seeds) confirms review-2's suspicion: success across pruners is within noise
  (63.8–65.2%) — at this scale the pruners differ in solve cost, not task success.
- **Minors:** criticality now scales the probability, not the logit (L3);
  `reliability_drop` with `node=None` applies globally (L4); dead constant removed.
  Selection-bias footnote for readout training acknowledged (left as future work).

---
---

# Review 3 — Constraint pressure, repaired baselines, and metric audit (2026-09-15)

Motivated by an external review of the paper: (a) the "guarantees" defense of the
QUBO pruner is never stress-tested because no experiment makes the constraints
matter; (b) constraint-blind baselines make any violations table a straw man;
(c) offload rate and pruning accuracy are missing metrics; (d) trust metrics are
selection-confounded across pruners. Suite: battle_test simulator/GNN/solver
sections re-run green (48 checks across G/I/K/L); full-suite run cut short only
by the 10-min tool budget (heavy CBC sections unchanged).

## Code changes

- **`simulation.py`**
  - New pruner arms `threshold_repair` and `topk_repair`: same greedy prune,
    then the same feasibility repair the QUBO path gets (degree caps, budget,
    mission-critical connectivity). Infeasible-budget repair falls back to a
    minimum-cost spanning tree (Kruskal) with residual violations reported
    honestly rather than collapsing to the empty graph.
  - New metrics in `summary()`: `offload_rate` (assigned slots / task slots),
    `delivery_fail_rate` (failed deliveries / total deliveries), and
    delivery-critical pruning precision/recall per cycle. Ground truth is a
    per-cycle counterfactual: an edge is delivery-critical iff removing it
    from the full candidate graph strictly reduces the widest-path delivery
    bottleneck of any mission-critical node.
  - New knobs: `degree_cap` (was hardcoded 3) and `budget_factor` (budget as a
    fraction of full-graph cost, so pressure is comparable across cycles).
  - Repair time now included in reported solve time.
- **`constraint_pressure.py`** (new): budget (β ∈ {0.35,0.5,0.75,1.0}),
  degree (D ∈ {2,3,4}), roster (|R| ∈ {3,5,8}) sweeps × 6 arms × 16 seeds ×
  20 cycles, resumable via `sweep_*.jsonl`; paired bootstrap CIs and
  permutation p-values per cell.
- **`benchmark_full.py`** (new): the paper's Table-2 benchmark at 16 seeds with
  all 6 arms, sequential for clean timings, resumable via `benchmark.jsonl`;
  supports both the nominal fixed K=15 and binding budget_factor modes.
- **`analyze_results.py`** (new): extracts paper-ready numbers from the JSONLs.

## Findings (all in the paper now)

1. **Audit caught the audit:** at the nominal fixed budget K=15 (10 rovers),
   the budget is below the cost of any connected subgraph — the QUBO's
   relaxation ladder silently used no-budget mode in 319/320 cycles. The old
   Table 2's QUBO row was effectively unconstrained. All constraint-relevant
   comparisons re-run at a binding, feasible budget (K = 0.75× full-graph
   cost; 320/320 full-constraint mode, zero violations).
2. **No success crossover:** across ten pressure levels the QUBO never beats
   the best repaired baseline on success. It ties at extreme budget pressure
   (β=0.35: −0.1pp, p=1.0) and loses 2–4pp elsewhere (significant at
   β ∈ {0.5, 0.75, 1.0} and roster 8). The honest value proposition is the
   guarantee itself (zero degree violations at every level; constraint-blind
   threshold violates budget 20/20 cycles at β=0.35), not success.
3. **Baseline fairness fixed:** vs constraint-aware repaired baselines the
   QUBO's success deficit is −4.1/−4.2pp (p=0.004/0.017) — the earlier
   −4.3pp gap against blind pruners overstated the gap by ~2pp.
4. **New metrics:** QUBO offload ≈100% vs 89.8% for top-K (which strands
   tasks by disconnecting mission nodes), but delivery-failure rate 4.4pp
   worse than threshold+repair (CI [+1.4,+8.0]) and 11.8pp worse than top-K —
   the soft objective misranks delivery-critical edges (recall 0.67 vs
   0.82–0.93).
5. **Trust-quality claim softened:** at 16 seeds the QUBO arm has the best
   Spearman (0.295 at binding budget) but not the best AUC (0.681 vs 0.683);
   and all cross-pruner trust comparisons are selection-confounded (each
   pruner generates its own observation stream) — flagged in the paper.
