# AGENTS.md: Complete Working Context for the quantum-gnn Repository

This file gives an incoming agent everything needed to work on this project
without re-discovering it. Read it fully before editing. For a
human-friendly narrative version, see `PROJECT_GUIDE.md`.

---

## 1. What this project is

Research codebase + LaTeX paper: "trust-aware GNN task allocation with
constrained QUBO edge pruning for dynamic rover networks".
GitHub remote: `ppmpreetham/quantum-gnn`.

Pipeline (per simulation cycle):

```
full dynamic graph
  -> EdgeSelectionQUBO (prune links, main.py)
  -> TrustGNN (score rovers on the pruned graph, trust_gnn.py)
  -> assign_tasks (greedy capacity-aware matching)
  -> outcomes sampled from hidden ground-truth reliability
  -> TrustModel update (feeds next cycle's features)
```

The project's stance is an **audit, not just a proposal**. Negative results
are reported deliberately and must stay in the paper. Do NOT tune numbers to
look better; the user has repeatedly endorsed the honest framing. When the
user says "make it better than other models", they mean make the
*comparison* rigorous, never game the results.

---

## 2. Environment facts (critical, cost us real time)

- OS: Linux, bash. No sandbox. Working dir is the project root.
- **System Python is 3.14** with numpy only. `python3 battle_test.py` runs,
  but sections H (neal/tabu/SQA) and M (MILP) need extra packages.
- **scipy is broken on this Python 3.14** (missing bundled OpenBLAS). That
  is exactly why the MILP baseline uses PuLP/CBC. Do not reintroduce scipy.
- **pip is PEP-668 blocked** system-wide. Always use a venv:
  ```bash
  python3 -m venv /tmp/gnnenv
  /tmp/gnnenv/bin/pip install numpy pulp dwave-neal dwave-tabu openjij
  /tmp/gnnenv/bin/python -u battle_test.py   # full suite, 15-20 min
  ```
- **`/tmp` gets wiped between days.** The venv and any downloaded tools
  there vanish. Recreate as above if missing. Check first:
  `ls /tmp/gnnenv/bin/python`.
- **`uv` is NOT installed** despite what README.md says. Use `python3`
  directly. `pyproject.toml` only declares `numpy>=2.5.3`.
- **No system TeX.** Build the paper with tectonic:
  ```bash
  cd paper && /tmp/tectonic main.tex
  ```
  Redownload tectonic 0.15.0 from GitHub releases if `/tmp/tectonic` is
  gone. The paper no longer uses the fiziko testbed figure, so tectonic
  alone suffices (`paper/build.sh` wraps it). The old
  `paper/fig_testbed.{tex,pdf}` files are kept but unused.

---

## 3. Repository map

| File | Lines | Role |
|---|---|---|
| `main.py` | 1240 | QUBO formulation + both annealers |
| `trust_gnn.py` | 355 | TrustGNN, TrustModel, assign_tasks, readout training |
| `simulation.py` | 941 | Causal end-to-end simulator, baselines, stats, CLI |
| `milp_baseline.py` | 169 | Exact MILP + exact linearized-QUBO solves (PuLP/CBC) |
| `scale_runner.py` | 181 | Fleet-size scaling experiments |
| `battle_test.py` | 1418 | 491-check hostile test suite, sections A-N |
| `example.py` | 92 | Minimal 4-rover worked example |
| `paper/main.tex` | 662 | The paper (article class, TikZ/pgfplots, booktabs) |
| `paper/fig_testbed.{tex,pdf}` | - | old fiziko testbed figure, no longer used by the paper |
| `paper/figs_doc/*.pdf` | - | matplotlib result figures (`make_figures.py`); the paper uses `fig_budget_sweep.pdf` |
| `paper/build.sh` | - | Paper build script |
| `changes.md` | - | Log of external review rounds and responses |
| `PROJECT_GUIDE.md` | - | Plain-language full explainer for humans |

`.gitignore`: `__pycache__/`, `*.pyc`, `.venv/`, `paper/fiziko/`.
`stray file`: `texput.log` (leftover, harmless).

---

### Web console (added 2026-10-16)

| Path | Role |
|---|---|
| `shared/fleet-config.json` | Param specs (scope: session/planner/client), defaults, calibrated presets, comparison catalogue + paper reference numbers. Single source for TS and Python. |
| `web/fleet_session.py` | `LiveSimulation(RoverSimulation)` with live positions and in-place tuning; `PlannerSession` (primary + 3 paired assignment followers sharing the primary's kept edges + optional solver baseline on the same Q); `SessionStore`. |
| `web/server.py` | stdlib HTTP: `GET /api/health`, `POST /api/step`. 400 bad input, 429 sessions full, 503 busy. Optional `tasks: [{id, x, z}]` (user-placed tasks; a list, even empty, replaces the planner's own tasks for that cycle). Port via `--port`; `npm run dev` passes `FLEET_API_PORT` (default 8765). |
| `web/showcase_scenarios.py` | Scripted "where we win" scenarios (params, per-cycle positions, faults). Replays them through `PlannerSession`, keeps only wins, writes `shared/showcase-scenarios.json` with the measured per-cycle trace. |
| `src/core/*.ts` | Pure TS, Node-testable (explicit `.ts` imports): kinematics, arena/spawn, graph (widest-path tree), `FleetEngine`, stats, `fleetView.ts` (the `FleetView` interface every renderer reads). |
| `src/modes/` | `modes.ts` is the single mode registry (console, photoreal, showcase; `mission: live | scripted`); `ModeHost` lazy-loads non-default modes; live modes share one `LiveMission`. |
| `src/modes/photoreal/` | Photoreal field site: `AssetLibrary` (GLTF + KTX2 + Meshopt, cached promises for React `use()`, disposes on exit), scanned terrain shader, instanced props, HDRI or physical sky with the sun measured from the HDRI, PCF shadows, N8AO/bloom/grain by tier, environment panel. |
| `src/modes/showcase/` | Split-screen replay of the scenarios (two scenes, one camera, scissor), `compare.ts` turns each response into marks and a headline, `ScriptedFleet.ts` implements `FleetView` for scripted motion. |
| `src/scene/` | R3F scene: `SceneCanvas` (tiers, perf, context loss), `FleetLayer`/`LiveRig`, `FleetMeshes` (instanced, LOD + shadow proxy), `LinkArcs`, `GroundDecals`, `Beacons`, `TaskPicker` (click to place), `CameraRig`, `Effects` tiers. |
| `src/ui/`, `src/state/` | HUD panels, minimap, roster radars; zustand store; engine/poses/view-bus context (`SceneProvider` for renderers, `FleetProvider` for the live mission). |
| `public/photoreal/` | 35 MB of processed CC0 / NASA assets, `manifest.json`, `CREDITS.md`. Rebuild with `scripts/photoreal-assets` (needs the KTX-Software `ktx` CLI). |
| `tests/*.test.ts`, `tests/test_web.py`, `tests/test_showcase.py` | `npm test` runs all. `FLEET_FULL_SHOWCASE=1` replays every showcase cycle (about 2 min). |

`simulation.py` gained additive, inspection-only hooks used by the console
(on top of the upstream `degree_cap` / `budget_factor` / repair-pruner work):
an `annealer_options` kwarg (defaults reproduce restarts=2, steps=2,
cooling=0.995), plus `last_prune`, `last_outcomes`, `last_path_factor`.
Results are byte-identical to the version without the hooks for the same
seeds, including the repair pruners and `budget_factor`.

Modes: console (default, unchanged), photoreal (same live mission on a
desert test site) and "where we win" (scripted replays, ours left, baseline
right, same positions/faults/seed through the paired followers or the
same-Q solver baseline). Showcase seeds were picked because we win on them;
the UI says so and keeps the MILP/threshold footnote. Never hand-edit
`shared/showcase-scenarios.json`: regenerate it with the script, and the
tests check the replay reproduces it.

Console rules: the comparison menu lists only baselines our pipeline beats
(trust-only, random, no-graph marked not significant; bit-SA, neal, tabu,
SQA). MILP and the threshold pruner beat us and are named in a footnote,
not hidden. `ProjectedEdgeAnnealing` counts the warm start as a restart, so
`restarts=1` with an infeasible warm start never finds a feasible start;
the console's minimum is 2.

## 4. Code architecture details

### main.py

- `EdgeFeatures` (frozen dataclass): trust, link_quality, battery,
  proximity, freshness, task_criticality, processing_cost; all in [0, 1].
- `QUBOConfig` (frozen): utility weights `w_*` (validated to sum to 1);
  soft terms `edge_penalty=0.08`, `stability_penalty=0.15`; hard penalties
  `degree_penalty=100`, `budget_penalty=100`, `flow_penalty=500`,
  `flow_capacity_penalty=500` (must dominate soft terms: breaking a
  constraint costs >= 100 while one edge is worth <= ~1); `cost_scale=10`
  quantizes processing_cost into integer budget units; SA params.
- `EdgeSelectionQUBO.__init__(nodes, edges, features, previous_decisions,
  hard_critical_edges, required_nodes, root, degree_limits, global_budget,
  config)`. Builds variables: one binary per optional edge, integer slack
  bits for degree/budget, binary flow bits per directed arc plus flow
  capacity slack bits per edge. Connectivity = single-commodity flow from
  `root` to all `required_nodes`. Key methods: `get_qubo() -> (Q, constant,
  var_names)`, `soft_energy(selected)`, `energy(z)`, `selected_edges(z)`,
  `check_constraints(z) -> List[str]`, `build_feasible_solution(edges) ->
  z` (validates feasibility, populates flow), `extract_solution(z) ->
  QUBOResult`.
- `QUBOResult`: energy, kept_edges, pruned_edges, solution, variable_names,
  violated_constraints.
- `SimulatedAnnealing(Q, constant, seed, sweeps_per_temperature, ...,
  initial_states, pair_probability)`: bit-level SA over raw bit strings,
  has `pair_probability` two-bit moves and an `energy_evals` counter.
  Known result: cannot cross penalty barriers; random restarts returned
  infeasible states in 19/25 instances. Used in the H benchmark only.
- `ProjectedEdgeAnnealing(builder, *, initial_temperature=1.0,
  final_temperature=1e-2, cooling_rate=0.998, steps_per_temperature=4,
  swap_probability=0.5, restarts=5, seed=42, initial_edge_sets)`:
  **the production solver**. Moves in feasible edge-set space: toggle one
  optional edge or swap two; candidates repaired/projected via
  `build_feasible_solution` (exact slack + spanning-tree flow recompute);
  unprojectable moves rejected. Restarts seeded by degree-cap-aware random
  spanning trees (`_random_spanning_tree`) because greedy growth cannot
  cross infeasible prefix states. Deterministic `_polish` alternates
  toggle and swap passes to a fixpoint. Tracks `energy_evals`.

### trust_gnn.py

- `NodeFeatures`: trust, battery, available compute (1 - load), capability.
  `N_FEATURES = 4`.
- `TaskRequirements`, `GNNWeights` (defaults implement: prefer high trust,
  battery, low load, strong neighborhood).
- `TrustGNN(weights=None)`: `_adjacency_weights`, `hidden_states` (one
  message-passing round, sigmoid), `score_from_hidden`, `scores`,
  `scores_for_tasks`. Runs only on the pruned graph. Pure NumPy by design
  (fog tier = Raspberry Pi; training is cloud tier only).
- `assign_tasks(...)`: capacity-aware greedy.
- `train_out_weights(...)`: cloud readout training. NULL RESULT (p=0.49,
  -3.9pp vs fixed weights); reported as such in the paper.
- `TrustModel`: EWMA per-node trust with aging; `get`, `relax`, `update`.

### simulation.py

- Free functions: `edge_utility` (shared with baselines so comparisons are
  fair), `spearman`, `auc`, `_timed_median_ms`, `paired_bootstrap_ci`
  (n_boot=10000), `permutation_pvalue` (n_perm=20000).
- `RoverState`, `FaultEvent`, `EdgeLifecycleTracker` (probation boost for
  re-added edges).
- `RoverSimulation(...)` key args: `n_rovers=6, tasks_per_cycle=2,
  comm_range=4.0, steps=10, seed=7, assigner in {gnn, trust_only, nograph,
  random}, pruner in {qubo, full, threshold, topk}, threshold_tau=0.4,
  topk_fraction=0.5, exploration=0.15, freshness_tau=3.0, packet_loss=0.05,
  faults, gnn_weights, collect_training, solve_interval=1 (QUBO every k
  cycles, cloud amortization), n_mission_critical=3,
  reliability_drift=0.0 (hidden random walk), qubo_budget=15`.
- Hidden ground truth: spatially correlated reliability field; outcomes are
  sampled from it, so mission success CAUSALLY depends on pruned topology
  (`_path_factors` maps kept topology to delivery success). This was a
  review-2 fix; do not regress it to a correlational model.
- CLI: `python3 simulation.py [benchmark|ablate|staleness|train]`, default
  single run with per-cycle log.
- `run_benchmark` (10 rovers, 20 steps, 10 seeds): pruner comparison with
  real per-stage timing (heuristics measured in microseconds, reported as
  `<0.01` ms in the paper, not 0).
- `run_ablation`: arms gnn / nograph / trust_only / random with paired
  bootstrap CI + permutation p.
- `run_staleness`: staleness margin experiment.
- `train_and_evaluate`: held-out eval of trained vs fixed readout.

### milp_baseline.py

- `solve_milp(...)`: exact MILP of the original constrained problem with
  CBC via PuLP. Single-commodity flow relaxation is integral for integer
  demands, so continuous flows are exact. Verified equal to brute-force
  enumeration on small instances.
- `solve_qubo_exact(Q, constant, time_limit)`: exact QUBO min via standard
  linearization (one aux var per nonzero quadratic term), CBC. **Raises
  ValueError unless CBC proves Optimality**; callers catch and skip.

### battle_test.py (491 checks, sections A-N)

- Harness: `check(name, condition, detail)`, `expect_raises`; final summary
  line `==== N passed, M failed ====`; exit code 1 on any failure.
- Helpers: `independent_energy` (independent re-derivation of the QUBO
  energy used to cross-verify the Q matrix), `brute_force`,
  `exact_edge_enumeration` (enumerates all 2^k optional-edge subsets with
  feasibility check; the small-instance ground truth),
  `random_connected_instance`, `fast_config(seed, restarts=6)`,
  `norm_gap(h_sa, h_star)` implementing `(H_SA - H*) / max(1, |H*|)`,
  `soft_lower_bound`.
- Sections: A construction invariants; B SA vs brute force; C analytic
  optima; D randomized scenario matrix; E validation/edge cases; F scaling
  sanity; G targeted domain scenarios; **H QUBO solver benchmark on the
  actual Q matrix** (bit-SA 56%, neal 60%, tabu 64%, SQA 48% exact; exact
  CBC linearized solve is the ground truth, cross-checked against
  enumeration only for Q<=60 vars, 5 instances; ProjectedEdgeAnnealing kept
  OUT of the scientific comparison, reported as engineering note with a
  90% exactness floor via `check("H.projected", ...)`); I Trust-GNN layer;
  J optimality-gap benchmark (small: exact H* by enumeration, 100% exact,
  enforced >= 90%; large: H* by MILP); K simulation harness; L review-2
  regressions; **M MILP ground truth** (projected annealer 95% exact, mean
  true gap 7e-5, floor check `M.exact-rate` >= 0.8); **N sensitivity
  sweeps** (temperatures, cooling, restarts, penalty scales,
  pair_probability; 87% exact across all perturbations).
- `HAVE_NEAL` gates optional dwave-neal use; tabu/openjij imports are
  try/except. Missing optional deps skip, not fail.

---

## 5. Verified numbers (as committed, do not regress)

- Solver audit (H): bit-SA 56%, neal 60%, tabu 64%, SQA 48% exact on 25
  instances with enumerated ground truth; random bit-SA restarts infeasible
  in 19/25.
- Production projected annealer: 100% exact on small instances vs
  enumeration; 95% exact vs MILP on 20 larger instances (mean true gap
  7e-5, worst 1.4e-3); 87% exact under all sensitivity perturbations;
  ~2.4e4 energy evals, ~2.1 s per solve.
- MILP (CBC): exact, ~100x faster (11-15 ms vs ~2 s). This DOMINATES the
  QUBO on classical hardware and the paper says so.
- Benchmark J: small instances 100% exact; large (up to 817 binaries) 6.9%
  mean gap vs best-of-4 annealer seeds (BKS).
- Ablation (10-rover): gnn 39.8%, nograph 38.1%, trust_only 31.7%, random
  26.0%. gnn - trust_only = +8.13pp, bootstrap 95% CI [+4.2, +11.7],
  permutation p=0.016. gnn - random = +13.8pp. gnn - nograph = +1.7pp,
  NOT significant (reported).
- Pruner table: QUBO 33.8% success vs threshold 38.2% (p=0.028): the QUBO
  pruner LOSES on raw success at 10-rover scale. Deliberate negative
  result; the paper explains it (redundancy helps bottleneck delivery).
- Staleness margin: +13.75pp.
- scale_runner: connectivity enforced only n<=20 (dense Q memory); n=10
  greedy repair infeasible (nan, reported not hidden).

---

## 6. Paper conventions (paper/main.tex)

- article 11pt, A4, geometry 2.4cm, amsmath, booktabs, multirow, TikZ +
  pgfplots (compat 1.18), hyperref, microtype, caption.
- Title: "Constrained Edge Pruning and Trust-Aware GNN Inference for Task
  Allocation in Dynamic Rover Networks: a QUBO Formulation with a
  Solver-Level Audit". Anonymous authors.
- **No em dashes anywhere in the paper** (explicit user requirement).
- Figures are TikZ/pgfplots inline except `figs_doc/fig_budget_sweep.pdf`
  (matplotlib, committed). The fiziko testbed figure was dropped in the
  2026-10-20 rewrite. The solver bar chart has bars for SQA/bit-SA/
  neal/tabu plus `exact (CBC)` at 100 as the ground-truth reference; the
  old `projected` bar was deliberately removed (user request).
- Captions must describe the figure only; no meta-commentary like "fiziko
  rendering when available" (user request).
- Citations are real: Sabater-Sierra 2005 (trust), CBC, D-Wave tools. The
  old `trustsurvey` placeholder was replaced. Do not add placeholder cites.
- Build: `cd paper && /tmp/tectonic main.tex` (runs twice if refs change).

---

## 7. Working agreements with the user

- Workflow pattern: user brings external review docs (ChatGPT exports in
  ~/Downloads or screenshots) -> fix the legitimate items -> run tests ->
  rebuild paper if main.tex changed -> commit and push.
- Push with `git add -A && git commit && git push origin HEAD`. The user
  asks for pushes explicitly; do not push unprompted.
- Latest commit: `419e355` ("Benchmark Q matrix against serious QUBO
  solvers; fix table/caption issues"). All work is committed and pushed.
- The user sometimes asks for quick verdicts; answer those in one line.
- Full battle suite before any push: `/tmp/gnnenv/bin/python -u
  battle_test.py` (use `run_in_background`, 15-20 min). 491/491 must pass.
- Style: match existing code (NumPy, dataclasses, type hints, no comments
  unless needed). No new dependencies beyond numpy/pulp/dwave-*/openjij
  without asking. No scipy.

---

## 8. Known sharp edges / pitfalls for agents

- `EdgeSelectionQUBO` validates hard: infeasible constructor inputs raise
  (e.g. unreachable required nodes, budget below cheapest spanning tree,
  features outside [0,1]). Tests in section E enumerate these.
- Hard-constraint penalties must dominate soft terms; if you change
  `QUBOConfig` weights or penalties, section A invariants and the analytic
  checks in C will catch violations. Keep weights summing to 1.
- `solve_qubo_exact` linearization grows quadratically; only used for
  Q<=60 vars in tests. Larger exact checks go through `solve_milp`.
- The pruner "0 ms" artifact was a real bug class (heuristics timed with
  too-coarse clock); timing uses `_timed_median_ms`. Keep real measured
  costs in any new table.
- The simulation's `random_assignment` flag is redundant with
  `assigner="random"` (both kept for backward compat).
- If you touch `main.py` encoding, run at minimum sections A, B, C, M
  before the full suite.
- `changes.md` is the review log; append new review rounds there when the
  user brings them.
