# Quantum GNN: The Whole Project, Explained Simply

This file is a complete, plain-language tour of the repository. It explains
what the project is trying to do, how every piece works, what the numbers
say (including the embarrassing ones), and how to run everything yourself.

---

## 1. What is this project, in one paragraph?

Imagine a team of small rovers on another planet (or a disaster zone). They
talk to each other over cheap, short-range radios (ESP-NOW), so each rover
can only hear a few nearby teammates, and the quality of those links changes
every few seconds as the rovers move, batteries drain, and dust or terrain
gets in the way. Every control cycle, someone has to decide two things:

1. **Which communication links should we actually use?** Keeping every
   possible link is wasteful and noisy. We want a smaller, reliable
   "working graph" that still keeps every rover connected.
2. **Which rover should do which task?** Some rovers are more trustworthy
   (they have a good track record), some have more battery, some are closer
   to the task.

This project answers those two questions with a pipeline:

```
full dynamic graph
      |
      v
[ QUBO edge pruning ]   <-- picks the best small set of links
      |
      v
[ Trust-aware GNN ]     <-- scores each rover on the pruned graph
      |
      v
[ task assignment ]     <-- greedy, capacity-aware matching of tasks to rovers
      |
      v
outcomes observed --> trust scores updated --> feed into next cycle
```

The "quantum" part of the name comes from the fact that the pruning problem
is written as a **QUBO**, which is the exact input format that quantum
annealers (like D-Wave's machines) accept. The honest finding of the project
is that classical computers solve this problem better and about 100x faster,
so the QUBO is really a *future hardware encoding*, not the recommended way
to solve it today. The project is as much an **audit** of this idea as it is
a proposal.

---

## 2. Key terms in plain language

- **Graph**: a list of rovers (nodes) and the radio links between them
  (edges). "Dynamic" means it changes every cycle.
- **QUBO** (Quadratic Unconstrained Binary Optimization): a standard puzzle
  format. You have yes/no variables (one per edge: "use this link or not"),
  and a cost function with single-variable terms and pair terms. Quantum
  annealers and many classical solvers eat exactly this format.
- **Penalty encoding**: QUBOs have no built-in way to say "this is not
  allowed". So rule-breaking (e.g. disconnecting a rover) is made extremely
  expensive instead. A valid solution that breaks a rule costs at least 100
  units, while any single edge is worth at most about 1, so the solver never
  prefers breaking rules.
- **Simulated annealing (SA)**: a classic "wiggle and keep improvements"
  optimizer. Start hot (accept random changes), cool down (only accept
  improvements). Cheap, general, but not guaranteed to find the best answer.
- **MILP** (Mixed-Integer Linear Program): the classical, exact way to solve
  this kind of constrained yes/no problem. The CBC solver (free, open
  source) finds provably optimal answers.
- **GNN** (Graph Neural Network): here, a very small message-passing layer.
  Each rover's score is updated using its own features plus its neighbors'
  features. Pure NumPy, tiny matrices, because it must run on a Raspberry
  Pi-class "fog" computer.
- **Trust model**: an exponentially-weighted track record per rover. Did it
  finish its last tasks? Did its links behave as predicted? Trust goes up
  and down over time and feeds back into both the QUBO and the GNN.
- **Optimality gap**: how much worse a solver's answer is than the true best
  answer, as a fraction: `(H_SA - H*) / max(1, |H*|)`. Zero means perfect.
- **Pruner**: any method that picks which edges to keep. We compare the QUBO
  pruner against three simple baselines: keep everything, keep links above a
  quality threshold, keep the top-K best links per node.

---

## 3. File-by-file: what each piece of code does

### `main.py` (1240 lines) - the QUBO heart

- **`EdgeFeatures`**: seven numbers per link, all in [0, 1]: trust,
  link quality, battery, proximity, freshness, task criticality,
  processing cost.
- **`QUBOConfig`**: every knob. Edge utility weights (must sum to 1),
  soft-objective penalties (a small cost per edge so we prefer fewer links,
  and a stability cost so we do not churn the graph every cycle), and the
  big hard-constraint penalties (100 for degree/budget violations, 500 for
  connectivity violations) that must dominate the soft terms.
- **`EdgeSelectionQUBO`**: builds the actual Q matrix. Given the current
  graph, per-edge features, and constraints (max links per rover, total
  compute budget, and "all required rovers must stay connected"), it
  produces `E(z) = z^T Q z + c` where `z` is one binary variable per
  optional edge. Connectivity is encoded with a single-commodity flow
  relaxation: pretend we push flow from a root rover to everyone else along
  chosen edges; if the flow cannot reach someone, the penalty fires.
- **`SimulatedAnnealing`**: a bit-level SA solver over raw bit strings, with
  a `pair_probability` option (moves that flip two bits at once) and an
  `energy_evals` counter so benchmarks can report solver effort honestly.
- **`ProjectedEdgeAnnealing`**: the production solver. Instead of flipping
  raw bits (which almost always lands in rule-breaking territory that the
  penalties make radioactive), it moves directly in the space of *valid*
  edge sets: toggle one edge, or swap two edges, then an exact repair
  procedure checks slacks and recomputes a spanning-tree flow. Moves that
  cannot be repaired are rejected. Restarts are seeded with
  degree-cap-aware random spanning trees, because greedy edge-by-edge growth
  can never cross the invalid intermediate states that connectivity rules
  create.

### `trust_gnn.py` (355 lines) - the trust-aware GNN

- **Design constraint**: Mist/Fog/Cloud architecture. Training happens in
  the cloud; the fog tier (Raspberry Pi) only evaluates, so everything is
  pure NumPy with tiny matrices.
- **`TrustModel`**: per-rover dynamic trust. Updated each cycle from
  observed task outcomes and link evidence, with trust aging (old evidence
  fades).
- **`TrustGNN`**: node features are trust, battery, available compute, and
  hardware capability. One message-passing round runs **only on the pruned
  graph** (the QUBO's output is literally the GNN's input graph), then a
  readout scores each rover for a given task's requirements.
- **`assign_tasks`**: capacity-aware greedy matching of tasks to rovers.
- **`train_out_weights`**: cloud-tier training of the readout layer.
  (Spoiler: it did not beat fixed hand-set weights. See section 6.)

### `simulation.py` (941 lines) - the end-to-end world simulator

Each cycle, in causal order:

1. Rovers move (proximity changes), links are observed subject to packet
   loss (unobserved links get staler), batteries drain.
2. The dynamic candidate graph is built with fresh per-edge features.
3. A pruner picks the working graph: the QUBO pruner (production) or a
   baseline (full / threshold / top-K).
4. The GNN scores rovers on the working graph, tasks are assigned, and
   scripted faults (e.g. a rover dies at cycle 5) may be active.
5. Task outcomes are sampled from hidden ground-truth reliability, so
   mission success is *causally* affected by the pruned topology, not just
   correlated with it. Trust is updated and feeds the next cycle.

Metrics: task success rate, per-stage wall-clock latency, trust-vs-truth
Spearman correlation, topology churn, edge lifecycle states, and fault
recovery time. CLI modes: `benchmark`, `ablate`, `staleness`, `train`.
Includes paired bootstrap confidence intervals and permutation tests so
claims come with statistics, not vibes.

### `milp_baseline.py` (169 lines) - the strong classical referee

- **`solve_milp`**: solves the exact same constrained edge-selection problem
  directly as a MILP with CBC (via PuLP). No penalty encoding, no annealing.
  Because the flow relaxation is integral for integer demands, continuous
  flow variables are exact here. Verified to agree with brute-force
  enumeration on small instances.
- **`solve_qubo_exact`**: exact minimum of the Q matrix itself, found by
  standard linearization (one helper variable per quadratic term) and CBC.
  Raises unless CBC proves optimality. This is the ground-truth machine for
  the solver benchmarks.

### `scale_runner.py` (181 lines) - how big can this get?

Builds range-limited sparse graphs (k-nearest neighbors, like real ESP-NOW
rovers) at growing fleet sizes and reports edge/variable counts, Q memory,
build and solve time, quality vs the greedy top-K baseline, and GNN latency
on the full vs pruned graph. Honest scope limit: connectivity encoding is
only enabled up to 20 rovers because the dense Q grows quadratically; at 10
rovers the greedy repair comparison is infeasible (reported as nan, not
hidden).

### `battle_test.py` (1418 lines, 491 checks) - the hostile auditor

A self-written torture suite, sections A through N, that tries to break
every claim:

- **Construction invariants**: weights sum to 1, penalties dominate, edge
  canonicalization, no self-loops.
- **Brute-force ground truth**: on small instances, enumerate all 2^k edge
  subsets and check the solvers and the MILP agree with reality.
- **H. QUBO solver benchmark**: the actual Q matrix against serious
  off-the-shelf QUBO solvers: local bit-SA, D-Wave `neal`, tabu search, and
  simulated quantum annealing (OpenJij). Result: they find the exact optimum
  only 48-64% of the time. The exact linearized CBC solve defines the
  optimum (100% by construction, cross-checked against enumeration on
  instances with up to 60 QUBO variables). The production projected
  annealer is deliberately *excluded* from this scientific comparison
  because its move set is richer; it is reported separately as an
  engineering note (100% on these small instances, and the suite enforces a
  90% floor so regressions fail the build).
- **J. Optimality-gap benchmark**: the headline metric
  `gap = (H_SA - H*) / max(1, |H*|)`. Small instances use exact enumeration
  for H*; larger instances use MILP H*. Result: 100% exact on small, 95%
  exact vs MILP on larger (mean true gap about 7e-5), and a 6.9% average
  gap against best-known solutions on instances up to 817 binary variables.
- **M. MILP comparison**: the uncomfortable truth section. CBC is exact and
  about 100x faster (roughly 11-15 ms vs about 2 seconds).
- **N. Sensitivity sweeps**: perturbs temperatures, cooling rates, restarts,
  penalty scales, and pair-move probability; the production solver stays
  87% exact across all perturbations, i.e. it is not a lucky-tuned artifact.
- Plus: scenario matrices, edge cases (disconnected inputs, zero optional
  edges, single rover), staleness handling, fault injection, and the GNN
  layer's own tests.

### `example.py` (92 lines)

A tiny 4-rover worked example you can read in one sitting. Start here.

### `paper/` - the research paper

- `main.tex` (662 lines): the full LaTeX paper, "Constrained Edge Pruning
  and Trust-Aware GNN Inference for Task Allocation in Dynamic Rover
  Networks: a QUBO Formulation with a Solver-Level Audit". TikZ/pgfplots
  figures, booktabs tables, real citations (Sabater-Sierra 2005 for trust,
  CBC, D-Wave tools).
- `fig_testbed.tex/pdf`: a physical testbed sketch built with the fiziko
  MetaPost library (with a TikZ fallback), committed as PDF so the paper
  builds without a full MetaPost install.
- `build.sh`: builds with tectonic.

### `changes.md`

The running log of external review rounds (two ChatGPT review exports plus
screenshot feedback) and what was changed in response. Useful for seeing how
the project hardened over time.

---

## 4. The results, honestly

The good:

- **The encoding is verified**: the Q matrix is checked against an
  independent re-derivation, exhaustive enumeration, and the exact
  linearized solve. They all agree.
- **The production solver is genuinely good**: 100% exact on small instances
  (verified by brute force), 95% exact against true MILP optima on larger
  instances (mean true gap 7e-5, worst 1.4e-3), 87% exact under every
  sensitivity perturbation.
- **Trust-aware assignment works end-to-end**: +13.8 percentage points
  mission success over random assignment, +8.1 over a trust-only baseline
  (paired bootstrap 95% CI [+4.2, +11.7], permutation p = 0.016).
- **Trust tracking is real**: per-cycle staleness margin of +13.75 points,
  and trust correlates with hidden ground-truth reliability.

The bad (reported in the paper with equal prominence, on purpose):

- **MILP crushes the QUBO on classical hardware**: CBC is exact and about
  100x faster. The QUBO's only justification is that it is the input format
  for quantum/annealing hardware.
- **Annealing samplers are unreliable on this landscape**: four serious
  samplers hit the optimum only 48-64% of the time on the penalty-encoded Q
  matrix. Constraint-breaking moves are basically never accepted (random
  restarts of bit-level SA returned infeasible states in 19 of 25
  instances). This is a finding about penalty-encoded constrained QUBOs in
  general, not just this codebase.
- **The QUBO pruner loses on raw mission success at 10-rover scale**: a
  simple threshold pruner beats it by about 4 points (33.8% vs 38.2%,
  p = 0.028). The QUBO optimally strips redundancy, but that redundancy is
  practically useful for pushing data through bottleneck paths.
- **The GNN's message passing barely matters at this scale**: +1.7 points
  over a no-graph logistic readout, statistically insignificant. The heavy
  lifting is done by the trust model and gating.
- **Cloud-trained readout does not beat fixed weights**: -3.9 points,
  p = 0.49. Null result, reported as such.

This "report the negative results too" stance is deliberate: it is what
makes the positive numbers believable.

---

## 5. How to run it

The README says `uv`, but `uv` is not actually installed on this machine;
plain Python works fine.

```bash
# quick demo
python3 example.py

# single simulation run with per-cycle log
python3 simulation.py

# pruner comparison table (QUBO vs full / threshold / top-K)
python3 simulation.py benchmark

# ablation arms (gnn / no-graph / trust-only / random)
python3 simulation.py ablate

# scaling experiments
python3 scale_runner.py

# the full 491-check battle suite
python3 battle_test.py
```

Sections H (neal / tabu / SQA) and M (MILP) need extra packages. They live
in a venv:

```bash
python3 -m venv /tmp/gnnenv
/tmp/gnnenv/bin/pip install numpy pulp dwave-neal dwave-tabu openjij
/tmp/gnnenv/bin/python -u battle_test.py   # full run, ~15-20 minutes
```

Notes: system Python is 3.14 and scipy is broken on it (missing bundled
OpenBLAS), which is exactly why the MILP baseline uses PuLP/CBC instead of
scipy. System pip is PEP-668 blocked, so always use a venv. `/tmp` gets
wiped between days, so the venv may need recreating.

To build the paper (no system TeX on this machine):

```bash
cd paper && /tmp/tectonic main.tex   # tectonic 0.15.0, redownload if wiped
```

`fig_testbed.pdf` is committed, so tectonic alone is enough; rebuilding the
fiziko figure needs lualatex + MetaPost and a clone of
github.com/jemmybutton/fiziko (gitignored).

---

## 6. Project history, briefly

1. Initial commit: the QUBO formulation and a basic annealer.
2. Trust-GNN layer, simulation harness, scaling runner, battle-test suite.
3. Review-2 fixes: connectivity-aware annealer starts, feasible greedy
   baseline, neal backend, robust timing.
4. The LaTeX paper with TikZ figures and the fiziko testbed sketch.
5. A hostile review round: the simulation was made causal (outcomes now
   actually depend on topology), the MILP baseline was added, ablations and
   real statistics were added, and the paper was rewritten to report
   negative results.
6. Solver audit round: the "projected" bar was removed from the solver
   figure, the actual Q matrix was benchmarked against serious QUBO solvers
   (neal, tabu, SQA), the 0 ms solve-time artifact in the pruner table was
   fixed (heuristics now show their real microsecond cost as `<0.01` ms),
   and meta-commentary was removed from figure captions.

---

## 7. If you only remember three things

1. **What it is**: prune a dynamic rover communication graph with a
   constrained QUBO, then assign tasks with a tiny trust-aware GNN that runs
   on the pruned graph, with trust updated online.
2. **What is genuinely new/verified**: the encoding is formally checked
   against enumeration and exact solvers; the feasibility-projected
   annealer is near-exact and robust; the whole claim set comes with paired
   statistics.
3. **What the honest catch is**: classical MILP beats the QUBO 100x on
   speed and is exact, annealing samplers are unreliable on
   penalty-encoded constrained QUBOs, and at small scale the fancy pruner
   and the graph convolution add less than simpler alternatives. The paper
   says all of this out loud.
