# How Every Test Works — The Theory, Explained Like You're 10 (but with the real math inside)

This is the complete theory behind `battle_test.py` — the 486-check regression
suite that guards the whole project. Fourteen test families (A through N), each
attacking the code from a different angle. Read this once and you'll be able to
explain any check to anyone.

---

## Table of contents

1. [The big idea: why so many tests?](#1-the-big-idea)
2. [The shared toolbox (read this first)](#2-the-shared-toolbox)
3. [A — Construction invariants: "is the matrix even a legal matrix?"](#a--construction-invariants)
4. [B — Brute force vs simulated annealing: "the answer-key duel"](#b--brute-force-vs-simulated-annealing)
5. [C — Analytic optima: "problems we can solve with a pencil"](#c--analytic-optima)
6. [D — The scenario matrix: "many random worlds, one promise"](#d--the-scenario-matrix)
7. [E — Validation and edge cases: "garbage in, clear error out"](#e--validation-and-edge-cases)
8. [F — Scaling sanity: "does it survive growing up?"](#f--scaling-sanity)
9. [G — Domain scenarios: "seven little stories that must end right"](#g--domain-scenarios)
10. [H — The solver audit: "the penalty-barrier discovery"](#h--the-solver-audit)
11. [I — The trust-GNN layer: "is the brain actually thinking?"](#i--the-trust-gnn-layer)
12. [J — The optimality-gap benchmark: "how close to perfect, with a safety net"](#j--the-optimality-gap-benchmark)
13. [K — The simulation harness: "does the whole world run?"](#k--the-simulation-harness)
14. [L — Regression tests: "bugs are not allowed to come back"](#l--regression-tests)
15. [M — The MILP ground truth: "the professional answer key"](#m--the-milp-ground-truth)
16. [N — Sensitivity: "what if we tuned the knobs wrong?"](#n--sensitivity)
17. [The philosophy behind it all](#17-the-philosophy-behind-it-all)

---

## 1. The big idea

Imagine you build a machine that picks which radio links a robot team should
use. How do you *know* it picks well? You can't just watch it run once and say
"looks fine" — that's how planes used to fall out of the sky.

So we interrogate the machine four different ways:

1. **Independent re-derivation** — a *second accountant* recomputes every
   number from the written specification, without looking at the machine's
   notes. If the two accountants ever disagree, someone is cheating (or buggy).
2. **Answer keys** — for small problems we can find the *provably perfect*
   answer by trying literally everything. The machine must match the key.
3. **Pencil problems** — some special problems are so simple you can solve
   them with a formula on paper. The machine must match the formula.
4. **Stories** — tiny scenarios with an obvious right ending ("a bridge to a
   stranded rover must be kept even if it's unpopular"). The machine must
   tell the story correctly.

Every family below is one of these four, wearing a different costume.

A `check(name, condition)` simply counts: if the condition is true, one point
for the machine; if false, the failure is printed with details and the suite
exits with an error at the end. **No failure is ever silently swallowed.**

---

## 2. The shared toolbox

These helpers are used by many families, so learn them once.

### 2.1 `random_connected_instance` — the world generator

We need random test problems, but a *random* graph is usually disconnected
(some nodes can't reach others), which makes it useless for connectivity
tests. Trick: **first build a random spanning tree** (a chain of connections
that touches every node — like dealing people in a circle so everyone is
linked to someone earlier), **then sprinkle extra random edges on top** with
probability `extra_edge_p`.

That guarantees:
- the graph is always connected (the tree is inside it),
- it's not trivially easy (extra edges create shortcuts and redundancy),
- degree caps and budget are set *relative to the tree* — the `tight=True`
  setting adds 0–1 spare units so the constraints actually bite.

It also randomizes: all 7 edge features (trust, link quality, battery,
proximity, freshness, task criticality + processing cost), the previous
cycle's decisions (for the stability term), which edges are hard-fixed
(25% of tree edges), the required-node set, and the root. Every trial uses a
fixed seed from a `np.random.default_rng(seed)`, so **the "random" worlds are
the same every run** — reproducible randomness, like a deal of cards you can
re-deal identically forever.

### 2.2 `independent_energy` — the second accountant

The QUBO machine computes energy as `E(z) = zᵀQz + c` — one big matrix
multiplication, fast but opaque. The independent accountant **never touches
Q**. It re-reads the specification and sums up each term by hand, one loop at
a time:

```
E = (soft objective)                     − utility + λ per kept edge
  + η × (kept − previous)²               switching cost
  + 100 × (degree_over)²                 degree penalty
  + 100 × (budget_over)²                 budget penalty
  + 500 × (flow imbalance)²              connectivity conservation
  + 500 × (flow capacity error)²         connectivity capacity
```

If the matrix Q encodes even one term slightly wrong, the fast path and the
slow path disagree. The tolerance is relative: `|got − want| < 1e-6 ×
max(1, |want|)` — one part in a million, scaled so big energies don't demand
impossible precision.

### 2.3 `brute_force` — try every door

A QUBO with `n` binary variables has exactly `2ⁿ` possible answers. For
`n ≤ 20` that's at most ~1 million states — cheap for a computer. The brute
forcer builds **all** states as a big table and computes every energy in one
shot (`einsum`), then keeps the minimum. No cleverness, no heuristics, no
bugs to worry about: the dumbest possible method, which is exactly why we
trust it as a *ground truth*.

### 2.4 `exact_edge_enumeration` — the smart answer key

Trying all `2ⁿ` *bit strings* is wasteful: most bits are bookkeeping (slack
and flow variables), not decisions. The smart enumeration instead tries all
`2^|optional edges|` **edge subsets** (capped at 20 → ≤ 1M subsets), and for
each one computes the energy in closed form:

- **soft terms**: sum utility, edge penalty, stability per edge;
- **degree/budget penalties**: computed directly from the subset — the slack
  bits always have a best value (cancel the residual exactly when legal), so
  we don't need the bits at all;
- **connectivity**: no flow assignment can fix a *disconnected* graph, so we
  just do a flood-fill (DFS) from the root and skip the subset if any
  required node is unreachable; if connected, the tree flow always fits, so
  flow penalties are zero.

The proof that slack/flow bits never change the ranking is the reason this is
*exact*: for a fixed edge subset, the best slack assignment is known in
closed form, and connectivity is binary (possible or not).

### 2.5 `fast_config` — the speed dial

The production annealer settings (20 restarts, slow cooling) are tuned for
quality, not test speed. `fast_config` uses hotter/faster settings so the
suite runs in minutes. Important honesty rule: the *fast* settings are used
for correctness tests (A–G), but the *production* settings are used for the
quality benchmarks (H, J, M) — we never grade the machine with easier rules
than production and then claim production-level scores.

### 2.6 `norm_gap` — grading on a curve

```
gap = (solver_energy − best_energy) / max(1, |best_energy|)
```

Raw energy differences mean nothing without scale ("100 worse than perfect"
is terrible if perfect is 5, excellent if perfect is 10,000). The normalized
gap is a percentage-ish score: 0 means exact, 0.10 means 10% off.

---

## A — Construction invariants

**Question: is the built Q matrix even a legal, self-consistent object?**

30 random instances, and for each one four things are checked:

| Check | Theory |
|---|---|
| `A.sym` | `Q = Qᵀ` (the matrix equals its own transpose). The energy `zᵀQz` only "sees" the symmetric part of Q, but the solver's ΔE shortcut `2δ(Qz)ᵢ + Qᵢᵢ` is derived *assuming* symmetry. A non-symmetric Q would make the solver's math subtly wrong. |
| `A.fin` | Every entry is a real number — no `NaN`, no `±∞`. One `NaN` would poison every downstream energy silently. |
| `A.uniq` | All variable names are unique. Variables are looked up **by name** when decoding solutions; a duplicate name means two variables share one slot and the solution becomes garbage. |
| `A.energy` | The big one: 5 random bit vectors per instance, and the machine's `zᵀQz` answer must equal the **independent accountant's** term-by-term sum. This catches *any* mis-encoded term — wrong weight, wrong sign, missing slack — on random inputs, not cherry-picked ones. |

Why random bit vectors and not real solutions? Because a bug that only shows
on weird states (all zeros, all ones, mixed) would hide from "sensible"
inputs. Random states probe the whole landscape.

---

## B — Brute force vs simulated annealing

**Question: when we know the perfect answer, does the solver find it?**

For 60 small random instances (3–5 nodes, so ≤ 20 variables):

1. **`B.cross` — the two answer keys must agree.** The bit-string brute force
   and the smart edge enumeration both claim to know the optimum. They are
   computed by *completely different code* (matrix einsum vs per-subset
   formulas). Agreement to `1e-9` means both keys are trustworthy. If this
   ever failed, the *keys* would be the bug, not the solver.
2. **`B.bf-feas` — no penalty leakage.** Take the brute-force winner and
   verify its constraint energy is ~0. Why? If the penalties were encoded too
   weakly, the "optimum" would be a solution that *breaks the rules* because
   breaking them was cheaper than obeying. The math: one unit of violation
   costs `100 × 1² = 100` points, but the best possible gain from an
   extra edge is utility (≤ 1) minus the penalty λ (0.08) — under 1 point.
   100 ≫ 1, so cheating can never win *if the encoding is right*. This check
   proves the "if" on every instance.
3. **The solver duel.** Bit-flip simulated annealing runs on the same
   instance (with a feasible warm start). Its result goes through the same
   repair path production uses (project back to feasibility), then:
   - **`B.gap`**: the 90th-percentile normalized gap must be < 0.5 — even
     the *worse* 10% of runs can't be off by more than half the optimum's
     scale. (We don't demand perfection from bit-SA; we demand "not
     embarrassing".)
   - **`B.repair`**: zero repair failures — the feasibility repair must never
     crash or give up.

The subtle theory point: bit-SA is *allowed* to be imperfect because it's a
diagnostic (Section H shows why it fundamentally can't be perfect here). But
its imperfection must be bounded, and the repair path must be a safety net
that never has a hole.

---

## C — Analytic optima

**Question: can we solve the problem with a pencil, and does the machine
agree with the pencil?**

These are the strongest checks in the suite, because the expected answer
comes from a formula, not from *any* code — not even the test's own
machinery.

**Case 1 (`C.uncon`): no constraints at all.** Strip degree limits, budget,
connectivity, and stability (η = 0). Now each edge is an independent choice:
keeping edge `e` changes the energy by

```
Δ(e) = −utility(e) + λ     (λ = 0.08 edge penalty)
```

Keep it if `Δ < 0` (it pays for itself), drop it otherwise. The optimal total
energy is `Σ min(0, Δ(e))`. Brute force must return exactly this number
(`1e-9`), on 20 random instances. If the QUBO encoding is correct, the
machine's global optimization *must* reproduce 20 independent little
greedy decisions.

**Case 2 (`C.stab`): add stability back.** Now each edge has two candidate
energies:

```
drop:  a0 = η × (0 − prev)²
keep:  a1 = −utility + λ + η × (1 − prev)²
```

and the optimum is `Σ min(a0, a1)`. This tests that the switching penalty
interacts correctly with the keep/drop decision — e.g., a previously-kept
edge gets a discount for staying.

If either case failed while B passed, it would mean the matrix encodes
*something* consistently (the two keys agree) but not the *intended*
something — exactly the bug class independent formulas are designed to catch.

---

## D — The scenario matrix

**Question: on bigger, wilder random worlds, does the guarantee hold?**

80 instances, 4–10 nodes, randomized density and tightness — too big to
brute force, so no answer key. Instead we test a different property: the
**feasibility guarantee**.

- Raw bit-SA is allowed to return infeasible states (we just *count* how
  often — this number feeds the penalty-barrier story).
- But after the repair path, **zero** instances may remain infeasible
  (`D.final`). The repair is the production safety net: it must have zero
  holes across every random world we can generate.

Think of it as: "the acrobat may wobble on the wire, but the net must never
miss."

---

## E — Validation and edge cases

**Question: what happens when the world is wrong, tiny, or degenerate?**
A library that *crashes mysteriously* on bad input is a library you can't
trust in production. Each probe asserts a specific, *designed* behavior:

**Garbage in → clear `ValueError` (12 probes):**

| Probe | Why it must fail loudly |
|---|---|
| self-loop `(0,0)` | a link from a rover to itself is meaningless |
| edge to unknown node | silent phantom nodes would corrupt the graph |
| feature = 1.5 or NaN | features are defined on [0,1]; out-of-range values would silently skew utilities |
| missing edge features | an un-scored edge would get a default utility nobody chose |
| weights not summing to 1 | utilities would silently change scale, breaking every threshold intuition |
| previous decision = 2 | decisions are binary |
| impossible connectivity (node 2 required but no edge to it) | better to refuse at construction than to fail at solve time |
| negative budget | nonsense input |
| root not in required set | the flow encoding *defines* root = source; a foreign root is a logic error |
| hard-critical edge that isn't an edge / has no features | fixed edges enter the constants of Q; a typo here would vanish silently |

**Behavioral probes:**

- **`E.degree-default`** — nodes *omitted* from `degree_limits` must be
  **unconstrained**, not capped at 0. This is a classic silent-wrong-answer
  bug: `dict.get(node, 0)` would treat every unlisted node as "no links
  allowed" and produce legal-looking but terrible solutions.
- **`E.budget-zero`** — with budget 0, even the cheapest edge (cost ≥ 1)
  can't fit. Brute force must confirm the *optimal* solution keeps nothing.
  Tests that the budget binds correctly at its most extreme.
- **`E.empty`** — a problem with zero variables must return a zero-length
  solution, not crash. (Real deployable code meets the empty graph
  eventually.)
- **`E.determinism`** — same seed ⇒ bit-identical result, twice. Reproducible
  randomness is a contract: an experiment you can't re-run exactly is an
  experiment you can't defend.

---

## F — Scaling sanity

**Question: does it stay correct and fast as graphs grow?**

At n = 8, 12, 16 nodes:
- **`F.feas`** — the feasibility guarantee holds at size (repair never fails).
- **`F.time`** — solve finishes under 60 s.

Also printed: variable count and **Q's memory footprint**. The theory worth
knowing: Q is `V × V` where V = edge vars + slack bits + flow bits. Flow
variables grow with edges × nodes, so **V grows quadratically-ish with
fleet size** — at n=16 the matrix already costs hundreds of MB. This test is
where the paper's "dense Q bounds us at tens of rovers" limitation comes
from, measured rather than guessed.

---

## G — Domain scenarios

**Question: does it make the obviously-right call in seven tiny stories?**
Each is hand-built so the right answer is unambiguous. The solver runs with
production settings and a warm start, and must produce the story's ending:

| Story | Setup | Required ending | The math of why |
|---|---|---|---|
| **G1** bridge | node 2 reachable *only* via a trust-0.05 edge | keep the bridge | dropping it costs a 500-point flow penalty; keeping it costs 0.01 utility points. Constraint ≫ preference. |
| **G2** weak parallel edge | a redundant 3rd edge with all features 0.05 | prune the weak one | both paths exist; the weak edge only pays (−utility + λ > 0) with no benefit. Sparsity works. |
| **G3** hysteresis | two *identical* utility-0.07 edges; one was kept last cycle, one wasn't | keep the old one, drop the new one | utility 0.07 < λ 0.08, so alone each edge is a bad deal (+0.01). Stability: the previously-kept edge costs 0.01 to keep vs 0.15 to drop ⇒ keep; the previously-absent edge costs 0.16 to keep vs 0 to drop ⇒ drop. The switching penalty acts as a tie-breaker that resists flapping. |
| **G4** star overload | center node with cap 2 and three candidate leaves | drop the worst leaf | degree penalty forces a choice; the optimizer must pick the two highest-utility leaves — a tiny "knapsack" the encoding must solve, not just survive. |
| **G5** identical features | all 10 edges identical, everything required | deterministic, feasible output | same input + same seed ⇒ same output; ties must not explode into randomness. |
| **G6** tight budget | two cheap good edges (cost 1) vs one expensive good edge (cost 10), budget 2 | keep the two cheap ones | equal utility, but cost 10 > budget 2. The budget must bind and the optimizer must do the value-per-cost reasoning. |
| **G7** chain | a 3-edge chain, all nodes required, all features terrible (0.01) | keep the whole chain | every link is the only path; connectivity forbids pruning regardless of utility. "Terrible but necessary" beats "great but disconnecting". |

These seven are the tests a reviewer *remembers*: "show me the bridge case."
They're also the fastest way to explain the whole system to a newcomer.

---

## H — The solver audit

**Question: can serious, off-the-shelf QUBO solvers actually solve our
matrix?** This is the paper's headline discovery, and the test is built like
a courtroom.

**The instances:** 25 random instances with ≤ 16 optional edges (so the
answer key is computable), tight constraints.

**The ground truth is verified twice:** for 5 small instances, the exact
linearized solve (CBC via `solve_qubo_exact`) must match the enumeration to
`1e-7` (`H.exact-agree`). Two *independent* keys agreeing is what makes the
upcoming failure numbers credible — we're not grading against a buggy key.

**The lineup** (each gets the actual Q matrix, 100 reads, same warm start,
same repair-or-warm scoring):

- local bit-flip SA
- D-Wave `neal`
- tabu search
- simulated quantum annealing (OpenJij SQA)

Translation detail: dimod-style samplers use the convention
`E = Σ Qᵢᵢxᵢ + Σ_{i<j} Qᵢⱼxᵢxⱼ`, while our matrix stores the full symmetric
form — hence `_dimod_qubo` doubles the off-diagonal entries before handing Q
over. Get this translation wrong and you'd benchmark a *distorted* problem.

**The result:** the four samplers hit the exact optimum in only **48–64%** of
instances. And the "why" is the test suite's most elegant piece of theory:

> From any feasible solution, every single-bit flip either
> (a) breaks a constraint → costs ≥ 100 points, or
> (b) stays feasible → changes the score by at most ~1.2
> (max utility 1.0 + λ 0.08 + η 0.15).
>
> The wall is ≥ 80× taller than any reward behind it. At any temperature
> where good moves are accepted, rule-breaking moves are rejected — so a
> bit-flip sampler **cannot leave a feasible basin**. It's not weak; the
> landscape is a prison.

Supporting evidence collected by the same test: with *random* (no warm)
starts, raw bit-SA returns infeasible states in **19 of 25** instances — the
samplers aren't finding wrong optima, they can't even find legal ones.

**The projected annealer** (our production solver, which moves whole edges
and projects back to feasibility) is deliberately *excluded* from the
comparison — its move set is richer, so beating bit-SA with it proves
nothing scientific. The test only asserts a floor: ≥ 90% exact
(`H.projected`). Honesty by construction.

---

## I — The trust-GNN layer

**Question: is the assignment "brain" mathematically sane, and does it
actually help?**

| Check | Theory |
|---|---|
| **I1 equivariance** | Relabel nodes (0,1,2 → 10,11,12) and every score must follow its node *exactly* (`1e-12`). A GNN's promise is that its output depends on the *graph*, not on the labels. This is the GNN version of the symmetry check in family A. |
| **I2 ordering** | A node with trust 0.9/battery 0.9 must score above a 0.5/0.5 node, which must score above a 0.2/0.2 node. The most basic sanity of any scorer. |
| **I3 gates** | Eligibility is a hard gate: low trust (0.1), empty battery (0.05), or an overloaded node (load 0.95) must all score **exactly 0.0** — not "low", zero. Gates are safety rules; they must behave like walls, not slopes. |
| **I4 neighborhood** | Two identical nodes, but one is linked to a strong neighbor: the connected one must score higher. This is the one thing that makes it a *graph* neural network — message passing must actually move information. |
| **I5 validation** | Missing node features, feature = 2.0, all-zero output weights → `ValueError`, loudly. |
| **I6 assignment** | The matcher must never assign one rover to two tasks (`len(set(a.values())) == len(a)`), must skip zero-score rovers, and must respect per-node capacity overrides. |
| **I7 trust dynamics** | The EWMA trust update `t ← t + α(outcome − t)` with α = 0.5: success from 0.5 → 0.75 → 0.875 (monotone up ✓); one failure → 0.4375 (drops below ✓); 50 straight successes → 1 − 0.5×0.5⁵⁰ ≈ 1.0 (converges ✓ — a geometric series with ratio 0.5); always within [0,1] ✓; an unknown node gets the neutral prior 0.5 ✓; outcome 1.5 → `ValueError` ✓. |
| **I8 integration** | The full simulator, GNN assignment vs *random* assignment, 8 paired seeds: GNN must beat random by > 3 points. This is the smallest end-to-end claim the paper is allowed to make. |
| **I9 canonical lookup** | A regression test for a real latent bug: asking for adjacency of edge `(1,0)` must find the features stored at `(0,1)` via canonicalization. Before the fix, a non-canonical edge silently got weight 1.0 — a wrong answer with no error message. The worst kind of bug; now it has a tripwire. |
| **I10 batch == single** | Scoring all tasks in one batched pass must equal per-task scoring to `1e-12`. Performance optimizations must never change results. |
| **I11 training** | Synthetic data where trust *causes* success (y = 1 iff trust > 0.5, plus noise). After logistic training, the trust weight must be the largest and > 1.0, and ranking AUC must exceed 0.9. Plus: training on a single class → `ValueError` (logistic loss is undefined there — fail loudly, not with NaN weights). |

---

## J — The optimality-gap benchmark

**Question: how close to perfect is the *production* solver — with a
certified safety net when perfection can't be computed?**

**Small instances (25):** enumeration gives exact H*. Two checks:
- `J.soft-eq` — the solver's reported energy must equal `soft_energy()`
  (the O(m) shortcut) to `1e-9` on the returned feasible set. Two ways of
  computing the same thing must agree.
- `J.small-exact` — ≥ 90% of instances solved exactly.

**Large instances (20, too big to enumerate):** now we need honesty without
an answer key, and this is the clever part — *bounds*:

1. **Best-known solution (BKS):** run the annealer 4 times with different
   seeds and keep the best. The gap to BKS *overestimates* the true gap
   (BKS ≥ true optimum). Reported, not certified.
2. **Certified bound via a lower bound:** for each edge independently,
   `min(keep_cost, drop_cost)` is the best that edge could ever contribute,
   *ignoring all constraints*. Since penalties are ≥ 0, the sum of these
   per-edge minima is a **certified lower bound** on the true optimum:
   `LB ≤ H* ≤ H_SA`. Therefore
   `(H_SA − LB)/max(1,|LB|)` is an **upper bound on the true gap** — if that
   number is small, the solver is provably close to perfect, no answer key
   needed.

Thresholds: mean BKS gap < 10%, 90th percentile < 25%. Both are floors on
quality, chosen so a regression (a code change that made the solver dumber)
trips them.

---

## K — The simulation harness

**Question: does the whole simulated world — pruners, faults, trust, edge
lifecycle — behave as designed?**

- **K1** — every pruner arm (`full`, `threshold`, `topk`, `qubo`) and the
  random-assignment arm runs end-to-end and returns a valid summary with
  timings. Nothing crashes, nothing returns NaN.
- **K2** — the default fault script produces recovery-time measurements (the
  fault/recovery machinery is *wired*, not just present).
- **K3** — under total packet loss (`packet_loss = 1.0`), a link's freshness
  must **decay** as it ages (5 cycles of no observations). Freshness is the
  `exp(−Δt/τ)` "how stale is this information" feature; this verifies it
  actually decays instead of silently freezing.
- **K4** — the Spearman correlation implementation: perfectly matching
  rankings → +1.0, perfectly reversed → −1.0, ties → still within [−1, 1].
  (This function grades trust quality everywhere; if it's wrong, every trust
  number in the paper is wrong.)
- **K5** — the edge lifecycle: prune an edge once → SUSPECT, again →
  PROBATION, again → REMOVED, then keep it → back to ACTIVE. Hysteresis
  theory: one bad cycle must never permanently delete a link (maybe the
  rover was just behind a hill), so removal takes three strikes — and
  rehabilitation is possible.
- **K6** — custom scripted faults (battery drain, global) apply without
  crashing.

---

## L — Regression tests

**Question: are the bugs we already fixed still dead?**

Each `L` test pins a specific historical bug with a minimal reproduction:

- **L1** (was bug E1): a pure chain graph with connectivity constraints and
  **no warm start** must still solve. The old annealer couldn't bootstrap —
  random starts grew edge-by-edge and never crossed the infeasible "half a
  chain" states (connectivity is *not prefix-monotone*: no single edge
  connects a chain, so greedy growth always failed). The fix seeds random
  starts with random **spanning trees**. This test is the proof the fix
  works and stays working.
- **L2** (was bug E2): take a greedy utility-top-17 selection on a 10-node
  instance — first *prove it violates constraints* (the test asserts the
  premise!), then run `repair_to_feasible` and require a legal result. The
  assert-the-premise pattern matters: a repair test where the input happens
  to be feasible proves nothing.
- **L3**: task criticality must scale the success *probability*, not the
  logit. The old version multiplied the logit, which for negative logits
  *raised* the score — a sign-error making critical tasks easier for
  unreliable rovers. Now: higher criticality may never increase any score.
- **L4**: a fault with `node=None` means "everyone" — verify every rover's
  reliability actually dropped. (Silently doing nothing was the old bug.)
- **L5**: the `measure_full_gnn` flag must gate the full-graph timing
  measurement (measurement code inside the production loop costs time; it
  must be switchable).

Regression tests are the memory of the suite: every `L` check is a scar that
won't let the same wound reopen.

---

## M — The MILP ground truth

**Question: what happens when the solver fights a *professional*?**

`solve_milp` formulates the **same** constrained problem — identical soft
objective, degree caps, budget, connectivity — directly as a mixed-integer
program. No penalty encoding, no slack bits, no Q matrix: just linear
constraints that CBC's branch-and-cut can reason about exactly. (Key
theoretical gift: single-commodity flow with integer demands has an
*integral* relaxation, so the flow variables don't even need to be integers.)

20 tight instances, ≤ 40 optional edges. Checks:

- **`M.obj-eq`** — CBC's objective value must equal `soft_energy()` of the
  edge set it returned (to `1e-6`). The two formulations must literally
  agree on what "score" means before we compare quality.
- **The gap**: the projected annealer must reach CBC's exact optimum on ≥ 80%
  of instances. Result: it does (95% on the paper's run, mean normalized gap
  7×10⁻⁵).
- **The timing** (printed, not asserted): CBC ≈ 15 ms vs annealer ≈ 2.1 s.
  This is the source of the paper's most humbling sentence: *on classical
  hardware, direct MILP is exact and ~100× faster* — the QUBO's job is to be
  the input format for quantum hardware, not to win on a CPU.

---

## N — Sensitivity

**Question: the weights and penalties are hand-chosen — how fragile are we
to choosing them wrong?**

15 fixed instances; 6 perturbed configurations:

| Perturbation | What it stresses |
|---|---|
| trust weight ×1.5 (0.25 → 0.375) | the "trust matters most" intuition |
| all weights uniform (1/6 each) | total loss of feature priorities |
| penalties 100/500 → 20/100 (weaker) | the barrier height from Section H |
| penalties → 500/2000 (stronger) | landscape stiffness |
| edge penalty 0.08 → 0.3 | how hard sparsity pushes |
| stability 0.15 → 0.6 | how hard anti-flapping pushes |

For each configuration, the answer key is **recomputed with that same
configuration** (enumeration with the perturbed penalties — the ground truth
moves with the knobs, which is the only correct way to do it), and the
production solver must hit it on ≥ 70% of instances.

Result: 87% under every perturbation. The solver is *robust* to the hand
tuning — which is what lets the paper say "parameters are hand-chosen, with
sensitivity analysis but no principled tuning procedure" and still sleep at
night.

---

## 17. The philosophy behind it all

If you remember nothing else, remember the five design rules:

1. **No single source of truth.** The optimum is computed three independent
   ways (bit brute force, smart enumeration, MILP) and they must *agree
   with each other* before either is used to grade the solver. A grader with
   a bug is worse than no grader.
2. **Independent re-derivation over shared machinery.** The `Q` matrix is
   re-implemented term-by-term by the "second accountant" (family A). Code
   checking itself is astrology; a second implementation is science.
3. **Reproducible randomness.** Every "random" world comes from a fixed
   seed. Randomness provides *coverage*; seeds provide *accountability*.
   Same failures every time means fixable failures.
4. **Assert the premise.** Tests like L2 first prove the input is broken
   (greedy *is* infeasible) before testing the fix. A test whose premise is
   accidentally false passes forever and proves nothing.
5. **Floors, not fans.** Thresholds (≥ 90% exact, < 10% gap, ≥ 70% under
   perturbation) are *floors* that catch regressions — never tuned to the
   exact current score. The paper's claims come from the *printed
   measurements*, and the checks only guarantee "never worse than this."

And the one-sentence summary of the whole suite:

> **Small problems are graded against proven-perfect answer keys computed
> three independent ways; large problems are graded against certified
> bounds; the solver's honest weaknesses are measured, printed, and
> published instead of hidden.**
