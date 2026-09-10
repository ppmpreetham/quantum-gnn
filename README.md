# Quantum GNN

Trust-aware task allocation for multi-rover networks: a constrained QUBO
prunes the dynamic communication graph, and a trust-aware GNN assigns tasks
on the pruned graph.

Before you run the project, make sure that you have UV installed. It's pretty easy to install, just follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

## Running the Project

Install dependencies and sync the environment:

```bash
uv sync
```

Then,

```bash
uv run example.py
```

## Modules

- `main.py` — constrained QUBO formulation for dynamic edge pruning
  (`EdgeSelectionQUBO`), a bit-level simulated annealing solver
  (`SimulatedAnnealing`), and the production solver
  (`ProjectedEdgeAnnealing`: annealing over feasibility-projected edge
  subsets; bit-level SA cannot cross the penalty barriers of the
  constrained QUBO).
- `trust_gnn.py` — trust-aware GNN (`TrustGNN`) scoring nodes on the pruned
  graph, capacity-aware greedy `assign_tasks`, a dynamic EWMA `TrustModel`
  with trust aging, and `train_out_weights` (cloud-tier readout training).
- `simulation.py` — time-stepped end-to-end simulation with pruning
  baselines (full / threshold / top-K / QUBO), per-stage latency timing,
  fault injection with recovery-time metrics, and trust-quality
  (Spearman) evaluation:
  - `uv run simulation.py` — single run with per-cycle log
  - `uv run simulation.py benchmark` — pruner comparison table
  - `uv run simulation.py train` — cloud-tier GNN readout training
- `scale_runner.py` — scaling experiments (fleet size vs edges, variables,
  solve time, energy vs greedy baseline, GNN latency full vs pruned).

## Battle Tests

To run the full test suite (construction invariants, brute-force ground
truth, scenario matrix, edge cases, scaling, GNN layer, simulation harness,
optimality-gap benchmark):

```bash
uv run battle_test.py
```

If `dwave-neal` is installed (optional), section H additionally benchmarks the
real neal QUBO sampler against the same Q matrices.
