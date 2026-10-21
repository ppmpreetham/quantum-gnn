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

## Web console

A live 3D console (React, react-three-fiber, pmndrs postprocessing) drives
the real Python pipeline with rover positions from a browser-side
differential-drive simulation of the Gazebo SDF rover.

```bash
npm install
npm run dev        # Python planner on :8765 and Vite on http://127.0.0.1:5173
npm test           # TypeScript motion tests and Python planner/HTTP tests
npm run build      # type-check and production bundle
```

`npm run dev` uses `.venv/bin/python` when it exists (set `FLEET_PYTHON` to
override). Installing `dwave-neal`, `dwave-tabu` or `openjij` into that
environment enables the matching solver baselines in the comparison menu.

- `shared/fleet-config.json` is the single source for parameter limits,
  presets (with measured cycle times), and the comparison catalogue with
  the published reference numbers. Both the TypeScript client and
  `web/fleet_session.py` read it.
- `web/fleet_session.py` wraps `RoverSimulation` so live positions replace
  its random walk, runs paired assignment baselines on the same pruned
  topology, and runs QUBO samplers on the same Q matrix as our annealer.
- `src/core/` holds the framework-free motion math (exact arc integration,
  wheel-limit saturation, collision-safe sequential commits, deadlock
  detours); `src/scene/` renders the fleet, links, beacons and ground in
  about seven draw calls with instancing; `src/ui/` is the HUD.
- Three modes, switched from the top bar (or `#mode=photoreal`,
  `#mode=showcase`): the console; a photoreal desert test site running the
  same live mission (NASA Perseverance model, Poly Haven scans and HDRIs,
  full light and lens controls, assets in `public/photoreal/` with
  `CREDITS.md`); and "where we win", which replays seeded scenarios from
  `shared/showcase-scenarios.json` side by side against a baseline.
- Tasks can be drawn by the planner (default) or placed by clicking the
  ground (Mission tab, "Task placement"); clicked tasks go out with the next
  cycle and the real pipeline assigns them.
- If port 8765 is taken, `FLEET_API_PORT=8766 npm run dev` moves the planner
  and the dev proxy together.

## Battle Tests

To run the full test suite (construction invariants, brute-force ground
truth, scenario matrix, edge cases, scaling, GNN layer, simulation harness,
optimality-gap benchmark):

```bash
uv run battle_test.py
```

If `dwave-neal` is installed (optional), section H additionally benchmarks the
real neal QUBO sampler against the same Q matrices.
