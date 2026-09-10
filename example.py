from main import (
    EdgeFeatures,
    EdgeSelectionQUBO,
    ProjectedEdgeAnnealing,
    QUBOConfig,
)


def main() -> None:
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 2)]

    features = {
        (0, 1): EdgeFeatures(0.9, 0.9, 0.8, 0.7, 0.9, 0.5, 0.2),
        (1, 2): EdgeFeatures(0.8, 0.7, 0.9, 0.6, 0.8, 0.4, 0.3),
        (2, 3): EdgeFeatures(0.4, 0.5, 0.5, 0.4, 0.4, 0.2, 0.8),
        (0, 3): EdgeFeatures(0.7, 0.8, 0.8, 0.7, 0.9, 0.6, 0.4),
        (0, 2): EdgeFeatures(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1),
    }

    previous = {
        (0, 1): 1,
        (1, 2): 1,
        (2, 3): 0,
        (0, 3): 1,
        (0, 2): 1,
    }

    config = QUBOConfig(
        w_trust=0.25,
        w_link=0.25,
        w_battery=0.10,
        w_proximity=0.10,
        w_freshness=0.15,
        w_task=0.15,
        edge_penalty=0.08,
        stability_penalty=0.15,
        degree_penalty=100.0,
        budget_penalty=100.0,
        flow_penalty=500.0,
        flow_capacity_penalty=500.0,
        cost_scale=10,
        initial_temperature=8.0,
        final_temperature=1e-3,
        cooling_rate=0.98,
        sweeps_per_temperature=2,
        restarts=15,
        seed=42,
    )

    builder = EdgeSelectionQUBO(
        nodes=nodes,
        edges=edges,
        features=features,
        previous_decisions=previous,
        hard_critical_edges=[(0, 1)],
        required_nodes=[0, 2, 3],
        root=0,
        degree_limits={0: 3, 1: 2, 2: 2, 3: 2},
        global_budget=15,
        config=config,
    )

    Q, constant, names = builder.get_qubo()

    # The following graph is a known feasible warm start for this example:
    # hard edge (0,1) + (0,2) + (0,3).
    warm_edges = {(0, 1), (0, 2), (0, 3)}

    # ProjectedEdgeAnnealing anneals directly over feasible edge subsets
    # (bit-level SA on the penalty-encoded Q cannot cross penalty barriers
    # with single-bit flips, so it would barely leave its warm start).
    solver = ProjectedEdgeAnnealing(
        builder,
        initial_edge_sets=[warm_edges],
        restarts=10,
        seed=config.seed,
    )
    result = solver.run()

    print(f"QUBO variables : {len(names)}")
    print(f"QUBO shape     : {Q.shape}")
    print(f"Q symmetric    : {(Q == Q.T).all()}")
    print("\n--- RESULTS ---")
    print(f"Energy         : {result.energy:.4f}")
    print(f"Kept edges     : {result.kept_edges}")
    print(f"Pruned edges   : {result.pruned_edges}")
    print(f"Constraint     : {result.violated_constraints or 'none'}")


if __name__ == "__main__":
    main()
