from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================
# Data models
# ============================================================

Edge = Tuple[int, int]


@dataclass(frozen=True)
class EdgeFeatures:
    """
    All values should be normalized to [0, 1], except cost,
    which is normalized separately.
    """

    trust: float
    link_quality: float
    battery: float
    proximity: float
    freshness: float
    task_criticality: float
    processing_cost: float


@dataclass
class QUBOConfig:
    # Utility weights
    w_trust: float = 0.25
    w_link: float = 0.25
    w_battery: float = 0.10
    w_proximity: float = 0.10
    w_freshness: float = 0.15
    w_task: float = 0.15

    # Penalize retaining expensive edges
    cost_weight: float = 0.20

    # Regularization
    stability_penalty: float = 0.10

    # Constraint penalties
    degree_penalty: float = 10.0
    budget_penalty: float = 10.0
    flow_penalty: float = 100.0

    # Simulated annealing
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.995
    sweeps_per_temperature: int = 5
    restarts: int = 5
    seed: int = 42


@dataclass
class QUBOResult:
    variables: List[str]
    solution: np.ndarray
    energy: float
    kept_edges: List[Edge]
    pruned_edges: List[Edge]


# ============================================================
# Utility functions
# ============================================================


def validate_probability(name: str, value: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def edge_utility(
    features: EdgeFeatures,
    cfg: QUBOConfig,
) -> float:
    """
    U_e =
        wT*T +
        wL*L +
        wB*B +
        wD*D +
        wA*A +
        wC*C
    """

    values = {
        "trust": validate_probability("trust", features.trust),
        "link_quality": validate_probability("link_quality", features.link_quality),
        "battery": validate_probability("battery", features.battery),
        "proximity": validate_probability("proximity", features.proximity),
        "freshness": validate_probability("freshness", features.freshness),
        "task_criticality": validate_probability(
            "task_criticality", features.task_criticality
        ),
        "processing_cost": validate_probability(
            "processing_cost", features.processing_cost
        ),
    }

    weights = np.array(
        [
            cfg.w_trust,
            cfg.w_link,
            cfg.w_battery,
            cfg.w_proximity,
            cfg.w_freshness,
            cfg.w_task,
        ]
    )

    if np.any(weights < 0):
        raise ValueError("Utility weights must be non-negative")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Utility weights must sum to 1.0, got {weights.sum()}")

    quality = float(
        weights
        @ np.array(
            [
                values["trust"],
                values["link_quality"],
                values["battery"],
                values["proximity"],
                values["freshness"],
                values["task_criticality"],
            ]
        )
    )

    return quality - cfg.cost_weight * values["processing_cost"]


# ============================================================
# QUBO builder
# ============================================================


class EdgeSelectionQUBO:
    """
    Builds

        H(z) = z^T Q z + constant

    where z contains:
        - x_e      : optional edge selection variables
        - slack    : binary encoded slack variables
        - flow     : binary encoded flow variables

    Hard-critical edges are removed from the optimization
    and forced to remain active.
    """

    def __init__(
        self,
        nodes: Sequence[int],
        edges: Sequence[Edge],
        features: Dict[Edge, EdgeFeatures],
        previous_decisions: Optional[Dict[Edge, int]],
        hard_critical_edges: Iterable[Edge],
        required_nodes: Optional[Sequence[int]],
        root: Optional[int],
        degree_limits: Dict[int, int],
        global_budget: int,
        config: Optional[QUBOConfig] = None,
    ):
        self.nodes = sorted(set(nodes))

        self.edges = [self._canonical_edge(e) for e in edges]

        self.edges = sorted(set(self.edges))

        self.features = {self._canonical_edge(e): f for e, f in features.items()}

        self.previous_decisions = {
            self._canonical_edge(e): int(v)
            for e, v in (previous_decisions or {}).items()
        }

        self.hard_critical = {self._canonical_edge(e) for e in hard_critical_edges}

        self.optional_edges = [e for e in self.edges if e not in self.hard_critical]

        self.required_nodes = (
            sorted(set(required_nodes)) if required_nodes is not None else []
        )

        self.root = root
        self.degree_limits = degree_limits
        self.global_budget = global_budget
        self.cfg = config or QUBOConfig()

        if self.root is not None and self.root not in self.required_nodes:
            raise ValueError("Root must be in required_nodes")

        self._validate_inputs()

        self.variables: List[str] = []
        self.var_index: Dict[str, int] = {}

        self.Q: Optional[np.ndarray] = None
        self.constant: float = 0.0

        self.edge_var: Dict[Edge, int] = {}

        self.degree_slack_vars: Dict[int, List[int]] = {}
        self.budget_slack_vars: List[int] = []

        self.flow_var: Dict[Tuple[int, int, int], List[int]] = {}
        self.flow_slack_vars: Dict[Edge, List[int]] = {}

        self._build_variables()
        self._build_qubo()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    @staticmethod
    def _canonical_edge(edge: Edge) -> Edge:
        u, v = edge

        if u == v:
            raise ValueError(f"Self-loop not allowed: {edge}")

        return (u, v) if u < v else (v, u)

    def _validate_inputs(self) -> None:
        graph_nodes = set(self.nodes)

        for u, v in self.edges:
            if u not in graph_nodes or v not in graph_nodes:
                raise ValueError(f"Edge {(u, v)} contains unknown node")

        missing = set(self.optional_edges) - set(self.features)
        if missing:
            raise ValueError(f"Missing features for optional edges: {missing}")

        for node, limit in self.degree_limits.items():
            if node not in graph_nodes:
                raise ValueError(f"Degree limit provided for unknown node {node}")

            if limit < 0:
                raise ValueError("Degree limits must be >= 0")

        if self.global_budget < 0:
            raise ValueError("Global budget must be >= 0")

        # We cannot require connectivity if no root is given.
        if self.required_nodes and self.root is None:
            raise ValueError("A root is required when required_nodes is non-empty")

        # Verify that the current graph contains a path from root
        # to every required node BEFORE constructing the QUBO.
        if self.required_nodes:
            reachable = self._reachable_nodes(
                root=self.root,
                edges=self.edges,
            )

            missing_nodes = set(self.required_nodes) - reachable

            if missing_nodes:
                raise ValueError(
                    "Required connectivity is physically infeasible. "
                    f"Unreachable nodes: {sorted(missing_nodes)}"
                )

    # --------------------------------------------------------
    # Graph helpers
    # --------------------------------------------------------

    def _reachable_nodes(
        self,
        root: int,
        edges: Sequence[Edge],
    ) -> set[int]:
        adjacency: Dict[int, List[int]] = {node: [] for node in self.nodes}

        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        visited = {root}
        stack = [root]

        while stack:
            u = stack.pop()

            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        return visited

    def _incident_optional_edges(
        self,
        node: int,
    ) -> List[Edge]:
        return [e for e in self.optional_edges if node in e]

    # --------------------------------------------------------
    # Variable management
    # --------------------------------------------------------

    def _add_variable(self, name: str) -> int:
        idx = len(self.variables)

        self.variables.append(name)
        self.var_index[name] = idx

        return idx

    def _binary_bits_for_integer(
        self,
        name: str,
        max_value: int,
    ) -> List[int]:
        if max_value <= 0:
            return []

        bits = []

        n_bits = int(np.ceil(np.log2(max_value + 1)))

        for k in range(n_bits):
            bits.append(self._add_variable(f"{name}[{k}]"))

        return bits

    def _build_variables(self) -> None:
        # --------------------------------------------
        # Edge variables
        # --------------------------------------------

        for edge in self.optional_edges:
            idx = self._add_variable(f"x_{edge[0]}_{edge[1]}")

            self.edge_var[edge] = idx

        # --------------------------------------------
        # Per-node degree slack
        # --------------------------------------------

        for node in self.nodes:
            limit = self.degree_limits.get(node, 0)

            bits = self._binary_bits_for_integer(
                name=f"s_degree_{node}",
                max_value=limit,
            )

            self.degree_slack_vars[node] = bits

        # --------------------------------------------
        # Global budget slack
        # --------------------------------------------

        self.budget_slack_vars = self._binary_bits_for_integer(
            name="s_budget",
            max_value=self.global_budget,
        )

        # --------------------------------------------
        # Connectivity flow
        #
        # For every undirected edge, create:
        #
        #     f_(u->v)
        #     f_(v->u)
        #
        # Each flow variable is integer-valued,
        # encoded with binary bits.
        # --------------------------------------------

        if self.required_nodes:
            max_flow = len(self.required_nodes) - 1

            for edge in self.edges:
                u, v = edge

                self.flow_var[(u, v, 1)] = self._binary_bits_for_integer(
                    f"f_{u}_{v}",
                    max_flow,
                )

                self.flow_var[(v, u, 1)] = self._binary_bits_for_integer(
                    f"f_{v}_{u}",
                    max_flow,
                )

                self.flow_slack_vars[edge] = self._binary_bits_for_integer(
                    f"q_flow_{u}_{v}",
                    max_flow * 2,
                )

    # --------------------------------------------------------
    # QUBO algebra
    # --------------------------------------------------------

    def _ensure_Q(self) -> None:
        if self.Q is None:
            n = len(self.variables)

            self.Q = np.zeros((n, n), dtype=np.float64)

    def _add_linear(
        self,
        i: int,
        coefficient: float,
    ) -> None:
        self._ensure_Q()
        assert self.Q is not None

        self.Q[i, i] += coefficient

    def _add_quadratic(
        self,
        i: int,
        j: int,
        coefficient: float,
    ) -> None:
        self._ensure_Q()
        assert self.Q is not None

        if i == j:
            self.Q[i, i] += coefficient
            return

        # With z^T Q z and symmetric Q:
        #
        # Q[i,j] z_i z_j + Q[j,i] z_j z_i
        #
        # therefore divide by 2.
        half = coefficient / 2.0

        self.Q[i, j] += half
        self.Q[j, i] += half

    def _add_square_penalty(
        self,
        terms: List[Tuple[int, float]],
        constant: float,
        penalty: float,
    ) -> None:
        """
        Adds:

            penalty * (constant + sum(a_i z_i))^2

        to the QUBO.
        """

        # Constant^2
        self.constant += penalty * constant * constant

        # 2 * constant * a_i * z_i
        for i, a_i in terms:
            self._add_linear(
                i,
                penalty * 2.0 * constant * a_i,
            )

        # a_i^2 z_i^2 = a_i^2 z_i
        for i, a_i in terms:
            self._add_linear(
                i,
                penalty * a_i * a_i,
            )

        # 2 * a_i * a_j z_i z_j
        for idx_a in range(len(terms)):
            i, a_i = terms[idx_a]

            for idx_b in range(idx_a + 1, len(terms)):
                j, a_j = terms[idx_b]

                self._add_quadratic(
                    i,
                    j,
                    penalty * 2.0 * a_i * a_j,
                )

    def _bits_as_terms(
        self,
        bits: Sequence[int],
        multiplier: float = 1.0,
    ) -> List[Tuple[int, float]]:
        return [(idx, multiplier * (2**k)) for k, idx in enumerate(bits)]

    # --------------------------------------------------------
    # Build QUBO
    # --------------------------------------------------------

    def _build_qubo(self) -> None:
        self._ensure_Q()

        # ============================================
        # 1. Edge utility
        #
        #       -V_e * x_e
        #
        # ============================================

        for edge in self.optional_edges:
            features = self.features[edge]

            V_e = edge_utility(
                features,
                self.cfg,
            )

            x = self.edge_var[edge]

            self._add_linear(
                x,
                -V_e,
            )

        # ============================================
        # 2. Temporal stability
        #
        # eta * (x - x_prev)^2
        #
        # If x_prev = 0:
        #       eta*x
        #
        # If x_prev = 1:
        #       eta*(1-x)
        #
        # ============================================

        for edge in self.optional_edges:
            x = self.edge_var[edge]

            prev = self.previous_decisions.get(
                edge,
                0,
            )

            if prev not in (0, 1):
                raise ValueError(f"Previous decision must be 0/1: {edge}")

            # x^2 = x for binary x
            if prev == 0:
                self._add_linear(
                    x,
                    self.cfg.stability_penalty,
                )
            else:
                self.constant += self.cfg.stability_penalty

                self._add_linear(
                    x,
                    -self.cfg.stability_penalty,
                )

        # ============================================
        # 3. Per-node degree constraints
        #
        # sum(x_e) + slack = D_i
        #
        # ============================================

        for node in self.nodes:
            degree_edges = self._incident_optional_edges(node)

            x_terms = [(self.edge_var[e], 1.0) for e in degree_edges]

            slack_terms = self._bits_as_terms(self.degree_slack_vars[node])

            # sum(x) + s - D = 0
            terms = x_terms + slack_terms

            self._add_square_penalty(
                terms=terms,
                constant=-float(self.degree_limits.get(node, 0)),
                penalty=self.cfg.degree_penalty,
            )

        # ============================================
        # 4. Global edge-processing budget
        #
        # sum(cost_e * x_e) + slack = K
        #
        # ============================================

        budget_terms = []

        for edge in self.optional_edges:
            x = self.edge_var[edge]

            # Integer/quantized approximation of cost.
            # For QUBO compatibility, use a positive integer.
            raw_cost = self.features[edge].processing_cost

            quantized_cost = max(
                1,
                int(round(raw_cost * 10)),
            )

            budget_terms.append((x, float(quantized_cost)))

        budget_terms.extend(
            self._bits_as_terms(
                self.budget_slack_vars,
            )
        )

        self._add_square_penalty(
            terms=budget_terms,
            constant=-float(self.global_budget),
            penalty=self.cfg.budget_penalty,
        )

        # ============================================
        # 5. Connectivity flow
        #
        # Every required node must receive one unit
        # of flow from the root.
        #
        # ============================================

        if self.required_nodes:
            self._add_connectivity_constraints()

    # --------------------------------------------------------
    # Connectivity constraints
    # --------------------------------------------------------

    def _flow_bits_to_terms(
        self,
        u: int,
        v: int,
        coefficient: float = 1.0,
    ) -> List[Tuple[int, float]]:
        edge = self._canonical_edge((u, v))

        # Determine direction.
        if (u, v) == edge:
            key = (u, v, 1)
        else:
            key = (u, v, 1)

        bits = self.flow_var[key]

        return self._bits_as_terms(
            bits,
            multiplier=coefficient,
        )

    def _all_flow_terms(
        self,
        node: int,
    ) -> List[Tuple[int, float]]:
        """
        Flow conservation expression at 'node':

            incoming - outgoing
        """

        terms = []

        for edge in self.edges:
            u, v = edge

            if node == u:
                # v -> u incoming
                terms.extend(
                    self._flow_bits_to_terms(
                        v,
                        u,
                        +1.0,
                    )
                )

                # u -> v outgoing
                terms.extend(
                    self._flow_bits_to_terms(
                        u,
                        v,
                        -1.0,
                    )
                )

            elif node == v:
                # u -> v incoming
                terms.extend(
                    self._flow_bits_to_terms(
                        u,
                        v,
                        +1.0,
                    )
                )

                # v -> u outgoing
                terms.extend(
                    self._flow_bits_to_terms(
                        v,
                        u,
                        -1.0,
                    )
                )

        return terms

    def _add_connectivity_constraints(self) -> None:
        required = set(self.required_nodes)
        root = self.root

        assert root is not None

        m = len(required) - 1

        # --------------------------------------------
        # Root:
        #
        # outgoing - incoming = m
        #
        # Equivalent:
        #
        # incoming - outgoing + m = 0
        # --------------------------------------------

        root_flow = self._all_flow_terms(root)

        self._add_square_penalty(
            terms=root_flow,
            constant=float(m),
            penalty=self.cfg.flow_penalty,
        )

        # --------------------------------------------
        # Each non-root required node:
        #
        # incoming - outgoing = 1
        #
        # -> incoming - outgoing - 1 = 0
        # --------------------------------------------

        for node in required:
            if node == root:
                continue

            flow_terms = self._all_flow_terms(node)

            self._add_square_penalty(
                terms=flow_terms,
                constant=-1.0,
                penalty=self.cfg.flow_penalty,
            )

        # --------------------------------------------
        # Flow capacity:
        #
        # f_uv + f_vu + q_e = M * x_e
        #
        # for optional edges.
        #
        # Mandatory edges are always active.
        # --------------------------------------------

        M = len(required) - 1

        for edge in self.edges:
            u, v = edge

            forward = self._flow_bits_to_terms(u, v)
            backward = self._flow_bits_to_terms(v, u)

            slack = self._bits_as_terms(self.flow_slack_vars[edge])

            terms = forward + backward + slack

            if edge in self.hard_critical:
                # No x variable.
                #
                # Flow <= M for mandatory edge.
                #
                # sum(flow) + slack = M
                constant = -float(M)

            else:
                x = self.edge_var[edge]

                terms.append((x, -float(M)))

                constant = 0.0

            self._add_square_penalty(
                terms=terms,
                constant=constant,
                penalty=self.cfg.flow_penalty,
            )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def energy(self, z: np.ndarray) -> float:
        if self.Q is None:
            raise RuntimeError("QUBO has not been built")

        if len(z) != len(self.variables):
            raise ValueError("Invalid solution size")

        return float(z @ self.Q @ z + self.constant)

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    def decode_edges(
        self,
        z: np.ndarray,
    ) -> Tuple[List[Edge], List[Edge]]:
        kept = list(self.hard_critical)
        pruned = []

        for edge in self.optional_edges:
            x = int(round(z[self.edge_var[edge]]))

            if x == 1:
                kept.append(edge)
            else:
                pruned.append(edge)

        kept = sorted(set(kept))
        pruned = sorted(set(pruned))

        return kept, pruned

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    def solve(
        self,
    ) -> QUBOResult:
        assert self.Q is not None

        annealer = SimulatedAnnealing(
            Q=self.Q,
            constant=self.constant,
            config=self.cfg,
        )

        z, energy = annealer.run()

        kept, pruned = self.decode_edges(z)

        return QUBOResult(
            variables=self.variables,
            solution=z,
            energy=energy,
            kept_edges=kept,
            pruned_edges=pruned,
        )


# ============================================================
# Simulated Annealing
# ============================================================


class SimulatedAnnealing:
    """
    Minimizes:

        E(z) = z^T Q z + c

    for binary z.
    """

    def __init__(
        self,
        Q: np.ndarray,
        constant: float,
        config: QUBOConfig,
    ):
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square")

        if not np.allclose(Q, Q.T):
            raise ValueError("Q must be symmetric")

        self.Q = Q
        self.constant = constant
        self.cfg = config

        self.rng = Random(config.seed)

    def energy(self, z: np.ndarray) -> float:
        return float(z @ self.Q @ z + self.constant)

    def delta_energy(
        self,
        z: np.ndarray,
        i: int,
    ) -> float:
        """
        Energy change if z_i flips.

        For:

            E = z^T Q z + c

        flipping z_i gives:

            ΔE = (1 - 2z_i)
                 * (2 * sum_j Q_ij z_j + Q_ii)
        """

        zi = z[i]

        interaction = (
            2.0
            * np.dot(
                self.Q[i],
                z,
            )
            - self.Q[i, i] * zi
        )

        # Equivalent and numerically stable derivation:
        #
        # E(new) - E(old)

        new_value = 1.0 - zi

        old_contribution = (
            self.Q[i, i] * zi
            + 2.0
            * np.dot(
                self.Q[i],
                z,
            )
            - 2.0 * self.Q[i, i] * zi
        )

        new_contribution = (
            self.Q[i, i] * new_value
            + 2.0
            * np.dot(
                self.Q[i],
                z,
            )
            - 2.0 * self.Q[i, i] * zi
        )

        return float(new_contribution - old_contribution)

    def _random_solution(
        self,
        n: int,
    ) -> np.ndarray:
        return np.array(
            [self.rng.randint(0, 1) for _ in range(n)],
            dtype=np.float64,
        )

    def _run_once(
        self,
    ) -> Tuple[np.ndarray, float]:

        n = self.Q.shape[0]

        z = self._random_solution(n)

        current_energy = self.energy(z)

        temperature = self.cfg.initial_temperature

        while temperature > self.cfg.final_temperature:
            for _ in range(self.cfg.sweeps_per_temperature):
                indices = list(range(n))

                self.rng.shuffle(indices)

                for i in indices:
                    delta = self.delta_energy(
                        z,
                        i,
                    )

                    if delta <= 0:
                        accept = True
                    else:
                        probability = exp(-delta / temperature)

                        accept = self.rng.random() < probability

                    if accept:
                        z[i] = 1.0 - z[i]
                        current_energy += delta

            temperature *= self.cfg.cooling_rate

        return z, current_energy

    def run(self) -> Tuple[np.ndarray, float]:

        best_z: Optional[np.ndarray] = None
        best_energy = float("inf")

        for _ in range(self.cfg.restarts):
            z, energy = self._run_once()

            if energy < best_energy:
                best_energy = energy
                best_z = z.copy()

        assert best_z is not None

        return best_z, best_energy


# ============================================================
# Example
# ============================================================


def build_example() -> EdgeSelectionQUBO:

    nodes = [0, 1, 2, 3]

    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    # These should come from the previous system cycle.
    features = {
        (0, 1): EdgeFeatures(
            trust=0.95,
            link_quality=0.92,
            battery=0.85,
            proximity=0.90,
            freshness=0.95,
            task_criticality=0.20,
            processing_cost=0.30,
        ),
        (0, 2): EdgeFeatures(
            trust=0.80,
            link_quality=0.80,
            battery=0.70,
            proximity=0.85,
            freshness=0.90,
            task_criticality=0.10,
            processing_cost=0.40,
        ),
        (0, 3): EdgeFeatures(
            trust=0.40,
            link_quality=0.30,
            battery=0.50,
            proximity=0.50,
            freshness=0.50,
            task_criticality=0.00,
            processing_cost=0.90,
        ),
        (1, 2): EdgeFeatures(
            trust=0.88,
            link_quality=0.85,
            battery=0.80,
            proximity=0.75,
            freshness=0.95,
            task_criticality=0.10,
            processing_cost=0.30,
        ),
        (1, 3): EdgeFeatures(
            trust=0.30,
            link_quality=0.35,
            battery=0.35,
            proximity=0.50,
            freshness=0.40,
            task_criticality=0.80,
            processing_cost=0.90,
        ),
        (2, 3): EdgeFeatures(
            trust=0.90,
            link_quality=0.88,
            battery=0.75,
            proximity=0.80,
            freshness=0.90,
            task_criticality=0.30,
            processing_cost=0.35,
        ),
    }

    previous = {
        (0, 1): 1,
        (0, 2): 1,
        (0, 3): 1,
        (1, 2): 1,
        (1, 3): 0,
        (2, 3): 1,
    }

    # Example: edge (2,3) is required by the mission.
    #
    # In the full system this should be determined
    # from the current task/connectivity requirements.
    hard_critical_edges = {
        (2, 3),
    }

    # Required nodes must remain mutually reachable.
    required_nodes = [0, 1, 2, 3]

    root = 0

    # Maximum OPTIONAL degree per robot.
    degree_limits = {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }

    # Quantized global optional-edge processing budget.
    #
    # Larger value = more edges can survive.
    global_budget = 30

    config = QUBOConfig(
        w_trust=0.25,
        w_link=0.25,
        w_battery=0.10,
        w_proximity=0.10,
        w_freshness=0.15,
        w_task=0.15,
        cost_weight=0.20,
        stability_penalty=0.10,
        degree_penalty=20.0,
        budget_penalty=20.0,
        flow_penalty=200.0,
        initial_temperature=5.0,
        final_temperature=0.01,
        cooling_rate=0.995,
        sweeps_per_temperature=5,
        restarts=10,
        seed=42,
    )

    return EdgeSelectionQUBO(
        nodes=nodes,
        edges=edges,
        features=features,
        previous_decisions=previous,
        hard_critical_edges=hard_critical_edges,
        required_nodes=required_nodes,
        root=root,
        degree_limits=degree_limits,
        global_budget=global_budget,
        config=config,
    )


if __name__ == "__main__":
    qubo = build_example()

    print(f"QUBO variables: {len(qubo.variables)}")
    print(f"QUBO shape: {qubo.Q.shape}")

    result = qubo.solve()

    print("\nEnergy:")
    print(result.energy)

    print("\nKept edges:")
    for edge in result.kept_edges:
        print("  KEEP  ", edge)

    print("\nPruned edges:")
    for edge in result.pruned_edges:
        print("  PRUNE ", edge)
