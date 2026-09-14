from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log2
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Node = int
Edge = Tuple[Node, Node]


def canonical_edge(edge: Edge) -> Edge:
    u, v = edge
    if u == v:
        raise ValueError(f"Self-loop is not allowed: {edge}")
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class EdgeFeatures:
    """All feature values are normalized to [0, 1]."""

    trust: float
    link_quality: float
    battery: float
    proximity: float
    freshness: float
    task_criticality: float
    processing_cost: float


@dataclass(frozen=True)
class QUBOConfig:
    # Edge utility weights: sum must equal 1.
    w_trust: float = 0.25
    w_link: float = 0.25
    w_battery: float = 0.10
    w_proximity: float = 0.10
    w_freshness: float = 0.15
    w_task: float = 0.15

    # Soft objective.
    edge_penalty: float = 0.08
    stability_penalty: float = 0.15

    # Hard-constraint penalties. Must dominate the soft objective.
    degree_penalty: float = 100.0
    budget_penalty: float = 100.0
    flow_penalty: float = 500.0
    flow_capacity_penalty: float = 500.0

    # Quantization of normalized processing_cost into integer budget units.
    cost_scale: int = 10

    # Simulated annealing.
    initial_temperature: float = 10.0
    final_temperature: float = 1e-3
    cooling_rate: float = 0.98
    sweeps_per_temperature: int = 2
    restarts: int = 20
    seed: int = 42


@dataclass
class QUBOResult:
    energy: float
    kept_edges: List[Edge]
    pruned_edges: List[Edge]
    solution: np.ndarray
    variable_names: List[str]
    violated_constraints: List[str]


class EdgeSelectionQUBO:
    """
    Constrained QUBO for dynamic graph sparsification.

    The decision variables select optional edges. Hard-critical edges are fixed
    to 1 and are not QUBO decision variables.

    Soft objective:
        maximize edge utility
        minimize number of retained edges
        minimize topology switching

    Hard constraints are encoded with binary slack variables:
        - per-node degree upper bound
        - global processing-cost budget
        - required-node connectivity via single-commodity flow

    Final QUBO form:
        H(z) = z.T @ Q @ z + constant
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        edges: Sequence[Edge],
        features: Dict[Edge, EdgeFeatures],
        previous_decisions: Optional[Dict[Edge, int]] = None,
        hard_critical_edges: Iterable[Edge] = (),
        required_nodes: Optional[Sequence[Node]] = None,
        root: Optional[Node] = None,
        degree_limits: Optional[Dict[Node, int]] = None,
        global_budget: Optional[int] = None,
        config: Optional[QUBOConfig] = None,
    ) -> None:
        self.nodes = sorted(set(nodes))
        self.edges = sorted({canonical_edge(e) for e in edges})
        self.features = {canonical_edge(e): value for e, value in features.items()}
        self.previous = {
            canonical_edge(e): int(value)
            for e, value in (previous_decisions or {}).items()
        }
        self.hard_critical = {canonical_edge(e) for e in hard_critical_edges}
        self.required_nodes = sorted(set(required_nodes or []))
        self.root = root
        self.degree_limits = dict(degree_limits or {})
        self.global_budget = global_budget
        self.cfg = config or QUBOConfig()

        self.optional_edges = [e for e in self.edges if e not in self.hard_critical]
        self.edge_cost_units: Dict[Edge, int] = {}

        self.variables: List[str] = []
        self.var_index: Dict[str, int] = {}
        self.edge_var: Dict[Edge, int] = {}
        self.degree_slack_bits: Dict[Node, List[int]] = {}
        self.budget_slack_bits: List[int] = []
        self.flow_bits: Dict[Tuple[Node, Node], List[int]] = {}
        self.flow_slack_bits: Dict[Edge, List[int]] = {}

        self.Q = np.zeros((0, 0), dtype=float)
        self.constant = 0.0

        self._validate_inputs()
        self._prepare_costs()
        self._build_variables()
        self.Q = np.zeros((len(self.variables), len(self.variables)), dtype=float)
        self._build_objective()
        self._build_degree_constraints()
        self._build_budget_constraint()
        self._build_connectivity_constraints()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_unit(name: str, value: float) -> None:
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value}")

    def _validate_inputs(self) -> None:
        if not self.nodes:
            raise ValueError("At least one node is required")

        node_set = set(self.nodes)
        for edge in self.edges:
            if edge[0] not in node_set or edge[1] not in node_set:
                raise ValueError(f"Unknown node in edge {edge}")

        edge_set = set(self.edges)

        unknown_hard = self.hard_critical - edge_set
        if unknown_hard:
            raise ValueError(
                f"hard_critical_edges not present in edges: {sorted(unknown_hard)}"
            )

        missing = edge_set - set(self.features)
        if missing:
            raise ValueError(f"Missing features for edges: {sorted(missing)}")

        unknown_feature_edges = set(self.features) - edge_set
        if unknown_feature_edges:
            raise ValueError(
                f"Features given for edges not in the graph: {sorted(unknown_feature_edges)}"
            )

        # previous_decisions are historical: entries for edges that have since
        # disappeared are ignored (dynamic graphs), but values must be binary.
        for edge, value in self.previous.items():
            if value not in (0, 1):
                raise ValueError(
                    f"previous_decisions[{edge}] must be 0 or 1, got {value}"
                )

        for edge, feature in self.features.items():
            for name, value in vars(feature).items():
                self._validate_unit(f"{edge}.{name}", value)

        if self.required_nodes:
            if self.root is None or self.root not in self.required_nodes:
                raise ValueError("root must be one of required_nodes")
            if not set(self.required_nodes).issubset(node_set):
                raise ValueError("required_nodes contains an unknown node")
            if not self._required_nodes_reachable():
                raise ValueError(
                    "Required connectivity is physically infeasible in the current graph"
                )

        for node, limit in self.degree_limits.items():
            if node not in node_set:
                raise ValueError(f"Degree limit given for unknown node {node}")
            if limit < 0:
                raise ValueError("Degree limits must be non-negative")

        if self.global_budget is not None and self.global_budget < 0:
            raise ValueError("global_budget must be non-negative")

        if not np.isclose(
            sum(
                [
                    self.cfg.w_trust,
                    self.cfg.w_link,
                    self.cfg.w_battery,
                    self.cfg.w_proximity,
                    self.cfg.w_freshness,
                    self.cfg.w_task,
                ]
            ),
            1.0,
        ):
            raise ValueError("Utility weights must sum to 1")

    def _required_nodes_reachable(self) -> bool:
        assert self.root is not None
        adjacency = {node: [] for node in self.nodes}
        for u, v in self.edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        seen = {self.root}
        stack = [self.root]
        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        return set(self.required_nodes).issubset(seen)

    def _prepare_costs(self) -> None:
        if self.cfg.cost_scale <= 0:
            raise ValueError("cost_scale must be positive")

        self.edge_cost_units = {
            edge: max(
                1,
                int(round(feature.processing_cost * self.cfg.cost_scale)),
            )
            for edge, feature in self.features.items()
        }

        fixed_degree = {node: 0 for node in self.nodes}
        fixed_cost = 0
        for u, v in self.hard_critical:
            fixed_degree[u] += 1
            fixed_degree[v] += 1
            fixed_cost += self.edge_cost_units[(u, v)]

        for node, limit in self.degree_limits.items():
            if fixed_degree[node] > limit:
                raise ValueError(
                    f"Hard-critical edges exceed degree limit for node {node}"
                )

        if self.global_budget is not None and fixed_cost > self.global_budget:
            raise ValueError("Hard-critical edges already exceed global budget")

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def add_variable(self, name: str) -> int:
        if name in self.var_index:
            raise ValueError(f"Duplicate variable {name}")
        idx = len(self.variables)
        self.variables.append(name)
        self.var_index[name] = idx
        return idx

    def add_integer_bits(self, prefix: str, max_value: int) -> List[int]:
        """Binary encode an integer in [0, max_value]."""
        if max_value < 0:
            raise ValueError("max_value must be non-negative")
        if max_value == 0:
            return []
        # Equality constraints prevent unused high values, so a power-of-two
        # range slightly larger than max_value is safe.
        n_bits = int(ceil(log2(max_value + 1)))
        return [self.add_variable(f"{prefix}[{k}]") for k in range(n_bits)]

    @staticmethod
    def bits_to_terms(bits: Sequence[int], coefficient: float = 1.0) -> List[Tuple[int, float]]:
        return [(idx, coefficient * float(2**k)) for k, idx in enumerate(bits)]

    def _build_variables(self) -> None:
        for edge in self.optional_edges:
            self.edge_var[edge] = self.add_variable(
                f"x_{edge[0]}_{edge[1]}"
            )

        fixed_degree = {node: 0 for node in self.nodes}
        for u, v in self.hard_critical:
            fixed_degree[u] += 1
            fixed_degree[v] += 1

        for node, limit in self.degree_limits.items():
            remaining = limit - fixed_degree[node]
            self.degree_slack_bits[node] = self.add_integer_bits(
                f"s_degree_{node}", remaining
            )

        if self.global_budget is not None:
            fixed_cost = sum(self.edge_cost_units[e] for e in self.hard_critical)
            remaining = self.global_budget - fixed_cost
            self.budget_slack_bits = self.add_integer_bits(
                "s_budget", remaining
            )

        if self.required_nodes:
            max_flow = len(self.required_nodes) - 1
            for u, v in self.edges:
                self.flow_bits[(u, v)] = self.add_integer_bits(
                    f"f_{u}_{v}", max_flow
                )
                self.flow_bits[(v, u)] = self.add_integer_bits(
                    f"f_{v}_{u}", max_flow
                )
                self.flow_slack_bits[(u, v)] = self.add_integer_bits(
                    f"q_{u}_{v}", max_flow
                )

    # ------------------------------------------------------------------
    # QUBO algebra
    # ------------------------------------------------------------------

    def _add_linear(self, i: int, coefficient: float) -> None:
        self.Q[i, i] += coefficient

    def _add_pair(self, i: int, j: int, coefficient: float) -> None:
        # z.T Q z counts symmetric off-diagonal terms twice.
        if i == j:
            self.Q[i, i] += coefficient
        else:
            half = coefficient / 2.0
            self.Q[i, j] += half
            self.Q[j, i] += half

    def _add_square(
        self,
        terms: Sequence[Tuple[int, float]],
        constant: float,
        penalty: float,
    ) -> None:
        """
        Add penalty * (constant + sum(a_i z_i))^2.
        """
        if penalty < 0:
            raise ValueError("Penalty must be non-negative")

        self.constant += penalty * constant * constant

        # Combine duplicate variable terms first. This is important because
        # repeated variables must contribute a_i^2 * z_i, not a spurious
        # z_i*z_i matrix term.
        combined: Dict[int, float] = {}
        for idx, coeff in terms:
            combined[idx] = combined.get(idx, 0.0) + coeff

        items = list(combined.items())

        for idx, coeff in items:
            self._add_linear(
                idx,
                penalty * (2.0 * constant * coeff + coeff * coeff),
            )

        for a in range(len(items)):
            i, coeff_i = items[a]
            for b in range(a + 1, len(items)):
                j, coeff_j = items[b]
                self._add_pair(
                    i,
                    j,
                    penalty * 2.0 * coeff_i * coeff_j,
                )

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _build_objective(self) -> None:
        cfg = self.cfg
        weights = np.array(
            [
                cfg.w_trust,
                cfg.w_link,
                cfg.w_battery,
                cfg.w_proximity,
                cfg.w_freshness,
                cfg.w_task,
            ],
            dtype=float,
        )

        for edge in self.optional_edges:
            f = self.features[edge]
            utility = float(
                weights
                @ np.array(
                    [
                        f.trust,
                        f.link_quality,
                        f.battery,
                        f.proximity,
                        f.freshness,
                        f.task_criticality,
                    ],
                    dtype=float,
                )
            )

            # -utility rewards good edges; +edge_penalty discourages density.
            self._add_linear(
                self.edge_var[edge],
                -utility + cfg.edge_penalty,
            )

            # eta * (x - x_prev)^2. For binary x this is linear up to a constant.
            previous = self.previous.get(edge, 0)
            if previous == 0:
                self._add_linear(self.edge_var[edge], cfg.stability_penalty)
            elif previous == 1:
                self.constant += cfg.stability_penalty
                self._add_linear(self.edge_var[edge], -cfg.stability_penalty)
            else:
                raise ValueError("previous_decisions values must be 0 or 1")

    # ------------------------------------------------------------------
    # Degree and budget constraints
    # ------------------------------------------------------------------

    def _build_degree_constraints(self) -> None:
        fixed_degree = {node: 0 for node in self.nodes}
        for u, v in self.hard_critical:
            fixed_degree[u] += 1
            fixed_degree[v] += 1

        for node, limit in self.degree_limits.items():
            rhs = limit - fixed_degree[node]
            terms = [
                (self.edge_var[edge], 1.0)
                for edge in self.optional_edges
                if node in edge
            ]
            terms += self.bits_to_terms(self.degree_slack_bits[node])
            self._add_square(
                terms,
                constant=-float(rhs),
                penalty=self.cfg.degree_penalty,
            )

    def _build_budget_constraint(self) -> None:
        if self.global_budget is None:
            return

        fixed_cost = sum(self.edge_cost_units[e] for e in self.hard_critical)
        rhs = self.global_budget - fixed_cost

        terms = [
            (self.edge_var[edge], float(self.edge_cost_units[edge]))
            for edge in self.optional_edges
        ]
        terms += self.bits_to_terms(self.budget_slack_bits)

        self._add_square(
            terms,
            constant=-float(rhs),
            penalty=self.cfg.budget_penalty,
        )

    # ------------------------------------------------------------------
    # Connectivity flow constraints
    # ------------------------------------------------------------------

    def _flow_terms_at_node(self, node: Node) -> List[Tuple[int, float]]:
        """Return incoming - outgoing flow expression."""
        terms: List[Tuple[int, float]] = []

        for u, v in self.edges:
            if node == u:
                terms += self.bits_to_terms(self.flow_bits[(v, u)], +1.0)
                terms += self.bits_to_terms(self.flow_bits[(u, v)], -1.0)
            elif node == v:
                terms += self.bits_to_terms(self.flow_bits[(u, v)], +1.0)
                terms += self.bits_to_terms(self.flow_bits[(v, u)], -1.0)

        return terms

    def _build_connectivity_constraints(self) -> None:
        if not self.required_nodes:
            return

        assert self.root is not None
        required_set = set(self.required_nodes)
        units = len(self.required_nodes) - 1

        # Root sends one unit to every other required node.
        # incoming - outgoing = -units
        self._add_square(
            self._flow_terms_at_node(self.root),
            constant=float(units),
            penalty=self.cfg.flow_penalty,
        )

        # Each required non-root node consumes one unit.
        # incoming - outgoing = +1
        for node in self.required_nodes:
            if node == self.root:
                continue
            self._add_square(
                self._flow_terms_at_node(node),
                constant=-1.0,
                penalty=self.cfg.flow_penalty,
            )

        # Non-required nodes can be Steiner/transshipment nodes.
        for node in self.nodes:
            if node in required_set:
                continue
            self._add_square(
                self._flow_terms_at_node(node),
                constant=0.0,
                penalty=self.cfg.flow_penalty,
            )

        # Flow capacity:
        #   f_uv + f_vu <= units * x_e
        # converted into equality with slack.
        for edge in self.edges:
            u, v = edge
            terms = []
            terms += self.bits_to_terms(self.flow_bits[(u, v)], +1.0)
            terms += self.bits_to_terms(self.flow_bits[(v, u)], +1.0)
            terms += self.bits_to_terms(self.flow_slack_bits[edge], +1.0)

            if edge in self.hard_critical:
                # x_e is fixed to 1:
                # flow + slack = units
                constant = -float(units)
            else:
                # flow + slack - units*x_e = 0
                terms.append((self.edge_var[edge], -float(units)))
                constant = 0.0

            self._add_square(
                terms,
                constant=constant,
                penalty=self.cfg.flow_capacity_penalty,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_qubo(self) -> Tuple[np.ndarray, float, List[str]]:
        return self.Q.copy(), float(self.constant), list(self.variables)

    def soft_energy(self, selected: Iterable[Edge]) -> float:
        """
        Soft-objective energy of an edge selection, assuming feasibility
        (all penalty terms are zero). For any feasible state z this equals
        energy(z), but costs O(m) instead of O(V^2).
        """
        cfg = self.cfg
        selected = {canonical_edge(e) for e in selected}
        weights = np.array(
            [
                cfg.w_trust,
                cfg.w_link,
                cfg.w_battery,
                cfg.w_proximity,
                cfg.w_freshness,
                cfg.w_task,
            ],
            dtype=float,
        )
        total = 0.0
        for edge in self.optional_edges:
            f = self.features[edge]
            utility = float(
                weights
                @ np.array(
                    [
                        f.trust,
                        f.link_quality,
                        f.battery,
                        f.proximity,
                        f.freshness,
                        f.task_criticality,
                    ],
                    dtype=float,
                )
            )
            x = int(edge in selected)
            if x:
                total += -utility + cfg.edge_penalty
            total += cfg.stability_penalty * (x - self.previous.get(edge, 0)) ** 2
        return total

    def energy(self, z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        if z.shape != (len(self.variables),):
            raise ValueError("Invalid solution shape")
        return float(z @ self.Q @ z + self.constant)

    def selected_edges(self, z: np.ndarray) -> List[Edge]:
        z = np.rint(z).astype(int)
        selected = set(self.hard_critical)
        selected.update(
            edge
            for edge in self.optional_edges
            if z[self.edge_var[edge]] == 1
        )
        return sorted(selected)

    def check_constraints(self, z: np.ndarray) -> List[str]:
        """Check graph-level feasibility of a decoded solution."""
        selected = self.selected_edges(z)
        issues: List[str] = []

        # Degree constraints.
        for node, limit in self.degree_limits.items():
            degree = sum(node in edge for edge in selected)
            if degree > limit:
                issues.append(f"degree[{node}]={degree}>{limit}")

        # Processing-cost budget.
        if self.global_budget is not None:
            cost = sum(self.edge_cost_units[e] for e in selected)
            if cost > self.global_budget:
                issues.append(f"budget={cost}>{self.global_budget}")

        # Required connectivity.
        if self.required_nodes:
            assert self.root is not None
            adjacency = {node: [] for node in self.nodes}
            for u, v in selected:
                adjacency[u].append(v)
                adjacency[v].append(u)

            seen = {self.root}
            stack = [self.root]
            while stack:
                u = stack.pop()
                for v in adjacency[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)

            missing = sorted(set(self.required_nodes) - seen)
            if missing:
                issues.append(f"connectivity_missing={missing}")

        return issues

    def _set_integer_bits(self, z: np.ndarray, bits: Sequence[int], value: int) -> None:
        value = int(value)
        max_representable = (1 << len(bits)) - 1
        if value < 0 or value > max_representable:
            raise ValueError(
                f"Cannot encode integer {value} with {len(bits)} bits"
            )
        for k, idx in enumerate(bits):
            z[idx] = float((value >> k) & 1)

    def build_feasible_solution(self, selected_edges: Iterable[Edge]) -> np.ndarray:
        """
        Construct a fully consistent binary QUBO state for a graph-level
        feasible edge set. This is used as a strong SA warm start and as a
        final repair/reconstruction step.
        """
        selected = {canonical_edge(e) for e in selected_edges}
        if not self.hard_critical.issubset(selected):
            raise ValueError("Selected edges must include every hard-critical edge")

        # Validate graph-level constraints first.
        z = np.zeros(len(self.variables), dtype=float)
        for edge in self.optional_edges:
            z[self.edge_var[edge]] = 1.0 if edge in selected else 0.0

        issues = self.check_constraints(z)
        if issues:
            raise ValueError(f"Selected graph is infeasible: {issues}")

        # Degree slack.
        for node, limit in self.degree_limits.items():
            fixed_degree = sum(node in e for e in self.hard_critical)
            optional_degree = sum(
                node in e and e not in self.hard_critical
                for e in selected
            )
            rhs = limit - fixed_degree
            slack_value = rhs - optional_degree
            self._set_integer_bits(z, self.degree_slack_bits[node], slack_value)

        # Global budget slack.
        if self.global_budget is not None:
            fixed_cost = sum(self.edge_cost_units[e] for e in self.hard_critical)
            optional_cost = sum(
                self.edge_cost_units[e]
                for e in selected
                if e in self.optional_edges
            )
            rhs = self.global_budget - fixed_cost
            slack_value = rhs - optional_cost
            self._set_integer_bits(z, self.budget_slack_bits, slack_value)

        if self.required_nodes:
            self._populate_connectivity_flow(z, selected)

        # Verify both graph-level and exact QUBO constraints by checking the
        # energy against the equations indirectly through a fresh residual check.
        if self._constraint_energy(z) > 1e-7:
            raise RuntimeError("Constructed feasible state does not satisfy QUBO constraints")

        return z

    def _populate_connectivity_flow(self, z: np.ndarray, selected: set[Edge]) -> None:
        assert self.root is not None
        required = set(self.required_nodes)
        units = len(self.required_nodes) - 1

        # Build a spanning tree of the selected graph rooted at self.root.
        adjacency = {node: [] for node in self.nodes}
        for u, v in selected:
            adjacency[u].append(v)
            adjacency[v].append(u)

        parent: Dict[Node, Optional[Node]] = {self.root: None}
        order = [self.root]
        for u in order:
            for v in adjacency[u]:
                if v not in parent:
                    parent[v] = u
                    order.append(v)

        missing = required - set(parent)
        if missing:
            raise ValueError(f"Cannot construct connectivity flow; missing {sorted(missing)}")

        # Demand is 1 for each non-root required node and 0 for other nodes.
        subtree_demand = {node: (1 if node in required and node != self.root else 0) for node in self.nodes}

        # Process reverse BFS/DFS order. Flow goes parent -> child.
        for node in reversed(order):
            parent_node = parent[node]
            if parent_node is None:
                continue
            amount = subtree_demand[node]
            subtree_demand[parent_node] += amount

            edge = canonical_edge((parent_node, node))
            if amount > units:
                raise RuntimeError("Flow amount exceeds theoretical maximum")

            if edge not in selected:
                raise RuntimeError("Flow tree used a non-selected edge")

            self._set_integer_bits(z, self.flow_bits[(parent_node, node)], amount)
            self._set_integer_bits(z, self.flow_bits[(node, parent_node)], 0)

            slack = units - amount
            self._set_integer_bits(z, self.flow_slack_bits[edge], slack)

        # Edges selected but not used by the spanning tree carry zero flow.
        used_tree_edges = {
            canonical_edge((parent[node], node))
            for node in order
            if parent[node] is not None
        }
        for edge in self.edges:
            if edge in used_tree_edges:
                continue
            self._set_integer_bits(z, self.flow_bits[(edge[0], edge[1])], 0)
            self._set_integer_bits(z, self.flow_bits[(edge[1], edge[0])], 0)
            # For a selected edge: slack = units. For a pruned edge: slack = 0.
            slack = units if edge in selected else 0
            self._set_integer_bits(z, self.flow_slack_bits[edge], slack)

        # Sanity: root should have total outgoing flow = units.
        if subtree_demand[self.root] != units:
            raise RuntimeError("Incorrect root flow")

    def _constraint_energy(self, z: np.ndarray) -> float:
        """
        Compute only the penalty contribution by rebuilding a clean copy of
        the soft objective-free constraints. This is used for feasibility
        verification and does not depend on penalty coefficient scaling.
        """
        # Degree residuals.
        total = 0.0
        selected = self.selected_edges(z)

        fixed_degree = {node: 0 for node in self.nodes}
        for u, v in self.hard_critical:
            fixed_degree[u] += 1
            fixed_degree[v] += 1

        for node, limit in self.degree_limits.items():
            optional_degree = sum(
                node in e and e not in self.hard_critical
                for e in selected
            )
            rhs = limit - fixed_degree[node]
            slack = self._decode_int(z, self.degree_slack_bits[node])
            total += (optional_degree + slack - rhs) ** 2

        if self.global_budget is not None:
            fixed_cost = sum(self.edge_cost_units[e] for e in self.hard_critical)
            optional_cost = sum(
                self.edge_cost_units[e]
                for e in selected
                if e in self.optional_edges
            )
            rhs = self.global_budget - fixed_cost
            slack = self._decode_int(z, self.budget_slack_bits)
            total += (optional_cost + slack - rhs) ** 2

        if self.required_nodes:
            # Verify flow conservation and capacity directly.
            assert self.root is not None
            units = len(self.required_nodes) - 1

            for node in self.nodes:
                incoming = 0
                outgoing = 0
                for u, v in self.edges:
                    if node == u:
                        incoming += self._decode_int(z, self.flow_bits[(v, u)])
                        outgoing += self._decode_int(z, self.flow_bits[(u, v)])
                    elif node == v:
                        incoming += self._decode_int(z, self.flow_bits[(u, v)])
                        outgoing += self._decode_int(z, self.flow_bits[(v, u)])

                if node == self.root:
                    total += (incoming - outgoing + units) ** 2
                elif node in self.required_nodes:
                    total += (incoming - outgoing - 1) ** 2
                else:
                    total += (incoming - outgoing) ** 2

            for edge in self.edges:
                flow = self._decode_int(z, self.flow_bits[edge]) + self._decode_int(z, self.flow_bits[(edge[1], edge[0])])
                slack = self._decode_int(z, self.flow_slack_bits[edge])
                x = 1 if edge in self.hard_critical else int(z[self.edge_var[edge]])
                total += (flow + slack - units * x) ** 2

        return float(total)

    def _decode_int(self, z: np.ndarray, bits: Sequence[int]) -> int:
        return sum(int(round(z[idx])) * (2**k) for k, idx in enumerate(bits))

    def extract_solution(self, z: np.ndarray) -> QUBOResult:
        z = np.rint(z).astype(int)
        kept = self.selected_edges(z)
        pruned = sorted(set(self.edges) - set(kept))
        issues = self.check_constraints(z)
        return QUBOResult(
            energy=self.energy(z),
            kept_edges=kept,
            pruned_edges=pruned,
            solution=z,
            variable_names=list(self.variables),
            violated_constraints=issues,
        )


class ProjectedEdgeAnnealing:
    """
    Simulated annealing over feasible edge subsets.

    Bit-level SA on the penalty-encoded QUBO cannot cross penalty barriers
    (~100+ per violated constraint) with single-bit flips, so it effectively
    never leaves its warm start. This solver instead toggles optional edges at
    graph level (single toggles and two-edge swaps) and projects every
    candidate back to a fully feasible QUBO state via
    EdgeSelectionQUBO.build_feasible_solution. Candidates that cannot be
    projected are rejected. The objective is the same QUBO energy.
    """

    def __init__(
        self,
        builder: EdgeSelectionQUBO,
        *,
        initial_temperature: float = 1.0,
        final_temperature: float = 1e-2,
        cooling_rate: float = 0.998,
        steps_per_temperature: int = 4,
        swap_probability: float = 0.5,
        restarts: int = 5,
        seed: int = 42,
        initial_edge_sets: Optional[Sequence[Iterable[Edge]]] = None,
    ) -> None:
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0,1)")
        if initial_temperature <= 0 or final_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if final_temperature >= initial_temperature:
            raise ValueError("final_temperature must be below initial_temperature")
        if steps_per_temperature <= 0 or restarts <= 0:
            raise ValueError("steps_per_temperature and restarts must be positive")
        if not 0.0 <= swap_probability <= 1.0:
            raise ValueError("swap_probability must be in [0,1]")

        self.builder = builder
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.steps = steps_per_temperature
        self.swap_probability = swap_probability
        self.restarts = restarts
        self.rng = np.random.default_rng(seed)
        self.energy_evals = 0
        self.initial_edge_sets = [
            {canonical_edge(e) for e in edges} for edges in (initial_edge_sets or [])
        ]

    def _project(self, selected: set[Edge]) -> Tuple[np.ndarray, float]:
        # build_feasible_solution validates feasibility; for feasible states
        # the QUBO energy equals the soft objective, which is O(m).
        self.energy_evals += 1
        z = self.builder.build_feasible_solution(selected)
        return z, self.builder.soft_energy(selected)

    def _random_spanning_tree(self) -> Optional[set[Edge]]:
        """
        Prim-style random spanning tree of the root's component that respects
        residual degree caps (hard edges pre-consume degree), so the result is
        feasible by construction whenever the caps allow any tree.
        """
        builder = self.builder
        assert builder.root is not None
        adjacency = {n: [] for n in builder.nodes}
        for u, v in builder.edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        residual = {
            n: builder.degree_limits.get(n, len(builder.nodes))
            - sum(n in e for e in builder.hard_critical)
            for n in builder.nodes
        }
        # reachable component of the root
        target = {builder.root}
        stack = [builder.root]
        while stack:
            for v in adjacency[stack.pop()]:
                if v not in target:
                    target.add(v)
                    stack.append(v)

        in_tree = {builder.root}
        tree: set[Edge] = set()
        while in_tree != target:
            frontier = [
                (u, v)
                for u in in_tree
                for v in adjacency[u]
                if v not in in_tree and residual[u] > 0 and residual[v] > 0
            ]
            if not frontier:
                return None
            u, v = frontier[int(self.rng.integers(len(frontier)))]
            tree.add(canonical_edge((u, v)))
            residual[u] -= 1
            residual[v] -= 1
            in_tree.add(v)
        return tree

    def _random_feasible_start(self) -> set[Edge]:
        builder = self.builder
        selected = set(builder.hard_critical)
        if builder.required_nodes:
            # Feasibility is NOT prefix-monotone under connectivity: growing
            # edge-by-edge can never pass through the infeasible partial sets
            # to reach a connected one. Seed with a random spanning tree of
            # the required component instead, retrying on degree/budget caps.
            for _ in range(10):
                tree = self._random_spanning_tree()
                if tree is None:
                    continue
                candidate = set(builder.hard_critical) | tree
                try:
                    builder.build_feasible_solution(candidate)
                except ValueError:
                    continue
                selected = candidate
                break
            else:
                return selected  # infeasible; run() will reject it
        order = self.rng.permutation(len(builder.optional_edges))
        for k in order:
            candidate = selected | {builder.optional_edges[k]}
            try:
                builder.build_feasible_solution(candidate)
            except ValueError:
                continue
            selected = candidate
        return selected

    def _polish(self, start: set[Edge], energy: float) -> Tuple[set[Edge], float]:
        """
        Deterministic local descent: alternate single-edge toggles and
        pairwise-swap passes until a fixpoint (bounded rounds). Removes the
        residual stochastic gap left by temperature scheduling.
        """
        options = self.builder.optional_edges
        current, current_energy = set(start), energy

        for _ in range(4):  # bounded fixpoint rounds
            improved = False

            toggle_improved = True
            while toggle_improved:
                toggle_improved = False
                for edge in options:
                    candidate = (
                        (current | {edge}) if edge not in current
                        else (current - {edge})
                    )
                    try:
                        _, e = self._project(candidate)
                    except ValueError:
                        continue
                    if e < current_energy - 1e-12:
                        current, current_energy = candidate, e
                        toggle_improved = True
                        improved = True

            for a in range(len(options)):
                for bidx in range(a + 1, len(options)):
                    e1, e2 = options[a], options[bidx]
                    candidate = current ^ {e1, e2}
                    try:
                        _, e = self._project(candidate)
                    except ValueError:
                        continue
                    if e < current_energy - 1e-12:
                        current, current_energy = candidate, e
                        improved = True

            if not improved:
                break
        return current, current_energy

    def _anneal(self, start: set[Edge]) -> Tuple[set[Edge], float]:
        builder = self.builder
        options = builder.optional_edges
        if not options:
            return set(start), builder.energy(builder.build_feasible_solution(start))

        current = set(start)
        _, current_energy = self._project(current)
        best, best_energy = set(current), current_energy

        temperature = self.initial_temperature
        while temperature > self.final_temperature:
            for _ in range(self.steps):
                candidate = set(current)
                edge = options[int(self.rng.integers(len(options)))]
                if edge in candidate:
                    candidate.discard(edge)
                else:
                    candidate.add(edge)
                if self.rng.random() < self.swap_probability and len(options) > 1:
                    edge2 = options[int(self.rng.integers(len(options)))]
                    if edge2 in candidate:
                        candidate.discard(edge2)
                    else:
                        candidate.add(edge2)
                try:
                    _, energy = self._project(candidate)
                except ValueError:
                    continue
                delta = energy - current_energy
                if delta <= 0.0 or self.rng.random() < exp(-delta / temperature):
                    current, current_energy = candidate, energy
                    if energy < best_energy:
                        best, best_energy = set(candidate), energy
            temperature *= self.cooling_rate
        return self._polish(best, best_energy)

    def run(self) -> QUBOResult:
        best_edges: Optional[set[Edge]] = None
        best_energy = float("inf")

        for restart in range(self.restarts):
            if restart < len(self.initial_edge_sets):
                start = self.initial_edge_sets[restart]
            else:
                start = self._random_feasible_start()
            try:
                self.builder.build_feasible_solution(start)
            except ValueError:
                continue
            edges, energy = self._anneal(start)
            if energy < best_energy:
                best_edges, best_energy = edges, energy

        if best_edges is None:
            raise ValueError(
                "No feasible edge subset exists for the given constraints"
            )
        z = self.builder.build_feasible_solution(best_edges)
        return self.builder.extract_solution(z)


class SimulatedAnnealing:
    """Binary simulated annealing for E(z)=z.T Q z + constant."""

    def __init__(
        self,
        Q: np.ndarray,
        constant: float,
        *,
        initial_temperature: float = 10.0,
        final_temperature: float = 1e-3,
        cooling_rate: float = 0.98,
        sweeps_per_temperature: int = 2,
        restarts: int = 20,
        seed: int = 42,
        initial_states: Optional[Sequence[np.ndarray]] = None,
        pair_probability: float = 0.0,
    ) -> None:
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be square")
        if not np.allclose(Q, Q.T, atol=1e-10):
            raise ValueError("Q must be symmetric")
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0,1)")
        if final_temperature <= 0:
            raise ValueError("final_temperature must be positive")
        if sweeps_per_temperature <= 0 or restarts <= 0:
            raise ValueError("sweeps_per_temperature and restarts must be positive")
        if not 0.0 <= pair_probability <= 1.0:
            raise ValueError("pair_probability must be in [0,1]")

        self.Q = Q
        self.constant = float(constant)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.sweeps = sweeps_per_temperature
        self.restarts = restarts
        self.rng = np.random.default_rng(seed)
        self.initial_states = list(initial_states or [])
        self.pair_probability = pair_probability

    def energy(self, z: np.ndarray) -> float:
        return float(z @ self.Q @ z + self.constant)

    def run(self) -> Tuple[np.ndarray, float]:
        n = self.Q.shape[0]
        diag = np.diag(self.Q)

        best_z = None
        best_energy = float("inf")

        for restart in range(self.restarts):
            if restart < len(self.initial_states):
                z = np.rint(self.initial_states[restart]).astype(float).copy()
                if z.shape != (n,):
                    raise ValueError("Warm-start state has wrong shape")
            else:
                z = self.rng.integers(0, 2, size=n).astype(float)

            qz = self.Q @ z
            current_energy = float(z @ qz + self.constant)
            local_best_z = z.copy()
            local_best_energy = current_energy
            temperature = self.initial_temperature

            while temperature > self.final_temperature:
                for _ in range(self.sweeps * n):
                    if self.pair_probability > 0.0 and n > 1 \
                            and self.rng.random() < self.pair_probability:
                        # two-bit move: tests whether the penalty barrier is
                        # a neighborhood artifact or a landscape property
                        i = int(self.rng.integers(n))
                        j = int(self.rng.integers(n - 1))
                        if j >= i:
                            j += 1
                        di = 1.0 - 2.0 * z[i]
                        dj = 1.0 - 2.0 * z[j]
                        d_energy = (
                            2.0 * di * qz[i] + diag[i]
                            + 2.0 * dj * qz[j] + diag[j]
                            + 2.0 * di * dj * self.Q[i, j]
                        )
                        if d_energy <= 0.0 or self.rng.random() < exp(-d_energy / temperature):
                            z[i] = 1.0 - z[i]
                            z[j] = 1.0 - z[j]
                            current_energy += d_energy
                            qz += di * self.Q[:, i] + dj * self.Q[:, j]
                            if current_energy < local_best_energy:
                                local_best_energy = current_energy
                                local_best_z = z.copy()
                        continue
                    i = int(self.rng.integers(n))
                    old = z[i]
                    delta_z = 1.0 - 2.0 * old

                    # For symmetric Q:
                    # ΔE = 2*delta_z*(Qz)_i + Q_ii.
                    d_energy = 2.0 * delta_z * qz[i] + diag[i]

                    if d_energy <= 0.0 or self.rng.random() < exp(-d_energy / temperature):
                        z[i] = 1.0 - old
                        current_energy += d_energy
                        qz += delta_z * self.Q[:, i]

                        if current_energy < local_best_energy:
                            local_best_energy = current_energy
                            local_best_z = z.copy()

                temperature *= self.cooling_rate

            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_z = local_best_z

        assert best_z is not None
        return best_z, best_energy
