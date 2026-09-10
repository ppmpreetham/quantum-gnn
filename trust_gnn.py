"""Trust-aware GNN task-assignment layer.

This sits directly after the QUBO pruner (main.py) in the pipeline:

    full graph -> EdgeSelectionQUBO (prune) -> TrustGNN (score) -> assignment

Design choices that match the Mist/Fog/Cloud architecture:
  - Pure NumPy, tiny matrices: inference must run on the Fog tier (Raspberry Pi).
  - Message passing runs ONLY on the pruned graph, so the QUBO output is the
    GNN's input graph.
  - Weights are configurable constants: training happens in the Cloud tier,
    the Fog tier only evaluates. Defaults below implement the intended policy
    (prefer high trust, high battery, low load, strong neighborhood).
  - Trust is dynamic: TrustModel updates per-node trust each cycle from
    observed task outcomes and link evidence, and the updated values feed the
    next cycle's QUBO features and GNN node features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from main import Edge, EdgeFeatures, Node, canonical_edge

# Feature vector layout used throughout the GNN.
#   0: trust            (from TrustModel)
#   1: battery          (0..1)
#   2: available compute (1 - load)
#   3: capability       (hardware class, 0..1)
N_FEATURES = 4


@dataclass(frozen=True)
class NodeFeatures:
    """All values normalized to [0, 1]."""

    trust: float
    battery: float
    load: float
    capability: float

    def vector(self) -> np.ndarray:
        return np.array(
            [self.trust, self.battery, 1.0 - self.load, self.capability],
            dtype=float,
        )


@dataclass(frozen=True)
class TaskRequirements:
    """What a task demands from its executing node."""

    min_trust: float = 0.3
    min_battery: float = 0.15
    compute_need: float = 0.3
    criticality: float = 0.5  # 0 = routine, 1 = mission-critical


@dataclass(frozen=True)
class GNNWeights:
    """
    Message passing:
        m_i   = sum_j a_ij * h_j / sum_j a_ij      (trust-weighted mean)
        h_i'  = relu(W_self h_i + W_neigh m_i + bias)
        score = sigmoid(v . h_i^L)
    """

    w_self: np.ndarray = field(
        default_factory=lambda: np.eye(N_FEATURES) * 0.8
    )
    w_neigh: np.ndarray = field(
        default_factory=lambda: np.eye(N_FEATURES) * 0.2
    )
    bias: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    # score weights: trust dominates (resources are mostly hard gates);
    # neighborhood robustness adds a smaller bonus
    out: np.ndarray = field(
        default_factory=lambda: np.array([0.60, 0.15, 0.10, 0.15])
    )
    out_bias: float = 0.0
    num_layers: int = 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TrustGNN:
    """Trust-weighted message passing over the pruned graph."""

    def __init__(self, weights: Optional[GNNWeights] = None) -> None:
        w = weights or GNNWeights()
        for name, mat in (("w_self", w.w_self), ("w_neigh", w.w_neigh)):
            if mat.shape != (N_FEATURES, N_FEATURES):
                raise ValueError(f"{name} must be {N_FEATURES}x{N_FEATURES}")
        if w.bias.shape != (N_FEATURES,) or w.out.shape != (N_FEATURES,):
            raise ValueError("bias and out must have shape (N_FEATURES,)")
        if w.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.w = w

    def _adjacency_weights(
        self,
        nodes: Sequence[Node],
        edges: Iterable[Edge],
        edge_features: Dict[Edge, EdgeFeatures],
    ) -> Dict[Node, List[Tuple[Node, float]]]:
        adj: Dict[Node, List[Tuple[Node, float]]] = {n: [] for n in nodes}
        for edge in edges:
            u, v = canonical_edge(edge)
            f = edge_features.get((u, v))
            # edge weight: how much we trust information flowing over this link
            a = (f.trust * f.link_quality) if f is not None else 1.0
            adj[u].append((v, a))
            adj[v].append((u, a))
        return adj

    def hidden_states(
        self,
        nodes: Sequence[Node],
        node_features: Dict[Node, NodeFeatures],
        edges: Iterable[Edge],
        edge_features: Optional[Dict[Edge, EdgeFeatures]] = None,
    ) -> Dict[Node, np.ndarray]:
        missing = set(nodes) - set(node_features)
        if missing:
            raise ValueError(f"Missing node features for nodes: {sorted(missing)}")
        for n in nodes:
            for name, value in vars(node_features[n]).items():
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"{n}.{name} must be finite and in [0, 1]")

        adj = self._adjacency_weights(nodes, edges, edge_features or {})
        h = {n: node_features[n].vector() for n in nodes}
        for _ in range(self.w.num_layers):
            new_h = {}
            for n in nodes:
                neigh = adj[n]
                if neigh:
                    weights = np.array([a for _, a in neigh])
                    total = weights.sum()
                    if total > 0.0:
                        msgs = np.array([h[j] for j, _ in neigh])
                        m = (weights @ msgs) / total
                    else:
                        m = np.zeros(N_FEATURES)
                else:
                    m = np.zeros(N_FEATURES)
                z = self.w.w_self @ h[n] + self.w.w_neigh @ m + self.w.bias
                new_h[n] = np.maximum(z, 0.0)  # relu
            h = new_h
        return h

    def score_from_hidden(
        self, n: Node, h_n: np.ndarray, f: NodeFeatures, task: TaskRequirements
    ) -> float:
        if (
            f.trust < task.min_trust
            or f.battery < task.min_battery
            or (1.0 - f.load) < task.compute_need
        ):
            return 0.0
        raw = float(self.w.out @ h_n) + self.w.out_bias
        s = float(_sigmoid(np.array(raw)))
        # mission-critical tasks amplify trust requirements; scale the
        # PROBABILITY (not the logit) so the direction is sign-independent
        if task.criticality > 0.5:
            s *= 0.5 + 0.5 * f.trust
        return s

    def scores(
        self,
        nodes: Sequence[Node],
        node_features: Dict[Node, NodeFeatures],
        edges: Iterable[Edge],
        task: TaskRequirements,
        edge_features: Optional[Dict[Edge, EdgeFeatures]] = None,
    ) -> Dict[Node, float]:
        """
        Per-node suitability score in [0, 1] for the given task.
        Ineligible nodes (below min_trust / min_battery / compute_need) score 0.
        """
        h = self.hidden_states(nodes, node_features, edges, edge_features)
        return {
            n: self.score_from_hidden(n, h[n], node_features[n], task)
            for n in nodes
        }

    def scores_for_tasks(
        self,
        nodes: Sequence[Node],
        node_features: Dict[Node, NodeFeatures],
        edges: Iterable[Edge],
        tasks: Dict[str, TaskRequirements],
        edge_features: Optional[Dict[Edge, EdgeFeatures]] = None,
    ) -> Dict[str, Dict[Node, float]]:
        """
        Score all tasks with a SINGLE message-passing pass (hidden states are
        task-independent; only the output gate depends on the task).
        Returns {task_id: {node: score}}.
        """
        h = self.hidden_states(nodes, node_features, edges, edge_features)
        return {
            tid: {
                n: self.score_from_hidden(n, h[n], node_features[n], task)
                for n in nodes
            }
            for tid, task in tasks.items()
        }


def assign_tasks(
    scores: Dict[Node, Dict[str, float]],
    capacity: Optional[Dict[Node, int]] = None,
) -> Dict[str, Node]:
    """
    Greedy conflict-free assignment: highest (score, task, node) first,
    respecting per-node capacity (default 1 task per node per cycle).

    scores: {node: {task_id: score}}; zero scores are never assigned.
    Returns {task_id: node}.
    """
    remaining = dict(capacity or {n: 1 for n in scores})
    flat: List[Tuple[float, str, Node]] = []
    for node, per_task in scores.items():
        for task_id, s in per_task.items():
            if s > 0.0:
                flat.append((s, task_id, node))
    flat.sort(key=lambda t: (-t[0], t[1]))

    assigned: Dict[str, Node] = {}
    for s, task_id, node in flat:
        if task_id in assigned or remaining.get(node, 0) <= 0:
            continue
        assigned[task_id] = node
        remaining[node] -= 1
    return assigned


def train_out_weights(
    samples: Sequence[Tuple[np.ndarray, float]],
    base: Optional[GNNWeights] = None,
    *,
    lr: float = 0.5,
    epochs: int = 1000,
    l2: float = 1e-3,
) -> GNNWeights:
    """
    Cloud-tier training: fit the output layer (out, out_bias) of the GNN by
    logistic regression on collected (hidden_state, task_success) samples.

    The message-passing matrices stay fixed (they encode the graph smoothing
    prior); only the readout is learned, which is enough to make the GNN a
    trained model rather than a hand-set heuristic. Samples are collected
    fog-side during operation and training happens offline (cloud).
    """
    base = base or GNNWeights()
    if len(samples) < 2:
        raise ValueError("Need at least 2 training samples")
    X = np.array([h for h, _ in samples], dtype=float)
    y = np.array([float(s) for _, s in samples], dtype=float)
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        raise ValueError(f"Hidden states must have {N_FEATURES} features")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Labels must be 0/1 task outcomes")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("Training data has only one outcome class")

    w = base.out.copy()
    b = float(base.out_bias)
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        err = p - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return GNNWeights(
        w_self=base.w_self,
        w_neigh=base.w_neigh,
        bias=base.bias,
        out=w,
        out_bias=b,
        num_layers=base.num_layers,
    )


class TrustModel:
    """
    EWMA trust update per node per cycle:

        T_new = alpha * T_old + (1 - alpha) * observation
        observation = w_success * outcome + w_link * link_evidence

    outcome: 1.0 if the node completed its assigned task, 0.0 otherwise
             (use 0.5 when the node was idle / unobserved).
    link_evidence: mean link_quality of the node's surviving edges this cycle.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        w_success: float = 0.7,
        w_link: float = 0.3,
        initial: float = 0.5,
        decay: float = 0.1,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if w_success < 0 or w_link < 0 or w_success + w_link == 0:
            raise ValueError("evidence weights must be non-negative and not both 0")
        total = w_success + w_link
        self.alpha = alpha
        self.w_success = w_success / total
        self.w_link = w_link / total
        self.initial = initial
        self.prior = initial
        self.decay = decay
        self.trust: Dict[Node, float] = {}

    def get(self, node: Node) -> float:
        return self.trust.get(node, self.initial)

    def relax(self, node: Node) -> float:
        """
        Trust aging for nodes with no new observation this cycle: drift
        slowly toward the prior. This rehabilitates nodes that failed once
        (a single unlucky failure must not blacklist a rover forever).
        """
        t = self.get(node) + self.decay * (self.prior - self.get(node))
        self.trust[node] = float(t)
        return self.trust[node]

    def update(
        self,
        node: Node,
        outcome: float,
        link_evidence: Optional[float] = None,
    ) -> float:
        if not 0.0 <= outcome <= 1.0:
            raise ValueError("outcome must be in [0, 1]")
        if link_evidence is None:
            obs = outcome
        else:
            if not 0.0 <= link_evidence <= 1.0:
                raise ValueError("link_evidence must be in [0, 1]")
            obs = self.w_success * outcome + self.w_link * link_evidence
        t = self.alpha * self.get(node) + (1.0 - self.alpha) * obs
        self.trust[node] = float(min(1.0, max(0.0, t)))
        return self.trust[node]
