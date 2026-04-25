"""GNN-based plan-quality scorer — Plan §7.1.

A small 2-layer graph convolutional encoder that scores the structural
quality of the agent's execution DAG against a bank of exemplar DAGs.

Design notes:
  • Node features (6-dim, hand-engineered, stable across tasks):
      [is_mcp, is_a2a, is_plan, is_memory, is_error, fanout_norm]
  • GraphSAGE-style mean-aggregator, 2 layers, 16-dim hidden.
  • Encoder weights are RANDOM-FIXED (seed=0) so scoring is deterministic
    with no training data required. The structural prior (depth,
    fanout-balanced, error-sparse, acyclic) comes from the exemplar set
    — not from learned weights.
  • Score = cosine(encoded_dag, mean(encoded_exemplars)), clipped [0,1].
  • If torch is not importable we return None from load() so callers fall
    back to the rule-based scorer in `signals.score_plan_quality`.

This is the signal the plan names as "the one nobody else will ship."
"""

from __future__ import annotations

from typing import Any, List, Optional, Dict
import math
import random

from ..models import DAGSnapshot, DAGNode


_EXEMPLAR_DAGS: List[Dict[str, Any]] = [
    # 1. Canonical research-synthesis shape: search → fetch → kb → submit
    {"nodes": [{"id": "n0", "op": "mcp:web.search",      "status": "ok"},
               {"id": "n1", "op": "mcp:web.fetch_url",   "status": "ok"},
               {"id": "n2", "op": "mcp:kb.lookup_fact",  "status": "ok"}],
     "edges": [["n0", "n1"], ["n1", "n2"]]},
    # 2. Discover + parallel lookups fanning in
    {"nodes": [{"id": "n0", "op": "mcp:web.search",      "status": "ok"},
               {"id": "n1", "op": "mcp:kb.lookup_fact",  "status": "ok"},
               {"id": "n2", "op": "mcp:kb.summarize",    "status": "ok"},
               {"id": "n3", "op": "mcp:web.fetch_url",   "status": "ok"}],
     "edges": [["n0", "n1"], ["n0", "n2"], ["n2", "n3"]]},
    # 3. Memory lookup on drift → recovery call
    {"nodes": [{"id": "n0", "op": "mcp:web.search",      "status": "error"},
               {"id": "n1", "op": "memory:query",        "status": "ok"},
               {"id": "n2", "op": "mcp:web.search_v2",   "status": "ok"}],
     "edges": [["n0", "n1"], ["n1", "n2"]]},
    # 4. A2A delegation path
    {"nodes": [{"id": "n0", "op": "mcp:kb.lookup_fact",  "status": "ok"},
               {"id": "n1", "op": "a2a:citer",           "status": "ok"}],
     "edges": [["n0", "n1"]]},
    # 5. Multi-step with SQL leg
    {"nodes": [{"id": "n0", "op": "mcp:data.list_tables", "status": "ok"},
               {"id": "n1", "op": "mcp:data.run_sql",     "status": "ok"},
               {"id": "n2", "op": "mcp:kb.summarize",     "status": "ok"}],
     "edges": [["n0", "n1"], ["n1", "n2"]]},
]


def _node_feat(op: str, status: str, fanout: int) -> List[float]:
    op = (op or "").lower()
    return [
        1.0 if op.startswith("mcp:") else 0.0,
        1.0 if op.startswith("a2a:") else 0.0,
        1.0 if op.startswith("plan") or "dag" in op else 0.0,
        1.0 if op.startswith("memory") else 0.0,
        1.0 if status == "error" else 0.0,
        min(1.0, fanout / 4.0),
    ]


class _NumpyEncoder:
    """Tiny 2-layer mean-aggregator GCN implemented in numpy so we don't
    require torch at runtime. Torch-based upgrade path documented below."""

    def __init__(self, in_dim: int = 6, hidden: int = 16, seed: int = 0):
        import numpy as np
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((in_dim, hidden)) / math.sqrt(in_dim)
        self.W2 = rng.standard_normal((hidden, hidden)) / math.sqrt(hidden)

    @staticmethod
    def _build_adj(nodes: List[Dict[str, Any]], edges: List[List[str]]):
        import numpy as np
        idx = {n["id"]: i for i, n in enumerate(nodes)}
        n = len(nodes)
        adj = np.eye(n)  # self-loops
        for a, b in edges:
            if a in idx and b in idx:
                adj[idx[a], idx[b]] = 1.0
                adj[idx[b], idx[a]] = 1.0
        d = adj.sum(axis=1, keepdims=True).clip(min=1.0)
        return adj / d

    def encode(self, nodes: List[Dict[str, Any]], edges: List[List[str]]):
        import numpy as np
        if not nodes:
            return np.zeros(self.W2.shape[0])
        fanout: Dict[str, int] = {n["id"]: 0 for n in nodes}
        for a, _ in edges:
            fanout[a] = fanout.get(a, 0) + 1
        X = np.array([_node_feat(n.get("op", ""), n.get("status", ""),
                                 fanout.get(n["id"], 0)) for n in nodes])
        A = self._build_adj(nodes, edges)
        H = np.tanh(A @ X @ self.W1)
        H = np.tanh(A @ H @ self.W2)
        return H.mean(axis=0)


class GNNPlanScorer:
    """Main entry point. Stateless, cheap to instantiate.

    Usage:
        scorer = GNNPlanScorer.load()        # None if numpy missing
        s = scorer.score(dag_snapshot)       # float in [0, 1]
    """

    def __init__(self, encoder: _NumpyEncoder):
        import numpy as np  # noqa: F401
        self.encoder = encoder
        self._exemplar_vec = self._mean_exemplar()

    @classmethod
    def load(cls, seed: int = 0) -> Optional["GNNPlanScorer"]:
        try:
            import numpy  # noqa: F401
        except Exception:
            return None
        return cls(_NumpyEncoder(seed=seed))

    def _mean_exemplar(self):
        import numpy as np
        vecs = [self.encoder.encode(ex["nodes"], ex["edges"])
                for ex in _EXEMPLAR_DAGS]
        if not vecs:
            return np.zeros(self.encoder.W2.shape[0])
        return np.stack(vecs, axis=0).mean(axis=0)

    def score(self, dag: DAGSnapshot) -> float:
        import numpy as np
        nodes = [n.model_dump() if hasattr(n, "model_dump") else dict(n)
                 for n in dag.nodes]
        edges = [list(e) for e in dag.edges]
        if not nodes:
            return 0.0
        v = self.encoder.encode(nodes, edges)
        u = self._exemplar_vec
        nu = np.linalg.norm(u) + 1e-8
        nv = np.linalg.norm(v) + 1e-8
        cos = float((u @ v) / (nu * nv))
        # Cosine is in [-1, 1]; map to [0, 1]
        return round(max(0.0, min(1.0, 0.5 * (cos + 1.0))), 3)


# Torch upgrade path (left as a comment so PyTorch Geometric can be dropped in
# without changing signatures):
#
#     from torch_geometric.nn import SAGEConv
#     class TorchGNNPlanScorer:
#         def __init__(self): ...
#         def score(self, dag: DAGSnapshot) -> float: ...
#
# When torch_geometric is importable, prefer it by trying first in `.load()`.
