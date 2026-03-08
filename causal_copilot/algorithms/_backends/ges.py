"""GES algorithm backend — imports causal-learn directly."""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from causallearn.search.ScoreBased.GES import ges as cl_ges

from causal_copilot.algorithms._backends._base import Backend


def _convert_to_adjacency_matrix(G) -> np.ndarray:
    """Convert causal-learn CausalGraph to our adjacency convention.

    Same encoding as PC: adj[i,j]=1 + adj[j,i]=-1 => j->i (directed).
    """
    adj_matrix = G.graph
    inferred_flat = np.zeros_like(adj_matrix)

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[j, i] == -1:
            # directed edge: j -> i
            inferred_flat[i, j] = 1

    indices = np.where(adj_matrix == -1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[j, i] == -1:
            # undirected edge: j -- i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 2

    return inferred_flat


class GESBackend(Backend):
    """GES algorithm using causal-learn."""

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        maxP = self._params.get("maxP", 4)
        if maxP is not None and maxP < 0:
            maxP = None

        record = cl_ges(
            data_values,
            score_func=self._params.get("score_func", "local_score_BIC"),
            maxP=maxP,
            parameters=self._params.get("parameters", None),
            node_names=node_names,
        )

        adj_matrix = _convert_to_adjacency_matrix(record["G"])

        info = {
            "score": record["score"],
            "update1": record["update1"],
            "update2": record["update2"],
        }

        return adj_matrix, info, record
