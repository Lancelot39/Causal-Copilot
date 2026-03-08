"""PC algorithm backend — imports causal-learn directly."""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from causallearn.search.ConstraintBased.PC import pc as cl_pc

from causal_copilot.algorithms._backends._base import Backend


def _convert_to_adjacency_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """Convert causal-learn's CPDAG encoding to our adjacency convention.

    causal-learn encodes edges as:
      adj[i,j]=1  and adj[j,i]=-1  =>  directed j -> i
      adj[i,j]=1  and adj[j,i]=1   =>  bidirected i <-> j
      adj[i,j]=-1 and adj[j,i]=-1  =>  undirected i -- j

    Our convention:
      mat[i,j]=1 => directed j -> i
      mat[i,j]=2 => undirected
      mat[i,j]=3 => bidirected
    """
    inferred_flat = np.zeros_like(adj_matrix)

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[j, i] == -1:
            # directed edge: j -> i
            inferred_flat[i, j] = 1
        elif adj_matrix[j, i] == 1:
            # bidirected edge: j <-> i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 3

    indices = np.where(adj_matrix == -1)
    for i, j in zip(indices[0], indices[1]):
        if adj_matrix[j, i] == -1:
            # undirected edge: j -- i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 2

    return inferred_flat


class PCBackend(Backend):
    """PC algorithm using causal-learn (CPU path only)."""

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        # Normalise indep_test: strip _cpu/_gpu suffixes for the vendored CPU path
        indep_test = self._params.get("indep_test", "fisherz")
        indep_test = indep_test.replace("_cpu", "").replace("_gpu", "")

        cg = cl_pc(
            data_values,
            alpha=self._params.get("alpha", 0.05),
            indep_test=indep_test,
            depth=self._params.get("depth", 4),
            stable=self._params.get("stable", True),
            uc_rule=self._params.get("uc_rule", 0),
            uc_priority=self._params.get("uc_priority", -1),
            mvpc=self._params.get("mvpc", False),
            correction_name=self._params.get("correction_name", "MV_Crtn_Fisher_Z"),
            background_knowledge=None,
            verbose=self._params.get("verbose", False),
            show_progress=self._params.get("show_progress", False),
            node_names=node_names,
        )

        adj_matrix = _convert_to_adjacency_matrix(cg.G.graph)

        info = {
            "sepset": cg.sepset if hasattr(cg, "sepset") else None,
            "definite_UC": cg.definite_UC if hasattr(cg, "definite_UC") else [],
            "definite_non_UC": cg.definite_non_UC if hasattr(cg, "definite_non_UC") else [],
        }

        return adj_matrix, info, cg
