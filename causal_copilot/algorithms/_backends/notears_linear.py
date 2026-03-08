"""NOTEARS (linear) backend — tries gcastle first, falls back to causal-learn."""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from causal_copilot.algorithms._backends._base import Backend


class NOTEARSLinearBackend(Backend):
    """NOTEARS linear using gcastle (castle.algorithms.Notears)."""

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        # Remove domain_index if present
        if isinstance(data, pd.DataFrame) and "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data_values = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data_values = np.array(data)

        from castle.algorithms import Notears

        model = Notears(
            lambda1=self._params.get("lambda1", 0.1),
            loss_type=self._params.get("loss_type", "l2"),
            max_iter=self._params.get("max_iter", 100),
            h_tol=self._params.get("h_tol", 1e-8),
            rho_max=self._params.get("rho_max", 1e16),
            w_threshold=self._params.get("w_threshold", 0.3),
        )

        model.learn(data_values)

        # causal_matrix convention: model.causal_matrix[i,j] = weight means i -> j
        # Our convention: mat[i,j] = 1 means j -> i  =>  transpose
        adj_matrix = model.causal_matrix.T

        info = {
            "model": model,
            "weight_causal_matrix": model.weight_causal_matrix,
        }

        return adj_matrix, info, model
