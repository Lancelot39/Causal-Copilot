"""Granger Causality backend using statsmodels."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class GrangerCausalityBackend(Backend):
    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        from statsmodels.tsa.stattools import grangercausalitytests

        node_names = list(data.columns)
        n_vars = len(node_names)
        max_lag = self._params.get("p", 10)
        alpha = self._params.get("alpha", 0.05)
        criterion = self._params.get("criterion", "ssr_ftest")

        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue  # No self-loops
                try:
                    test_result = grangercausalitytests(
                        data[[node_names[i], node_names[j]]].values,
                        maxlag=max_lag,
                        verbose=False,
                    )
                    p_values = [
                        test_result[lag + 1][0][criterion][1]
                        for lag in range(max_lag)
                    ]
                    if min(p_values) < alpha:
                        adj_matrix[i, j] = 1  # j Granger-causes i
                except Exception:
                    pass  # Some pairs may fail

        info = {"lag": max_lag, "nodes": node_names}
        return adj_matrix, info, None
