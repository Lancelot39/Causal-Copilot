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

        # Guard: need enough observations for the requested lag order
        n_obs = len(data)
        if n_obs <= max_lag + 1:
            raise ValueError(
                f"Granger causality requires more observations than lag order + 1. "
                f"Got {n_obs} rows with max_lag={max_lag}."
            )

        adj_matrix = np.zeros((n_vars, n_vars), dtype=int)
        n_pairs = n_vars * (n_vars - 1)
        n_failed = 0

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
                except (ValueError, KeyError, RuntimeError) as exc:
                    n_failed += 1
                    if n_failed >= n_pairs:
                        raise RuntimeError(
                            f"All {n_pairs} Granger tests failed. Last error: {exc}"
                        ) from exc

        info = {"lag": max_lag, "nodes": node_names, "n_failed_pairs": n_failed}
        return adj_matrix, info, None
