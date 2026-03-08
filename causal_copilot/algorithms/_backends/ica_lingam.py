"""ICALiNGAM backend using causal-learn."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class ICALiNGAMBackend(Backend):
    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        from causallearn.search.FCMBased.lingam.ica_lingam import ICALiNGAM as CLICALiNGAM

        model = CLICALiNGAM(
            random_state=self._params.get("random_state", None),
            max_iter=self._params.get("max_iter", 1000),
        )
        model.fit(data.values)
        adj_matrix = np.where(model.adjacency_matrix_ != 0, 1, 0)
        info = {"causal_order": model.causal_order_}
        return adj_matrix, info, model
