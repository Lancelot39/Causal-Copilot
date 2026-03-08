"""DirectLiNGAM backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd
from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM as CLDirectLiNGAM

from causal_copilot.algorithms._backends._base import Backend


class DirectLiNGAMBackend(Backend):
    """DirectLiNGAM using causal-learn (CPU path only)."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        data_values = data.values

        # Build constructor params (exclude non-constructor keys)
        model = CLDirectLiNGAM(
            random_state=self._params.get("random_state", None),
            prior_knowledge=self._params.get("prior_knowledge", None),
            apply_prior_knowledge_softly=self._params.get("apply_prior_knowledge_softly", False),
            measure=self._params.get("measure", "pwling"),
        )
        model.fit(data_values)

        # Convert weighted adjacency to binary
        adj_matrix = np.where(model.adjacency_matrix_ != 0, 1, 0)

        info = {
            "causal_order": model.causal_order_,
        }

        return adj_matrix, info, model
