"""Minimal base class for vendored backends."""

from typing import Any

import numpy as np
import pandas as pd


class Backend:
    """Thin wrapper around an upstream causal discovery library."""

    def __init__(self, params: dict[str, Any]):
        self._params = dict(params)

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        raise NotImplementedError
