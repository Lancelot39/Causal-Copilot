"""Minimal base class for vendored backends."""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


class Backend:
    """Thin wrapper around an upstream causal discovery library."""
    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        raise NotImplementedError
