"""Abstract base classes for causal discovery and inference algorithms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class CausalDiscoveryBase(ABC):
    """
    Unified interface for all causal discovery algorithms.

    Every discovery algorithm must implement fit() with this exact signature.
    This contract enables: apples-to-apples benchmarks, contributor onboarding,
    and planner-agnostic execution.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params: Dict[str, Any] = params or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm identifier, e.g. 'PC', 'GES', 'NOTEARSLinear'."""
        ...

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run causal discovery on data.

        Args:
            data: Input dataset (n_samples x n_features).

        Returns:
            adj_matrix: Adjacency matrix where mat[i,j]=1 means j->i.
            metadata: Algorithm-specific info (e.g. p-values, scores, model).
        """
        ...

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for this algorithm."""
        ...

    def get_params(self) -> Dict[str, Any]:
        """Return current parameters."""
        return dict(self._params)

    def set_params(self, **params) -> None:
        """Update parameters."""
        self._params.update(params)


class CausalInferenceBase(ABC):
    """
    Unified interface for all causal inference / treatment effect estimators.

    Covers DML, DRL, IV, MetaLearners, and Uplift families.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        outcome: str = "",
        treatment: str = "",
        covariates: Optional[list] = None,
    ):
        self._params: Dict[str, Any] = params or {}
        self.outcome = outcome
        self.treatment = treatment
        self.covariates = covariates or []

    @property
    @abstractmethod
    def name(self) -> str:
        """Estimator identifier, e.g. 'CausalForestDML'."""
        ...

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the estimator on data."""
        ...

    @abstractmethod
    def estimate_ate(self, data: pd.DataFrame) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Estimate Average Treatment Effect.

        Returns:
            ate: Point estimate.
            ci: Optional (lower, upper) confidence interval.
        """
        ...

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters."""
        ...

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)

    def set_params(self, **params) -> None:
        self._params.update(params)
