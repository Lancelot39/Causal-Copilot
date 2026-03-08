"""Rule-based algorithm selection planner (offline, deterministic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PlannerDecision:
    """Output of algorithm selection — typed, inspectable, serializable."""

    algorithm: str
    hyperparams: dict[str, Any]
    reason: str


def detect_data_properties(data: pd.DataFrame) -> dict[str, Any]:
    """Detect basic statistical properties of input data for algorithm selection."""
    n_samples, n_features = data.shape

    # Check for time-series indicators
    is_time_series = False
    if isinstance(data.index, pd.DatetimeIndex):
        is_time_series = True

    # Check linearity heuristic: correlation between feature pairs
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] >= 2:
        corr = numeric_data.corr().abs()
        np.fill_diagonal(corr.values, 0)
        avg_corr = corr.mean().mean()
        likely_linear = avg_corr > 0.3  # heuristic
    else:
        likely_linear = True

    # Check for missing values
    has_missing = data.isnull().any().any()
    missing_ratio = data.isnull().sum().sum() / (n_samples * n_features) if n_samples > 0 else 0

    # Check gaussianity heuristic (simple: skewness-based)
    try:
        skewness = numeric_data.skew().abs().mean()
        likely_gaussian = skewness < 1.0
    except Exception:
        likely_gaussian = True

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "is_time_series": is_time_series,
        "likely_linear": likely_linear,
        "likely_gaussian": likely_gaussian,
        "has_missing": has_missing,
        "missing_ratio": missing_ratio,
    }


def _get_defaults(name: str) -> dict[str, Any]:
    """Read default hyperparams from the canonical registry."""
    from causal_copilot.algorithms.registry import REGISTRY

    return dict(REGISTRY[name].default_params)


def rule_based_select(properties: dict[str, Any]) -> PlannerDecision:
    """
    Select algorithm based on data properties using a simple decision tree.

    Decision logic:
    1. Time-series -> PCMCI
    2. Non-gaussian errors -> DirectLiNGAM (functional model exploits non-gaussianity)
    3. Small-medium data + likely linear -> PC (constraint-based, well-understood)
    4. Large data or many features -> NOTEARSLinear (continuous optimization, scalable)
    5. Default fallback -> GES (score-based, general purpose)
    """
    n = properties["n_samples"]
    p = properties["n_features"]

    if properties["is_time_series"]:
        return PlannerDecision(
            algorithm="PCMCI",
            hyperparams=_get_defaults("PCMCI"),
            reason="Time-series data detected — PCMCI handles temporal causal discovery.",
        )

    if not properties["likely_gaussian"]:
        if p <= 50:
            return PlannerDecision(
                algorithm="DirectLiNGAM",
                hyperparams=_get_defaults("DirectLiNGAM"),
                reason="Non-gaussian errors detected — DirectLiNGAM exploits non-gaussianity for identifiability.",
            )

    if properties["likely_linear"] and p <= 30 and n <= 5000:
        return PlannerDecision(
            algorithm="PC",
            hyperparams=_get_defaults("PC"),
            reason="Small-medium linear data — PC is well-understood with strong theoretical guarantees.",
        )

    if p > 30 or n > 5000:
        return PlannerDecision(
            algorithm="NOTEARSLinear",
            hyperparams=_get_defaults("NOTEARSLinear"),
            reason="Large data or many features — NOTEARSLinear scales via continuous optimization.",
        )

    # Default fallback
    return PlannerDecision(
        algorithm="GES",
        hyperparams=_get_defaults("GES"),
        reason="General-purpose score-based method — GES is a robust default for medium-sized data.",
    )
