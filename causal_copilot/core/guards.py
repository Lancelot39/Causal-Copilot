"""Data validation guards — enforce boundaries before pipeline execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

# v0.1 boundaries (laptop-first)
MAX_SAMPLES = 100_000
MAX_FEATURES = 200
MAX_MISSING_RATIO = 0.5


class DataValidationError(ValueError):
    """Raised when input data fails validation."""

    pass


def validate_data(data: pd.DataFrame) -> list[str]:
    """
    Validate input data and return list of warnings.
    Raises DataValidationError for hard failures.
    """
    warnings = []

    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(f"Expected pandas DataFrame, got {type(data).__name__}")

    if data.empty:
        raise DataValidationError("Input data is empty")

    n_samples, n_features = data.shape

    if n_samples < 10:
        raise DataValidationError(f"Too few samples ({n_samples}). Minimum is 10.")

    if n_features < 2:
        raise DataValidationError(f"Too few features ({n_features}). Minimum is 2.")

    if n_samples > MAX_SAMPLES:
        raise DataValidationError(
            f"Too many samples ({n_samples:,}). v0.1 maximum is {MAX_SAMPLES:,}. Consider subsampling."
        )

    if n_features > MAX_FEATURES:
        raise DataValidationError(
            f"Too many features ({n_features}). v0.1 maximum is {MAX_FEATURES}. Consider feature selection."
        )

    # Check missing values
    missing_ratio = data.isnull().sum().sum() / (n_samples * n_features)
    if missing_ratio > MAX_MISSING_RATIO:
        raise DataValidationError(
            f"Missing value ratio ({missing_ratio:.1%}) exceeds maximum ({MAX_MISSING_RATIO:.0%})."
        )
    if missing_ratio > 0.05:
        warnings.append(f"High missing value ratio ({missing_ratio:.1%}). Results may be less reliable.")

    # Check numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        raise DataValidationError(
            f"Need at least 2 numeric columns, found {len(numeric_cols)}. Causal discovery requires numeric data."
        )

    non_numeric = set(data.columns) - set(numeric_cols)
    if non_numeric:
        warnings.append(f"Non-numeric columns will be dropped: {sorted(non_numeric)}")

    # Check constant columns
    constant_cols = [c for c in numeric_cols if data[c].nunique() <= 1]
    if constant_cols:
        warnings.append(f"Constant columns detected (will be dropped): {constant_cols}")

    return warnings
