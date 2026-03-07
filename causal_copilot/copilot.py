"""CausalCopilot — the main entry point for causal analysis."""
from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from causal_copilot import __version__
from causal_copilot.core.base import CausalDiscoveryBase
from causal_copilot.core.guards import DataValidationError, validate_data
from causal_copilot.core.planner import PlannerDecision, detect_data_properties, rule_based_select
from causal_copilot.core.result import CausalResult, Provenance


class TimeoutError(Exception):
    """Raised when algorithm execution exceeds timeout."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Algorithm execution timed out")


# Lazy registry — maps algorithm name to import path
_ALGORITHM_REGISTRY = {
    "PC": ("causal_copilot.algorithms.adapters", "PCAdapter"),
    "GES": ("causal_copilot.algorithms.adapters", "GESAdapter"),
    "NOTEARSLinear": ("causal_copilot.algorithms.adapters", "NOTEARSLinearAdapter"),
    "DirectLiNGAM": ("causal_copilot.algorithms.adapters", "DirectLiNGAMAdapter"),
    "PCMCI": ("causal_copilot.algorithms.adapters", "PCMCIAdapter"),
}


def _load_algorithm(name: str, params: Dict[str, Any]) -> CausalDiscoveryBase:
    """Lazily import and instantiate an algorithm adapter."""
    if name not in _ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name!r}. Available: {sorted(_ALGORITHM_REGISTRY)}")
    module_path, class_name = _ALGORITHM_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(params=params)


class CausalCopilot:
    """
    Main entry point for causal analysis.

    Usage:
        copilot = CausalCopilot()
        result = copilot.analyze("data.csv", planner="rule", seed=42)
    """

    def __init__(self, planner: str = "rule"):
        """
        Args:
            planner: Algorithm selection strategy. "rule" (default, offline) or "llm" (optional).
        """
        if planner not in ("rule", "llm"):
            raise ValueError(f"Unknown planner: {planner!r}. Use 'rule' or 'llm'.")
        if planner == "llm":
            raise NotImplementedError("LLM planner is not yet available in v0.1. Use planner='rule'.")
        self.planner = planner

    def analyze(
        self,
        data: str | Path | pd.DataFrame,
        *,
        planner: Optional[str] = None,
        algorithm: Optional[str] = None,
        timeout: int = 300,
        seed: int = 42,
    ) -> CausalResult:
        """
        Run causal discovery on data.

        Args:
            data: CSV file path or DataFrame.
            planner: Override instance planner ("rule" or "llm").
            algorithm: Force a specific algorithm (bypasses planner).
            timeout: Max seconds for algorithm execution (default 300).
            seed: Random seed for reproducibility (default 42).

        Returns:
            CausalResult with graph, provenance, assumptions, and warnings.
        """
        start_time = time.monotonic()
        active_planner = planner or self.planner
        warnings = []

        # Set random seed
        np.random.seed(seed)

        # Load data
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                return CausalResult(
                    status="failed",
                    summary=f"File not found: {path}",
                    warnings=[f"File not found: {path}"],
                )
            df = pd.read_csv(path)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return CausalResult(
                status="failed",
                summary=f"Unsupported data type: {type(data).__name__}",
            )

        # Validate data
        try:
            data_warnings = validate_data(df)
            warnings.extend(data_warnings)
        except DataValidationError as e:
            return CausalResult(
                status="failed",
                summary=str(e),
                warnings=[str(e)],
            )

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Drop constant columns
        constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
        if constant_cols:
            numeric_df = numeric_df.drop(columns=constant_cols)

        # Hash data for provenance
        data_hash = Provenance.hash_data(numeric_df)

        # Select algorithm
        if algorithm:
            decision = PlannerDecision(
                algorithm=algorithm,
                hyperparams={},
                reason=f"User specified algorithm: {algorithm}",
            )
        else:
            properties = detect_data_properties(numeric_df)
            decision = rule_based_select(properties)

        # Load and run algorithm
        try:
            algo = _load_algorithm(decision.algorithm, decision.hyperparams)
        except (ValueError, ImportError) as e:
            return CausalResult(
                status="failed",
                summary=f"Failed to load algorithm {decision.algorithm}: {e}",
                warnings=warnings + [str(e)],
            )

        # Execute with timeout
        try:
            # Set timeout (Unix only; on Windows this is a no-op)
            old_handler = None
            if hasattr(signal, 'SIGALRM'):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(timeout)

            adj_matrix, metadata, model = algo.fit(numeric_df)

            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        except TimeoutError:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} timed out after {timeout}s",
                warnings=warnings + [f"Timeout after {timeout}s"],
                provenance=Provenance(
                    dataset_hash=data_hash, seed=seed,
                    algorithm=decision.algorithm, algorithm_version=__version__,
                    package_version=__version__,
                    hyperparams=Provenance.freeze_params(decision.hyperparams),
                    planner=active_planner, runtime_seconds=elapsed,
                    timestamp=Provenance.now_utc(),
                    environment=Provenance.get_environment(),
                ),
                algorithm_selection_reason=decision.reason,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} failed: {e}",
                warnings=warnings + [str(e)],
                provenance=Provenance(
                    dataset_hash=data_hash, seed=seed,
                    algorithm=decision.algorithm, algorithm_version=__version__,
                    package_version=__version__,
                    hyperparams=Provenance.freeze_params(decision.hyperparams),
                    planner=active_planner, runtime_seconds=elapsed,
                    timestamp=Provenance.now_utc(),
                    environment=Provenance.get_environment(),
                ),
                algorithm_selection_reason=decision.reason,
            )

        elapsed = time.monotonic() - start_time

        # Build graph if networkx available
        graph = None
        try:
            import networkx as nx
            graph = nx.DiGraph()
            cols = list(numeric_df.columns)
            graph.add_nodes_from(cols)
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if adj_matrix[i, j] == 1:  # j -> i
                        graph.add_edge(cols[j], cols[i])
        except ImportError:
            pass

        provenance = Provenance(
            dataset_hash=data_hash,
            seed=seed,
            algorithm=decision.algorithm,
            algorithm_version=__version__,
            package_version=__version__,
            hyperparams=Provenance.freeze_params(decision.hyperparams),
            planner=active_planner,
            runtime_seconds=elapsed,
            timestamp=Provenance.now_utc(),
            environment=Provenance.get_environment(),
        )

        n_edges = int(np.sum(adj_matrix == 1)) if adj_matrix is not None else 0

        return CausalResult(
            status="ok",
            adjacency_matrix=adj_matrix,
            graph=graph,
            summary=f"Discovered {n_edges} directed edges using {decision.algorithm} "
                    f"on {numeric_df.shape[0]} samples × {numeric_df.shape[1]} features "
                    f"in {elapsed:.1f}s.",
            warnings=warnings,
            provenance=provenance,
            algorithm_selection_reason=decision.reason,
        )
