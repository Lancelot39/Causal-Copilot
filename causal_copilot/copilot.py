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

_VALID_PLANNERS = ("rule",)  # "llm" will be added in a future version


class AlgorithmTimeoutError(Exception):
    """Raised when algorithm execution exceeds timeout."""
    pass


def _timeout_handler(signum, frame):
    raise AlgorithmTimeoutError("Algorithm execution timed out")


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


def _build_graph(adj_matrix: np.ndarray, columns: list):
    """Build networkx graph from adjacency matrix, handling all edge types.

    Edge encoding: 0=none, 1=directed, 2=undirected, 3=bidirected.
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            val = adj_matrix[i, j]
            if val == 1:  # directed: j -> i
                graph.add_edge(columns[j], columns[i], edge_type="directed")
            elif val == 2:  # undirected: i -- j (add both directions)
                graph.add_edge(columns[j], columns[i], edge_type="undirected")
                graph.add_edge(columns[i], columns[j], edge_type="undirected")
            elif val == 3:  # bidirected: i <-> j
                graph.add_edge(columns[j], columns[i], edge_type="bidirected")
                graph.add_edge(columns[i], columns[j], edge_type="bidirected")
    return graph


def _make_provenance(data_hash, seed, decision, active_planner, elapsed):
    return Provenance(
        dataset_hash=data_hash, seed=seed,
        algorithm=decision.algorithm, algorithm_version=__version__,
        package_version=__version__,
        hyperparams=Provenance.freeze_params(decision.hyperparams),
        planner=active_planner, runtime_seconds=elapsed,
        timestamp=Provenance.now_utc(),
        environment=Provenance.get_environment(),
    )


class CausalCopilot:
    """
    Main entry point for causal analysis.

    Usage:
        copilot = CausalCopilot()
        result = copilot.analyze("data.csv", seed=42)
    """

    def __init__(self, planner: str = "rule"):
        if planner not in _VALID_PLANNERS:
            raise ValueError(f"Unknown planner: {planner!r}. Available: {_VALID_PLANNERS}")
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
            planner: Override instance planner (must be valid).
            algorithm: Force a specific algorithm (bypasses planner).
            timeout: Max seconds for algorithm execution (default 300).
            seed: Random seed for reproducibility (default 42).

        Returns:
            CausalResult with graph, provenance, assumptions, and warnings.
        """
        start_time = time.monotonic()
        warnings: list[str] = []

        # Validate planner
        active_planner = planner or self.planner
        if active_planner not in _VALID_PLANNERS:
            return CausalResult(
                status="failed",
                summary=f"Unknown planner: {active_planner!r}. Available: {_VALID_PLANNERS}",
            )

        # Set random seed (scoped — use Generator for future thread safety)
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
            try:
                df = pd.read_csv(path)
            except Exception as e:
                return CausalResult(
                    status="failed",
                    summary=f"Failed to read CSV: {e}",
                    warnings=[str(e)],
                )
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return CausalResult(
                status="failed",
                summary=f"Unsupported data type: {type(data).__name__}",
            )

        # Validate raw data
        try:
            data_warnings = validate_data(df)
            warnings.extend(data_warnings)
        except DataValidationError as e:
            return CausalResult(
                status="failed",
                summary=str(e),
                warnings=[str(e)],
            )

        # Clean: keep only numeric, drop constants
        numeric_df = df.select_dtypes(include=[np.number])
        constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Dropped constant columns: {constant_cols}")
            numeric_df = numeric_df.drop(columns=constant_cols)

        # Re-validate after cleaning
        if numeric_df.shape[1] < 2:
            return CausalResult(
                status="failed",
                summary=f"After cleaning, only {numeric_df.shape[1]} numeric column(s) remain. Need at least 2.",
                warnings=warnings,
            )

        # Drop rows with NaN (simple strategy for v0.1)
        n_before = len(numeric_df)
        numeric_df = numeric_df.dropna()
        n_dropped = n_before - len(numeric_df)
        if n_dropped > 0:
            warnings.append(f"Dropped {n_dropped} rows with missing values ({n_dropped/n_before:.1%}).")
        if len(numeric_df) < 10:
            return CausalResult(
                status="failed",
                summary=f"After dropping NaN rows, only {len(numeric_df)} samples remain. Need at least 10.",
                warnings=warnings,
            )

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

        # Load algorithm
        try:
            algo = _load_algorithm(decision.algorithm, decision.hyperparams)
        except (ValueError, ImportError) as e:
            return CausalResult(
                status="failed",
                summary=f"Failed to load algorithm {decision.algorithm}: {e}",
                warnings=warnings + [str(e)],
            )

        # Execute with timeout (signal.alarm is Unix-only and main-thread-only)
        old_handler = None
        alarm_set = False
        try:
            try:
                if hasattr(signal, 'SIGALRM'):
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout)
                    alarm_set = True
            except (ValueError, OSError):
                # Not in main thread or signal not available — skip timeout
                pass

            adj_matrix, metadata, model = algo.fit(numeric_df)

        except AlgorithmTimeoutError:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} timed out after {timeout}s",
                warnings=warnings + [f"Timeout after {timeout}s"],
                provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                algorithm_selection_reason=decision.reason,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} failed: {e}",
                warnings=warnings + [str(e)],
                provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                algorithm_selection_reason=decision.reason,
            )
        finally:
            # Always clean up signal state
            if alarm_set:
                signal.alarm(0)
            if old_handler is not None:
                try:
                    signal.signal(signal.SIGALRM, old_handler)
                except (ValueError, OSError):
                    pass

        elapsed = time.monotonic() - start_time

        # Build graph handling all edge types (1=directed, 2=undirected, 3=bidirected)
        cols = list(numeric_df.columns)
        graph = _build_graph(adj_matrix, cols)

        provenance = _make_provenance(data_hash, seed, decision, active_planner, elapsed)

        # Count edges by type
        n_directed = int(np.sum(adj_matrix == 1))
        n_undirected = int(np.sum(adj_matrix == 2)) // 2  # counted twice in matrix
        n_bidirected = int(np.sum(adj_matrix == 3)) // 2

        edge_parts = [f"{n_directed} directed"]
        if n_undirected:
            edge_parts.append(f"{n_undirected} undirected")
        if n_bidirected:
            edge_parts.append(f"{n_bidirected} bidirected")
        edge_summary = ", ".join(edge_parts)

        return CausalResult(
            status="ok",
            adjacency_matrix=adj_matrix,
            graph=graph,
            summary=f"Discovered {edge_summary} edges using {decision.algorithm} "
                    f"on {numeric_df.shape[0]} samples × {numeric_df.shape[1]} features "
                    f"in {elapsed:.1f}s.",
            warnings=warnings,
            provenance=provenance,
            algorithm_selection_reason=decision.reason,
        )
