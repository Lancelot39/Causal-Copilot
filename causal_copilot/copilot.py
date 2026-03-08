"""CausalCopilot — the main entry point for causal analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_copilot import __version__
from causal_copilot.core.base import CausalDiscoveryBase
from causal_copilot.core.guards import DataValidationError, validate_data
from causal_copilot.core.planner import PlannerDecision, detect_data_properties, rule_based_select
from causal_copilot.core.result import CausalResult, Provenance

_VALID_PLANNERS = ("rule",)  # "llm" will be added in a future version

# Inline script executed in a fresh Python process by _run_in_subprocess.
# Reads JSON config from stdin, runs algo.fit(), writes JSON result to stdout.
_WORKER_SCRIPT = r"""
import io, json, sys
import numpy as np
import pandas as pd

cfg = json.load(sys.stdin)
from causal_copilot.algorithms.registry import REGISTRY
from causal_copilot.core.result import _json_safe

spec = REGISTRY.get(cfg["algo_name"])
if spec is None:
    json.dump({"status": "error", "message": f"Unknown algorithm: {cfg['algo_name']}"}, sys.stdout)
    sys.exit(0)
adapter = spec.adapter_cls(params=cfg["algo_params"])
df = pd.read_json(io.StringIO(cfg["data_json"]))
try:
    adj, meta, _ = adapter.fit(df)
    json.dump({"status": "ok", "adj": adj.tolist(), "meta": _json_safe(meta)}, sys.stdout)
except Exception as e:
    json.dump({"status": "error", "message": str(e)}, sys.stdout)
"""


def _run_in_subprocess(
    algo: CausalDiscoveryBase, data: pd.DataFrame, timeout: int
) -> tuple[np.ndarray, dict]:
    """Run algo.fit(data) in a fresh subprocess with kill-based timeout.

    Launches a new Python interpreter via subprocess.Popen, avoiding all
    fork-safety issues (deadlocks in multi-threaded parents, macOS spawn
    restrictions). Data crosses the process boundary as JSON over stdin/stdout.

    Returns (adj_matrix, metadata) or raises TimeoutError / RuntimeError.
    """
    cfg = json.dumps({
        "algo_name": algo.name,
        "algo_params": algo.get_params(),
        "data_json": data.to_json(),
    })

    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(input=cfg.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise TimeoutError(f"Algorithm timed out after {timeout}s")

    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess exited with code {proc.returncode}: {stderr.decode()[:500]}")

    result = json.loads(stdout.decode())
    if result["status"] == "ok":
        return np.array(result["adj"]), result.get("meta", {})
    raise RuntimeError(result.get("message", "Unknown error in subprocess"))


def _load_algorithm(name: str, params: dict[str, Any]) -> CausalDiscoveryBase:
    """Instantiate an algorithm adapter from the canonical registry."""
    from causal_copilot.algorithms.registry import REGISTRY

    if name not in REGISTRY:
        raise ValueError(f"Unknown algorithm: {name!r}. Available: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    return spec.adapter_cls(params=params)


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
    from causal_copilot.algorithms.registry import REGISTRY

    spec = REGISTRY.get(decision.algorithm)
    algo_version = spec.algorithm_version if spec else "unknown"
    return Provenance(
        dataset_hash=data_hash,
        seed=seed,
        algorithm=decision.algorithm,
        algorithm_version=algo_version,
        package_version=__version__,
        hyperparams=Provenance.freeze_params(decision.hyperparams),
        planner=active_planner,
        runtime_seconds=elapsed,
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
        planner: str | None = None,
        algorithm: str | None = None,
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
            warnings.append(f"Dropped {n_dropped} rows with missing values ({n_dropped / n_before:.1%}).")
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
            # For forced algorithms with empty hyperparams, record effective defaults
            if not decision.hyperparams:
                decision = PlannerDecision(
                    algorithm=decision.algorithm,
                    hyperparams=algo.default_params(),
                    reason=decision.reason,
                )
        except (ValueError, ImportError) as e:
            return CausalResult(
                status="failed",
                summary=f"Failed to load algorithm {decision.algorithm}: {e}",
                warnings=warnings + [str(e)],
            )

        # Execute algorithm (with process-based timeout)
        try:
            try:
                adj_matrix, metadata = _run_in_subprocess(algo, numeric_df, timeout)
            except (OSError, AttributeError, TypeError) as sub_err:
                if isinstance(sub_err, TimeoutError):
                    raise  # TimeoutError is a subclass of OSError — don't swallow it
                # Subprocess couldn't start: fork unavailable (OSError), mock algo
                # missing .name/.get_params (AttributeError), or serialization
                # failure (TypeError). Fall back to direct execution (no timeout).
                adj_matrix, metadata, _ = algo.fit(numeric_df)
        except TimeoutError:
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

        elapsed = time.monotonic() - start_time

        # Guard: adjacency matrix dimensions must match feature count
        cols = list(numeric_df.columns)
        n_vars = len(cols)
        if adj_matrix.shape != (n_vars, n_vars):
            # Some wrappers (e.g. CDNOD) drop domain_index internally,
            # producing a smaller matrix than expected.
            if adj_matrix.shape[0] == adj_matrix.shape[1] and adj_matrix.shape[0] < n_vars:
                # Square but smaller — trim node_names to match
                # (e.g. CDNOD drops domain_index internally)
                cols = cols[: adj_matrix.shape[0]]
                warnings.append(
                    f"Adjacency matrix is {adj_matrix.shape[0]}x{adj_matrix.shape[0]} "
                    f"but data had {n_vars} columns; node_names trimmed to match matrix."
                )
            else:
                return CausalResult(
                    status="failed",
                    summary=f"Adjacency matrix shape {adj_matrix.shape} does not match {n_vars} features",
                    warnings=warnings,
                    provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                )

        # Build graph handling all edge types (1=directed, 2=undirected, 3=bidirected)
        graph = _build_graph(adj_matrix, cols)

        provenance = _make_provenance(data_hash, seed, decision, active_planner, elapsed)

        # Count edges by type — undirected/bidirected may be one-sided or symmetric
        n_directed = int(np.sum(adj_matrix == 1))
        # Count unique unordered pairs for symmetric edge types
        n_undirected = sum(
            1
            for i in range(adj_matrix.shape[0])
            for j in range(i + 1, adj_matrix.shape[1])
            if adj_matrix[i, j] == 2 or adj_matrix[j, i] == 2
        )
        n_bidirected = sum(
            1
            for i in range(adj_matrix.shape[0])
            for j in range(i + 1, adj_matrix.shape[1])
            if adj_matrix[i, j] == 3 or adj_matrix[j, i] == 3
        )

        edge_parts = [f"{n_directed} directed"]
        if n_undirected:
            edge_parts.append(f"{n_undirected} undirected")
        if n_bidirected:
            edge_parts.append(f"{n_bidirected} bidirected")
        edge_summary = ", ".join(edge_parts)

        return CausalResult(
            status="ok",
            adjacency_matrix=adj_matrix,
            node_names=cols,
            graph=graph,
            discovery_metadata=metadata if isinstance(metadata, dict) else {},
            summary=f"Discovered {edge_summary} edges using {decision.algorithm} "
            f"on {numeric_df.shape[0]} samples × {numeric_df.shape[1]} features "
            f"in {elapsed:.1f}s.",
            warnings=warnings,
            provenance=provenance,
            algorithm_selection_reason=decision.reason,
        )
