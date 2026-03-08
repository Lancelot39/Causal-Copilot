"""Integration tests for CausalCopilot.analyze() — the main entry point."""
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_copilot.copilot import CausalCopilot, _build_graph, AlgorithmTimeoutError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """50-row, 4-column numeric DataFrame — well within validation limits."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(size=n) * 0.3
    z = x + y + rng.normal(size=n) * 0.5
    w = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": w})


@pytest.fixture
def csv_path(simple_df, tmp_path):
    """Write simple_df to a CSV and return its path."""
    p = tmp_path / "test_data.csv"
    simple_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_planner(self):
        c = CausalCopilot()
        assert c.planner == "rule"

    def test_invalid_planner_raises(self):
        with pytest.raises(ValueError, match="Unknown planner"):
            CausalCopilot(planner="bogus")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_file_not_found(self):
        result = CausalCopilot().analyze("/nonexistent/path.csv")
        assert result.status == "failed"
        assert "not found" in result.summary.lower()

    def test_csv_loading(self, csv_path):
        # Mock the algorithm to avoid needing real wrappers
        with _mock_algorithm():
            result = CausalCopilot().analyze(csv_path, seed=0)
        assert result.status == "ok"

    def test_dataframe_input(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"

    def test_unsupported_type(self):
        result = CausalCopilot().analyze(42)
        assert result.status == "failed"
        assert "Unsupported" in result.summary

    def test_malformed_csv(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03")
        result = CausalCopilot().analyze(bad)
        # Should fail gracefully, not raise
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Validation & cleaning
# ---------------------------------------------------------------------------

class TestValidationAndCleaning:
    def test_too_few_numeric_columns(self):
        df = pd.DataFrame({"a": [1] * 20, "b": ["x"] * 20})
        result = CausalCopilot().analyze(df)
        assert result.status == "failed"

    def test_constant_columns_dropped(self, simple_df):
        simple_df["const"] = 7.0
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert any("constant" in w.lower() for w in result.warnings)

    def test_nan_rows_dropped(self, simple_df):
        simple_df.iloc[0, 0] = np.nan
        simple_df.iloc[1, 1] = np.nan
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert any("missing" in w.lower() for w in result.warnings)

    def test_all_nan_fails(self):
        df = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20})
        result = CausalCopilot().analyze(df)
        assert result.status == "failed"

    def test_too_few_rows_after_dropna(self):
        # 20 rows, 12 have NaN in one column → 8 remain after dropna → fail (<10)
        # Keep missing ratio under 50% so validate_data passes
        vals = [float(i) for i in range(8)] + [np.nan] * 12
        df = pd.DataFrame({
            "a": vals,
            "b": [float(i) for i in range(20)],
        })
        result = CausalCopilot().analyze(df)
        assert result.status == "failed"
        assert "remain" in result.summary.lower()


# ---------------------------------------------------------------------------
# Algorithm selection & execution
# ---------------------------------------------------------------------------

class TestAlgorithmSelection:
    def test_forced_algorithm(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", seed=0)
        assert result.status == "ok"
        assert result.provenance.algorithm == "PC"
        assert "User specified" in result.algorithm_selection_reason

    def test_unknown_algorithm(self, simple_df):
        result = CausalCopilot().analyze(simple_df, algorithm="FakeAlgo")
        assert result.status == "failed"
        assert "FakeAlgo" in result.summary

    def test_planner_override(self, simple_df):
        result = CausalCopilot().analyze(simple_df, planner="bogus")
        assert result.status == "failed"
        assert "Unknown planner" in result.summary

    def test_algorithm_failure_captured(self, simple_df):
        """If algorithm.fit() raises, result is failed with message."""
        def _failing_fit(data, **kwargs):
            raise RuntimeError("algo exploded")

        with _mock_algorithm(fit_side_effect=_failing_fit):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "failed"
        assert "exploded" in result.summary
        assert result.provenance is not None


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_ok_result_has_all_fields(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert result.adjacency_matrix is not None
        assert result.adjacency_matrix.shape[0] == result.adjacency_matrix.shape[1]
        assert result.provenance is not None
        assert result.provenance.seed == 0
        assert result.provenance.runtime_seconds > 0
        assert result.provenance.dataset_hash
        assert result.provenance.planner == "rule"
        assert len(result.assumptions) >= 1
        assert result.summary

    def test_provenance_uses_frozen_params(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert isinstance(result.provenance.hyperparams, tuple)

    def test_node_names_populated(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.node_names == ["x", "y", "z", "w"]

    def test_discovery_metadata_populated(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.discovery_metadata == {"mock": True}

    def test_forced_algo_has_effective_hyperparams(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", seed=0)
        # Should have default_params, not empty {}
        params = dict(result.provenance.hyperparams)
        assert params  # non-empty
        assert "mock_param" in params

    def test_matrix_smaller_than_data_trims_node_names(self, simple_df):
        # Simulate a wrapper (e.g. CDNOD) that drops domain_index internally,
        # producing a 3x3 matrix for 4-column data.
        small_adj = np.zeros((3, 3))
        with _mock_algorithm(adj_matrix=small_adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert result.node_names == ["x", "y", "z"]  # trimmed to match matrix
        assert any("trimmed" in w for w in result.warnings)

    def test_non_square_matrix_fails(self, simple_df):
        bad_adj = np.zeros((3, 4))
        with _mock_algorithm(adj_matrix=bad_adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "failed"
        assert "non-square" in result.summary.lower()

    def test_to_dict_roundtrip(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        d = result.to_dict()
        assert d["status"] == "ok"
        assert "adjacency_matrix" in d
        assert "node_names" in d
        assert d["node_names"] == ["x", "y", "z", "w"]
        assert "discovery_metadata" in d
        assert "provenance" in d
        assert d["provenance"]["seed"] == 0


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_directed_edges(self):
        # mat[i,j]=1 means j→i
        mat = np.array([[0, 1], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        assert g is not None
        assert g.has_edge("B", "A")
        assert not g.has_edge("A", "B")

    def test_undirected_edges(self):
        mat = np.array([[0, 2], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        # Undirected: both directions
        assert g.has_edge("B", "A")
        assert g.has_edge("A", "B")
        assert g["B"]["A"]["edge_type"] == "undirected"

    def test_bidirected_edges(self):
        mat = np.array([[0, 3], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        assert g.has_edge("B", "A")
        assert g.has_edge("A", "B")
        assert g["B"]["A"]["edge_type"] == "bidirected"

    def test_empty_graph(self):
        mat = np.zeros((3, 3))
        g = _build_graph(mat, ["A", "B", "C"])
        assert g is not None
        assert g.number_of_edges() == 0
        assert g.number_of_nodes() == 3

    def test_mixed_edges(self):
        mat = np.array([
            [0, 1, 0],
            [0, 0, 2],
            [0, 0, 0],
        ])
        g = _build_graph(mat, ["A", "B", "C"])
        assert g.has_edge("B", "A")  # directed
        assert g["B"]["A"]["edge_type"] == "directed"
        assert g.has_edge("C", "B")  # undirected
        assert g.has_edge("B", "C")  # undirected (both)


# ---------------------------------------------------------------------------
# Edge counting in summary
# ---------------------------------------------------------------------------

class TestEdgeCounting:
    def test_directed_count(self, simple_df):
        adj = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "2 directed" in result.summary

    def test_undirected_count_symmetric(self, simple_df):
        """Undirected edges encoded symmetrically: both (i,j) and (j,i) = 2."""
        adj = np.array([
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "1 undirected" in result.summary

    def test_undirected_count_one_sided(self, simple_df):
        """Undirected edges encoded one-sided: only (i,j) = 2 (PC wrapper style)."""
        adj = np.array([
            [0, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "1 undirected" in result.summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockAlgo:
    """Minimal mock that satisfies the algorithm interface."""
    def __init__(self, adj_matrix=None, fit_side_effect=None):
        self._adj = adj_matrix
        self._side_effect = fit_side_effect

    def default_params(self):
        return {"mock_param": True}

    def fit(self, data, **kwargs):
        if self._side_effect:
            return self._side_effect(data, **kwargs)
        n = data.shape[1]
        adj = self._adj if self._adj is not None else np.zeros((n, n))
        return adj, {"mock": True}, None


class _mock_algorithm:
    """Context manager that patches _load_algorithm to return a mock."""
    def __init__(self, adj_matrix=None, fit_side_effect=None):
        self._adj = adj_matrix
        self._side_effect = fit_side_effect
        self._patcher = None

    def __enter__(self):
        mock_algo = _MockAlgo(self._adj, self._side_effect)
        self._patcher = patch(
            "causal_copilot.copilot._load_algorithm",
            return_value=mock_algo,
        )
        self._patcher.start()
        return mock_algo

    def __exit__(self, *args):
        self._patcher.stop()
