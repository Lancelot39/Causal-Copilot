"""Tests for process-based algorithm timeout."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from causal_copilot.copilot import CausalCopilot


class TestTimeout:
    @pytest.fixture
    def simple_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})

    def test_normal_execution_within_timeout(self, simple_df):
        """Normal algorithm should succeed within timeout."""
        result = CausalCopilot().analyze(simple_df, timeout=60, seed=0)
        assert result.status == "ok"

    def test_timeout_returns_failed_status(self, simple_df):
        """If we mock _run_in_subprocess to raise TimeoutError, result is failed."""
        with patch("causal_copilot.copilot._run_in_subprocess") as mock_run:
            mock_run.side_effect = TimeoutError("timed out after 1s")
            result = CausalCopilot().analyze(simple_df, timeout=1, seed=0)
        assert result.status == "failed"
        assert "timed out" in result.summary.lower() or "timeout" in result.summary.lower()

    def test_timeout_has_provenance(self, simple_df):
        """Even on timeout, provenance should be recorded."""
        with patch("causal_copilot.copilot._run_in_subprocess") as mock_run:
            mock_run.side_effect = TimeoutError("timed out")
            result = CausalCopilot().analyze(simple_df, timeout=1, seed=0)
        assert result.provenance is not None

    def test_subprocess_seed_reproducibility(self, simple_df):
        """Same seed should produce identical results across runs (forces ICALiNGAM)."""
        r1 = CausalCopilot().analyze(simple_df, algorithm="ICALiNGAM", seed=123, timeout=60)
        r2 = CausalCopilot().analyze(simple_df, algorithm="ICALiNGAM", seed=123, timeout=60)
        assert r1.status == "ok"
        assert r2.status == "ok"
        np.testing.assert_array_equal(r1.adjacency_matrix, r2.adjacency_matrix)
