"""Real algorithm smoke tests — execute actual wrappers on synthetic data.

Run with: pytest tests/test_algorithms_slow.py -m slow -v
These tests are SKIPPED if algorithm dependencies are not installed.
"""
import numpy as np
import pandas as pd
import pytest


def _make_linear_data(n=100, seed=42):
    """Simple X → Y → Z linear data."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n) * 0.3
    z = 0.6 * y + rng.normal(size=n) * 0.4
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _try_import_adapter(name):
    """Try to import and instantiate an adapter, skip if deps missing."""
    try:
        from causal_copilot.algorithms.adapters import STABLE_ALGORITHMS
        cls = STABLE_ALGORITHMS[name]
        adapter = cls()
        return adapter
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        pytest.skip(f"Dependencies not available for {name}: {e}")


def _run_fit(adapter, df, name):
    """Run adapter.fit(), skip if runtime deps are missing."""
    try:
        return adapter.fit(df)
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        pytest.skip(f"Runtime dependencies not available for {name}: {e}")


@pytest.mark.slow
class TestRealAlgorithms:
    """Each test instantiates a real adapter, runs fit(), and validates output shape."""

    def test_pc_real(self):
        adapter = _try_import_adapter("PC")
        df = _make_linear_data()
        adj, meta, model = _run_fit(adapter, df, "PC")
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)
        assert isinstance(meta, dict)

    def test_ges_real(self):
        adapter = _try_import_adapter("GES")
        df = _make_linear_data()
        adj, meta, model = _run_fit(adapter, df, "GES")
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)
        assert isinstance(meta, dict)

    def test_notears_linear_real(self):
        adapter = _try_import_adapter("NOTEARSLinear")
        df = _make_linear_data()
        adj, meta, model = _run_fit(adapter, df, "NOTEARSLinear")
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)
        assert isinstance(meta, dict)

    def test_direct_lingam_real(self):
        adapter = _try_import_adapter("DirectLiNGAM")
        df = _make_linear_data()
        adj, meta, model = _run_fit(adapter, df, "DirectLiNGAM")
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)
        assert isinstance(meta, dict)

    def test_pcmci_real(self):
        adapter = _try_import_adapter("PCMCI")
        # PCMCI needs time-series-like data (more rows)
        df = _make_linear_data(n=200)
        adj, meta, model = _run_fit(adapter, df, "PCMCI")
        assert isinstance(adj, np.ndarray)
        assert adj.shape == (3, 3)
        assert isinstance(meta, dict)


@pytest.mark.slow
class TestEndToEndQuickstart:
    """Run the full quickstart path without mocks."""

    def test_quickstart_e2e(self, tmp_path):
        """Full CausalCopilot.analyze() on synthetic data with real algorithm."""
        try:
            from causal_copilot import CausalCopilot
            import json

            df = _make_linear_data(n=200)
            copilot = CausalCopilot()
            result = copilot.analyze(df, seed=42)

            assert result.status == "ok"
            assert result.adjacency_matrix is not None
            assert result.node_names == ["X", "Y", "Z"]
            assert result.provenance is not None
            assert result.provenance.seed == 42
            assert dict(result.provenance.hyperparams)  # non-empty

            # Verify JSON serialization works
            out_path = tmp_path / "result.json"
            out_path.write_text(json.dumps(result.to_dict(), indent=2))
            data = json.loads(out_path.read_text())
            assert data["status"] == "ok"
            assert data["node_names"] == ["X", "Y", "Z"]

        except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
            pytest.skip(f"Algorithm dependencies not available: {e}")
