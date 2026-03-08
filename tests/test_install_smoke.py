"""Smoke tests verifying the package works without repo-root path hacks."""

import numpy as np
import pandas as pd


class TestPublicAPI:
    def test_import_causal_copilot(self):
        from causal_copilot import CausalCopilot

        assert CausalCopilot is not None

    def test_version_string(self):
        from causal_copilot import __version__

        assert isinstance(__version__, str) and len(__version__) > 0

    def test_cli_entry_point(self):
        from causal_copilot.cli import main

        assert callable(main)


class TestNoRepoRootDependency:
    def test_adapters_no_repo_root_hacks(self):
        """Adapters should use _backends, not causal_discovery/ tree."""
        import inspect

        from causal_copilot.algorithms import adapters

        source = inspect.getsource(adapters)
        assert "_REPO_ROOT" not in source
        assert "_WRAPPERS_DIR" not in source
        assert "importlib.util" not in source


class TestAnalyzeWithoutRepoRoot:
    def test_analyze_synthetic_data(self):
        from causal_copilot import CausalCopilot

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "X": (x := rng.normal(size=n)),
                "Y": 0.8 * x + rng.normal(size=n) * 0.3,
                "Z": 0.6 * (0.8 * x) + rng.normal(size=n) * 0.4,
            }
        )
        result = CausalCopilot().analyze(df, seed=42)
        # Must succeed — if this returns "failed" the backend is broken
        assert result.status == "ok", f"Expected ok, got {result.status}: {result.summary}"
        assert result.adjacency_matrix is not None
        assert result.node_names == ["X", "Y", "Z"]

    def test_result_serialization(self):
        import json

        from causal_copilot import CausalCopilot

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})
        result = CausalCopilot().analyze(df, seed=42)
        serialized = json.dumps(result.to_dict())
        assert isinstance(serialized, str)
