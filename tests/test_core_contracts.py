"""Tests for core interface contracts -- CausalResult, Provenance, base classes."""

import json

import numpy as np
import pytest

from causal_copilot.core.base import CausalDiscoveryBase, CausalInferenceBase
from causal_copilot.core.result import CausalResult, Provenance, TreatmentEffect, _json_safe


class TestProvenance:
    def test_hash_data_ndarray(self):
        data = np.array([[1, 2], [3, 4]])
        h = Provenance.hash_data(data)
        assert isinstance(h, str) and len(h) == 64

    def test_hash_data_deterministic(self):
        data = np.array([[1, 2], [3, 4]])
        assert Provenance.hash_data(data) == Provenance.hash_data(data)

    def test_get_environment(self):
        env = Provenance.get_environment()
        assert "python=" in env
        assert "numpy=" in env

    def test_frozen(self):
        p = Provenance(
            dataset_hash="abc",
            seed=42,
            algorithm="PC",
            algorithm_version="0.1",
            package_version="0.1.0",
            hyperparams=Provenance.freeze_params({"alpha": 0.05}),
            planner="rule",
            runtime_seconds=1.0,
            timestamp="2026-01-01T00:00:00Z",
            environment="test",
        )
        with pytest.raises(AttributeError):
            p.seed = 99

    def test_freeze_thaw_params(self):
        original = {"alpha": 0.05, "depth": 4}
        frozen = Provenance.freeze_params(original)
        assert isinstance(frozen, tuple)
        thawed = Provenance.thaw_params(frozen)
        assert thawed == original


class TestCausalResult:
    def test_default_assumptions(self):
        r = CausalResult(status="ok")
        assert len(r.assumptions) >= 1
        assert "exploratory" in r.assumptions[0].lower()

    def test_to_dict_minimal(self):
        r = CausalResult(status="failed", summary="No data")
        d = r.to_dict()
        assert d["status"] == "failed"
        assert "adjacency_matrix" not in d

    def test_to_dict_with_matrix(self):
        r = CausalResult(status="ok", adjacency_matrix=np.eye(3))
        d = r.to_dict()
        assert d["adjacency_matrix"] == np.eye(3).tolist()

    def test_schema_version(self):
        r = CausalResult(status="ok")
        assert r.schema_version == "0.1.0"


class TestJsonSafe:
    def test_ndarray(self):
        result = _json_safe(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_nested_dict(self):
        data = {"a": np.array([1.0]), "b": {"c": np.int64(5)}}
        result = _json_safe(data)
        assert result == {"a": [1.0], "b": {"c": 5}}
        json.dumps(result)  # must not raise

    def test_non_serializable_dropped(self):
        result = _json_safe({"model": object(), "score": 0.5})
        assert result["score"] == 0.5
        assert "<object>" in result["model"]

    def test_primitives(self):
        assert _json_safe("hello") == "hello"
        assert _json_safe(42) == 42
        assert _json_safe(None) is None


class TestCausalResultNodeNames:
    def test_node_names_in_to_dict(self):
        r = CausalResult(status="ok", node_names=["A", "B"], adjacency_matrix=np.eye(2))
        d = r.to_dict()
        assert d["node_names"] == ["A", "B"]

    def test_discovery_metadata_in_to_dict(self):
        r = CausalResult(status="ok", discovery_metadata={"lag_matrix": np.zeros((2, 2))})
        d = r.to_dict()
        assert "lag_matrix" in d["discovery_metadata"]
        json.dumps(d)  # must be serializable


class TestTreatmentEffect:
    def test_basic(self):
        te = TreatmentEffect(treatment="X", outcome="Y", method="DML", ate=0.5)
        assert te.ate == 0.5
        assert te.att is None


class TestDiscoveryBaseContract:
    """Verify that CausalDiscoveryBase cannot be instantiated without implementing abstract methods."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            CausalDiscoveryBase()

    def test_concrete_subclass(self):
        class DummyDiscovery(CausalDiscoveryBase):
            @property
            def name(self) -> str:
                return "Dummy"

            def fit(self, data, **kwargs):
                return np.eye(data.shape[1]), {"info": "dummy"}, None

            def default_params(self):
                return {"alpha": 0.05}

        algo = DummyDiscovery(params={"alpha": 0.01})
        assert algo.name == "Dummy"
        mat, meta, model = algo.fit(np.random.randn(100, 5))
        assert mat.shape == (5, 5)
        assert model is None
        assert algo.get_params() == {"alpha": 0.01}


class TestInferenceBaseContract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            CausalInferenceBase()

    def test_concrete_subclass(self):
        class DummyInference(CausalInferenceBase):
            @property
            def name(self):
                return "DummyDML"

            def fit(self, data):
                pass

            def estimate_ate(self, data):
                return 0.5, (0.3, 0.7)

            def default_params(self):
                return {}

        est = DummyInference(outcome="Y", treatment="X")
        assert est.name == "DummyDML"
        ate, ci = est.estimate_ate(None)
        assert ate == 0.5
