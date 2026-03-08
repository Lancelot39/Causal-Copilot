"""Tests for the unified algorithm registry."""

from causal_copilot.algorithms.registry import REGISTRY


class TestAlgorithmSpec:
    def test_registry_has_5_algorithms(self):
        assert len(REGISTRY) == 5

    def test_all_specs_have_required_fields(self):
        for name, spec in REGISTRY.items():
            assert spec.name == name
            assert spec.adapter_cls is not None
            assert spec.family in ("constraint", "score", "functional", "hybrid", "timeseries")
            assert isinstance(spec.default_params, dict)
            assert len(spec.default_params) > 0
            assert isinstance(spec.upstream_packages, list)
            assert isinstance(spec.algorithm_version, str)

    def test_pc_spec(self):
        spec = REGISTRY["PC"]
        assert spec.family == "constraint"
        assert "alpha" in spec.default_params
        assert "causal-learn" in spec.upstream_packages

    def test_pcmci_spec(self):
        spec = REGISTRY["PCMCI"]
        assert spec.family == "timeseries"
        assert "tigramite" in spec.upstream_packages

    def test_spec_default_params_match_adapter(self):
        """Every spec.default_params must match adapter.default_params()."""
        for name, spec in REGISTRY.items():
            adapter = spec.adapter_cls()
            assert adapter.default_params() == spec.default_params, f"{name}: registry defaults != adapter defaults"

    def test_spec_instantiation(self):
        """Every spec.adapter_cls() should produce a valid adapter."""
        for name, spec in REGISTRY.items():
            adapter = spec.adapter_cls()
            assert adapter.name == name


class TestRegistryIsCanonical:
    def test_copilot_no_local_registry(self):
        import inspect

        from causal_copilot import copilot

        source = inspect.getsource(copilot)
        assert "_ALGORITHM_REGISTRY" not in source

    def test_planner_no_local_defaults(self):
        import inspect

        from causal_copilot.core import planner

        source = inspect.getsource(planner)
        assert "_DEFAULT_HYPERPARAMS" not in source
