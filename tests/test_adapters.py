"""Tests for algorithm adapters — verify ABC conformance and adapter pattern."""

import pytest

from causal_copilot.algorithms.adapters import (
    STABLE_ALGORITHMS,
    DirectLiNGAMAdapter,
    GESAdapter,
    NOTEARSLinearAdapter,
    PCAdapter,
    PCMCIAdapter,
)
from causal_copilot.core.base import CausalDiscoveryBase


class TestAdapterConformance:
    """Verify all adapters conform to CausalDiscoveryBase ABC."""

    @pytest.mark.parametrize(
        "cls",
        [
            PCAdapter,
            GESAdapter,
            NOTEARSLinearAdapter,
            DirectLiNGAMAdapter,
            PCMCIAdapter,
        ],
    )
    def test_is_subclass(self, cls):
        assert issubclass(cls, CausalDiscoveryBase)

    @pytest.mark.parametrize(
        "cls",
        [
            PCAdapter,
            GESAdapter,
            NOTEARSLinearAdapter,
            DirectLiNGAMAdapter,
            PCMCIAdapter,
        ],
    )
    def test_can_instantiate(self, cls):
        adapter = cls()
        assert isinstance(adapter, CausalDiscoveryBase)
        assert isinstance(adapter.name, str) and len(adapter.name) > 0

    @pytest.mark.parametrize(
        "cls",
        [
            PCAdapter,
            GESAdapter,
            NOTEARSLinearAdapter,
            DirectLiNGAMAdapter,
            PCMCIAdapter,
        ],
    )
    def test_default_params(self, cls):
        adapter = cls()
        params = adapter.default_params()
        assert isinstance(params, dict)
        assert len(params) > 0

    @pytest.mark.parametrize(
        "cls",
        [
            PCAdapter,
            GESAdapter,
            NOTEARSLinearAdapter,
            DirectLiNGAMAdapter,
            PCMCIAdapter,
        ],
    )
    def test_params_override(self, cls):
        adapter = cls(params={"custom_key": 42})
        assert adapter.get_params()["custom_key"] == 42


class TestRegistry:
    def test_registry_has_5_algorithms(self):
        assert len(STABLE_ALGORITHMS) == 5

    def test_registry_names(self):
        expected = {"PC", "GES", "NOTEARSLinear", "DirectLiNGAM", "PCMCI"}
        assert set(STABLE_ALGORITHMS.keys()) == expected

    def test_registry_values_are_classes(self):
        for name, cls in STABLE_ALGORITHMS.items():
            assert issubclass(cls, CausalDiscoveryBase), f"{name} is not a CausalDiscoveryBase subclass"
