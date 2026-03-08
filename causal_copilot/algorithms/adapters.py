"""Adapters wrapping vendored backends to CausalDiscoveryBase ABC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.core.base import CausalDiscoveryBase


class PCAdapter(CausalDiscoveryBase):
    """Adapter for the PC algorithm."""

    @property
    def name(self) -> str:
        return "PC"

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 0.05, "indep_test": "fisherz", "stable": True, "depth": 4}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.pc import PCBackend

        backend = PCBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GESAdapter(CausalDiscoveryBase):
    """Adapter for the GES algorithm."""

    @property
    def name(self) -> str:
        return "GES"

    def default_params(self) -> dict[str, Any]:
        return {"score_func": "local_score_BIC"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.ges import GESBackend

        backend = GESBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class NOTEARSLinearAdapter(CausalDiscoveryBase):
    """Adapter for the NOTEARSLinear algorithm."""

    @property
    def name(self) -> str:
        return "NOTEARSLinear"

    def default_params(self) -> dict[str, Any]:
        return {"lambda1": 0.1, "max_iter": 100, "h_tol": 1e-8, "w_threshold": 0.3}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.notears_linear import NOTEARSLinearBackend

        backend = NOTEARSLinearBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class DirectLiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the DirectLiNGAM algorithm."""

    @property
    def name(self) -> str:
        return "DirectLiNGAM"

    def default_params(self) -> dict[str, Any]:
        return {"measure": "pwling"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.direct_lingam import DirectLiNGAMBackend

        backend = DirectLiNGAMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class PCMCIAdapter(CausalDiscoveryBase):
    """Adapter for the PCMCI time-series algorithm."""

    @property
    def name(self) -> str:
        return "PCMCI"

    def default_params(self) -> dict[str, Any]:
        return {"tau_min": 0, "tau_max": 2, "pc_alpha": 0.05, "alpha_level": 0.05}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.pcmci import PCMCIBackend

        backend = PCMCIBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class ICALiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the ICALiNGAM algorithm."""

    @property
    def name(self) -> str:
        return "ICALiNGAM"

    def default_params(self) -> dict[str, Any]:
        return {"max_iter": 1000}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.ica_lingam import ICALiNGAMBackend

        backend = ICALiNGAMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GrangerCausalityAdapter(CausalDiscoveryBase):
    """Adapter for the GrangerCausality algorithm."""

    @property
    def name(self) -> str:
        return "GrangerCausality"

    def default_params(self) -> dict[str, Any]:
        return {"p": 10, "alpha": 0.05, "criterion": "ssr_ftest"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.granger import GrangerCausalityBackend

        backend = GrangerCausalityBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


# Registry for programmatic access
STABLE_ALGORITHMS = {
    "PC": PCAdapter,
    "GES": GESAdapter,
    "NOTEARSLinear": NOTEARSLinearAdapter,
    "DirectLiNGAM": DirectLiNGAMAdapter,
    "PCMCI": PCMCIAdapter,
    "ICALiNGAM": ICALiNGAMAdapter,
    "GrangerCausality": GrangerCausalityAdapter,
}
