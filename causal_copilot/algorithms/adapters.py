"""Adapters wrapping existing causal_discovery wrappers to CausalDiscoveryBase ABC."""
from __future__ import annotations

import importlib.util
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from causal_copilot.core.base import CausalDiscoveryBase

# Ensure the repo root is on sys.path so existing wrapper imports work
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure externals are accessible (matches existing wrapper convention)
_EXTERNALS = os.path.join(_REPO_ROOT, "externals")
_CAUSAL_LEARN = os.path.join(_EXTERNALS, "causal-learn")
for p in [_EXTERNALS, _CAUSAL_LEARN]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

_WRAPPERS_DIR = os.path.join(_REPO_ROOT, "causal_discovery", "wrappers")


def _import_wrapper(module_name: str):
    """Import a wrapper module directly by file, bypassing wrappers/__init__.py.

    This avoids the eager import of torch and all 20+ algorithm modules
    that __init__.py triggers.
    """
    file_path = os.path.join(_WRAPPERS_DIR, f"{module_name}.py")
    if not os.path.isfile(file_path):
        raise ImportError(f"Wrapper file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(
        f"causal_discovery.wrappers.{module_name}", file_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PCAdapter(CausalDiscoveryBase):
    """Adapter for the PC algorithm wrapper."""

    @property
    def name(self) -> str:
        return "PC"

    def default_params(self) -> Dict[str, Any]:
        return {"alpha": 0.05, "indep_test": "fisherz", "stable": True, "depth": 4}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        mod = _import_wrapper("pc")
        PC = mod.PC
        wrapper = PC(params={**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return wrapper.fit(df)


class GESAdapter(CausalDiscoveryBase):
    """Adapter for the GES algorithm wrapper."""

    @property
    def name(self) -> str:
        return "GES"

    def default_params(self) -> Dict[str, Any]:
        return {"score_func": "local_score_BIC"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        mod = _import_wrapper("ges")
        GES = mod.GES
        wrapper = GES(params={**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return wrapper.fit(df)


class NOTEARSLinearAdapter(CausalDiscoveryBase):
    """Adapter for the NOTEARSLinear algorithm wrapper."""

    @property
    def name(self) -> str:
        return "NOTEARSLinear"

    def default_params(self) -> Dict[str, Any]:
        return {"lambda1": 0.1, "max_iter": 100, "h_tol": 1e-8, "w_threshold": 0.3}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        mod = _import_wrapper("notears_linear")
        NOTEARSLinear = mod.NOTEARSLinear
        wrapper = NOTEARSLinear(params={**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return wrapper.fit(df)


class DirectLiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the DirectLiNGAM algorithm wrapper."""

    @property
    def name(self) -> str:
        return "DirectLiNGAM"

    def default_params(self) -> Dict[str, Any]:
        return {"measure": "pwling"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        mod = _import_wrapper("direct_lingam")
        DirectLiNGAM = mod.DirectLiNGAM
        wrapper = DirectLiNGAM(params={**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return wrapper.fit(df)


class PCMCIAdapter(CausalDiscoveryBase):
    """Adapter for the PCMCI time-series algorithm wrapper."""

    @property
    def name(self) -> str:
        return "PCMCI"

    def default_params(self) -> Dict[str, Any]:
        return {"tau_min": 0, "tau_max": 2, "pc_alpha": 0.05, "alpha_level": 0.05}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        mod = _import_wrapper("pcmci")
        PCMCI = mod.PCMCI
        wrapper = PCMCI(params={**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return wrapper.fit(df)


# Registry for programmatic access
STABLE_ALGORITHMS = {
    "PC": PCAdapter,
    "GES": GESAdapter,
    "NOTEARSLinear": NOTEARSLinearAdapter,
    "DirectLiNGAM": DirectLiNGAMAdapter,
    "PCMCI": PCMCIAdapter,
}
