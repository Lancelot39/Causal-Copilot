"""Canonical algorithm registry -- single source of truth.

Every algorithm's metadata, adapter class, defaults, and upstream info
lives here. copilot.py, planner.py, cli.py all read from this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    adapter_cls: type
    family: str  # constraint, score, functional, hybrid, timeseries
    default_params: dict[str, Any]
    upstream_packages: list[str]
    algorithm_version: str  # e.g. "pc-v1|causallearn=0.1.3"
    tags: tuple = ()
    optional_deps: tuple = ()


def _build_registry() -> dict[str, AlgorithmSpec]:
    from causal_copilot.algorithms.adapters import (
        DirectLiNGAMAdapter,
        GESAdapter,
        GrangerCausalityAdapter,
        ICALiNGAMAdapter,
        NOTEARSLinearAdapter,
        PCAdapter,
        PCMCIAdapter,
    )

    return {
        "PC": AlgorithmSpec(
            name="PC",
            adapter_cls=PCAdapter,
            family="constraint",
            default_params={"alpha": 0.05, "indep_test": "fisherz", "stable": True, "depth": 4},
            upstream_packages=["causal-learn"],
            algorithm_version="pc-v1",
            tags=("cpdag-output", "linear", "gaussian"),
        ),
        "GES": AlgorithmSpec(
            name="GES",
            adapter_cls=GESAdapter,
            family="score",
            default_params={"score_func": "local_score_BIC"},
            upstream_packages=["causal-learn"],
            algorithm_version="ges-v1",
            tags=("cpdag-output", "linear"),
        ),
        "NOTEARSLinear": AlgorithmSpec(
            name="NOTEARSLinear",
            adapter_cls=NOTEARSLinearAdapter,
            family="score",
            default_params={"lambda1": 0.1, "max_iter": 100, "h_tol": 1e-8, "w_threshold": 0.3},
            upstream_packages=["castle", "causal-learn"],
            algorithm_version="notears-linear-v1",
            tags=("dag-output", "linear", "continuous-optimization"),
            optional_deps=("algorithms",),
        ),
        "DirectLiNGAM": AlgorithmSpec(
            name="DirectLiNGAM",
            adapter_cls=DirectLiNGAMAdapter,
            family="functional",
            default_params={"measure": "pwling"},
            upstream_packages=["causal-learn"],
            algorithm_version="direct-lingam-v1",
            tags=("dag-output", "linear", "non-gaussian"),
        ),
        "PCMCI": AlgorithmSpec(
            name="PCMCI",
            adapter_cls=PCMCIAdapter,
            family="timeseries",
            default_params={"tau_min": 0, "tau_max": 2, "pc_alpha": 0.05, "alpha_level": 0.05},
            upstream_packages=["tigramite"],
            algorithm_version="pcmci-v1",
            tags=("timeseries", "constraint"),
            optional_deps=("algorithms",),
        ),
        "ICALiNGAM": AlgorithmSpec(
            name="ICALiNGAM",
            adapter_cls=ICALiNGAMAdapter,
            family="functional",
            default_params={"max_iter": 1000},
            upstream_packages=["causal-learn"],
            algorithm_version="ica-lingam-v1",
            tags=("dag-output", "linear", "non-gaussian"),
        ),
        "GrangerCausality": AlgorithmSpec(
            name="GrangerCausality",
            adapter_cls=GrangerCausalityAdapter,
            family="timeseries",
            default_params={"p": 10, "alpha": 0.05, "criterion": "ssr_ftest"},
            upstream_packages=["statsmodels"],
            algorithm_version="granger-v1",
            tags=("timeseries", "linear", "pairwise"),
        ),
    }


REGISTRY: dict[str, AlgorithmSpec] = _build_registry()
