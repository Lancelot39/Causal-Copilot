"""Typed result contracts for Causal-Copilot pipeline outputs."""
from __future__ import annotations

import hashlib
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None


@dataclass(frozen=True)
class Provenance:
    """Full reproducibility record for a single pipeline run."""
    dataset_hash: str                          # SHA256 of input data
    seed: int                                  # random seed used
    algorithm: str                             # which algo ran
    algorithm_version: str                     # package commit or wrapper version
    package_version: str                       # causal-copilot version
    hyperparams: Dict[str, Any]                # exact params used
    planner: str                               # "llm" | "rule" | "oracle" | "random"
    runtime_seconds: float                     # wall-clock time
    timestamp: str                             # ISO 8601 UTC
    environment: str                           # OS, Python version, key deps
    planner_model: Optional[str] = None        # e.g. "gpt-4o-2024-05-13"
    prompt_version: Optional[str] = None       # hash of prompt template

    @staticmethod
    def hash_data(data) -> str:
        """Compute SHA256 hash of input data (DataFrame or ndarray)."""
        if hasattr(data, 'to_csv'):
            raw = data.to_csv(index=False).encode('utf-8')
        elif isinstance(data, np.ndarray):
            raw = data.tobytes()
        else:
            raw = str(data).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def get_environment() -> str:
        """Capture current runtime environment string."""
        parts = [
            f"os={platform.system()}-{platform.release()}",
            f"python={sys.version.split()[0]}",
            f"numpy={np.__version__}",
        ]
        try:
            import pandas as pd
            parts.append(f"pandas={pd.__version__}")
        except ImportError:
            pass
        try:
            import sklearn
            parts.append(f"sklearn={sklearn.__version__}")
        except ImportError:
            pass
        return "; ".join(parts)

    @staticmethod
    def now_utc() -> str:
        return datetime.now(timezone.utc).isoformat()


@dataclass
class TreatmentEffect:
    """Result of a single treatment effect estimation."""
    treatment: str
    outcome: str
    method: str                               # "DML", "DRL", "IV", etc.
    ate: Optional[float] = None
    ate_ci: Optional[tuple] = None            # (lower, upper)
    att: Optional[float] = None
    att_ci: Optional[tuple] = None
    cate: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalResult:
    """
    Canonical output of every Causal-Copilot pipeline run.

    Design principles:
    - status is always set (never None)
    - assumptions + warnings enable "do no harm" guardrails
    - provenance enables exact reproducibility
    - schema_version enables forward-compatible evolution
    """
    status: Literal["ok", "partial", "failed"]
    schema_version: str = "0.1.0"

    # Core outputs
    adjacency_matrix: Optional[np.ndarray] = None
    graph: Optional[Any] = None                # nx.DiGraph when available
    effects: Dict[str, TreatmentEffect] = field(default_factory=dict)

    # Transparency (every output MUST have these)
    assumptions: List[str] = field(default_factory=lambda: [
        "Causal discovery results are exploratory, not confirmatory.",
        "Results depend on algorithm assumptions — see provenance for details.",
    ])
    warnings: List[str] = field(default_factory=list)
    summary: str = ""

    # Reproducibility
    provenance: Optional[Provenance] = None

    # Optional extras
    report_path: Optional[Path] = None
    algorithm_selection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dictionary."""
        d = {
            "status": self.status,
            "schema_version": self.schema_version,
            "summary": self.summary,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "algorithm_selection_reason": self.algorithm_selection_reason,
        }
        if self.adjacency_matrix is not None:
            d["adjacency_matrix"] = self.adjacency_matrix.tolist()
        if self.effects:
            d["effects"] = {
                k: {"treatment": v.treatment, "outcome": v.outcome,
                     "method": v.method, "ate": v.ate}
                for k, v in self.effects.items()
            }
        if self.provenance:
            d["provenance"] = {
                "dataset_hash": self.provenance.dataset_hash,
                "seed": self.provenance.seed,
                "algorithm": self.provenance.algorithm,
                "planner": self.provenance.planner,
                "runtime_seconds": self.provenance.runtime_seconds,
                "timestamp": self.provenance.timestamp,
                "environment": self.provenance.environment,
                "hyperparams": self.provenance.hyperparams,
                "package_version": self.provenance.package_version,
            }
        return d
