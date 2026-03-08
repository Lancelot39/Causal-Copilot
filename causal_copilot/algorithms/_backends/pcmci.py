"""PCMCI backend — imports tigramite directly."""
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple

from tigramite.pcmci import PCMCI as PCMCI_model
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr

from causal_copilot.algorithms._backends._base import Backend


def _build_cond_ind_test(test_name: str):
    """Instantiate a tigramite independence test by name."""
    if test_name == "parcorr":
        return ParCorr()
    elif test_name == "robustparcorr":
        return RobustParCorr()
    elif test_name == "gpdc":
        try:
            import torch
            if torch.cuda.is_available():
                from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
                return GPDC()
        except ImportError:
            pass
        from tigramite.independence_tests.gpdc import GPDC
        return GPDC(significance="analytic", gp_params=None)
    elif test_name == "gsq":
        from tigramite.independence_tests.gsquared import Gsquared
        return Gsquared(significance="analytic")
    elif test_name == "regression":
        from tigramite.independence_tests.regressionCI import RegressionCI
        return RegressionCI(significance="analytic")
    elif test_name == "cmi":
        from tigramite.independence_tests.cmiknn import CMIknn
        return CMIknn(
            significance="shuffle_test",
            knn=0.1,
            shuffle_neighbors=5,
            transform="ranks",
            sig_samples=5,
        )
    else:
        # Default fallback
        return ParCorr()


class PCMCIBackend(Backend):
    """PCMCI+ time-series causal discovery using tigramite."""

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        indep_test_name = self._params.get("indep_test", "parcorr")
        cond_ind_test = _build_cond_ind_test(indep_test_name)

        data_t = pp.DataFrame(data.values, var_names=node_names)
        pcmci = PCMCI_model(dataframe=data_t, cond_ind_test=cond_ind_test)

        # Build run_pcmciplus kwargs
        run_params = {
            "tau_min": self._params.get("tau_min", 0),
            "tau_max": self._params.get("tau_max", 2),
            "pc_alpha": self._params.get("pc_alpha", 0.05),
            "contemp_collider_rule": self._params.get("contemp_collider_rule", "majority"),
            "conflict_resolution": self._params.get("conflict_resolution", True),
            "reset_lagged_links": self._params.get("reset_lagged_links", False),
            "fdr_method": self._params.get("fdr_method", "none"),
            "link_assumptions": self._params.get("link_assumptions", None),
            "max_conds_dim": self._params.get("max_conds_dim", None),
            "max_combinations": self._params.get("max_combinations", 1),
            "max_conds_py": self._params.get("max_conds_py", None),
            "max_conds_px": self._params.get("max_conds_px", None),
        }

        results = pcmci.run_pcmciplus(**run_params)

        # Post-process: FDR correction
        fdr_method = self._params.get("fdr_method", "none")
        pc_alpha = self._params.get("pc_alpha", 0.05)

        if fdr_method != "none":
            q_matrix = pcmci.get_corrected_pvalues(
                p_matrix=results["p_matrix"],
                fdr_method=fdr_method,
                exclude_contemporaneous=False,
            )
        else:
            q_matrix = results["p_matrix"]

        # Build lag matrix and collapse to summary
        matrices = (q_matrix <= pc_alpha).astype(int)
        lag_matrix = np.array([matrices[:, :, lag].T for lag in range(matrices.shape[2])])
        summary_matrix = np.any(lag_matrix, axis=0).astype(int)

        info = {
            "val_matrix": results["val_matrix"],
            "p_matrix": results["p_matrix"],
            "conf_matrix": results["conf_matrix"],
            "alpha": pc_alpha,
            "q_matrix": q_matrix,
            "lag_matrix": lag_matrix,
        }

        return summary_matrix, info, results
