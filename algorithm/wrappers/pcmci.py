import numpy as np
import pandas as pd
from typing import Dict, Tuple

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from tigramite.pcmci import PCMCI as PCMCI_model
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
import torch
cuda_available = torch.cuda.is_available()
from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator
from algorithm.wrappers.utils.ts_utils import generate_stationary_linear, column_type

class PCMCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'indep_test': 'parcorr',
            'tau_min': 0,
            'tau_max': 10,
            'pc_alpha': 0.05,
            'contemp_collider_rule': 'majority',
            'conflict_resolution': True,
            'reset_lagged_links': False,
            'fdr_method': 'none',
            'link_assumptions': None,
            'max_conds_dim': None,
            'max_combinations': 1,
            'max_conds_py': None,
            'max_conds_px': None,
        }
        self._params.update(params)

    @property
    def name(self):
        return "PCMCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['indep_test', 'tau_min', 'tau_max', 'pc_alpha']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['contemp_collider_rule', 'conflict_resolution', 'reset_lagged_links', 'fdr_method', 'link_assumptions', 'max_conds_dim', 'max_combinations', 'max_conds_py', 'max_conds_px']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, PCMCI_model]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        # PCMCI
        node_names = list(data.columns)
        data_types = None 
        if self._params['indep_test'] == 'parcorr':
            cond_ind_test = ParCorr()
        elif self._params['indep_test'] == 'robustparcorr':
            cond_ind_test = RobustParCorr()
        elif self._params['indep_test'] == 'gpdc':
            if cuda_available:
                from tigramite.independence_tests.gpdc_torch import GPDCtorch as GPDC
                cond_ind_test = GPDC()
            if not cuda_available:
                from tigramite.independence_tests.gpdc import GPDC
                print("CUDA is not available, will not use GPU acceleration")
                cond_ind_test = GPDC(significance='analytic', gp_params=None)
        elif self._params['indep_test'] == 'gsq':
            from tigramite.independence_tests.gsquared import Gsquared
            cond_ind_test = Gsquared(significance='analytic')
        elif self._params['indep_test'] == 'regression':
            from tigramite.independence_tests.regressionCI import RegressionCI
            cond_ind_test = RegressionCI(significance='analytic')
            data_types = column_type(data)
        elif self._params['indep_test'] == 'cmi':
            from tigramite.independence_tests.cmiknn import CMIknn
            cond_ind_test = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', sig_samples=5)

        data_t = pp.DataFrame(data.values, var_names=node_names, data_type=data_types)
        pcmci = PCMCI_model(dataframe=data_t, cond_ind_test=cond_ind_test)
        params = {**self.get_primary_params(), **self.get_secondary_params()}
        # pop indep_test from params since it's already used to initialize cond_ind_test
        params.pop('indep_test')
        results = pcmci.run_pcmciplus(**params)
        if self._params['fdr_method'] !='none':
            q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], 
                                                fdr_method=self._params['fdr_method'],
                                                exclude_contemporaneous=False)
        else:
            q_matrix = results['p_matrix']

        matrices = (q_matrix <= self._params['pc_alpha']).astype(int)
        lag_matrix = np.array([matrices[:, :, lag].T for lag in range(matrices.shape[2])])
        
        # Prepare additional information
        info = {
            'val_matrix': results['val_matrix'],
            'p_matrix': results['p_matrix'],
            'conf_matrix': results['conf_matrix'],
            'alpha': self._params['pc_alpha'],
            'q_matrix': q_matrix,
            'lag_matrix': lag_matrix
        }
        summary_matrix = np.any(lag_matrix, axis=0).astype(int)
 
        return summary_matrix, info, results

    def test_algorithm(self):
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph, gt_summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing PCMCI algorithm with pandas DataFrame:")
        print("Ground truth graph\n", gt_graph)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(1):
            adj_matrix, _, _ = self.fit(df)
            print("Prediction\n", adj_matrix)
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_summary, adj_matrix)
            f1_scores.append(metrics['f1'])
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            shds.append(metrics['shd'])
            
        # Calculate average and standard deviation
        avg_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        avg_precision = np.mean(precisions)
        std_precision = np.std(precisions)
        avg_recall = np.mean(recalls)
        std_recall = np.std(recalls)
        avg_shd = np.mean(shds)
        std_shd = np.std(shds)

        print("\nAverage Metrics:")
        print(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"SHD: {avg_shd:.4f} ± {std_shd:.4f}")

if __name__ == "__main__":
    params = {
        'cond_ind_test': 'robustparcorr',
        'tau_min': 0,
        'tau_max': 2,
        'pc_alpha': 1,
        'alpha_level': 0.07,
        'fdr_method': 'fdr_bh'
    }
    pcmci_algo = PCMCI(params)
    pcmci_algo.test_algorithm() 

