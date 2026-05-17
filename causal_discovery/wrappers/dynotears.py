import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)

from causal_discovery.wrappers.utils.ts_utils import dict_to_adjacency_matrix, generate_stationary_linear
from causal_discovery.wrappers.base import CausalDiscoveryAlgorithm
from causal_discovery.evaluation.evaluator import GraphEvaluator

class DYNOTEARS(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'p': 10,
            'lambda_w': 0.01,
            'lambda_a': 0.01,
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.05
        }
        self._params.update(params)

    @property
    def name(self):
        return "DYNOTEARS"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['p', 'lambda_w', 'lambda_a', 'max_iter', 'w_threshold']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        try:
            from causalnex.structure.dynotears import from_pandas_dynamic
        except ImportError as exc:
            raise ImportError(
                "DYNOTEARS requires the optional 'causalnex' package, which is not "
                "installed by default because it conflicts with the NumPy version "
                "used by the current application requirements."
            ) from exc

        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values
        max_lag = self._params['p']
        #init graph_dict for result aggregation
        graph_dict = dict()
        for name in node_names:
            graph_dict[name] = []

        # call DYNOTEARS
        sm = from_pandas_dynamic(data, **self.get_primary_params(), **self.get_secondary_params())
        tname_to_name_dict = dict()
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name_dict[tname] = data.columns[idx_name]
            if count_lag == max_lag:
                idx_name = idx_name +1
                count_lag = -1
            count_lag = count_lag +1
            
        for ce in sm.edges:
            c = ce[0]
            e = ce[1]
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            t = tc - te
            if (tname_to_name_dict[c], -t) not in graph_dict[tname_to_name_dict[e]]:
                graph_dict[tname_to_name_dict[e]].append((tname_to_name_dict[c], -t))
                
        summary_matrix, lag_matrix = dict_to_adjacency_matrix(graph_dict, len(node_names), max_lag)
        
        info = {
            'lag_matrix': lag_matrix,
            'lag': max_lag,
            'nodes': node_names
        }

        return summary_matrix, info, sm

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        n_nodes = 3
        lag = 2
        
        df, gt_graph, summary, graph_net = generate_stationary_linear(
            n_nodes,
            n_samples,
            lag,
            degree_intra=1,
            degree_inter=2,
        )
        print("Testing DYNOTEARS algorithm with pandas DataFrame:")
        print("Ground truth graph", gt_graph)
        # Initialize lists to store metrics
        f1_scores = []
        precisions = []
        recalls = []
        shds = []
        
        # Run the algorithm
        for _ in range(2):
            adj_matrix,_,_ = self.fit(df)
            print("Prediction", adj_matrix)
            evaluator = GraphEvaluator()
            metrics = evaluator._compute_single_metrics(gt_graph, adj_matrix)
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
        'p': 2,
        'lambda_w': 0.1,
        'lambda_a': 0.07,
        'max_iter': 1000,
        'w_threshold':0.01
    }
    dynotears_algo = DYNOTEARS(params)
    dynotears_algo.test_algorithm()
