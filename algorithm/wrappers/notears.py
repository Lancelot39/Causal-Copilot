import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')
sys.path.append('algorithm')

from causalnex.structure.notears import from_numpy, from_pandas, from_numpy_lasso, from_pandas_lasso
from algorithm.evaluation.evaluator import GraphEvaluator

from .base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

class NOTEARS(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.0,
            'tabu_edges': None,
            'tabu_parent_nodes': None,
            'tabu_child_nodes': None,
            'beta': 0.1,
            'sparse': True
        }
        self._params.update(params)

    @property
    def name(self):
        return "NOTEARS"

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['max_iter', 'sparse', 'beta']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol', 'w_threshold', 'tabu_edges', 
                                    'tabu_parent_nodes', 'tabu_child_nodes']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        sparse = self._params.pop('sparse', False)
        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=node_names)

        if sparse:
            sm = from_pandas_lasso(data, **self.get_primary_params(), **self.get_secondary_params())
        else:
            secondary_params = self.get_secondary_params()
            secondary_params.pop('beta', None)
            sm = from_pandas(data, **self.get_primary_params(), **secondary_params)
        
        # Add the 'sparse' parameter back
        self._params['sparse'] = sparse 

        # Convert the StructureModel to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(sm, node_names)

        # Prepare additional information
        info = {
            'structure_model': sm,
        }

        return adj_matrix, info, sm
    
    def convert_to_adjacency_matrix(self, sm, node_names: List[str]) -> np.ndarray:
        """
        Convert NOTEARS StructureModel to adjacency matrix.
        
        Args:
            sm: NOTEARS StructureModel
            node_names: List of node names
            
        Returns:
            numpy array with adjacency matrix
        """
        n = len(node_names)
        adj_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if sm.has_edge(node_names[i], node_names[j]):  # i->j
                    adj_matrix[j, i] = 1
        
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing NOTEARS algorithm with pandas DataFrame:")
        params = {
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.0,
            'sparse': True
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

        # Ground truth graph
        gt_graph = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0]
        ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

if __name__ == "__main__":
    notears_algo = NOTEARS({})
    notears_algo.test_algorithm() 