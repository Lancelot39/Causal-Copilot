import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple

from causalnex.structure.notears import from_numpy, from_pandas, from_numpy_lasso, from_pandas_lasso

from .base import CausalDiscoveryAlgorithm

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

    def get_params(self):
        return self._params
    
    def get_primary_params(self):
        self._primary_param_keys = ['max_iter', 'sparse', 'beta']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}
    
    def get_secondary_params(self):
        self._secondary_param_keys = ['h_tol', 'w_threshold','tabu_edges', 'tabu_parent_nodes', 'tabu_child_nodes']
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
            secondary_params.pop('beta')
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
        n = len(node_names)
        adj_matrix = np.zeros((n, n))
        for i, node_i in enumerate(node_names):
            for j, node_j in enumerate(node_names):
                if sm.has_edge(node_j, node_i):
                    # only keep the determined edges (j -> i)
                    adj_matrix[i, j] = 1 # sm.get_edge_data(node_j, node_i)['weight']
        return adj_matrix

    def test_algorithm(self):
        # Generate some sample data
        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X = np.random.randn(n_samples, n_features)
        
        # Test with numpy array
        print("Testing NOTEARS algorithm with numpy array:")
        params = {  
            'max_iter': 100,
            'h_tol': 1e-8,
            'w_threshold': 0.0,
            'sparse': True
        }
        adj_matrix, info = self.fit(X)
        print("Adjacency Matrix:")
        print(adj_matrix)
        
        # Test with pandas DataFrame
        print("\nTesting NOTEARS algorithm with pandas DataFrame:")
        df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
        adj_matrix, info = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

if __name__ == "__main__":
    notears_algo = NOTEARS({})
    notears_algo.test_algorithm()

