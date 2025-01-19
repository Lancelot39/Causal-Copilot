import numpy as np
import pandas as pd
from typing import Dict, Tuple
import time

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')
sys.path.append('algorithm')

from xges import XGES as Xges

from .base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

class XGES(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 2.0
        }
        self._params.update(params)

    @property
    def name(self):
        return "XGES"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        node_names = list(data.columns)
        data_values = data.values

        # Run XGES algorithm
        model = Xges(**self.get_primary_params())
        result = model.fit(data_values)

        # Convert the graph to adjacency matrix
        adj_pdag = result.to_adjacency_matrix()
        adj_matrix = self.convert_to_adjacency_matrix(adj_pdag)

        # Prepare additional information
        info = {
            # 'score': result['score']
        }

        return adj_matrix, info, result

    def convert_to_adjacency_matrix(self, G: np.ndarray) -> np.ndarray:
        n = G.shape[0]
        custom_matrix = np.zeros_like(G, dtype=int)

        indices = np.where(G == 1)

        for i, j in zip(indices[0], indices[1]):
            if G[j, i] == 1:
                if i > j:
                    custom_matrix[i, j] = 2
            else:
                custom_matrix[j, i] = 1

        return custom_matrix

    def test_algorithm(self):
        # Generate sample data with linear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)

        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing XGES algorithm with pandas DataFrame:")
        params = {
            'alpha': 2.0
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