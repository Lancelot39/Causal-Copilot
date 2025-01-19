import numpy as np
import pandas as pd
from typing import Dict, Tuple

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')
sys.path.append('algorithm')

from acceleration.fges.fges import FGES as Fges
from acceleration.fges.SEMScore import SEMBicScore

from .base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator


class FGES(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'sparsity': 10,
        }
        self._params.update(params)

    @property
    def name(self):
        return "FGES"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['sparsity']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = []
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run FGES algorithm
        score = SEMBicScore(**self.get_primary_params(), dataset=data_values)
        variables = list(range(len(node_names)))
        model = Fges(variables, score, filename='input', checkpoint_frequency=0, save_name='result')

        result = model.search()

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(result)

        # Prepare additional information
        info = {
            'knowledge': result['knowledge'],
            'sparsity': result['sparsity']
        }

        return adj_matrix, info, result

    def convert_to_adjacency_matrix(self, G: dict) -> np.ndarray:
        graph = G['graph']  

        n = len(graph.nodes)
        node_list = list(graph.nodes)
        index_map = {node: i for i, node in enumerate(node_list)}

        adj_matrix = np.zeros((n, n), dtype=int)

        for u, v in graph.edges():
            i, j = index_map[u], index_map[v]
            if (v, u) in graph.edges():
                if i > j:  # Only set for i > j to avoid symmetry
                    adj_matrix[i, j] = 2
            else:
                adj_matrix[j, i] = 1

        return adj_matrix

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

        print("Testing FGES algorithm with pandas DataFrame:")
        params = {
            'sparsity': 10,
            'filename': 'fges', 
            'checkpoint_frequency': 0, 
            'save_name': 'result'
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