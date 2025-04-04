import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# use the local causal-learn package
import sys

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(causal_learn_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.FCI import fci as cl_fci

from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

class FCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': 4, # -1,
            'max_path_length': -1,
            'verbose': False,
            'background_knowledge': None,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "FCI"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['max_path_length', 'verbose', 'background_knowledge', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Tuple[CausalGraph, List]]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run FCI algorithm
        graph, edges = cl_fci(data_values, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(graph)

        # Prepare additional information
        info = {
            'edges': edges,
            'graph': graph,
        }

        return adj_matrix, info, (graph, edges)

    def convert_to_adjacency_matrix(self, adj_matrix: CausalGraph) -> np.ndarray:
        adj_matrix = adj_matrix.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 2:
                # bidirected edge: j o-> i
                inferred_flat[i, j] = 4
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == 2)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == 2:
                # undirected edge: j o-o i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 6
        return inferred_flat

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

        # Define the dataset path
        dataset_name = "1_basic_scenarios/mlp_gaussian_extra_large_sample/20250401_222842_seed_0_nodes10_samples10000"
        base_dir = os.path.join(root_dir, "simulated_data", "copilot_benchmark_v3")
        
        # Load the data
        data_file = [os.path.join(base_dir, dataset_name, file) for file in os.listdir(os.path.join(base_dir, dataset_name)) if file.endswith(".csv")][0]
        graph_file = [os.path.join(base_dir, dataset_name, file) for file in os.listdir(os.path.join(base_dir, dataset_name)) if file.endswith(".npy")][0]
        
        # Load the data and ground truth graph
        df = pd.read_csv(data_file)
        gt_graph = np.load(graph_file)
        
        print(f"Loaded dataset from {dataset_name}")
        print(f"Data shape: {df.shape}")
        print(f"Ground truth graph shape: {gt_graph.shape}")

        print("Testing FCI algorithm with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'indep_test': 'rcit',
            'verbose': False,
            'show_progress': True
        }
        self._params.update(params)
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

        # # # Ground truth graph
        # gt_graph = np.array([
        #     [0, 0, 0, 0, 0],
        #     [1, 0, 0, 0, 0],
        #     [1, 1, 0, 0, 0],
        #     [0, 1, 0, 0, 0],
        #     [0, 0, 1, 1, 0]
        # ])

        # Use GraphEvaluator to compute metrics
        evaluator = GraphEvaluator()
        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)

        print("\nMetrics:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"SHD: {metrics['shd']:.4f}")

if __name__ == "__main__":
    fci = FCI()
    fci.test_algorithm()