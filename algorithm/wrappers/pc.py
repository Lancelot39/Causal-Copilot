import numpy as np
import pandas as pd
import cupy as cp
from typing import Dict, Tuple

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')
sys.path.append('algorithm')

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc as cl_pc

from .base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator


class KCI_PC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'kci',
            'depth': -1,
            'stable': True,
            'uc_rule': 0,
            'uc_priority': -1,
            'mvpc': False,
            'correction_name': 'MV_Crtn_KCI',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
            'gamma': 0.5  # Kernel bandwidth for KCI
        }
        self._params.update(params)

    @property
    def name(self):
        return "KCI_PC_GPU"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth', 'gamma']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvpc', 'correction_name',
                                      'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def kci_test_gpu(self, X, Y, Z, data, gamma):
        """
        GPU-accelerated KCI test using CuPy for kernel computations.
        """
        n_samples = data.shape[0]

        if len(Z) == 0:
            # No conditioning set
            X_data = cp.asarray(data[:, X]).reshape(-1, 1)
            Y_data = cp.asarray(data[:, Y]).reshape(-1, 1)
        else:
            Z_data = cp.asarray(data[:, Z])
            X_data = cp.asarray(data[:, X])
            Y_data = cp.asarray(data[:, Y])

            # Regress X on Z
            beta_X = cp.linalg.lstsq(Z_data, X_data, rcond=None)[0]
            X_residual = X_data - Z_data @ beta_X
            X_data = X_residual.reshape(-1, 1)

            # Regress Y on Z
            beta_Y = cp.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
            Y_residual = Y_data - Z_data @ beta_Y
            Y_data = Y_residual.reshape(-1, 1)

        # Compute kernel matrices
        K_X = cp.exp(-gamma * cp.linalg.norm(X_data[:, None] - X_data, axis=2) ** 2)
        K_Y = cp.exp(-gamma * cp.linalg.norm(Y_data[:, None] - Y_data, axis=2) ** 2)

        # Centering
        H = cp.eye(n_samples) - cp.ones((n_samples, n_samples)) / n_samples
        Kc_X = H @ K_X @ H
        Kc_Y = H @ K_Y @ H

        # Compute HSIC statistic
        hsic_stat = cp.trace(Kc_X @ Kc_Y) / ((n_samples - 1) ** 2)

        # Placeholder for p-value calculation (use permutation or asymptotic approximation)
        p_value = 1.0  # Assuming independence by default

        return p_value > self._params['alpha']  # True means independent

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Wrap the KCI test for GPU acceleration
        all_params['indep_test'] = lambda X, Y, Z: self.kci_test_gpu(X, Y, Z, data_values, self._params['gamma'])

        # Run PC algorithm
        cg = cl_pc(data_values, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'sepset': cg.sepset,
            'definite_UC': cg.definite_UC,
            'definite_non_UC': cg.definite_non_UC,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info, cg

    def test_algorithm(self):
        # Generate sample data with nonlinear relationships
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.sin(X1) + np.random.normal(0, 0.1, n_samples)
        X3 = np.cos(X1 + X2) + np.random.normal(0, 0.1, n_samples)
        X4 = np.exp(-X2) + np.random.normal(0, 0.1, n_samples)
        X5 = X3 ** 2 + X4 + np.random.normal(0, 0.1, n_samples)

        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5})

        print("Testing GPU-accelerated KCI-PC algorithm on pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'depth': 2,
            'indep_test': 'kci',
            'gamma': 0.5,
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("\nAdditional Info:")
        print(f"PC elapsed time: {info['PC_elapsed']:.4f} seconds")
        print(f"Number of definite unshielded colliders: {len(info['definite_UC'])}")
        print(f"Number of definite non-unshielded colliders: {len(info['definite_non_UC'])}")

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
    kci_pc_algo = KCI_PC({})
    kci_pc_algo.test_algorithm()
