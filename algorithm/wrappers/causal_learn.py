import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score

# use the local causal-learn package
import sys

sys.path.insert(0, 'causal-learn')

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from causallearn.search.ConstraintBased.FCI import fci as cl_fci
from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod
from causallearn.search.ScoreBased.GES import ges as cl_ges
from causallearn.search.FCMBased.lingam.direct_lingam import DirectLiNGAM as CLDirectLiNGAM
from causallearn.search.FCMBased.lingam.ica_lingam import ICALiNGAM as CLICALiNGAM

from .base import CausalDiscoveryAlgorithm


class PC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': -1,
            'stable': True,
            'uc_rule': 0,
            'uc_priority': -1,
            'mvpc': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "PC"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvpc', 'correction_name',
                                      'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

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
    
    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        # save all the determined edges (j -> i) and convert (j -- i) to (j -> i) and (i -> j)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
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

        print("Testing PC algorithm with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'depth': 2,
            'indep_test': 'fisherz',
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

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix.flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix.flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix.flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


class FCI(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'depth': -1,
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
        # save all the determined edges (j -> i) and convert (j -- i) to (j -> i) and (i -> j)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1 or adj_matrix[j, i] == 2:
                inferred_flat[i, j] = 1
        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
        indices = np.where(adj_matrix == 2)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == 2:
                inferred_flat[i, j] = 1
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

        print("Testing FCI algorithm with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix.flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix.flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix.flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


class CDNOD(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'stable': True,
            'uc_rule': 0,
            'uc_priority': 2,
            'depth': -1,
            'mvcdnod': False,
            'correction_name': 'MV_Crtn_Fisher_Z',
            'background_knowledge': None,
            'verbose': False,
            'show_progress': False,
        }
        self._params.update(params)

    @property
    def name(self):
        return "CDNOD"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['alpha', 'indep_test', 'depth']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        secondary_param_keys = ['stable', 'uc_rule', 'uc_priority', 'mvcdnod', 'correction_name',
                                'background_knowledge', 'verbose', 'show_progress']
        return {k: v for k, v in self._params.items() if k in secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        node_names = list(data.columns)
        # Extract c_indx (assuming it's the last column)
        c_indx = data['domain_index'].values.reshape(-1, 1)
        data = data.drop(columns=['domain_index'])
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run CD-NOD algorithm
        cg = cl_cdnod(data_values, c_indx, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(cg)

        # Prepare additional information
        info = {
            'graph': cg,
            'PC_elapsed': cg.PC_elapsed,
        }

        return adj_matrix, info, cg

    def convert_to_adjacency_matrix(self, cg: CausalGraph) -> np.ndarray:
        adj_matrix = cg.G.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        # save all the determined edges (j -> i) and convert (j -- i) to (j -> i) and (i -> j)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
        return inferred_flat

    def test_algorithm(self):
        # Generate sample data with linear relationships and domain index
        np.random.seed(42)
        n_samples = 1000
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.5 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.3 * X1 + 0.7 * X2 + np.random.normal(0, 0.3, n_samples)
        X4 = 0.6 * X2 + np.random.normal(0, 0.4, n_samples)
        X5 = 0.4 * X3 + 0.5 * X4 + np.random.normal(0, 0.2, n_samples)
        domain_index = np.ones_like(X1)
        
        df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'domain_index': domain_index})

        print("Testing CD-NOD algorithm with pandas DataFrame:")
        params = {
            'alpha': 0.05,
            'indep_test': 'fisherz',
            'verbose': False,
            'show_progress': False
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"CD-NOD elapsed time: {info['PC_elapsed']:.4f} seconds")

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix[:-1, :-1].flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix[:-1, :-1].flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix[:-1, :-1].flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


class GES(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'score_func': 'local_score_BIC',
            'maxP': None,
            'parameters': None,
        }
        self._params.update(params)

    @property
    def name(self):
        return "GES"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['score_func', 'maxP']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['parameters']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}

        # Run GES algorithm
        record = cl_ges(data_values, **all_params)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(record['G'])

        # Prepare additional information
        info = {
            'score': record['score'],
            'update1': record['update1'],
            'update2': record['update2'],
        }

        return adj_matrix, info, record

    def convert_to_adjacency_matrix(self, G: CausalGraph) -> np.ndarray:
        adj_matrix = G.graph
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        # save all the determined edges (j -> i) and convert (j -- i) to (j -> i) and (i -> j)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                inferred_flat[i, j] = 1
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

        print("Testing GES algorithm with pandas DataFrame:")
        params = {
            'score_func': 'local_score_BIC',
            'maxP': None,
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print(f"GES score: {info['score']}")

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix.flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix.flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix.flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


class DirectLiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'random_state': None,
            'prior_knowledge': None,
            'apply_prior_knowledge_softly': False,
            'measure': 'pwling'
        }
        self._params.update(params)

    @property
    def name(self):
        return "DirectLiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['measure']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['random_state', 'prior_knowledge', 'apply_prior_knowledge_softly']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CLDirectLiNGAM]:
        node_names = list(data.columns)
        data_values = data.values

        # Combine primary and secondary parameters
        all_params = {**self.get_primary_params(), **self.get_secondary_params()}

        # Run DirectLiNGAM algorithm
        model = CLDirectLiNGAM(**all_params)
        model.fit(data_values)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(model.adjacency_matrix_)

        # Prepare additional information
        info = {
            'causal_order': model.causal_order_
        }

        return adj_matrix, info, model
    
    def convert_to_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        adj_matrix = np.where(adjacency_matrix != 0, 1, 0)
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

        print("Testing DirectLiNGAM algorithm with pandas DataFrame:")
        params = {
            'measure': 'pwling',
            'random_state': 42
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix.flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix.flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix.flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")


class ICALiNGAM(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'random_state': None,
            'max_iter': 1000
        }
        self._params.update(params)

    @property
    def name(self):
        return "ICALiNGAM"

    def get_params(self):
        return self._params

    def get_primary_params(self):
        self._primary_param_keys = ['max_iter']
        return {k: v for k, v in self._params.items() if k in self._primary_param_keys}

    def get_secondary_params(self):
        self._secondary_param_keys = ['random_state']
        return {}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CLICALiNGAM]:
        node_names = list(data.columns)
        data_values = data.values

        # Run ICALiNGAM algorithm
        model = CLICALiNGAM(**self.get_primary_params(), **self.get_secondary_params())
        model.fit(data_values)

        # Convert the graph to adjacency matrix
        adj_matrix = self.convert_to_adjacency_matrix(model.adjacency_matrix_)

        # Prepare additional information
        info = {
            'causal_order': model.causal_order_
        }
        return adj_matrix, info, model

    def convert_to_adjacency_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        adj_matrix = np.where(adjacency_matrix != 0, 1, 0)
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

        print("Testing ICALiNGAM algorithm with pandas DataFrame:")
        params = {
            'random_state': 42,
            'max_iter': 1000
        }
        adj_matrix, info, _ = self.fit(df)
        print("Adjacency Matrix:")
        print(adj_matrix)
        print("Causal Order:")
        print(info['causal_order'])

        # Calculate metrics
        gt_graph = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ]).T
        
        f1 = f1_score(gt_graph.flatten(), adj_matrix.flatten())
        precision = precision_score(gt_graph.flatten(), adj_matrix.flatten())
        recall = recall_score(gt_graph.flatten(), adj_matrix.flatten())

        print("\nMetrics:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
if __name__ == "__main__":
    pc_algo = PC({})
    pc_algo.test_algorithm()

    fci_algo = FCI({})
    fci_algo.test_algorithm()

    cdnod_algo = CDNOD({})
    cdnod_algo.test_algorithm()

    ges_algo = GES({})
    ges_algo.test_algorithm()

    direct_lingam_algo = DirectLiNGAM({})
    direct_lingam_algo.test_algorithm()

    ica_lingam_algo = ICALiNGAM({})
    ica_lingam_algo.test_algorithm()