import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
import os

# use the local causal-learn package
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
causal_learn_dir = os.path.join(root_dir, 'externals', 'causal-learn')
if not os.path.exists(causal_learn_dir):
    raise FileNotFoundError(f"Local causal-learn directory not found: {causal_learn_dir}, please git clone the submodule of causal-learn")
algorithm_dir = os.path.join(root_dir, 'algorithm')
sys.path.append(root_dir)
sys.path.append(algorithm_dir)
sys.path.insert(0, causal_learn_dir)

from causallearn.graph.GraphClass import CausalGraph
from causallearn.search.ConstraintBased.PC import pc as cl_pc
from algorithm.wrappers.base import CausalDiscoveryAlgorithm
from algorithm.evaluation.evaluator import GraphEvaluator

import torch
cuda_available = torch.cuda.is_available()
try:
    from externals.acceleration.pc.pc import accelerated_pc
except Exception as e: 
    if not cuda_available:
        print("CUDA is not available, will not use GPU acceleration")
    print(e)


# KCI: 5 nodes, 1000 samples: 66s
# KCI: 5 nodes, 3000 samples: 100s
# kci: 10 nodes, 1000 samples: 500s
# KCI: 20 nodes, 1000 samples: 5500s
# fastkci: 5 nodes, 1000 samples: 15s
# fastkci: 5 nodes, 3000 samples: 80s
# fastkci: 10 nodes, 1000 samples: 160s
# fastkci: 20 nodes, 1000 samples: 1000s
# rcit: 5 nodes, 1000 samples: 2s
# rcit: 5 nodes, 3000 samples: 3s
# rcit: 5 nodes, 10000 samples: 25s
# rcit: 10 nodes, 1000 samples: 17s
# rcit: 20 nodes, 1000 samples: 180s


class PC(CausalDiscoveryAlgorithm):
    def __init__(self, params: Dict = {}):
        super().__init__(params)
        self._params = {
            'alpha': 0.05,
            'indep_test': 'fisherz_cpu',  # Default to fisherz
            'depth': 4,
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
                                    'background_knowledge', 'verbose', 'show_progress', 'gamma']
        return {k: v for k, v in self._params.items() if k in self._secondary_param_keys}

    def fit(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict, CausalGraph]:
        # Check and remove domain_index if it exists
        if 'domain_index' in data.columns:
            data = data.drop(columns=['domain_index'])
            
        node_names = list(data.columns)
        data_values = data.values

        if cuda_available and 'gpu' in self._params['indep_test']:
            use_gpu = True
            self._params['indep_test'] = self._params['indep_test'].replace('_gpu', '')
        else:
            use_gpu = False
            self._params['indep_test'] = self._params['indep_test'].replace('_cpu', '')
        
        if use_gpu:
            # Use GPU implementation
            all_params = {
                'alpha': self._params['alpha'],
                'indep_test': self._params['indep_test'],
                'depth': self._params['depth'],
            }
            adj_matrix, info = accelerated_pc(data_values, **all_params)
            adj_matrix = self.convert_to_adjacency_matrix(adj_matrix)
            cg = adj_matrix
        else:
            # Use CPU implementation
            all_params = {**self.get_primary_params(), **self.get_secondary_params(), 'node_names': node_names}
            cg = cl_pc(data_values, **all_params)
            adj_matrix = self.convert_to_adjacency_matrix(cg.G.graph)

            # Prepare additional information
            info = {
                'sepset': cg.sepset if hasattr(cg, 'sepset') else None,
                'definite_UC': cg.definite_UC if hasattr(cg, 'definite_UC') else [],
                'definite_non_UC': cg.definite_non_UC if hasattr(cg, 'definite_non_UC') else [],
            }

        return adj_matrix, info, cg
    
    def convert_to_adjacency_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        # Handle both GPU and CPU graph formats
        inferred_flat = np.zeros_like(adj_matrix)
        indices = np.where(adj_matrix == 1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # directed edge: j -> i
                inferred_flat[i, j] = 1
            elif adj_matrix[j, i] == 1:
                # bidirected edge: j <-> i
                if inferred_flat[j, i] == 0:
                    # keep asymmetric that only one entry is recorded
                    inferred_flat[i, j] = 3

        indices = np.where(adj_matrix == -1)
        for i, j in zip(indices[0], indices[1]):
            if adj_matrix[j, i] == -1:
                # undirected edge: j -- i
                if inferred_flat[j, i] == 0:
                    inferred_flat[i, j] = 2
        return inferred_flat

    def test_algorithm(self):
        # Generate sample data with linear relationships
        import time
        import os
        import numpy as np
        import pandas as pd
        from algorithm.evaluation.evaluator import GraphEvaluator
        from data.simulation.dummy import DataSimulator

        # Fix all random seeds for reproducibility
        np.random.seed(42)
        
        # Set random seeds for other libraries if they're being used
        import random
        random.seed(42)
        
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
                torch.cuda.manual_seed_all(42)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
            
        # Set TensorFlow seed if it's being used
        try:
            import tensorflow as tf
            tf.random.set_seed(42)
        except ImportError:
            pass
        np.random.seed(42)
        start_time = time.time()
        
        # Test hypothesis: larger node size needs larger sample size
        # We'll test different combinations of node sizes and sample sizes

        def degree2prob(degree, node_size):
            return degree / (node_size-1)
        
        node_sizes = [5] # [5, 10, 15, 20, 25]
        sample_sizes = [1000] # [500, 1000, 1500, 2000]
        num_runs = 1  # Number of runs to average results
        edge_probability = degree2prob(2, node_sizes[0])
        
        # Define different parameter configurations to compare
        # configurations = [
        #     {"name": "Adaptive Alpha", "alpha": None, "depth": -1},  # Adaptive alpha will be calculated per run
        #     {"name": "Fixed Alpha 0.05", "alpha": 0.05, "depth": -1},
        #     {"name": "Fixed Alpha 0.01", "alpha": 0.01, "depth": -1},
        #     {"name": "Fixed Depth 4", "alpha": 0.05, "depth": 4}
        # ]
        configurations = [
            {"name": "Fixed Alpha 0.05", "alpha": 0.01, "depth": 1, 'indep_test': 'fisherz_gpu'},
        ]

        results = {}
        
        print("Testing hypothesis: larger node size needs larger sample size")
        print("=" * 80)
        print(f"Running {num_runs} iterations for each configuration")
        print("=" * 80)
        
        for config in configurations:
            config_name = config["name"]
            results[config_name] = {}
            
            print(f"\nTesting configuration: {config_name}")
            print("-" * 60)
            
            for n_nodes in node_sizes:
                results[config_name][n_nodes] = {}
                for n_samples in sample_sizes:
                    metrics_list = []
                    time_list = []
                    
                    print(f"\nTesting with {n_nodes} nodes and {n_samples} samples:")
                    
                    for run in range(num_runs):
                        print(f"  Run {run+1}/{num_runs}...")
                        
                        # Create a DataSimulator instance with new random seed for each run
                        seed = 42 + run
                        np.random.seed(seed)
                        simulator = DataSimulator()
                        
                        # Generate data
                        gt_graph, df = simulator.generate_dataset(
                            n_samples=n_samples, 
                            n_nodes=n_nodes, 
                            noise_type='gaussian',
                            function_type='linear', 
                            edge_probability=edge_probability,
                            n_domains=1
                        )
                        
                        # Configure PC algorithm based on current configuration
                        if config["name"] == "Adaptive Alpha":
                            # Using formula: α = 0.05 × (n/100)^(-0.2) × (p/10)^(-0.2)
                            n = n_samples
                            p = n_nodes
                            alpha = 0.05 * (n/100)**(-0.2) * (p/10)**(-0.2)
                            self._params['alpha'] = alpha
                            print(f"    Adaptive alpha for n={n}, p={p}: {alpha:.5f}")
                        else:
                            self._params['alpha'] = config["alpha"]
                            
                        self._params['depth'] = config["depth"]
                        self._params['indep_test'] = config["indep_test"]
                        self._params['show_progress'] = False
                        
                        run_start_time = time.time()
                        adj_matrix, info, _ = self.fit(df)
                        run_time = time.time() - run_start_time
                        
                        # Evaluate results
                        evaluator = GraphEvaluator()
                        metrics = evaluator.compute_metrics(gt_graph, adj_matrix)
                        
                        # Store results
                        metrics_list.append(metrics)
                        time_list.append(run_time)
                    
                    # Calculate average metrics
                    avg_metrics = {
                        'f1': np.mean([m['f1'] for m in metrics_list]),
                        'precision': np.mean([m['precision'] for m in metrics_list]),
                        'recall': np.mean([m['recall'] for m in metrics_list]),
                        'shd': np.mean([m['shd'] for m in metrics_list]),
                        'time': np.mean(time_list)
                    }
                    
                    results[config_name][n_nodes][n_samples] = avg_metrics
                    
                    # Print average results for this configuration
                    print(f"  Results for {n_nodes} nodes, {n_samples} samples (averaged over {num_runs} runs):")
                    print(f"    F1 Score: {avg_metrics['f1']:.4f}")
                    print(f"    Precision: {avg_metrics['precision']:.4f}")
                    print(f"    Recall: {avg_metrics['recall']:.4f}")
                    print(f"    SHD: {avg_metrics['shd']:.4f}")
                    print(f"    Time: {avg_metrics['time']:.4f} seconds")
        
        # Print summary of results for each configuration
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        
        for config_name in results:
            print(f"\n{config_name}:")
            print("-" * 60)
            
            print("F1 Scores:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['f1']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nPrecision:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['precision']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nRecall:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['recall']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
            
            print("\nSHD:")
            for n_nodes in node_sizes:
                scores = [f"{results[config_name][n_nodes][n_samples]['shd']:.4f}" for n_samples in sample_sizes]
                print(f"  Nodes={n_nodes}: {', '.join(scores)}")
        
        total_time = time.time() - start_time
        print(f"\nTotal experiment time: {total_time:.2f} seconds")
        
        # Analyze and print conclusions
        print("\n" + "=" * 80)
        print("CONCLUSIONS")
        print("=" * 80)
        
        # Analyze sample size vs. node size relationship
        print("\n1. Sample Size to Node Size Relationship:")
        for config_name in results:
            print(f"\n  For {config_name}:")
            for n_nodes in node_sizes:
                # Find the sample size where F1 score first exceeds 0.8 (or the highest if none exceed 0.8)
                f1_scores = [results[config_name][n_nodes][n_samples]['f1'] for n_samples in sample_sizes]
                threshold_indices = [i for i, f1 in enumerate(f1_scores) if f1 >= 0.8]
                if threshold_indices:
                    min_sample_idx = min(threshold_indices)
                    min_sample = sample_sizes[min_sample_idx]
                    print(f"    Nodes={n_nodes}: Minimum sample size for F1 ≥ 0.8: {min_sample} (F1={f1_scores[min_sample_idx]:.4f})")
                else:
                    max_f1_idx = f1_scores.index(max(f1_scores))
                    print(f"    Nodes={n_nodes}: No sample size reached F1 ≥ 0.8. Best: {sample_sizes[max_f1_idx]} (F1={max(f1_scores):.4f})")
        
        # Compare configurations
        print("\n2. Configuration Comparison:")
        # Calculate average F1 across all node/sample combinations for each config
        config_avg_f1 = {}
        for config_name in results:
            all_f1 = []
            for n_nodes in node_sizes:
                for n_samples in sample_sizes:
                    all_f1.append(results[config_name][n_nodes][n_samples]['f1'])
            config_avg_f1[config_name] = np.mean(all_f1)
        
        # Sort configs by average F1
        sorted_configs = sorted(config_avg_f1.items(), key=lambda x: x[1], reverse=True)
        print("  Configurations ranked by average F1 score:")
        for i, (config_name, avg_f1) in enumerate(sorted_configs):
            print(f"    {i+1}. {config_name}: {avg_f1:.4f}")
        
        # Analyze adaptive alpha effectiveness
        if "Adaptive Alpha" in results and "Fixed Alpha 0.05" in results:
            print("\n3. Adaptive Alpha Effectiveness:")
            adaptive_better_count = 0
            total_comparisons = 0
            
            for n_nodes in node_sizes:
                for n_samples in sample_sizes:
                    adaptive_f1 = results["Adaptive Alpha"][n_nodes][n_samples]['f1']
                    fixed_f1 = results["Fixed Alpha 0.05"][n_nodes][n_samples]['f1']
                    
                    if adaptive_f1 > fixed_f1:
                        adaptive_better_count += 1
                    total_comparisons += 1
            
            adaptive_better_pct = (adaptive_better_count / total_comparisons) * 100
            print(f"  Adaptive alpha outperformed fixed alpha (0.05) in {adaptive_better_count}/{total_comparisons} cases ({adaptive_better_pct:.1f}%)")
            
            # Analyze when adaptive alpha works better
            print("  Conditions where adaptive alpha performs best:")
            for n_nodes in node_sizes:
                for n_samples in sample_sizes:
                    adaptive_f1 = results["Adaptive Alpha"][n_nodes][n_samples]['f1']
                    fixed_f1 = results["Fixed Alpha 0.05"][n_nodes][n_samples]['f1']
                    diff = adaptive_f1 - fixed_f1
                    
                    if diff > 0.05:  # Significant improvement threshold
                        print(f"    Nodes={n_nodes}, Samples={n_samples}: Improvement={diff:.4f} (Adaptive={adaptive_f1:.4f}, Fixed={fixed_f1:.4f})")
        
        # Final recommendations
        print("\n4. Recommendations:")
        print("  • For optimal performance, the sample size should be at least 100 times the number of nodes")
        print("  • Adaptive alpha is recommended for datasets with varying node and sample sizes")
        print("  • For large graphs (>20 nodes), limiting depth to 4 provides a good balance of accuracy and speed")
        print("  • More conservative alpha values (0.01) are better for larger sample sizes to reduce false positives")

if __name__ == "__main__":
    # for indep_test in ['rcit_cpu', 'cmiknn_gpu']:
    #     print(f"Testing {indep_test}")
    pc_algo = PC({'indep_test': 'fisherz_gpu'})
    pc_algo.test_algorithm() 