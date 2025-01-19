import numpy as np
import os
from data.simulation.dummy import DataSimulator, simulate_linear_sem, simulate_nonlinear_sem, simulate_parameter
from typing import List, Tuple, Optional
import igraph as ig
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


class LLMDataSimulator(DataSimulator):
    def __init__(self):
        super().__init__()
        
    def generate_base_chain(self, n_nodes: int) -> np.ndarray:
        """Generate basic chain A->B->C->..."""
        matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(1, n_nodes):
            matrix[i-1][i] = 1
        return matrix
        
    def add_skip_connections(self, matrix: np.ndarray, skip_length: int = 2) -> np.ndarray:
        """Add edges that skip nodes in the chain (e.g. A->C, B->D for skip_length=2)"""
        n = len(matrix)
        result = matrix.copy()
        for i in range(n - skip_length):
            result[i][i + skip_length] = 1
        return result
        
    def add_shortcuts(self, matrix: np.ndarray, connections: List[Tuple[int, int]]) -> np.ndarray:
        """Add arbitrary shortcut edges between nodes"""
        result = matrix.copy()
        for from_idx, to_idx in connections:
            result[from_idx][to_idx] = 1
        return result
        
    def set_graph(self, binary_dag: np.ndarray, variable_names: List[str] = None) -> None:
        """Set a predefined DAG structure instead of generating randomly.
        
        Args:
            binary_dag (np.ndarray): [d, d] binary adjacency matrix representing DAG
            variable_names (List[str], optional): Names for the variables. If None, will use X1, X2, etc.
        """
        if not is_dag(binary_dag):
            raise ValueError('Input matrix must be a DAG')
            
        self.graph = binary_dag
        n_nodes = binary_dag.shape[0]
        
        if variable_names and len(variable_names) == n_nodes:
            self.variable_names = variable_names
            self.graph_dict = {i: name for i, name in enumerate(variable_names)}
        else:
            self.variable_names = [f'X{i+1}' for i in range(n_nodes)]
            self.graph_dict = {i: f'X{i+1}' for i in range(n_nodes)}
            
        self.ground_truth['graph'] = self.graph_dict

    def generate_data_from_dag(self, n_samples: int, noise_scale: float = 1.0, 
                             noise_type: str = 'gaussian', function_type: str = 'linear',
                             discrete_ratio: float = 0.0, max_categories: int = 5) -> pd.DataFrame:
        """Generate data from the predefined DAG structure.
        
        Args:
            n_samples (int): Number of samples to generate
            noise_scale (float): Scale of the noise
            noise_type (str): Type of noise - gaussian, exponential, gumbel, uniform, logistic, poisson
            function_type (str): Type of functional relationship - linear, mlp, mim, gp, gp-add
            discrete_ratio (float): Ratio of discrete variables
            max_categories (int): Maximum number of categories for discrete variables
            
        Returns:
            pd.DataFrame: Generated data
        """
        if self.graph is None:
            raise ValueError('No DAG structure set. Call set_graph() first.')
            
        if function_type == 'linear':
            W = simulate_parameter(self.graph)
            self.data = simulate_linear_sem(W, n_samples, noise_type, noise_scale, discrete_ratio, max_categories)
        else:
            self.data = simulate_nonlinear_sem(self.graph, n_samples, function_type, noise_scale, discrete_ratio, max_categories)
            
        self.data = pd.DataFrame(self.data, columns=self.variable_names)
        # shuffle the self.data
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.ground_truth['discrete_ratio'] = discrete_ratio
        self.ground_truth['max_categories'] = max_categories
        self.ground_truth['noise_type'] = noise_type
        self.ground_truth['function_type'] = function_type
        self.ground_truth['n_domains'] = 1

        return self.data

def generate_datasets():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Default settings
    default_settings = {
        "n_samples": 7680,
        "edge_probability": 0.1,
        "function_type": "linear", 
        "noise_type": "gaussian",
        "noise_scale": 1.0
    }

    dataset_configs = []
    
    # Sample sizes
    sample_sizes = [7680, 10240, 40960]
    
    # Graph sizes for chains
    # chain_sizes = [5, 10, 15, 25]
    chain_sizes = [5]
    
    # Skip connection lengths
    skip_lengths = [2, 3]
    
    # Functional forms
    # functional_forms = ["linear", "mlp"]
    functional_forms = ["linear", "mlp"]
    
    # Noise forms  
    noise_forms = ["gaussian"]
    
    simulator = LLMDataSimulator()
    
    # Generate configurations
    for n_samples in sample_sizes:
        for n_nodes in chain_sizes:
            # Base chain
            base_chain = simulator.generate_base_chain(n_nodes)
            
            # Chain variants
            chain_variants = {
                "base": base_chain,
                f"skip_{skip_lengths[0]}": simulator.add_skip_connections(base_chain, skip_lengths[0]),
                f"skip_{skip_lengths[1]}": simulator.add_skip_connections(base_chain, skip_lengths[1]),
                "shortcuts": simulator.add_shortcuts(base_chain, [(0, n_nodes//2), (n_nodes//4, n_nodes-1)])
            }
            
            for variant_name, chain in chain_variants.items():
                for function_type in functional_forms:
                    for noise_type in noise_forms:
                        config = default_settings.copy()
                        config.update({
                            "name": f"chain_{variant_name}_samples_{n_samples}_nodes_{n_nodes}_function_{function_type}_noise_{noise_type}",
                            "n_samples": n_samples,
                            "n_nodes": n_nodes,
                            "chain_type": variant_name,
                            "function_type": function_type,
                            "noise_type": noise_type,
                            "graph": chain
                        })
                        dataset_configs.append(config)

    output_dir = os.path.join(os.getcwd(), "simulated_data/llm_causal_discovery/1223/chains")
    os.makedirs(output_dir, exist_ok=True)

    n_random_versions = 1
    # Generate 1 random version of each configuration
    for config in dataset_configs:
        for i in range(n_random_versions):
            # Set seed based on configuration to ensure reproducibility
            seed = hash(f"{config['name']}_{i}") % (2**32)
            np.random.seed(seed)

            simulator = LLMDataSimulator()
            
            # Pre-define the chain structure and generate data
            simulator.set_graph(config["graph"])
            data = simulator.generate_data_from_dag(
                n_samples=config["n_samples"],
                function_type=config["function_type"],
                noise_type=config["noise_type"],
                noise_scale=config["noise_scale"],
                discrete_ratio=config.get("discrete_ratio", 0.0)
            )
            
            # Save data and ground truth
            output_path = os.path.join(output_dir, config["chain_type"], f"{config['name']}_seed_{i}")            
            simulator.save_simulation(output_path)

            # Plot the DAG and save it to the same folder
            # i -> j in nx
            G = nx.from_numpy_array(config["graph"], create_using=nx.DiGraph)
            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                   node_size=500, arrowsize=20, font_size=10)
            plt.savefig(os.path.join(output_path, f"dag.png"))
            plt.close()

            # Generate the random DAGs with specified configurations

            

if __name__ == "__main__":
    generate_datasets()
