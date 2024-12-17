import numpy as np
import os
from data.simulation.dummy import DataSimulator

def generate_datasets():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    base_simulator = DataSimulator()
    
    dataset_configs = [
        # Easy
        {
            "name": "Linear-Gaussian-Sparse",
            "function_type": "linear", 
            "noise_type": "gaussian",
            "heterogeneous": False,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Linear-Gaussian-Medium", 
            "function_type": "linear",
            "noise_type": "gaussian", 
            "heterogeneous": False,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Linear-Gaussian-Dense", 
            "function_type": "linear",
            "noise_type": "gaussian", 
            "heterogeneous": False,
            "edge_probability": 0.3  # Dense
        },
        {
            "name": "Linear-Non-gaussian-Sparse",
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": False,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Linear-Non-gaussian-Medium",
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": False,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Linear-Non-gaussian-Dense",
            "function_type": "linear", 
            "noise_type": "uniform",
            "heterogeneous": False,
            "edge_probability": 0.3  # Dense
        },
        # Medium
        {
            "name": "Heterogenous-Linear-Non-gaussian-Sparse",
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": True,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Heterogenous-Linear-Non-gaussian-Medium",
            "function_type": "linear",
            "noise_type": "uniform", 
            "heterogeneous": True,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Heterogenous-Linear-Non-gaussian-Dense",
            "function_type": "linear",
            "noise_type": "uniform", 
            "heterogeneous": True,
            "edge_probability": 0.3  # Dense
        },
        {
            "name": "Heterogenous-Linear-Gaussian-Sparse",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Heterogenous-Linear-Gaussian-Medium",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Heterogenous-Linear-Gaussian-Dense",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.3  # Dense
        },
        {
            "name": "Non-Linear-Gaussian-Sparse",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": False,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Non-Linear-Gaussian-Medium",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": False,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Non-Linear-Gaussian-Dense",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": False,
            "edge_probability": 0.3  # Dense
        },
        {
            "name": "Heterogenous-Non-Linear-Gaussian-Sparse",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.1  # Sparse
        },
        {
            "name": "Heterogenous-Non-Linear-Gaussian-Medium",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.2  # Medium
        },
        {
            "name": "Heterogenous-Non-Linear-Gaussian-Dense",
            "function_type": "mlp",
            "noise_type": "gaussian",
            "heterogeneous": True,
            "edge_probability": 0.3  # Dense
        },
    ]

    output_dir = os.path.join(os.getcwd(), "simulated_data")
    os.makedirs(output_dir, exist_ok=True)

    node_sample_combos = [
        (5, 1000),
        (5, 2500),
        (5, 5000), 
        (10, 1000),
        (10, 2500),
        (10, 5000),
        (25, 2500),
        (25, 10000),
        (50, 5000),
        (50, 10000)
    ]
    
    domain_sizes = [2, 5, 10]
    
    for config in dataset_configs:
        for i, (n_nodes, n_samples) in enumerate(node_sample_combos):
            # Set seed based on configuration to ensure reproducibility
            seed = hash(f"{config['name']}_{n_nodes}_{n_samples}_{i}") % (2**32)
            np.random.seed(seed)
            
            edge_probability = config["edge_probability"]

            if config["heterogeneous"]:
                for n_domains in domain_sizes:
                    base_simulator.generate_and_save_dataset(
                        n_nodes=n_nodes,
                        n_samples=n_samples, 
                        edge_probability=edge_probability,
                        function_type=config["function_type"],
                        noise_type=config["noise_type"],
                        n_domains=n_domains,
                        prefix=f"{config['name']}_nodes{n_nodes}_samples{n_samples}_domains{n_domains}_id_{i}",
                        output_dir=output_dir
                    )
            else:
                base_simulator.generate_and_save_dataset(
                    n_nodes=n_nodes,
                    n_samples=n_samples,
                    edge_probability=edge_probability, 
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    prefix=f"{config['name']}_nodes{n_nodes}_samples{n_samples}_id_{i}",
                    output_dir=output_dir
                )

if __name__ == "__main__":
    generate_datasets()

