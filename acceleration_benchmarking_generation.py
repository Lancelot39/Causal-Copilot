import numpy as np
import os
from data.simulation.dummy import DataSimulator

def generate_datasets():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    base_simulator = DataSimulator()
    
    dataset_configs = [
        # Linear Gaussian
        {
            "name": "linear/gaussian",
            "function_type": "linear",
            "noise_type": "gaussian",
            "heterogeneous": False
        },
        # Linear Uniform
        {
            "name": "linear/uniform", 
            "function_type": "linear",
            "noise_type": "uniform",
            "heterogeneous": False
        },
    ]

    import datetime
    output_dir = os.path.join(os.getcwd(), f"simulated_data/{datetime.datetime.now().strftime('%m%d')}_acceleration_benchmarking")
    os.makedirs(output_dir, exist_ok=True)

    # Configurations
    node_sizes = [5, 25, 50, 100]
    sample_sizes = [500, 2500, 10000]
    edge_probabilities = [0.1, 0.3, 0.5]

    for config in dataset_configs:
        # Create subdirectory for this function/noise type
        config_dir = os.path.join(output_dir, config["name"])
        os.makedirs(config_dir, exist_ok=True)
        
        # Variable Scaling
        for n_nodes in node_sizes:
            for repeat in range(3):  # Repeat 3 times
                seed = hash(f"{config['name']}_var_{n_nodes}_{repeat}") % (2**32)
                np.random.seed(seed)
                base_simulator.generate_and_save_dataset(
                    n_nodes=n_nodes,
                    n_samples=5000,
                    edge_probability=0.2,
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    prefix=f"var_seed{repeat}",
                    output_dir=config_dir
                )
        
        # Sample Size Scaling
        for n_samples in sample_sizes:
            for repeat in range(3):  # Repeat 3 times
                seed = hash(f"{config['name']}_sample_{n_samples}_{repeat}") % (2**32)
                np.random.seed(seed)
                base_simulator.generate_and_save_dataset(
                    n_nodes=25,
                    n_samples=n_samples,
                    edge_probability=0.2,
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    prefix=f"sample_seed{repeat}",
                    output_dir=config_dir
                )
        
        # Graph Density Impact
        for edge_probability in edge_probabilities:
            for repeat in range(3):  # Repeat 3 times
                seed = hash(f"{config['name']}_density_{edge_probability}_{repeat}") % (2**32)
                np.random.seed(seed)
                base_simulator.generate_and_save_dataset(
                    n_nodes=25,
                    n_samples=5000,
                    edge_probability=edge_probability,
                    function_type=config["function_type"],
                    noise_type=config["noise_type"],
                    prefix=f"density_edgeprob{edge_probability}_seed{repeat}",
                    output_dir=config_dir
                )

if __name__ == "__main__":
    generate_datasets()