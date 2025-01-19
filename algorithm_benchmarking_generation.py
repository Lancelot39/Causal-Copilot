import numpy as np
import os
from data.simulation.dummy import DataSimulator

def generate_datasets():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Default settings
    default_settings = {
        "n_nodes": 25,
        "n_samples": 5000,
        "edge_probability": 0.2,
        "function_type": "linear",
        "noise_type": "gaussian",
        "noise_scale": 1.0
    }

    dataset_configs = []
    
    # Default configuration
    dataset_configs.append({
        **default_settings,
        "name": "default",
    })
    
    # Scale comparison - varying node counts
    for n_nodes in [5, 10, 50, 100]:  # Skip 25 as it's in default
        dataset_configs.append({
            **default_settings,
            "name": f"scale_nodes_{n_nodes}",
            "n_nodes": n_nodes
        })
    
    # Scale comparison - varying sample sizes 
    for n_samples in [1000, 2500, 10000]:  # Skip 5000 as it's in default
        dataset_configs.append({
            **default_settings,
            "name": f"scale_samples_{n_samples}",
            "n_samples": n_samples
        })

    # Graph density comparison
    for edge_prob in [0.1, 0.3]:  # Skip 0.2 as it's in default
        dataset_configs.append({
            **default_settings,
            "name": f"density_{int(edge_prob*100)}",
            "edge_probability": edge_prob
        })

    # Function type comparison
    dataset_configs.append({
        **default_settings,
        "name": "nonlinear",
        "function_type": "mlp"
    })

    # Noise type comparison
    dataset_configs.append({
        **default_settings,
        "name": "uniform_noise",
        "noise_type": "uniform"
    })

    # Data quality - Discrete ratios
    for ratio in [0.1, 0.2]:
        dataset_configs.append({
            **default_settings,
            "name": f"discrete_{int(ratio*100)}",
            "discrete_ratio": ratio
        })

    # Data quality - Measurement error
    for error_rate in [0.1, 0.3, 0.5]:
        dataset_configs.append({
            **default_settings,
            "name": f"measurement_{int(error_rate*100)}",
            "add_measurement_error": True,
            "error_std": 0.1,
            "error_rate": error_rate
        })

    # Data quality - Missing values
    for missing_rate in [0.05, 0.10, 0.15]:
        dataset_configs.append({
            **default_settings,
            "name": f"missing_{int(missing_rate*100)}",
            "add_missing_values": True,
            "missing_rate": missing_rate
        })

    # Heterogeneity comparison
    for n_domains in [2, 5, 10]:
        dataset_configs.append({
            **default_settings,
            "name": f"heterogeneous_{n_domains}",
            "n_domains": n_domains
        })

    output_dir = os.path.join(os.getcwd(), "simulated_data/algorithm_benchmarking")
    os.makedirs(output_dir, exist_ok=True)

    # Generate 3 random versions of each configuration
    for config in dataset_configs:
        for i in range(3):
            # Set seed based on configuration to ensure reproducibility
            seed = hash(f"{config['name']}_{i}") % (2**32)
            np.random.seed(seed)

            simulator = DataSimulator()
            simulator.generate_and_save_dataset(
                n_nodes=config["n_nodes"],
                n_samples=config["n_samples"],
                edge_probability=config["edge_probability"],
                function_type=config["function_type"],
                noise_type=config["noise_type"],
                noise_scale=config["noise_scale"],
                discrete_ratio=config.get("discrete_ratio", 0.0),
                add_measurement_error=config.get("add_measurement_error", False),
                add_missing_values=config.get("add_missing_values", False),
                error_std=config.get("error_std", 0.1),
                error_rate=config.get("error_rate", 0.1),
                missing_rate=config.get("missing_rate", 0.0),
                n_domains=config.get("n_domains", 1),
                output_dir=output_dir,
                prefix=f"{config['name']}_seed_{i}"
            )
            

if __name__ == "__main__":
    generate_datasets()
