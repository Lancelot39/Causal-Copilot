import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import sys
import random
from typing import Dict, List, Tuple, Union
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# Import the TimeSeriesSimulator from dummy.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dummy import TimeSeriesSimulator, set_random_seed

def simulate_time_series(config, n_samples=1000, seed=42, noise_type='linear-gauss', output_dir=None):
    """
    Simulate time series data based on the provided configuration.
    
    Args:
        config (dict): Configuration parameters for the time series simulation
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        noise_type (str): Type of noise to use in the simulation
        save_path (str): Path to save the generated data and ground truth
        
    Returns:
        tuple: (graph, data) - The ground truth graph and the generated data
    """
    set_random_seed(seed)
    
    # Extract configuration parameters
    num_nodes = config["num_nodes"]
    lag = config["lag"]
    degree_inter = config["degree_inter"]
    degree_intra = config["degree_intra"]
    
    # Create the time series simulator
    ts_simulator = TimeSeriesSimulator()
    
    # Generate the data
    ts_simulator.generate_and_save_time_series(
        num_nodes=num_nodes,
        lag=lag,
        degree_inter=degree_inter,
        degree_intra=degree_intra,
        sample_size=n_samples,
        noise_type=noise_type,
        seed=seed,
        output_dir=output_dir
    )
    print(f"Generated time series data with shape: {ts_simulator.data.shape}")
    print(f"Summary adjacency matrix shape: {ts_simulator.ground_truth['summary_adjacency'].shape}")
    print(f"Lagged adjacency matrix shape: {ts_simulator.ground_truth['lagged_adjacency'].shape}")
    
    return ts_simulator.ground_truth, ts_simulator.data

# Define the simulation configuration
simulation_config = {
  "default_settings": {
    "num_nodes": 20,
    "lag": 5,
    "degree_inter": 4.0,
    "degree_intra": 3.0,
    "n_samples": 2000,
    "noise_type": "linear-gauss"
  },
  "variations": {
    "node_counts": [5, 10, 25, 50, 100],
    "lag_values": [3, 5, 10, 15, 20],
    "degree_inter_values": [2.0, 4.0, 8.0, 12.0, 16.0],
    "intra_inter_combinations": [
      {"num_nodes": 10, "lag": 3, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 20, "lag": 3, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 30, "lag": 3, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 10, "lag": 10, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 20, "lag": 10, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 30, "lag": 10, "degree_inter": 3.0, "degree_intra": 0.0},
      {"num_nodes": 10, "lag": 3, "degree_inter": 3.0, "degree_intra": 3.0},
      {"num_nodes": 20, "lag": 3, "degree_inter": 3.0, "degree_intra": 3.0},
      {"num_nodes": 30, "lag": 3, "degree_inter": 3.0, "degree_intra": 3.0},
      {"num_nodes": 10, "lag": 10, "degree_inter": 3.0, "degree_intra": 3.0},
      {"num_nodes": 20, "lag": 10, "degree_inter": 3.0, "degree_intra": 3.0},
      {"num_nodes": 30, "lag": 10, "degree_inter": 3.0, "degree_intra": 3.0}
    ]
  },
  "output_dir": "simulated_data/ts_evaluation",
  "seeds_per_config": 3
}

if __name__ == "__main__":
    # Set a base random seed for reproducibility
    base_seed = 0
    n_samples = simulation_config["default_settings"]["n_samples"]
    seed_per_config = simulation_config["seeds_per_config"]
    output_dir = simulation_config["output_dir"]
    
    # Save the simulation configuration
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "simulation_config.json"), "w") as f:
        json.dump(simulation_config, f, indent=2)
    
    # Generate configurations
    all_configs = []
    
    # Default configuration
    default_config = [{
        "num_nodes": simulation_config["default_settings"]["num_nodes"],
        "lag": simulation_config["default_settings"]["lag"],
        "degree_inter": simulation_config["default_settings"]["degree_inter"],
        "degree_intra": simulation_config["default_settings"]["degree_intra"]
    }]
    all_configs.append(("default", default_config, os.path.join(output_dir, "default")))
    
    # Node scaling configuration
    scale_config_nodes = []
    for nodes in simulation_config["variations"]["node_counts"]:
        scale_config_nodes.append({
            "num_nodes": nodes,
            "lag": 3,
            "degree_inter": 3.0,
            "degree_intra": 2.0
        })
    all_configs.append(("node_scaling", scale_config_nodes, os.path.join(output_dir, "node_scaling")))
    
    # Lag scaling configuration
    scale_config_lags = []
    for lag in simulation_config["variations"]["lag_values"]:
        scale_config_lags.append({
            "num_nodes": 10,
            "lag": lag,
            "degree_inter": 3.0,
            "degree_intra": 2.0
        })
    all_configs.append(("lag_scaling", scale_config_lags, os.path.join(output_dir, "lag_scaling")))
    
    # Intra edges configuration (degree_intra = 0)
    intra_config = [config for config in simulation_config["variations"]["intra_inter_combinations"] 
                   if config["degree_intra"] == 0.0]
    all_configs.append(("intra_edges", intra_config, os.path.join(output_dir, "intra_edges")))
    
    # Inter edges configuration (degree_intra = 3.0)
    inter_config = [config for config in simulation_config["variations"]["intra_inter_combinations"] 
                   if config["degree_intra"] == 3.0]
    all_configs.append(("inter_edges", inter_config, os.path.join(output_dir, "inter_edges")))
    
    # Edge density configuration
    edge_config = []
    for degree in simulation_config["variations"]["degree_inter_values"]:
        edge_config.append({
            "num_nodes": 20,
            "lag": 5,
            "degree_inter": degree,
            "degree_intra": 3.0
        })
    all_configs.append(("edge_density", edge_config, os.path.join(output_dir, "edge_density")))
    
    # Loop through all configurations
    for config_name, configs, save_dir in all_configs:
        print(f"\nRunning {config_name} benchmark...")
        
        # Loop through each configuration in the set
        for i, config in enumerate(configs):
            print(f"\nRunning simulation for {config_name} config {i}:")
            print(f"Configuration: {config}")
            
            # Create a subfolder for this specific configuration
            config_folder = os.path.join(save_dir, f"n_nodes_{config['num_nodes']}_lag_{config['lag']}_degree_inter_{config['degree_inter']}_degree_intra_{config['degree_intra']}")
            
            # Run the simulation for each seed
            for seed_idx in range(seed_per_config):
                # Calculate the actual seed
                current_seed = base_seed + i + seed_idx
                
                # Create a seed-specific subfolder
                seed_folder = f"{config_folder}_seed_{seed_idx}"
                os.makedirs(seed_folder, exist_ok=True)
                
                # Create a unique identifier for this simulation
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                print(f"Running seed {seed_idx} (seed value: {current_seed})")
                
                # Run the simulation
                ground_truth, data = simulate_time_series(
                    config=config,
                    n_samples=n_samples,
                    seed=current_seed,
                    output_dir=seed_folder
                )
                
                # Print some statistics
                print(f"Completed simulation for {config_name} config {i}, seed {seed_idx}")
                print(f"Data shape: {data.shape}")
    
    print("\nAll benchmarks completed successfully!")
