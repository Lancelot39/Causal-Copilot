import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime
from tqdm import tqdm
import argparse
import pandas as pd
import copy
import os
import json

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from algorithm.hyperparameter_selector import HyperparameterSelector
from postprocess.judge import Judge
# from postprocess.visualization import Visualization, convert_to_edges
from preprocess.eda_generation import EDA
from global_setting.Initialize_state import global_state_initialization, load_data
from dataclasses import dataclass, field, asdict


from algorithm.evaluation.evaluator import GraphEvaluator
from benchmarking_integrated_config import get_config

from dotenv import load_dotenv

load_dotenv(root_dir + '/.env')

def augment_query_with_data_properties(query: str, config: Dict) -> str:
    """Augment the user query with data properties in a natural tone"""
    augmentations = []

    if "lag" in config:
        augmentations.append(f"This dataset is a time series dataset.")
    
    # Graph density
    if "edge_probability" in config:
        edge_prob = config["edge_probability"]
        if edge_prob < 0.15:
            augmentations.append("The dataset likely has a sparse causal structure with few connections between variables.")
        elif edge_prob > 0.3:
            augmentations.append("The dataset might have a dense causal structure with many connections between variables.")

    if "n_domains" in config and config["n_domains"] > 1:
        augmentations.append("This is a heterogeneous dataset, so you need to consider the domain information (leverage the column of dataset called 'domain_index') in the dataset when finding the causal relation.")

    
    # # Function type
    # if "function_type" in config:
    #     if config["function_type"] == "linear":
    #         augmentations.append("The relationships between variables might be mostly linear.")
    #     elif config["function_type"] == "mlp":
    #         augmentations.append("There could be non-linear relationships between variables in this dataset.")
    
    # # Noise type
    # if "noise_type" in config:
    #     if config["noise_type"] == "gaussian":
    #         augmentations.append("The noise in the data appears to follow a Gaussian distribution.")
    #     elif config["noise_type"] == "uniform":
    #         augmentations.append("The noise in the data seems to be uniformly distributed.")
    
    # Discrete variables
    if "discrete_ratio" in config and config["discrete_ratio"] > 0:
        augmentations.append(f"There are discrete variables in the dataset.")
    
    # Measurement error
    if "add_measurement_error" in config and config["add_measurement_error"]:
        augmentations.append("The data might contain some observational errors or noise.")
    
    # Missing values
    if "add_missing_values" in config and config["add_missing_values"]:
        missing_rate = config.get("missing_rate", 0.0)
        if missing_rate > 0:
            augmentations.append(f"There are some missing data in the dataset, the value 0 is used to represent the missing data.")
    
    # # Number of domains
    # if "n_domains" in config and config["n_domains"] > 1:
    #     augmentations.append(f"The data comes from {config['n_domains']} different domains or environments.")
    
    # # Sample size considerations
    # if "n_samples" in config:
    #     n_samples = config["n_samples"]
    #     if n_samples < 1000:
    #         augmentations.append("The sample size is relatively small, which might affect the reliability of causal discovery.")
    #     elif n_samples > 5000:
    #         augmentations.append("The dataset has a large sample size, which should help with reliable causal discovery.")
    
    # # Network size considerations
    # if "n_nodes" in config:
    #     n_nodes = config["n_nodes"]
    #     if n_nodes > 50:
    #         augmentations.append("This is a high-dimensional dataset with many variables.")
    
    # Combine augmentations
    if augmentations:
        augmented_query = query + "Based on my analysis of the data: " + " ".join(augmentations)   
        return augmented_query
    
    return query

class BenchmarkRunner:
    def __init__(self, debug_mode: bool = False):
        self.config = get_config()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = self.config['data_dir']
        self.output_dir = os.path.join(self.config['output_dir'], self.timestamp)
        self.evaluator = GraphEvaluator()
        self.debug_mode = debug_mode
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Benchmarking data directory: {self.data_dir}")
        print(f"Benchmarking output directory: {self.output_dir}")
        # Load all algorithm classes from wrappers
        # self.algorithm_candidates = self.config['algorithms']
        # self.algorithm_candidates = [self.algorithm]
        # self.algorithms = self._load_algorithms()
        
        # Index all available datasets
        self.dataset_index = self._index_datasets()

    def _load_algorithms(self) -> Dict:
        """Dynamically load all algorithm classes from wrappers directory"""
        algorithms = {}
        wrappers_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'wrappers')
        
        # Import all algorithm classes from __init__.py
        import algorithm.wrappers as wrappers
        
        # Get all available algorithm classes
        available_algorithms = {}
        for name in wrappers.__all__:
            available_algorithms[name] = getattr(wrappers, name)
            
        # Only keep algorithms specified in config
        for algo_name in self.algorithm_candidates:
            if algo_name in available_algorithms:
                algorithms[algo_name] = available_algorithms[algo_name]
                # print(f"Added algorithm: {algo_name}")
        
        print(f"Loaded {len(algorithms)} algorithms, {' | '.join(list(algorithms.keys()))}")
                        
        return algorithms

    def _index_datasets(self) -> List[Tuple[str, str, Dict]]:
        """Index all available datasets, including absolute paths and configs"""
        dataset_index = []
        
        # Check if the data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist")
            return dataset_index
            
        # Recursively walk through all directories
        for root, dirs, files in os.walk(self.data_dir):
            # Check if this directory contains dataset files (config.json)
            config_files = [f for f in files if f.endswith('config.json')]
            if not config_files:
                continue
                
            # Process each config file in the directory
            for config_file in config_files:
                config_path = os.path.join(root, config_file)
                dataset_dir_abs = os.path.dirname(config_path)
                
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Extract seed number using regex
                    import re
                    seed_match = re.search(r'seed_(\d+)', config_path)
                    seed_num = int(seed_match.group(1)) if seed_match else 0
                    config['seed'] = seed_num
                    
                    # Create dataset name with key configuration parameters
                    dataset_name = f"nodes{config.get('n_nodes', 'unknown')}_samples{config.get('n_samples', 'unknown')}_domains{config.get('n_domains', 'unknown')}_noise{config.get('noise_type', 'unknown')}_func{config.get('function_type', 'unknown')}"
                    
                    if 'edge_probability' in config and config['edge_probability'] is not None:
                        dataset_name += f"_edgeprob{config['edge_probability']}"
                    if 'experiment_type' in config:
                        dataset_name = f"{config['experiment_type']}_{dataset_name}"
                    if 'discrete_ratio' in config and config['discrete_ratio'] is not None:
                        dataset_name += f"_discrete{config['discrete_ratio']}"
                    if 'measurement_error' in config and config['measurement_error'] is not None:
                        measurement_error = config['measurement_error_value']
                        dataset_name += f"_measurement{measurement_error}"
                    if 'missing_rate' in config and config['missing_rate'] is not None:
                        missing_rate = config['missing_rate_value']
                        dataset_name += f"_missing{missing_rate}"
                    
                    dataset_name += f"_seed{seed_num}"
                    
                    # Add scenario/category information from directory structure
                    rel_path = os.path.relpath(dataset_dir_abs, self.data_dir)
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1 and path_parts[0] != '.':
                        scenario = path_parts[0]
                        if len(path_parts) > 2:
                            scenario += f"/{path_parts[1]}"
                        dataset_name = f"{scenario}_{dataset_name}"
                    
                    dataset_index.append((dataset_name, dataset_dir_abs, config))
                    
                except Exception as e:
                    print(f"Error processing config file {config_path}: {e}")
        
        print(f"Found {len(dataset_index)} datasets")
        
        # Sort by n_nodes first, then by n_samples
        dataset_index.sort(key=lambda x: (
            x[2].get('n_nodes', 0), 
            x[2].get('n_samples', 0)
        ))
            
        return dataset_index

    def _load_dataset(self, dataset_dir: str) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """Load data, true graph, and config from a dataset directory"""
        data_file = [file for file in os.listdir(dataset_dir) if 'data.csv' in file][0]
        data = pd.read_csv(os.path.join(dataset_dir, data_file))
        true_graph = [file for file in os.listdir(dataset_dir) if 'graph.npy' in file][0]
        true_graph = np.load(os.path.join(dataset_dir, true_graph))
        config_file = [file for file in os.listdir(dataset_dir) if 'config.json' in file][0]
        config = json.load(open(os.path.join(dataset_dir, config_file)))

        return data, true_graph, config

    def _run_single_experiment(self, dataset_dir, data: pd.DataFrame, 
                             true_graph: np.ndarray, config: Dict) -> Dict:
        """Run a single experiment with given algorithm and configuration"""


        # fill the missing value if it is existed
        data = data.fillna(0)

        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')
        parser.add_argument(
            '--initial_query',
            type=str,
            default="Help me to find causal relation in this dataset, try your best with your knowledge. Make sure to get the most accurate result.",
            help='Initial query for the algorithm'
        )
        parser.add_argument(
            '--demo_mode',
            type=bool,
            default=False,
            help='Demo mode'
        )

        parser.add_argument(
            '--data-file',
            type=str,
            default=f"{dataset_dir}",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='/integrated/dataset/sachs/output_report',
            help='Directory to save the output report'
    )

        parser.add_argument(
            '--query_augmentation',
            type=bool,
            default=True,
            help='Whether to augment the query with data properties'
        )

        args = parser.parse_args()

        initial_query = args.initial_query

        # Augment the query with data properties if enabled
        if args.query_augmentation:
            initial_query = augment_query_with_data_properties(initial_query, config)
            print(f"Query augmented with data properties: {initial_query}")

        # Update the args with the augmented query
        args.initial_query = initial_query

        global_state = global_state_initialization(args)
        global_state.user_data.raw_data = data

        # if 'domain_index' in data.columns:
        #     global_state.statistics.heterogeneous = True
        #     global_state.statistics.domain_index = 'domain_index'

        global_state.user_data.processed_data = global_state.user_data.raw_data
        global_state.user_data.visual_selected_features = global_state.user_data.processed_data.columns
        global_state.user_data.selected_features = global_state.user_data.processed_data.columns
    
        global_state = stat_info_collection(global_state)
        global_state = knowledge_info(args, global_state)

        # Convert statistics to text
        global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

        print(global_state.statistics.description)

        #############EDA###################
        # my_eda = EDA(global_state)
        # my_eda.generate_eda()
        
        # Algorithm selection and deliberation
        filter = Filter(args)
        global_state = filter.forward(global_state)

        reranker = Reranker(args)
        global_state = reranker.forward(global_state)

        hp_selector = HyperparameterSelector(args)
        global_state = hp_selector.forward(global_state)

        start_time = time.time()
        programmer = Programming(args)
        global_state = programmer.forward(global_state)
        runtime = time.time() - start_time

        # algo_instance = global_state.algorithm.selected_algorithm
        # adj_matrix, info, _ = algo_instance.fit(data)
        adj_matrix = global_state.results.converted_graph

        # Check if it's a time-series dataset
        is_time_series = "lag" in config
        
        if is_time_series:
            # Compute metrics using the evaluator
            metrics = self.evaluator._compute_single_metrics(true_graph, adj_matrix)
            
        else:
            # For non-time-series data, use standard evaluation
            metrics = self.evaluator.compute_metrics(true_graph, adj_matrix)
        
        global_state_dict = asdict(global_state)

        result = {
            "success": True,
            "runtime": runtime,
            "metrics": metrics,
            "data_config": config,
            "global_state": global_state_dict
        }

        print(f"metrics: {metrics} | runtime: {runtime}")
            
        return result

    def _save_result(self, dataset_name: str, result: Dict) -> None:
        """Save the result of a single experiment to a JSON file"""
        output_file = os.path.join(self.output_dir, f"{dataset_name}_results.json")
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
        else:
            existing_results = []
        
        # Make a deep copy of the result to avoid modifying the original
        result_copy = copy.deepcopy(result)
        
        # Convert any dictionary with tuple keys to use string keys instead
        def convert_tuple_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    if isinstance(k, tuple):
                        new_dict[str(k)] = convert_tuple_keys(v)
                    else:
                        new_dict[k] = convert_tuple_keys(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_tuple_keys(item) for item in obj]
            else:
                return obj
        
        result_copy = convert_tuple_keys(result_copy)
        existing_results.append(result_copy)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=2, default=str)

    def run_all_experiments(self) -> None:
        """Run all benchmark experiments"""
        all_results = {}
        # Run experiments for all datasets
        for dataset_name, dataset_dir, config in tqdm(self.dataset_index, desc=f"Running {len(self.dataset_index)} experiments"):
            data, true_graph, _ = self._load_dataset(dataset_dir)
            result = self._run_single_experiment(dataset_dir, data, true_graph, config)
            result['dataset'] = dataset_name
            self._save_result(dataset_name, result)
                
            if self.debug_mode:
                break  # Only test the first dataset in debug mode
                
        print(f"Benchmark results saved to: {self.output_dir}")

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    runner = BenchmarkRunner(debug_mode=args.debug)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()
