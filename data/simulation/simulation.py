import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from .dummy import DataSimulator
import networkx as nx
from openai import OpenAI

class SimulationManager:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        self.base_simulator = DataSimulator()

    def generate_simulation_config(self) -> Dict[str, Any]:
        """
        Use LLM to generate a simulation configuration.
        """
        prompt = self._load_simulation_prompt()
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in data simulation for scientific research. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _load_simulation_prompt(self) -> str:
        with open('data/simulation/context/simulation.txt', 'r') as f:
            return f.read()

    def simulate_data(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
        """
        Simulate data based on the provided configuration.
        """
        simulation_function = getattr(self, config['simulation_function'])
        graph, data = simulation_function(**config['arguments'])
        
        # Convert graph to numpy array
        graph_array = nx.to_numpy_array(graph).transpose()
        
        return config, data, graph_array

    def generate_dataset(self) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray]:
        """
        Generate a dataset using LLM-generated configuration.
        """
        config = self.generate_simulation_config()
        print("Generated Configuration: ", config)
        return self.simulate_data(config)

    # Simulation functions

    def simulate_base_data(self, n_nodes: int, n_samples: int, edge_probability: float,
                           noise_scale: float, noise_type: str,
                           function_type: Dict[str, str],
                           add_categorical: bool = False, add_measurement_error: bool = True,
                           add_selection_bias: bool = False, add_confounding: bool = False,
                           add_missing_values: bool = True, n_domains: int = 1,
                           variable_names: List[str] = None) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Simulate base data using the DataSimulator.
        """
        return self.base_simulator.generate_dataset(
            n_nodes=n_nodes,
            n_samples=n_samples,
            edge_probability=edge_probability,
            noise_scale=noise_scale,
            noise_type=noise_type,
            function_type=function_type,
            add_categorical=add_categorical,
            add_measurement_error=add_measurement_error,
            add_selection_bias=add_selection_bias,
            add_confounding=add_confounding,
            add_missing_values=add_missing_values,
            n_domains=n_domains,
            variable_names=variable_names
        )
