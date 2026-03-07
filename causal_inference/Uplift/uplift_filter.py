from llm import LLMClient
import json
import os
import pandas as pd

class UpliftFilter:
    """
    Class to recommend appropriate uplift modeling algorithms based on data characteristics
    """
    def __init__(self, args):
        self.args = args
        self.client = LLMClient(args)
        
    def load_algo_context(self, outcome_type):
        """Load the appropriate algorithm context based on outcome type"""
        if outcome_type == 'binary':
            return open("causal_inference/Uplift/context/uplift_algo_binary_tree.txt", "r").read()
        else:
            return open("causal_inference/Uplift/context/uplift_algo_continuous_tree.txt", "r").read()
    
    def load_select_prompt(self, outcome_type):
        """Load the appropriate prompt template based on outcome type"""
        if outcome_type == 'binary':
            return open("causal_inference/Uplift/context/uplift_select_prompt_binary_tree.txt", "r").read()
        else:
            return open("causal_inference/Uplift/context/uplift_select_prompt_continuous.txt", "r").read()

    def create_prompt(self, data, statistics_desc, question, outcome_type):
        """Create a prompt for the LLM to suggest an appropriate uplift algorithm"""
        columns = ', '.join(data.columns)
        
        algo_context = self.load_algo_context(outcome_type)
        prompt_template = self.load_select_prompt(outcome_type)
        
        # Load the causal graph template if available
        try:
            causal_graph = open("causal_inference/Uplift/context/causal_graph_template.txt", "r").read()
        except Exception:
            causal_graph = "No causal graph available"
        
        replacements = {
            "[COLUMNS]": columns,
            "[STATISTICS_DESC]": statistics_desc,
            "[ALGO_CONTEXT]": algo_context,
            "[QUESTION]": question,
            "[CAUSAL_GRAPH]": causal_graph
        }
        
        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)
            
        return prompt_template
    
    def determine_outcome_type(self, data, outcome_col):
        """Determine if the outcome variable is binary or continuous"""
        unique_values = data[outcome_col].nunique()
        if unique_values <= 2:
            return 'binary'
        else:
            return 'continuous'
    
    def forward(self, global_state, query="What is the best uplift modeling algorithm for this dataset?"):
        """Select the appropriate uplift algorithm based on data characteristics"""
        # Determine if the outcome is binary or continuous
        outcome_type = self.determine_outcome_type(
            global_state.user_data.processed_data, 
            global_state.inference.outcome_col
        )
        
        # Create prompt with the appropriate context
        prompt = self.create_prompt(
            global_state.user_data.processed_data, 
            global_state.statistics.description, 
            query,
            outcome_type
        )
        
        # Get algorithm suggestions from LLM
        response = self.client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for Uplift uplift_filter.",
            json_response=True
        )
        
        # Parse and store the algorithm suggestion
        algo_candidates = response
        global_state.inference.uplift_algo_json = algo_candidates
        
        return global_state