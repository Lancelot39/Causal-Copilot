import json
import os
import torch
from algorithm.llm_client import LLMClient

TOP_K = 2

class Filter(object):
    def __init__(self, args):
        self.args = args
        self.llm_client = LLMClient(args)

    def forward(self, global_state):
        if global_state.algorithm.selected_algorithm is not None:
            print(f"User has already selected the algorithm: {global_state.algorithm.selected_algorithm}, skip the filtering process.")
            global_state.algorithm.algorithm_candidates = {
                f"{global_state.algorithm.selected_algorithm}": {
                    "description": "",
                    "justification": f"The user has already selected the algorithm: {global_state.algorithm.selected_algorithm}",
                }
            }
            global_state.logging.select_conversation.append({
                "prompt": "User has already selected the algorithm: {global_state.algorithm.selected_algorithm}, skip the filtering process.",
                "response": ""
            })  
            return  global_state
        prompt = self.create_prompt(global_state)

        output = self.llm_client.chat_completion(
            prompt=f"Please choose the algorithms that provide most reliable and accurate results, up to top {TOP_K}, for the data user provided. ",
            system_prompt=prompt,
            json_response=True,
            model="gpt-4o",
            temperature=0.0
        )
        print(output)
        algorithm_candidates = self.parse_response(output)


        global_state.algorithm.algorithm_candidates = algorithm_candidates
        global_state.logging.select_conversation.append({
            "prompt": prompt,
            "response": output
        })

        return global_state

    def load_prompt_context(self, global_state):
        # Load algorithm context
        if global_state.statistics.time_series:
            tagging_path = "algorithm/context/algos/ts_tagging.txt"
        else:
            tagging_path = "algorithm/context/algos/tab_tagging.txt"
        
        with open(tagging_path, "r", encoding="utf-8") as f:
            algo_context = f.read()
        
        # Load select prompt template
        select_prompt = open(f"algorithm/context/algo_select_prompt.txt", "r", encoding="utf-8").read()
        
        return algo_context, select_prompt
    

    def create_prompt(self, global_state):
        algo_context, prompt_template = self.load_prompt_context(global_state)
        replacements = {
            "[USER_QUERY]": global_state.user_data.initial_query,
            # "[TABLE_NAME]": self.args.data_file,
            "[WAIT_TIME]": str(global_state.algorithm.waiting_minutes),
            "[COLUMNS]": ', '.join(global_state.user_data.processed_data.columns),
            "[STATISTICS_DESC]": global_state.statistics.description,
            "[ALGO_CONTEXT]": algo_context,
            "[CUDA_WARNING]": "Current machine supports CUDA, some algorithms can be accelerated by GPU if needed." if torch.cuda.is_available() else "\nCurrent machine doesn't support CUDA, do not choose any GPU-powered algorithms.",
            "[TOP_K]": str(TOP_K),
            "[ACCEPT_CPDAG]": "The user accepts the output graph including undirected edges/undeterministic directions (CPDAG/PAG)" if global_state.user_data.accept_CPDAG else "The user does not accept the output graph including undirected edges/undeterministic directions (CPDAG/PAG), so the output graph should be a DAG."
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template

    def parse_response(self, algo_candidates):
        return {algo['name']: {
            'description': algo['description'],
            'justification': algo['justification']
        } for algo in algo_candidates['algorithms']}