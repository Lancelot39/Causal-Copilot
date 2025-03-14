from openai import OpenAI
import json
import os
import torch
from algorithm.llm_client import LLMClient

TOP_K = 3

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
        prompt = self.create_prompt(global_state.user_data.initial_query, global_state.user_data.processed_data, global_state.statistics.description, global_state.user_data.accept_CPDAG)

        # save the prompt to a file for debugging
        with open(os.path.join(os.path.dirname(__file__), "prompt.txt"), "w") as f:
            f.write(prompt)

        output = self.llm_client.chat_completion(
            prompt=prompt,
            system_prompt="",
            json_response=True
        )

        algorithm_candidates = self.parse_response(output)

        print(algorithm_candidates)

        global_state.algorithm.algorithm_candidates = algorithm_candidates
        global_state.logging.select_conversation.append({
            "prompt": prompt,
            "response": output
        })

        return global_state

    def load_prompt_context(self):
        # Load algorithm context
        guidelines_path = "algorithm/context/algos/guidelines.txt"
        tagging_path = "algorithm/context/algos/tagging.txt"
        algos_folder = "algorithm/context/algos"

        # with open(guidelines_path, "r") as f:
        #     guidelines = f.read()
        
        with open(tagging_path, "r") as f:
            tags = f.read()

        # algo_context = "Here are the guidelines for causal discovery algorithms:\n" + guidelines + "\n\nHere are the tags for causal discovery algorithms:\n" + tags

        algo_context = tags
        
        # Load select prompt template
        select_prompt = open(f"algorithm/context/algo_select_prompt.txt", "r").read()
        
        return algo_context, select_prompt
    


    def create_prompt(self, user_query, data, statistics_desc, accept_CPDAG):
        algo_context, prompt_template = self.load_prompt_context()
        replacements = {
            "[USER_QUERY]": user_query,
            "[COLUMNS]": ', '.join(data.columns),
            "[STATISTICS_DESC]": statistics_desc,
            "[ALGO_CONTEXT]": algo_context,
            "[CUDA_WARNING]": "" if torch.cuda.is_available() else "\nCurrent machine doesn't support CUDA, do not choose any GPU-powered algorithms.",
            "[TOP_K]": str(TOP_K),
            "[ACCEPT_CPDAG]": "Yes" if accept_CPDAG else "No"
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template

    def parse_response(self, algo_candidates):
        return {algo['name']: {
            'description': algo['description'],
            'justification': algo['justification']
        } for algo in algo_candidates['algorithms']}