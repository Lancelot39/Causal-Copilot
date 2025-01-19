from openai import OpenAI
import json
import os
from algorithm.llm_client import LLMClient

class Filter(object):
    def __init__(self, args):
        self.args = args
        self.llm_client = LLMClient(args)

    def forward(self, global_state):
        prompt = self.create_prompt(global_state.user_data.processed_data, global_state.statistics.description)

        output = self.llm_client.chat_completion(
            prompt=prompt,
            system_prompt="You are a causal discovery expert. Provide your response in JSON format.",
            json_response=True
        )

        algorithm_candidates = self.parse_response(output)

        global_state.algorithm.algorithm_candidates = algorithm_candidates
        global_state.logging.select_conversation.append({
            "prompt": prompt,
            "response": output
        })

        return global_state

    def load_prompt_context(self):
        # Load algorithm context
        guidelines_path = "algorithm/context/algos/guidelines.txt"
        algos_folder = "algorithm/context/algos"

        with open(guidelines_path, "r") as f:
            guidelines = f.read()
        algo_files_content = []
        for filename in os.listdir(algos_folder):
            if filename.endswith(".txt") and filename != "guidelines.txt":
                file_path = os.path.join(algos_folder, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as algo_file:
                        algo_files_content.append(algo_file.read())

        algo_context = guidelines + "\n" + "\n".join(algo_files_content)
        
        # Load select prompt template
        select_prompt = open(f"algorithm/context/algo_select_prompt.txt", "r").read()
        
        return algo_context, select_prompt

    def create_prompt(self, data, statistics_desc):
        columns = '\t'.join(data.columns)
        algo_context, prompt_template = self.load_prompt_context()
        replacements = {
            "[COLUMNS]": columns,
            "[STATISTICS_DESC]": statistics_desc,
            "[ALGO_CONTEXT]": algo_context,
        }

        for placeholder, value in replacements.items():
            prompt_template = prompt_template.replace(placeholder, value)

        return prompt_template

    def parse_response(self, response):
        try:
            algo_candidates = json.loads(response)
            return {algo['name']: {
                'description': algo['description'],
                'justification': algo['justification']
            } for algo in algo_candidates['algorithms']}
        except json.JSONDecodeError:
            print("Error: Unable to parse JSON response")
            return {}