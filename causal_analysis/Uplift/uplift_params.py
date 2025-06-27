from llm import LLMClient
import json
import os
import sklearn
from sklearn import ensemble

class UpliftParamSelector:
    def __init__(self, args):
        self.args = args

    def prompt_generation(self,  global_state):

        prompt_path = 'causal_analysis/Uplift/context/model_select_prompt_tree.txt' #New prompt file
        algo_text_path = 'causal_analysis/Uplift/context/model_tree.txt'#New context file.
        prompt = open(prompt_path, "r").read()
        algo_text = open(algo_text_path, "r").read()

        replacement = {
                "[COLUMNS]": '\t'.join(global_state.user_data.processed_data.columns._data),
                "[STATISTICS INFO]": global_state.statistics.description,
                "[ALGO_CONTEXT]": algo_text,
                "[ALGORITHM_NAME]" : global_state.inference.uplift_algo_json['name']
                }
        for placeholder, value in replacement.items():
            prompt = prompt.replace(placeholder, value)
        return prompt
    
    def model_suggestion(self, client, prompt):
        response = client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for Uplift uplift_params.",
            json_response=True
        )
        return response

    def get_model(self, model_name):
        #Focus on tree based models
        if hasattr(ensemble, model_name):
            return getattr(ensemble, model_name)()
        else:
            print(f"Model {model_name} not found, default to RandomForestRegressor")
            return ensemble.RandomForestRegressor()

    def forward(self, global_state):
        client = LLMClient(self.args)
        model_prompt = self.prompt_generation(global_state)

        global_state.inference.uplift_model_json = None
        while not global_state.inference.uplift_model_json:
            global_state.inference.uplift_model_json = self.model_suggestion(client, model_prompt)

        model_name = global_state.inference.uplift_model_json['name']

        #The model suggested by the LLM
        model = self.get_model(model_name)
        #Wrap the base model
        global_state.inference.uplift_model_param = {
            'model': model
        }
        return global_state