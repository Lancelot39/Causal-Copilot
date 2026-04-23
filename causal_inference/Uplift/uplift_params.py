from llm import LLMClient
import json
import os
import sklearn
from sklearn import ensemble
from causal_inference.prompt_utils import render_dataset_prompt

class UpliftParamSelector:
    def __init__(self, args):
        self.args = args

    def prompt_generation(self,  global_state):

        prompt_path = 'causal_inference/Uplift/context/model_select_prompt_tree.txt' #New prompt file
        algo_text_path = 'causal_inference/Uplift/context/model_tree.txt'#New context file.
        return render_dataset_prompt(
            prompt_path,
            algo_text_path,
            global_state,
            extra_replacements={
                "[ALGORITHM_NAME]": global_state.inference.uplift_algo_json['name'],
            },
        )
    
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
            from utils.logger import logger
            logger.warning(f"Model {model_name} not found, defaulting to RandomForestRegressor")
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
