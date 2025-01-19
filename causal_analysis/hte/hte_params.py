import json
import numpy as np
import os
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm

# Class to get models in DML two stages suggested by LLM
class HTE_Param_Selector(object):
    def __init__(self, args, y_col: str, T_col: str, X_col: list, W_col: list=None):
        self.args = args
        self.y_col = y_col
        self.T_col = T_col
        self.X_col = X_col
        self.W_col = W_col

    def prompt_generation(self, target_node, global_state):
        node_type = global_state.statistics.data_type_column[target_node]

        if node_type =='continuous':
            prompt_path = 'causal_analysis/hte/context/regressor_select_prompt.txt'
            algo_text_path = 'causal_analysis/hte/context/regressor.txt'
        else:
            prompt_path = 'causal_analysis/hte/context/classifier_select_prompt.txt'
            algo_text_path = 'causal_analysis/hte/context/classifier.txt'
        prompt = open(prompt_path, "r").read()
        algo_text = open(algo_text_path, "r").read()
        
        replacement = {
                "[COLUMNS]": '\t'.join(global_state.user_data.processed_data.columns._data),
                "[STATISTICS INFO]": global_state.statistics.description,
                "[TARGET_NODE]": self.y_col,
                "[ALGO_CONTEXT]": algo_text
                }
        for placeholder, value in replacement.items():
            prompt = prompt.replace(placeholder, value)
        return prompt
    
    def model_suggestion(self, client, prompt):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a causal discovery expert. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        print("Model Response: ", response.choices[0].message.content)
        model_suggest_json = json.loads(response.choices[0].message.content)
        return model_suggest_json
    
    def get_model(self, model_name):
        if hasattr(sklearn.linear_model, model_name):
            return getattr(sklearn.linear_model, model_name)()
        elif hasattr(sklearn.ensemble, model_name):
            return getattr(sklearn.ensemble, model_name)()
        elif hasattr(sklearn.svm, model_name):
            return getattr(sklearn.svm, model_name)()
        else:
            raise ValueError(f"Model {model_name} not found")

    # workflow
    # 1. for the best algorithm, select the best hyperparameters
    #   a. give suggestions for the primary hyperparameters, use default values of the secondary hyperparameters
    #     i. for each primary hyperparameters, we might need to have some context about their meaning and possible values
    # 4. return the selected algorithm and its hyperparameters
    def forward(self, global_state):
        '''
        :param global_state: The global state containing the processed data, selected algorithm, statistics description, and knowledge documents
        :return: A doc containing the selected algorithm and its hyperparameter settings
        '''
        from openai import OpenAI
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        # Set up the Hyperparameters
        # Load hyperparameters prompt template
        import json
        import hte.wrappers as wrappers

        y_prompt = self.prompt_generation(self.y_col, global_state)
        T_prompt = self.prompt_generation(self.T_col, global_state)
        final_prompt = open('causal_analysis/hte/context/final_stage_select_prompt.txt', "r").read()

        global_state.inference.hte_model_y_json = None
        while not global_state.inference.hte_model_y_json:
            global_state.inference.hte_model_y_json = self.model_suggestion(client, y_prompt)
        y_model_name = global_state.inference.hte_model_y_json['name']
        y_model = self.get_model(y_model_name)

        global_state.inference.hte_model_T_json = None
        while not global_state.inference.hte_model_T_json:
            global_state.inference.hte_model_T_json = self.model_suggestion(client, T_prompt)
        T_model_name = global_state.inference.hte_model_T_json['name']
        T_model = self.get_model(T_model_name)

        global_state.inference.hte_model_final_json = None
        while not global_state.inference.hte_model_final_json:
            global_state.inference.hte_model_final_json = self.model_suggestion(client, final_prompt)
        final_model_name = global_state.inference.hte_model_final_json['name']
        final_model = self.get_model(final_model_name)

        global_state.inference.hte_model_param = {
            'model_y': y_model,
            'model_t': T_model, 
            'model_final': final_model
        }

        return global_state
