import json
import numpy as np
import os
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
from causal_inference.prompt_utils import render_dataset_prompt
from llm import LLMClient

# Class to get models in IV two stages suggested by LLM
class HTE_Param_Selector(object):
    def __init__(self, args, y_col: str, T_col: str, Z_col:str, X_col: list, W_col: list=None):
        self.args = args
        self.y_col = y_col
        self.T_col = T_col
        self.X_col = X_col
        self.W_col = W_col
        self.Z_col = Z_col

    def check_discrete(self, target_node, global_state):
        node_type = global_state.statistics.data_type_column[target_node]
        if node_type =='Continuous':
            prompt_path = 'causal_inference/IV/context/regressor_select_prompt.txt'
            algo_text_path = 'causal_inference/IV/context/regressor.txt'
            discrete = False
        else:
            prompt_path = 'causal_inference/IV/context/classifier_select_prompt.txt'
            algo_text_path = 'causal_inference/IV/context/classifier.txt'
            discrete = True

        return prompt_path, algo_text_path, discrete

    def prompt_generation(self, prompt_path, algo_text_path, target_node, global_state):
        return render_dataset_prompt(
            prompt_path,
            algo_text_path,
            global_state,
            target_node=target_node,
        )
    
    def model_suggestion(self, client, prompt):
        response = client.chat_completion(
            prompt=prompt,
            system_prompt="You are a helpful assistant for IV hte_params.",
            json_response=True
        )
        return response
    
    def get_model(self, model_name, **kwargs):
        if hasattr(sklearn.linear_model, model_name):
            return getattr(sklearn.linear_model, model_name)(**kwargs)
        elif hasattr(sklearn.ensemble, model_name):
            return getattr(sklearn.ensemble, model_name)(**kwargs)
        elif hasattr(sklearn.svm, model_name):
            return getattr(sklearn.svm, model_name)(**kwargs)
        else:
            raise ValueError(f"Model {model_name} not found")
        
    def get_params(self, model_name: str) -> dict:
        """
        Returns a dictionary of robust hyperparameters for the specified model.
        """
        if model_name == "ElasticNet":
            return {
                "alpha": 1.0,
                "l1_ratio": 0.5,
                "max_iter": 1000,
                "tol": 1e-4,
                "random_state": 42
            }
        elif model_name == "LinearRegression":
            return {
                "fit_intercept": True
            }
        elif model_name == "SGDRegressor":
            return {
                "loss": "huber",
                "penalty": "l2",
                "alpha": 0.0001,
                "max_iter": 1000,
                "tol": 1e-3,
                "random_state": 42
            }
        elif model_name == "RandomForestRegressor":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "random_state": 42,
            }
        elif model_name == "HistGradientBoostingRegressor":
            return {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 30,
                "min_samples_leaf": 20,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "tol": 1e-4,
                "random_state": 42
            }
        elif model_name == "HuberRegressor":
            return {
                "epsilon": 1.35,
                "alpha": 0.0001,
                "max_iter": 100,
                "tol": 1e-05,
                "fit_intercept": True,
                "warm_start": False
            }
        elif model_name == "LogisticRegression":
            return {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": 42
            }
        elif model_name == "RandomForestClassifier":
            return {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1
            }
        elif model_name == "HistGradientBoostingClassifier":
            return {
                "learning_rate": 0.1,
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "n_iter_no_change": 10,
                "tol": 1e-4,
                "random_state": 42
            }
        elif model_name == "SVC":
            return {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            }
        elif model_name == "SGDClassifier":
            return {
                "loss": "hinge",
                "penalty": "l2",
                "alpha": 0.0001,
                "max_iter": 1000,
                "tol": 1e-3,
                "random_state": 42
            }
        else:
            raise ValueError(f"No default parameters defined for model '{model_name}'.")

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
        client = LLMClient(self.args)
        # Check if discrete
        prompt_path_y, algo_text_path_y, discrete_y = self.check_discrete(self.y_col, global_state)
        prompt_path_T, algo_text_path_T, discrete_T = self.check_discrete(self.T_col, global_state)
        prompt_path_z, algo_text_path_z, discrete_z = self.check_discrete(self.Z_col, global_state)

        # Generate prompts
        y_prompt = self.prompt_generation(prompt_path_y, algo_text_path_y, self.y_col, global_state)
        T_prompt = self.prompt_generation(prompt_path_T, algo_text_path_T, self.T_col, global_state)
        z_prompt = self.prompt_generation(prompt_path_z, algo_text_path_z, self.Z_col, global_state)
        final_prompt = render_dataset_prompt(
            'causal_inference/IV/context/final_stage_select_prompt.txt',
            'causal_inference/IV/context/regressor.txt',
            global_state,
        )
        
        # model_y_xw - outcome
        global_state.inference.hte_model_y_json = None
        while not global_state.inference.hte_model_y_json:
            global_state.inference.hte_model_y_json = self.model_suggestion(client, y_prompt)
        y_xw_model_name = global_state.inference.hte_model_y_json['name']
        y_xw_model = self.get_model(y_xw_model_name, **self.get_params(y_xw_model_name))

        # model_t_xw - treatment
        global_state.inference.hte_model_T_json = None
        while not global_state.inference.hte_model_T_json:
            global_state.inference.hte_model_T_json = self.model_suggestion(client, T_prompt)
        t_xw_model_name = global_state.inference.hte_model_T_json['name']
        t_xw_model = self.get_model(t_xw_model_name, **self.get_params(t_xw_model_name))

        # model_z_xw - iv
        global_state.inference.hte_model_Z_json = None
        while not global_state.inference.hte_model_Z_json:
            global_state.inference.hte_model_Z_json = self.model_suggestion(client, z_prompt)
        z_xw_model_name = global_state.inference.hte_model_Z_json['name']
        z_xw_model = self.get_model(z_xw_model_name, **self.get_params(z_xw_model_name))

        # TODO: Figure out why these are not working
        # # model_t_xwz - treatment to all
        # global_state.inference.hte_model_TALL_json = None
        # while not global_state.inference.hte_model_TALL_json:
        #     global_state.inference.hte_model_TALL_json = self.model_suggestion(client, T_prompt)
        # t_xwz_model_name = global_state.inference.hte_model_TALL_json['name']
        # t_xwz_model = self.get_model(t_xwz_model_name)

        # # model_tz_xw - covraiance to features
        # global_state.inference.hte_model_TZ_json = None
        # while not global_state.inference.hte_model_TZ_json:
        #     global_state.inference.hte_model_TZ_json = self.model_suggestion(client, T_prompt)
        # tz_xw_model_name = global_state.inference.hte_model_TZ_json['name']
        # tz_xw_model = self.get_model(tz_xw_model_name)

        # global_state.inference.hte_model_final_json = None
        # while not global_state.inference.hte_model_final_json:
        #     global_state.inference.hte_model_final_json = self.model_suggestion(client, final_prompt)
        # final_model_name = global_state.inference.hte_model_final_json['name']
        # final_model = self.get_model(final_model_name)

        global_state.inference.hte_model_param = {
            'model_y_xw': y_xw_model,
            'model_t_xw': t_xw_model, 
            'model_z_xw': z_xw_model,
            'model_t_xwz': "auto",
            'model_tz_xw': "auto",
            # 'model_final': final_model
        }
        
        if discrete_y:
            global_state.inference.hte_model_param['discrete_outcome'] = True 
        if discrete_T:
            global_state.inference.hte_model_param['discrete_treatment'] = True
        if discrete_z:
            global_state.inference.hte_model_param['discrete_instrument'] = True

        return global_state
