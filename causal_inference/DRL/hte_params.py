import json
import numpy as np
import os
import sklearn.linear_model
import sklearn.ensemble
import sklearn.svm
from causal_inference.prompt_utils import render_dataset_prompt
from llm import LLMClient

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
        if node_type =='Continuous':
            prompt_path = 'causal_inference/DRL/context/regressor_select_prompt.txt'
            algo_text_path = 'causal_inference/DRL/context/regressor.txt'
            discrete = False
        else:
            prompt_path = 'causal_inference/DRL/context/classifier_select_prompt.txt'
            algo_text_path = 'causal_inference/DRL/context/classifier.txt'
            discrete = True
        prompt = render_dataset_prompt(
            prompt_path,
            algo_text_path,
            global_state,
            target_node=target_node,
        )
        return prompt, discrete
    
    def model_suggestion(self, client, prompt):
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a causal discovery expert. Provide your response in JSON format."},
                {"role": "user", "content": prompt}
            ],
            json_response=True
        )

        from utils.logger import logger
        logger.debug("Model response received", "HTE")
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
        Returns a dictionary of default hyperparameters for the specified model.
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
        # Set up the Hyperparameters
        # Load hyperparameters prompt template
        y_prompt, discrete_y = self.prompt_generation(self.y_col, global_state)
        T_prompt, discrete_T = self.prompt_generation(self.T_col, global_state)
        final_prompt = render_dataset_prompt(
            'causal_inference/DRL/context/final_stage_select_prompt.txt',
            'causal_inference/DRL/context/regressor.txt',
            global_state,
        )

        global_state.inference.hte_model_y_json = None
        while not global_state.inference.hte_model_y_json:
            global_state.inference.hte_model_y_json = self.model_suggestion(client, y_prompt)
        y_model_name = global_state.inference.hte_model_y_json['name']
        y_model_params = self.get_params(y_model_name)
        y_model = self.get_model(y_model_name, **y_model_params)

        global_state.inference.hte_model_T_json = None
        while not global_state.inference.hte_model_T_json:
            global_state.inference.hte_model_T_json = self.model_suggestion(client, T_prompt)
        T_model_name = global_state.inference.hte_model_T_json['name']
        T_model_params = self.get_params(T_model_name)
        T_model = self.get_model(T_model_name, **T_model_params)

        global_state.inference.hte_model_final_json = None
        while not global_state.inference.hte_model_final_json:
            global_state.inference.hte_model_final_json = self.model_suggestion(client, final_prompt)
        final_model_name = global_state.inference.hte_model_final_json['name']
        final_model_params = self.get_params(final_model_name)
        final_model = self.get_model(final_model_name, **final_model_params)

        global_state.inference.hte_model_param = {
            'model_regression': y_model,
            'model_propensity': T_model,
            'model_final': final_model
        }
        if discrete_y:
            global_state.inference.hte_model_param['discrete_outcome'] = True 
        # if discrete_T:
        #     global_state.inference.hte_model_param['discrete_treatment'] = True

        return global_state
