import json
import os
from openai import OpenAI
import algorithm.wrappers as wrappers

class HyperparameterSelector:
    def __init__(self, args):
        self.args = args

    def load_hp_context(self):
        # Load hyperparameters context
        hp_context = {}
        hyperparameters_folder = "algorithm/context/hyperparameters"
        for filename in os.listdir(hyperparameters_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(hyperparameters_folder, filename)
                with open(file_path, "r") as f:
                    algo_hp = json.load(f)
                    algo_name = algo_hp.pop("algorithm_name")
                    hp_context[algo_name] = algo_hp
            
        # Load additional context files for parameters that have them
        for algo in hp_context:
            for param in hp_context[algo]:
                if 'context_file' in hp_context[algo][param]:
                    context_file_path = hp_context[algo][param]['context_file']
                    with open(context_file_path, "r") as cf:
                        hp_context[algo][param]['context_content'] = cf.read()
        
        return hp_context

    def create_prompt(self, global_state, selected_algo, algo2des_cond_hyper, time_info_cdnod=""):
        with open("algorithm/context/hyperparameter_select_prompt.txt", "r") as f:
            hp_prompt = f.read()
        
        algo_description = algo2des_cond_hyper[selected_algo]
        primary_params = getattr(wrappers, selected_algo)().get_primary_params()
        hp_info_str = str([selected_algo])
        table_columns = '\t'.join(global_state.user_data.processed_data.columns._data)
        knowledge_info = '\n'.join(global_state.user_data.knowledge_docs)
        
        hp_prompt = hp_prompt.replace("[COLUMNS]", table_columns)
        hp_prompt = hp_prompt.replace("[KNOWLEDGE_INFO]", knowledge_info)
        hp_prompt = hp_prompt.replace("[STATISTICS INFO]", global_state.statistics.description)
        hp_prompt = hp_prompt.replace("[ALGORITHM_NAME]", selected_algo)
        hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algo_description)
        hp_prompt = hp_prompt.replace("[PRIMARY_HYPERPARAMETERS]", str(primary_params))
        hp_prompt = hp_prompt.replace("[HYPERPARAMETER_INFO]", hp_info_str)

        return hp_prompt, primary_params
        
    def select_hyperparameters(self, global_state, client, selected_algo, algo2des_cond_hyper, time_info_cdnod=""):
        if global_state.algorithm.algorithm_arguments is not None:
            print("User has already selected the hyperparameters, skip the hyperparameter selection process.")
            return global_state.algorithm.algorithm_arguments
        
        hp_prompt, primary_params = self.create_prompt(global_state, selected_algo, algo2des_cond_hyper, time_info_cdnod)

        # if selected_algo == 'CDNOD' and global_state.statistics.linearity == False:
        #     kci_prompt = (f'\nAs it is nonlinear data, it is suggested to use kci for indep_test. '
        #                 f'As the user can wait for {global_state.algorithm.waiting_minutes} minutes for the algorithm execution. If kci can not exceed it, we MUST select it:\n\n'
        #                 f'The estimated time costs of CDNOD algorithms using the two indep_test settings are: {time_info_cdnod}')
        #     hp_prompt = hp_prompt + kci_prompt
        
        response = client.chat_completion(hp_prompt, system_prompt="You are a causal discovery expert. Provide your response in JSON format.", json_response=True)

        hyper_suggest = json.loads(response)
        global_state.algorithm.algorithm_arguments_json = hyper_suggest
        hyper_suggest = {k: v['value'] for k, v in hyper_suggest['hyperparameters'].items() if k in primary_params}

        global_state.logging.argument_conversation.append({
            "prompt": hp_prompt,
            "response": response
        })

        print("-"*25, "\n", "Hyperparameter Response: ", "\n", hyper_suggest)

        return hyper_suggest 