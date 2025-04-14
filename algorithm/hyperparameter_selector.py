import json
import torch
import algorithm.wrappers as wrappers
from algorithm.llm_client import LLMClient
from .context.algos.utils.json2txt import create_filtered_benchmarking_results

class HyperparameterSelector:
    def __init__(self, args):
        self.args = args
        self.llm_client = LLMClient(args)

    def forward(self, global_state):
        selected_algo = global_state.algorithm.selected_algorithm
        hp_context = self.load_hp_context(selected_algo)
        try:
            algorithm_description = global_state.algorithm.algorithm_candidates[selected_algo]['description']
            algorithm_optimum_reason = global_state.algorithm.algorithm_optimum['reason']
            algorithm_optimum_reason = algorithm_description + "\n" + algorithm_optimum_reason
        except:
            algorithm_optimum_reason = "User specifies this algorithm."

        # Select hyperparameters
        hyper_suggest = self.select_hyperparameters(global_state, selected_algo, hp_context, algorithm_optimum_reason)
        global_state.algorithm.algorithm_arguments = hyper_suggest
        return global_state
        

    def load_hp_context(self, selected_algo):
        # Load hyperparameters context
        with open(f"algorithm/context/hyperparameters/{selected_algo}.json", "r") as f:
            hp_context = json.load(f)
        
        # Convert the hyperparameters context to natural language
        def convert_to_natural_language(hp_context):
            natural_language_context = ""
            # Skip the algorithm_name field
            for param, details in hp_context.items():
                if param == "algorithm_name":
                    continue
                natural_language_context += f"**Parameter:** {param}\n"
                if isinstance(details, dict) and "meaning" in details:
                    natural_language_context += f"- **Meaning:** {details['meaning']}\n"
                    natural_language_context += "- **Available Values:**\n"
                    for value in details['available_values']:
                        natural_language_context += f"  - {value}\n"
                    natural_language_context += f"- **Expert Suggestion:** {details['expert_suggestion']}\n\n"
            return natural_language_context
        
        return convert_to_natural_language(hp_context)

    def create_prompt(self, global_state, selected_algo, hp_context, algorithm_optimum_reason):
        with open(f"algorithm/context/benchmarking/algorithm_performance_analysis.json", "r", encoding="utf-8") as f:
            algorithm_benchmarking_results = json.load(f)
            algorithm_benchmarking_results = create_filtered_benchmarking_results(algorithm_benchmarking_results, [selected_algo])

        with open("algorithm/context/hyperparameter_select_prompt.txt", "r", encoding="utf-8") as f:
            hp_prompt = f.read()
        
        print(selected_algo)
        primary_params = list(getattr(wrappers, selected_algo)().get_primary_params().keys())
        hp_info_str = json.dumps(hp_context)
        table_columns = '\t'.join(global_state.user_data.processed_data.columns._data)
        knowledge_info = '\n'.join(global_state.user_data.knowledge_docs)
        
        hp_prompt = hp_prompt.replace("[USER_QUERY]", global_state.user_data.initial_query)
        hp_prompt = hp_prompt.replace("[WAIT_TIME]", str(global_state.algorithm.waiting_minutes))
        hp_prompt = hp_prompt.replace("[ALGORITHM_NAME]", selected_algo)
        # hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algorithm_optimum_reason)
        hp_prompt = hp_prompt.replace("[ALGORITHM_PERFORMANCE]", algorithm_benchmarking_results)
        hp_prompt = hp_prompt.replace("[COLUMNS]", table_columns)
        hp_prompt = hp_prompt.replace("[KNOWLEDGE_INFO]", knowledge_info)
        hp_prompt = hp_prompt.replace("[STATISTICS INFO]", global_state.statistics.description)
        hp_prompt = hp_prompt.replace("[CUDA_WARNING]", "Current machine supports CUDA, some algorithms can be accelerated by GPU if needed." if torch.cuda.is_available() else "\nCurrent machine doesn't support CUDA, do not choose any GPU-powered algorithms.")
        # hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algorithm_optimum_reason)
        hp_prompt = hp_prompt.replace("[PRIMARY_HYPERPARAMETERS]", ', '.join(primary_params))
        hp_prompt = hp_prompt.replace("[HYPERPARAMETER_INFO]", hp_info_str)

        with open(f"algorithm/context/hp_rerank_prompt_test.txt", "w", encoding="utf-8") as f:
            f.write(hp_prompt)

        return hp_prompt, primary_params
        
    def select_hyperparameters(self, global_state, selected_algo, hp_context, algorithm_optimum_reason):
        if global_state.algorithm.algorithm_arguments is not None:
            print("User has already selected the hyperparameters, skip the hyperparameter selection process.")
            return global_state.algorithm.algorithm_arguments
        
        hp_prompt, primary_params = self.create_prompt(global_state, selected_algo, hp_context, algorithm_optimum_reason)

        # if selected_algo == 'CDNOD' and global_state.statistics.linearity == False:
        #     kci_prompt = (f'\nAs it is nonlinear data, it is suggested to use kci for indep_test. '
        #                 f'As the user can wait for {global_state.algorithm.waiting_minutes} minutes for the algorithm execution. If kci can not exceed it, we MUST select it:\n\n'
        #                 f'The estimated time costs of CDNOD algorithms using the two indep_test settings are: {time_info_cdnod}')
        #     hp_prompt = hp_prompt + kci_prompt

        print(hp_prompt)
        
        response = self.llm_client.chat_completion("Please select the best hyperparameters for the algorithm.",
                                                    system_prompt=hp_prompt, json_response=True,  temperature=0.0,
                                                    model="gpt-4o")

        hyper_suggest = response
        global_state.algorithm.algorithm_arguments_json = hyper_suggest
        print('hyper_suggest', hyper_suggest)
        hyper_suggest = {k: v['value'] for k, v in hyper_suggest['hyperparameters'].items() if k in primary_params}

        global_state.logging.argument_conversation.append({
            "prompt": hp_prompt,
            "response": response
        })

        print("-"*25, "\n", "Hyperparameter Response: ", "\n", hyper_suggest)

        return hyper_suggest 