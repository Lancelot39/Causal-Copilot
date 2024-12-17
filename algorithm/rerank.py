import json
import numpy as np
class Reranker(object):
    # Kun Zhou Implemented
    def __init__(self, args):
        self.args = args
        self.algo2time_cost = json.load(open('algorithm/context/algo2time_cost.json', encoding='utf-8'))

    def statistics_dict2string(self, statistics_dict):
        return str(statistics_dict)

    def algo_can2string(self, algo_candidates, hp_context):
        algo_candidate_string = ''
        algo2des_cond_hyper = {}
        for algo_name in algo_candidates:
            if algo_name in hp_context:
                algo_des, algo_justify = algo_candidates[algo_name]['description'], algo_candidates[algo_name]['justification']
                algo_string = algo_name + ':\n' + algo_des + '\nJustification: ' + algo_justify
                algo2des_cond_hyper[algo_name] = algo_string
                algo_candidate_string += algo_string + "\n\n"
        return algo_candidate_string, algo2des_cond_hyper

    def time_estimate(self, algo_can, n_sample, variable):
        def fitting(data, a, b):
            x, y = data
            return a * np.log(x) * y ** 2 + b
        def find_neighbor(x, x_list):
            x_list = list(x_list)
            x_list.sort()
            for ele in x_list:
                if x <= ele:
                    return ele
            return x_list[-1]

        algo_cost = self.algo2time_cost[algo_can]['cost']
        algo_para = self.algo2time_cost[algo_can]['hyper']
        n_sample_list = [int(ele) for ele in algo_cost.keys()]
        if n_sample > max(n_sample_list):
            return fitting((n_sample, variable), algo_para[0], algo_para[1])
        neighbor_n_sample = find_neighbor(n_sample, n_sample_list)

        variable_list = [int(ele) for ele in algo_cost[str(neighbor_n_sample)].keys()]
        if variable > max(variable_list):
            return fitting((n_sample, variable), algo_para[0], algo_para[1])
        neighbor_variable = find_neighbor(variable, variable_list)
        return algo_cost[str(neighbor_n_sample)][str(neighbor_variable)]

    def algo_cans2time_string(self, algo_cans, n_sample, variable):
        prompt = ""
        CDNOD_prompt = ""
        for algo_can in algo_cans:
            try:
                time_cost = self.time_estimate(algo_can, n_sample, variable)
                prompt += algo_can + ": " + str(time_cost) + "min\n"
                if algo_can == 'CDNOD':
                    CDNOD_prompt += "CDNOD using fisherz for indep_test: " + str(time_cost) + "min\n"
                    time_cost = self.time_estimate('CDNOD-kci', n_sample, variable)
                    CDNOD_prompt += "CDNOD using kci for indep_test: " + str(time_cost) + "min\n"
            except:
                prompt += algo_can + ": " + 'Unknown Time' + "\n"
                print(f"Meeting Error for {algo_can}")
        return prompt, CDNOD_prompt

    def extract(self, output, start_str, end_str):
        if start_str in output and end_str in output:
            try:
                algo = output.split(start_str)[1].split(end_str)[0]
            except:
                algo = ''
            return algo
        else:
            return ''

    # workflow
    # 1. select candidate algorithms -> algorithm names and their descriptions
    # 2. rerank the algorithms based on more criteria -> pick the best one
    # 3. for the best algorithm, select the best hyperparameters
    #   a. give suggestions for the primary hyperparameters, use default values of the secondary hyperparameters
    #     i. for each primary hyperparameters, we might need to have some context about their meaning and possible values
    # 4. return the selected algorithm and its hyperparameters
    def forward(self, global_state):
        '''
        :param global_state: The global state containing the processed data, algorithm candidates, statistics description, and knowledge documents
        :return: A doc containing the selected algorithm and its hyperparameter settings
        '''
        from openai import OpenAI
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        # Set up the Hyperparameters
        # Load hyperparameters prompt template
        import json
        import algorithm.wrappers as wrappers

        data = global_state.user_data.processed_data
        if global_state.statistics.heterogeneous != True:
            algo_candidates = {algo:global_state.algorithm.algorithm_candidates[algo] for algo in global_state.algorithm.algorithm_candidates if algo != 'CDNOD'}
            if global_state.algorithm.selected_algorithm == 'CDNOD':
                print("Sorry! As the data is not heterogeneous, CDNOD algorithm should not be used! "
                      "Causality-Copilot will continue to select the best-suited algorithm for you!")
                global_state.algorithm.selected_algorithm = None
        else:
            algo_candidates = global_state.algorithm.algorithm_candidates

        statistics_desc = global_state.statistics.description
        knowledge_docs = global_state.user_data.knowledge_docs

        # Load hyperparameters context
        with open("algorithm/context/hyperparameters.json", "r") as f:
            hp_context = json.load(f)

        # Load additional context files for parameters that have them
        for algo in hp_context:
            for param in hp_context[algo]:
                if 'context_file' in hp_context[algo][param]:
                    context_file_path = hp_context[algo][param]['context_file']
                    with open(context_file_path, "r") as cf:
                        hp_context[algo][param]['context_content'] = cf.read()

        if global_state.algorithm.selected_algorithm is not None:
            algo_candidates = {global_state.algorithm.selected_algorithm: {'description': '', 'justification': ''}}
        algo_info, algo2des_cond_hyper = self.algo_can2string(algo_candidates, hp_context)

        table_name = self.args.data_file
        table_columns = '\t'.join(data.columns._data)
        knowledge_info = '\n'.join(knowledge_docs)
        statistics_info = statistics_desc
        wait_time = global_state.algorithm.waiting_minutes

        with open("algorithm/context/hyperparameters_prompt.txt", "r") as f:
            hp_prompt = f.read()

        if global_state.algorithm.selected_algorithm is None:
            time_info, time_info_cdnod = self.algo_cans2time_string(algo_candidates, global_state.statistics.sample_size, global_state.statistics.feature_number)

            # Select the Best Algorithm
            prompt = (("I will conduct causal discovery on the Tabular Dataset %s containing the following Columns:\n\n"
                    "%s\n\nThe Detailed Background Information is listed below:\n\n"
                    "%s\n\nThe Statistics Information about the dataset is:\n\n"
                    "%s\n\nBased on the above information, please select the best-suited algorithm from the following candidate:\n\n"
                    "%s\n\nNote that the user can wait for %f minutes for the algorithm execution, please ensure the time cost of the selected algorithm would not exceed it!\n"
                    "The estimated time costs of the following algorithms are:\n\n%s\n\n"
                    "Tips:\n1.If the data is nonstationary or heterogeneous across domains/time, Use CDNOD as the first choice. "
                    "Note that if the data is not heterogeneous, DO NOT select CDNOD!!!"
                    "\n2.If the noise is non-Gaussian, Try DirectLiNGAM or ICALiNGAM first;"
                    "Note that if the data is non-linear, DO NOT select DirectLiNGAM or ICALNGAM!!!"
                    "\n3.If the data is linear, Try GES first;"
                    "\n4.If the data is large (i.e., > 100 variables), Start with PC algorithm;"
                    "\n5.If the data is non-linear, Start with PC or FCI algorithms, and NEVER use DirectLiNGAM or ICALNGAM;"
                    "\n\nPlease highlight the selected algorithm name using the following template <Algo>Name</Algo> in the ending of the output") %
                    (table_name, table_columns, knowledge_info, statistics_info, algo_info, wait_time, time_info))
            selected_algo = ''
            print("Keys in algo2des_cond_hyper:", algo2des_cond_hyper.keys())
            while selected_algo not in algo2des_cond_hyper:
                #print("The used prompt for rerank is: -------------------------------------------------------------------------")
                #print(prompt)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                output = response.choices[0].message.content
                print("The received answer for rerank is: -------------------------------------------------------------------------")
                print(output)
                selected_algo = self.extract(output, '<Algo>', '</Algo>')
            global_state.algorithm.selected_algorithm = selected_algo
            global_state.logging.select_conversation.append({
                "prompt": prompt,
                "response": response.choices[0].message.content
            })
        else:
            print("User has already selected the algorithm, skip the reranking process.")
            selected_algo = global_state.algorithm.selected_algorithm
            if selected_algo == 'CDNOD':
                time_info, time_info_cdnod = self.algo_cans2time_string([selected_algo], global_state.statistics.sample_size, global_state.statistics.feature_number)
            else:
                time_info, time_info_cdnod = "", ""

        print("Selected Algorithm: ", global_state.algorithm.selected_algorithm)

        if global_state.algorithm.selected_algorithm is not None and global_state.algorithm.algorithm_arguments is None:
            # Get algorithm description and hyperparameters

            algo_description = algo2des_cond_hyper[selected_algo]
            primary_params = getattr(wrappers, selected_algo)().get_primary_params()

            # Prepare hyperparameter information
            hp_info_str = str(hp_context[selected_algo])

            # Create the hyperparameter selection prompt
            hp_prompt = hp_prompt.replace("[COLUMNS]", table_columns)
            hp_prompt = hp_prompt.replace("[KNOWLEDGE_INFO]", knowledge_info)
            hp_prompt = hp_prompt.replace("[STATISTICS INFO]", statistics_desc)
            hp_prompt = hp_prompt.replace("[ALGORITHM_NAME]", selected_algo)
            hp_prompt = hp_prompt.replace("[ALGORITHM_DESCRIPTION]", algo_description)
            hp_prompt = hp_prompt.replace("[PRIMARY_HYPERPARAMETERS]", str(primary_params))
            hp_prompt = hp_prompt.replace("[HYPERPARAMETER_INFO]", hp_info_str)

            if selected_algo == 'CDNOD' and global_state.statistics.linearity == False:
                kci_prompt = (f'\nAs it is nonlinear data, it is suggested to use kci for indep_test. '
                              f'As the user can wait for {wait_time} minutes for the algorithm execution. If kci can not exceed it, we MUST select it:\n\n'
                              f'The estimated time costs of CDNOD algorithms using the two indep_test settings are: {time_info_cdnod}')
                hp_prompt = hp_prompt + kci_prompt

            # Get hyperparameter suggestions from GPT-4
            #print("Hyperparameter Prompt: ", hp_prompt)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a causal discovery expert. Provide your response in JSON format."},
                    {"role": "user", "content": hp_prompt}
                ],
                response_format={"type": "json_object"}
            )

            print("Hyperparameter Response: ", response.choices[0].message.content)
            hyper_suggest = json.loads(response.choices[0].message.content)
            global_state.algorithm.algorithm_arguments_json = hyper_suggest

            # only use the hyperparameter keys and values, explanation is added later
            hyper_suggest = {k: v['value'] for k, v in hyper_suggest['hyperparameters'].items() if k in primary_params}

            global_state.algorithm.algorithm_arguments = hyper_suggest
            print("Selected Algorithm: ", selected_algo)
            print("Hyperparameter Suggestions: ", hyper_suggest)

            global_state.logging.argument_conversation.append({
                "prompt": hp_prompt,
                "response": response.choices[0].message.content
            })
        else:
            print("User has already selected the hyperparameters, skip the hyperparameter selection process.")
            print("Selected Hyperparameters: ", global_state.algorithm.algorithm_arguments)

        return global_state
