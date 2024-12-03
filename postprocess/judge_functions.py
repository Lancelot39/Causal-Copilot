import numpy as np
from sympy.stats.rv import probability
import ast
import json 
from openai import OpenAI
from postprocess.visualization import Visualization
from collections import Counter


def bootstrap_iteration(data, ts, algorithm, hyperparameters):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param ts: Indicator of time-series
    :param algorithm: String representing the algorithm name
    :param hyperparameters: Dictionary of hyperparameter names and values
    :return: Bootstrap result of one iteration
    '''
    import random
    import math
    import pandas as pd
    import algorithm.wrappers as wrappers

    n = data.shape[0]

    # Choose bootstrap method based on the ts flag
    if not ts:
        # General bootstrapping
        boot_index = random.choices(range(n), k=n)
        boot_sample = data.iloc[boot_index, :]
    else:
        # Moving block bootstrapping for time-series
        block_size = 10
        block_num = math.ceil(n / block_size)
        block_start = random.sample(range(n - block_size + 1), block_num)

        blocks = [list(range(start, start + block_size)) for start in block_start]
        subsets = [data.iloc[block] for block in blocks]

        boot_sample = pd.concat(subsets, ignore_index=True).iloc[0:n]

    # Get the algorithm function from wrappers
    algo_func = getattr(wrappers, algorithm)

    # Execute the algorithm with data and hyperparameters
    converted_graph, info, raw_result = algo_func(hyperparameters).fit(boot_sample)

    if algorithm == 'PC':
        boot_graph = raw_result.G.graph
    elif algorithm == 'FCI':
        boot_graph = raw_result[0].graph
    elif algorithm == "GES":
        boot_graph = raw_result['G'].graph
    elif algorithm == 'CDNOD':
        boot_graph = raw_result.G.graph
    else:
        boot_graph = converted_graph

    return boot_graph

def bootstrap_probability(boot_result, algorithm):

    m = boot_result.shape[1]

    certain_edges_prob = np.zeros((m,m))  # -> and use converted graph
    uncertain_edges_prob = np.zeros((m,m))  # -
    bi_edges_prob = np.zeros((m,m))  # <->
    half_edges_prob = np.zeros((m,m))  # o->
    none_edges_prob = np.zeros((m,m))  # o-o
    none_exist_prob = np.zeros((m,m)) # did not exist edge

    for i in range(m):
        for j in range(m):
            if i == j: continue
            else:
                elements_ij = boot_result[:, i, j]
                elements_ji = boot_result[:, j, i]

                if algorithm in ['PC','GES','CDNOD','FCI']:
                    # j -> i
                    certain_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == -1))
                    # i - j
                    uncertain_edges_prob[i, j] = np.mean((elements_ij == -1) & (elements_ji == -1))
                    # i <-> j
                    bi_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == 1))
                else:
                    # j -> i
                    certain_edges_prob[i, j] = np.mean(elements_ij == 1)

                # no existence of edge
                none_exist_prob[i, j] = np.mean(elements_ij == 0)

                if algorithm == 'FCI':
                    # j o-> i
                    half_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == 2))
                    # i o-o j
                    none_edges_prob[i, j] = np.mean((elements_ij == 2) & (elements_ji == 2))


    edges_prob = np.stack((certain_edges_prob, uncertain_edges_prob, bi_edges_prob, half_edges_prob, none_edges_prob, none_exist_prob), axis=0)

    return edges_prob



def bootstrap(data, full_graph, algorithm, hyperparameters, boot_num, ts, parallel):
    '''
    :param data: Given Tabular Data in Pandas DataFrame format
    :param full_graph: An adjacent matrix in Numpy Ndarray format -
                       causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
    :param algorithm: String representing the algorithm name
    :param hyperparameters: Dictionary of hyperparameter names and values
    :param boot_num: Number of bootstrap iterations
    :param ts: An indicator of time-series data
    :param parallel: indicator of parallel computing
    :return: a dict of obvious errors in causal analysis results based on bootstrap,
             e.g. {"X->Y: "Forced", "Y->Z: "Forbidden"};
             a matrix records bootstrap probability of directed edges, Matrix[i,j] records the
             bootstrap probability of the existence of edge j -> i.
    '''

    from multiprocessing import Pool

    m = data.shape[1]
    errors = {}
    raw_graph = full_graph

    # try:
    #     if algorithm == 'PC':
    #         raw_graph = full_graph.raw_result.G.graph
    #     elif algorithm == 'FCI':
    #         raw_graph = full_graph.raw_result[0].graph
    #     elif algorithm == "GES":
    #         raw_graph = full_graph.raw_result['G'].graph
    #     elif algorithm == 'CDNOD':
    #         raw_graph = full_graph.raw_result.G.graph
    #     else:
    #         raw_graph = full_graph.converted_graph
    # except:
    #     if algorithm == 'PC':
    #         raw_graph = full_graph.G.graph
    #     elif algorithm == 'FCI':
    #         raw_graph = full_graph[0].graph
    #     elif algorithm == "GES":
    #         raw_graph = full_graph['G'].graph
    #     elif algorithm == 'CDNOD':
    #         raw_graph = full_graph.G.graph
    #     else:
    #         raw_graph = full_graph

    boot_effect_save = []  # Save graphs based on bootstrapping

    if not parallel:
        for boot_time in range(boot_num):
            boot_graph = bootstrap_iteration(data, ts, algorithm, hyperparameters)
            boot_effect_save.append(boot_graph)

    if parallel:
        pool = Pool()

        # Prepare arguments for each process
        args = [(data, ts, algorithm, hyperparameters) for _ in range(boot_num)]
        boot_effect_save = pool.starmap(bootstrap_iteration, args)

        pool.close()
        pool.join()

    boot_effect_save_array = np.array(boot_effect_save)

    # Each layer of edges_prob represents:
    # 0. certain_edges_prob: ->
    # 1. uncertain_edges_prob: -
    # 2. bi_edges_prob: <->
    # 3. half_edges_prob: o->
    # 4. none_edges_prob: o-o
    # 5. none_exist_prob: x
    edges_prob = bootstrap_probability(boot_effect_save_array, algorithm)

    recommend = ['->', '-', '<->', 'o->', 'o-o', 'Forbid']

    boot_recommend = {}

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            else:
                element_ij = raw_graph[i, j]
                element_ji = raw_graph[j, i]
                prob_ij = edges_prob[:, i, j]

                if algorithm in ['PC','GES','CDNOD','FCI']:
                    certain_edge_raw = (element_ij == 1 and element_ji == -1)
                    uncertain_edge_raw = (element_ij == -1 and element_ji == -1)
                    bi_edge_raw = (element_ij == 1 and element_ji == 1)
                    non_exist_raw = (element_ij == 0 or element_ji == 0)

                    cond0 = certain_edge_raw and (edges_prob[0, i, j] < 0.05)  # j -> i
                    cond1 = uncertain_edge_raw and (edges_prob[1, i, j] < 0.05)  # j - i
                    cond2 = bi_edge_raw and (edges_prob[2, i, j] < 0.05)  # j <-> i
                    cond5 = non_exist_raw and (edges_prob[5, i, j] < 0.05)  # j x i

                    if algorithm == 'FCI':
                        half_edge_raw = (element_ij == 1 and element_ji == 2)
                        none_edge_raw = (element_ij == 2 and element_ji == 1)

                        cond3 = half_edge_raw and (edges_prob[3, i, j] < 0.05)  # j o-> i
                        cond4 = none_edge_raw and (edges_prob[4, i, j] < 0.05)  # j o-o i
                else:
                    certain_edge_raw = (element_ij == 1)
                    non_exist_raw = (element_ij == 0)

                    cond0 = certain_edge_raw and (edges_prob[0, i, j] < 0.05)  # j -> i
                    cond5 = non_exist_raw and (edges_prob[5, i, j] < 0.05)  # j x i

                # Bootstrap probability is less than 0.05
                if algorithm in ['PC', 'GES', 'CDNOD']:
                    if cond0 or cond1 or cond2 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'
                elif algorithm == 'FCI':
                    if cond0 or cond1 or cond2 or cond3 or cond4 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'
                else:
                    if cond0 or cond5:
                        boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(np.max(prob_ij)) + ')'

                # Bootstrap probability is greater than 0.95
                if (not certain_edge_raw) and (edges_prob[0, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = '->' + '(' + str(edges_prob[0, i, j]) + ')'
                elif (not non_exist_raw) and (edges_prob[5, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'Forbid' + '(' + str(edges_prob[5, i, j]) + ')'

                if algorithm in ['PC', 'GES', 'CDNOD','FCI']:
                    if (not uncertain_edge_raw) and (edges_prob[1, i, j] > 0.95):
                        boot_recommend[str(j) + '-' + str(i)] = '-' + '(' + str(edges_prob[1, i, j]) + ')'
                    elif (not bi_edge_raw) and (edges_prob[2, i, j] > 0.95):
                        boot_recommend[str(j) + '-' + str(i)] = '<->' + '(' + str(edges_prob[2, i, j]) + ')'

                    if algorithm == 'FCI':
                        if (not half_edge_raw) and (edges_prob[3, i, j] > 0.95):
                            boot_recommend[str(j) + '-' + str(i)] = 'o->' + '(' + str(edges_prob[3, i, j]) + ')'
                        elif (not none_edge_raw) and (edges_prob[4, i, j] > 0.95):
                            boot_recommend[str(j) + '-' + str(i)] = 'o-o' + '(' + str(edges_prob[4, i, j]) + ')'

    # Convert edges_prob to a dict
    boot_edges_prob = {'certain_edges': edges_prob[0,:,:],
                       'uncertain_edges': None,
                       'bi_edges': None,
                       'half_edges': None,
                       'non_edges': None,
                       'non_existence':edges_prob[5, :, :]}

    if algorithm in ['PC','GES','CDNOD','FCI']:
        boot_edges_prob['uncertain_edges'] = edges_prob[1,:,:]
        boot_edges_prob['bi_edges'] = edges_prob[2,:,:]
        if algorithm == 'FCI':
            boot_edges_prob['half_edges'] = edges_prob[3,:,:]
            boot_edges_prob['non_edges'] = edges_prob[4,:,:]

    return boot_recommend, boot_edges_prob

def bootstrap_recommend(raw_graph, boot_edges_prob):
    direct_prob_mat =  boot_edges_prob['certain_edges']
    high_prob_idx_direct = np.where(direct_prob_mat >= 0.9)
    high_prob_edges_direct = list(zip(high_prob_idx_direct[0], high_prob_idx_direct[1]))
    low_prob_idx_direct = np.where(direct_prob_mat <= 0.1)
    low_prob_edges_direct = list(zip(low_prob_idx_direct[0], low_prob_idx_direct[1]))
    middle_prob_idx_direct = np.where((direct_prob_mat < 0.9) & (direct_prob_mat > 0.1))
    middle_prob_edges_direct = list(zip(middle_prob_idx_direct[0], middle_prob_idx_direct[1]))

    undirect_prob_mat =  boot_edges_prob['uncertain_edges']
    high_prob_idx_undirect = np.where(undirect_prob_mat >= 0.9)
    high_prob_edges_undirect = list(zip(high_prob_idx_undirect[0], high_prob_idx_undirect[1]))
    middle_prob_idx_undirect = np.where((undirect_prob_mat < 0.9) & (undirect_prob_mat > 0.1))
    middle_prob_edges_undirect = list(zip(middle_prob_idx_undirect[0], middle_prob_idx_undirect[1]))
    
    middle_prob_edges = list(set(middle_prob_edges_direct+high_prob_edges_undirect+middle_prob_edges_undirect))
    
    print('middle_prob_edges',middle_prob_edges)

    bootstrap_check_dict = {
        'high_prob_edges':{
            'exist':[], # cannot be deleted
            'non-exist': [] # Add it and use it as a constraint in the next iteration
        },
        'low_prob_edges':{
            'exist':[], # delete it
            'non-exist': [] # correct and do not edit
        },
        'middle_prob_edges':{
            'exist':[], # Double-check by LLM, delete if it should not exist
            'non-exist': [] # Double-check by LLM, orientate if it should exist
        }
    }
    def exist_check(prob_edges, dict_key):
        for pair in prob_edges:
            if raw_graph[pair[0],pair[1]]==1 and raw_graph[pair[1],pair[0]]==-1:
                bootstrap_check_dict[dict_key]['exist'].append(pair)
            else:
                bootstrap_check_dict[dict_key]['non-exist'].append(pair)
    exist_check(high_prob_edges_direct, 'high_prob_edges')
    exist_check(low_prob_edges_direct, 'low_prob_edges')
    exist_check(middle_prob_edges, 'middle_prob_edges')

    return bootstrap_check_dict


def get_json(args, prompt):
        client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
        jsons = response.choices[0].message.content
        jsons_cleaned = jsons.replace('```json', '').replace('```', '').strip()
        try:
            jsons_cleaned = json.loads(jsons_cleaned)
        except:
            print('The returned LLM JSON file is wrong, try again')
            jsons_cleaned = get_json(args,prompt)
        return jsons_cleaned

def call_llm(args, prompt,prompt_type):
    cot_context = """
            Question and Answer templete:
            Here is the Question and Answer templete, you should learn and reference it when answering my following questions
            Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'], 
            your task is to double check these relationship between 'smoker' and 'cancer' from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  
            if the 'smoker' causes the 'cancer', the "result" is 'A'. If the 'cancer' causes the 'smoker', the "result" should be 'B'. If there is no relationship, the "result" is 'C'. If you are not sure, the result is 'D'
            'xray' and 'cancer'
            'pollution' and 'cancer'
            For each node pair, 
            Please note that Correlation doesn't mean Causation! For example ice cream sales increase in summer alongside higher rates of drowning, where both are influenced by warmer weather rather than one causing the other.
            Please note hidden confounders, for example a study finds a correlation between coffee consumption and heart disease, but fails to account for smoking, which influences both coffee habits and heart disease risk.
            Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
            Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

            Response me following the template below. Do not include anything else. explanations should only include one to two sentences.
            A or B or C or D: explanations ;
            Answer: 
            A: Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath.

            Question and Answer templete:
            Here is the Question and Answer templete, you should learn and reference it when answering my following questions
            Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'], 
            your task is to double check these relationship between 'xray' and 'cancer' from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  
            if the 'xray'' causes the 'cancer', the "result" is 'A'. If the 'cancer' causes the 'xray', the "result" should be 'B'. If there is no relationship, the "result" is 'C'. If you are not sure, the result is 'D'
            'pollution' and 'cancer'
            For each node pair, 
            Please note that Correlation doesn't mean Causation! For example ice cream sales increase in summer alongside higher rates of drowning, where both are influenced by warmer weather rather than one causing the other.
            Please note hidden confounders, for example a study finds a correlation between coffee consumption and heart disease, but fails to account for smoking, which influences both coffee habits and heart disease risk.
            Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
            Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

            Response me following the template below. Do not include anything else. explanations should only include one to two sentences.
            A or B or C or D: explanations ;
            Answer: 
            B: The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care.
            
            Question and Answer templete:
            Here is the Question and Answer templete, you should learn and reference it when answering my following questions
            Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'], 
            your task is to double check these relationship between 'pollution' and 'cancer' from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  
            if the 'pollution' causes the 'cancer', the "result" is 'A'. If the 'cancer' causes the 'pollution', the "result" should be 'B'. If there is no relationship, the "result" is 'C'. If you are not sure, the result is 'D'
            Please note that Correlation doesn't mean Causation! For example ice cream sales increase in summer alongside higher rates of drowning, where both are influenced by warmer weather rather than one causing the other.
            Please note hidden confounders, for example a study finds a correlation between coffee consumption and heart disease, but fails to account for smoking, which influences both coffee habits and heart disease risk.
            Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
            Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

            Response me following the template below. Do not include anything else. explanations should only include one to two sentences.
            A or B or C or D: explanations ;
            Answer: 
            A: The causal effect of pollution on cancer is that air pollution contains carcinogens (cancercausing substances) that may be absorbed into the body when inhaled and damage the DNA of cells. Therefore air pollution may cause cancer.
            """
    normal_context = """
    Response Example
    A: Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath.
    B: The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care.
    C: These two variables are independent because the height of a person does not influence or affect the color of a car they own or drive.
    D: The relationship between the amount of sleep a person gets and their performance in a job can be complex and is not definitively understood.
    """
    prompt += normal_context
    # initiate a client
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in Causal Discovery."}])
    if 'cot' in prompt_type:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": cot_context}])
    # get response          
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}])
    contents = response.choices[0].message.content
    try:
        result, explanation = contents.split(':')[0].strip().upper(), contents.split(':')[1].strip()
        llm_answer = {'result': result,
                      'explanation': explanation}
    except:
        print('The returned LLM evaluation response is wrong, try again')
        llm_answer = call_llm(args,prompt)
    return llm_answer

def call_llm_new(args, prompt, prompt_type):
    cot_context = """
                Question and Answer templete:
                Here is the Question and Answer templete, you should learn and reference it when answering my following questions
                Question: For a causal graph used to model relationship of various factors and outcomes related to cancer with the following nodes: ['Pollution', 'Cancer', 'Smoker', 'Xray', 'Dyspnoea'], 
                your task is to double check these relationships about node 'Cancer' from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  
                Firstly, determine the relationship between
                'smoker' and 'cancer'
                'xray' and 'cancer'
                'pollution' and 'cancer'
                For each node pair, if the left node causes the right node, the "result" is 'A'. If the right node causes the left node, the "result" should be 'B'. If there is no relationship, the "result" is 'C'. If you are not sure, the result is 'D'
                Please note that Correlation doesn't mean Causation! For example ice cream sales increase in summer alongside higher rates of drowning, where both are influenced by warmer weather rather than one causing the other.
                Please note hidden confounders, for example a study finds a correlation between coffee consumption and heart disease, but fails to account for smoking, which influences both coffee habits and heart disease risk.
                Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
                Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

                Response me following the template below. Do not include anything else. explanations should only include one to two sentences.
                ('smoker', 'cancer'): A or B or C or D: explanations ;
                ('xray' and 'cancer'): A or B or C or D: explanations ;
                ('pollution' and 'cancer') : A or B or C or D: explanations ;
                Answer: 
                ('smoker', 'cancer'): A: Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath.
                ('xray', 'cancer'): B: The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care.
                ('pollution', 'cancer') : A: The causal effect of pollution on cancer is that air pollution contains carcinogens (cancercausing substances) that may be absorbed into the body when inhaled and damage the DNA of cells. Therefore air pollution may cause cancer.
                """
    normal_context = """
    Response Example
    (smoker, cancer): A: Smoking introduces harmful substances into the respiratory system, leading to cellular damage and mutation, which significantly raises the likelihood of cancer development in the lungs or respiratory tract, subsequently impacting the occurrence of respiratory problems like shortness of breath.
    (xray, cancer): B: The causal effect of cancer on X-ray is that X-rays are often used to diagnose or detect cancer in different parts of the body, such as the bones, lungs, breasts, or kidneys123. Therefore, having cancer may increase the likelihood of getting an X-ray as part of the diagnostic process or follow-up care.
    (height, color): C: These two variables are independent because the height of a person does not influence or affect the color of a car they own or drive.
    (sleep, job performance): D: The relationship between the amount of sleep a person gets and their performance in a job can be complex and is not definitively understood.
    """
    prompt += normal_context
    # initiate a client
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in Causal Discovery."}])
    if 'cot' in prompt_type:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": cot_context}])
    # get response          
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}])
    contents = response.choices[0].message.content
    # parse response
    llm_answer = {}
    contents = contents.replace('\n', ';')
    lines = [line.strip() for line in contents.split(';') if line.strip()]
    for line in lines:
        try:
            pair, result, explanation = line.split(':')[0].strip(), line.split(':')[1].strip().upper(), line.split(':')[2].strip()
            llm_answer[pair] = {'result': result,
                                'explanation': explanation}
        except:
            print('The returned LLM evaluation response is wrong, try again')
            print(lines)
            llm_answer = call_llm_new(args, prompt, prompt_type)
    return llm_answer

def llm_evaluation(data, args, edges_dict, boot_edges_prob, bootstrap_check_dict, prompt_type, voting_num):
    """
    Here we let LLM double check the result of initial graph, and make edition (determine direction & delete edge)
    Provided Info:
    1. Structure of whole graph
    (ancestor decendant ... relationship?)
    2. Bootstrap Probability
    3. Pair relationship in the original graph
    Return: Result dicts with domain knowledge and explanation
    """
    direct_dict = {}
    forbid_dict = {}
    relation_text_dict, relation_text = edges_to_relationship(data, edges_dict, boot_edges_prob)
    #print('relation_text_dict:',relation_text_dict)
    #print('edges_dict: ', edges_dict)
    ### conbine uncertain edges
    uncertain_edges_exist = bootstrap_check_dict['middle_prob_edges']['exist']
    uncertain_edges_nonexist = bootstrap_check_dict['middle_prob_edges']['non-exist']
    combined_uncertain_edges = uncertain_edges_exist + uncertain_edges_nonexist
    # Remove duplicate tuples
    combined_uncertain_edges = list(set(tuple(sorted((i, j))) for (i, j) in combined_uncertain_edges))
    for idx_i, idx_j in combined_uncertain_edges:
        var_j = data.columns[idx_j]
        var_i = data.columns[idx_i]
        directed_exist_texts_j = ', '.join([text for text in relation_text_dict['certain_edges']+relation_text_dict['uncertain_edges'] if var_j in text])
        directed_exist_texts_i = ', '.join([text for text in relation_text_dict['certain_edges']+relation_text_dict['uncertain_edges'] if var_i in text])
        # Basic Prompt: No infos and ask relationships directly
        prompt_pruning = f"""
        Context: We want to carry out causal discovery analysis, considering these variables: {data.columns.tolist()}. 
        """
        # All Pairwise Relationships
        if 'all_relations' in prompt_type:
            prompt_pruning += f"""
            This is a context for your reference: 
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            {relation_text}, but it may not be correct. 
            """
        # Markov Blanket Context
        if 'markov_blanket' in prompt_type:
            prompt_pruning += f"""
            This is a context for your reference, and your answer cannot add cycles and colliders to these relationships
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            Edges of node {var_j}:
            {directed_exist_texts_j}
            Edges of node {var_i}:
            {directed_exist_texts_i}
            """
            # Relationships for related node
            # Extract tuples containing main node
            tuples_with_j = [t for t in edges_dict['uncertain_edges']+edges_dict['certain_edges'] if var_j in t]
            related_nodes_j = [item for t in tuples_with_j for item in t if item != var_j]
            tuples_with_i = [t for t in edges_dict['uncertain_edges']+edges_dict['certain_edges'] if var_i in t]
            related_nodes_i = [item for t in tuples_with_i for item in t if item != var_i]
            related_nodes_all = list(set(related_nodes_j+related_nodes_i))
            for node in related_nodes_all:
                directed_exist_texts_related = ', '.join([text for text in relation_text_dict['certain_edges'] if node in text])
                undirected_exist_texts_related = ', '.join([text for text in relation_text_dict['uncertain_edges'] if node in text])
                # print('directed_exist_texts_related',directed_exist_texts_related)
                # print('undirected_exist_texts_related',undirected_exist_texts_related)
                prompt_pruning += f"""
                Edges of node {node}:
                {directed_exist_texts_related} and {undirected_exist_texts_related}
                """

        prompt_pruning += f"""
        Question: your task is to double check these relationship between {var_j} and {var_i} from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  

        If {var_j} causes {var_i}, the "result" is 'A'. If {var_i} causes {var_j}, the "result" should be 'B'. If you are really sure there is no relationship, the "result" is 'C'. Otherwise the "result" is 'D'.
        Please note that Correlation doesn't mean Causation! For example ice cream sales increase in summer alongside higher rates of drowning, where both are influenced by warmer weather rather than one causing the other.
        Please note hidden confounders, for example a study finds a correlation between coffee consumption and heart disease, but fails to account for smoking, which influences both coffee habits and heart disease risk.
        Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
        Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.

        Response me following the template below. Do not include anything else. explanations should only include one to two sentences
        A or B or C or D: explanations 
            """
        #print('prompt_pruning:', prompt_pruning)
        
        ### Ask with Voting ###
        if voting_num == 1:
            llm_answer = call_llm(args, prompt_pruning, prompt_type)
        else:
            llm_answer_merge = []
            llm_answer = {}
            for i_vote in range(voting_num):
                llm_answer_i = call_llm(args, prompt_pruning, prompt_type)
                llm_answer_merge.append(llm_answer_i)
            result_list = [single_vote['result'] for single_vote in llm_answer_merge]
            print('result_list',result_list)
            explanation_list = [single_vote['explanation'] for single_vote in llm_answer_merge]
            majority_result = Counter(result_list).most_common(1)[0][0]
            majority_explanation = explanation_list[result_list.index(majority_result)]
            llm_answer={'result': majority_result,
                        'explanation': majority_explanation}
        ########### end of voting #################### 
        result, explanation = llm_answer['result'], llm_answer['explanation']
        print((var_j, var_i), result)
        print('response: ',llm_answer)
        if result == 'A':
            if (var_j, var_i) not in edges_dict['certain_edges']:
                direct_dict[(idx_j, idx_i)] = ((var_j, var_i), explanation)
        elif result == 'B':
            if (var_i, var_j) not in edges_dict['certain_edges']:
                direct_dict[(idx_i, idx_j)] = ((var_i, var_j), explanation)
        elif result == 'C':
            forbid_dict[(idx_j, idx_i)] = ((var_j, var_i), explanation)
    return direct_dict, forbid_dict

def llm_evaluation_new(data, args, edges_dict, boot_edges_prob, bootstrap_check_dict, prompt_type, vote_num=3):
    """
    Here we let LLM double check the result of initial graph, and make edition (determine direction & delete edge)
    Provided Info:
    1. Structure of whole graph
    (ancestor decendant ... relationship?)
    2. Bootstrap Probability
    3. Pair relationship in the original graph
    Return: Result dicts with domain knowledge and explanation
    """

    ### conbine uncertain edges
    uncertain_edges_exist = bootstrap_check_dict['middle_prob_edges']['exist']
    uncertain_edges_nonexist = bootstrap_check_dict['middle_prob_edges']['non-exist']
    combined_uncertain_edges = uncertain_edges_exist + uncertain_edges_nonexist
    # Remove duplicate tuples
    combined_uncertain_edges = list(set(tuple(sorted((i, j))) for (i, j) in combined_uncertain_edges))

    # Convert edges into node group
    # Initialize an empty dictionary
    grouped_dict = {}
    # Iterate over the list of tuples
    for idx_i, idx_j in combined_uncertain_edges:
        # Convert the first element to a string for the key
        key = data.columns[idx_i]
        # Append the tuple to the corresponding key in the dictionary
        if key not in grouped_dict:
            grouped_dict[key] = []  # Create a new list if the key doesn't exist
        grouped_dict[key].append((data.columns[idx_i], data.columns[idx_j]))
    print('grouped_dict',grouped_dict)

    direct_dict = {}
    forbid_dict = {}
    ##############iteration##################
    def check_node_relationship(main_node):
        # Relationships for main node
        relation_text_dict, relation_text = edges_to_relationship(data, edges_dict, boot_edges_prob)
        directed_exist_texts_mainnode = ', '.join([text for text in relation_text_dict['certain_edges'] if main_node in text])
        undirected_exist_texts_mainnode = ', '.join([text for text in relation_text_dict['uncertain_edges'] if main_node in text])
        # print('directed_exist_texts_mainnode',directed_exist_texts_mainnode)
        # print('undirected_exist_texts_mainnode',undirected_exist_texts_mainnode)
        
        related_pairs = grouped_dict[main_node]
        # Basic Prompt: No infos and ask relationships directly
        prompt_pruning = f"""
        **Context**: 
        We want to carry out causal discovery analysis, considering these variables: {data.columns.tolist()}. 
        """
        # All Pairwise Relationships
        if 'all_relations' in prompt_type:
            prompt_pruning += f"""
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            {relation_text}
            According to the results shown above, it has been determined that {directed_exist_texts_mainnode} and {undirected_exist_texts_mainnode}, but it may not be correct. 
            """
        # Markov Blanket Context
        if 'markov_blanket' in prompt_type:
            prompt_pruning += f"""
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            Edges of node {main_node}:
            {directed_exist_texts_mainnode} and {undirected_exist_texts_mainnode}
            """
            # Relationships for related node
            # Extract tuples containing main node
            tuples_with_mainnode = [t for t in edges_dict['uncertain_edges']+edges_dict['certain_edges'] if main_node in t]
            related_nodes = [item for t in tuples_with_mainnode for item in t if item != main_node]
            for node in related_nodes:
                directed_exist_texts_related = ', '.join([text for text in relation_text_dict['certain_edges'] if node in text])
                undirected_exist_texts_related = ', '.join([text for text in relation_text_dict['uncertain_edges'] if node in text])
                # print('directed_exist_texts_related',directed_exist_texts_related)
                # print('undirected_exist_texts_related',undirected_exist_texts_related)
                prompt_pruning += f"""
                Edges of node {node}:
                {directed_exist_texts_related} and {undirected_exist_texts_related}
                """

        prompt_pruning += f"""
        **Important Considerations**:

        1. **Correlation vs. Causation**:
        - Remember that statistical correlation does not imply causation. A detected association between variables may not indicate a causal link.
        - Base your reasoning on domain knowledge and logical inference rather than statistical correlations.

        2. **Direction of Causation**:
        - The direction of causation is crucial. Ensure that the proposed causal direction is logical and consistent with established domain knowledge.
        - Avoid assuming causation without proper justification.

        **Your Task**:
        Your task is to double check these causal relationships about node {main_node} from a domain knowledge perspective and determine whether this statistically suggested hypothesis is plausible in the context of the domain.  

        Firstly, determine the causal relationship between
        """
        for node_i, node_j in related_pairs:
            prompt_pruning += f" {node_i} and {node_j},"
        prompt_pruning += f"""\n
        **Options**
        - For each node pair, if I intervene the left node, it directly causes changes in the right node, the "result" is 'A'. 
        - If I intervene the right node, it directly causes changes in the left node, the "result" should be 'B'. 
        - If you are pretty sure there is no relationship, the "result" is 'C'. 
        - If you do not know established evidence, the "result" is 'D'.
        
        **Do Not**:
        - Conflate correlation with causation.
        - Include indirect relationships
        - Include circular relationships (e.g., "A causes B because B causes A")

        Secondly, please provide an explanation of your result, leveraging your expert knowledge on the causal relationship between the left node and the right node, please use only one to two sentences. 
        Your response should consider the relevant factors and provide a reasoned explanation based on your understanding of the domain.
        - **Avoid**:
        - Circular reasoning (e.g., "A causes B because B causes A").
        - Vague explanations lacking domain-specific details.
        - Use statistical terms such as "correlates with" or "is associated with" in your explanation.
        - Base your reasoning on data patterns or algorithm outputs.

        **Response Format**:
        Response me following the template below. Do not include anything else. explanations should only include one to two sentences. \n
        Please seperate your answers for each pair with semicolon ;
        """
        for node_i, node_j in related_pairs:
            prompt_pruning += f"""
            ({node_i}, {node_j}): A or B or C or D: explanations ; \n
        """
        #print('prompt_pruning: ',prompt_pruning)
        
        ### Ask with Voting ###
        if vote_num == 1:
            llm_answer = call_llm_new(args, prompt_pruning, prompt_type)
        else:
            llm_answer_merge = []
            llm_answer = {}
            for i_vote in range(vote_num):
                llm_answer_i = call_llm_new(args, prompt_pruning, prompt_type)
                llm_answer_merge.append(llm_answer_i)
            merged_dict ={}
            for d in llm_answer_merge:
                for key, value in d.items():
                    merged_dict.setdefault(key,[]).append(value)
            for pair_i in merged_dict.keys():
                result_list = [single_vote['result'] for single_vote in merged_dict[pair_i]]
                print('result_list',result_list)
                explanation_list = [single_vote['explanation'] for single_vote in merged_dict[pair_i]]
                majority_result = Counter(result_list).most_common(1)[0][0]
                majority_explanation = explanation_list[result_list.index(majority_result)]
                llm_answer[pair_i]={'result': majority_result,
                                    'explanation': majority_explanation}
        ########### end of voting #################### 
        print('response: ',llm_answer)
        # Update revised graph and edge dict
        for pair in llm_answer.keys():
            var_j, var_i = tuple(item.strip().strip('"').strip("'") for item in pair.strip('()').split(','))
            idx_j, idx_i = data.columns.str.lower().get_loc(var_j.lower()), data.columns.str.lower().get_loc(var_i.lower())
            if llm_answer[pair]['result'] == 'A':
                #if (var_j, var_i) not in edges_dict['certain_edges']:
                if True:
                    direct_dict[(idx_j, idx_i)] = ((var_j, var_i), llm_answer[pair]['explanation'])
                    edges_dict['certain_edges'].append((var_j, var_i))
            elif llm_answer[pair]['result'] == 'B':
                #if (var_i, var_j) not in edges_dict['certain_edges']:
                if True:
                    direct_dict[(idx_i, idx_j)] = ((var_i, var_j), llm_answer[pair]['explanation'])
                    edges_dict['certain_edges'].append((var_i, var_j))
            elif llm_answer[pair]['result'] == 'C':
                forbid_dict[(idx_j, idx_i)] = ((var_j, var_i), llm_answer[pair]['explanation'])
                if (var_j, var_i) in edges_dict['certain_edges']:
                    edges_dict['certain_edges'].remove((var_j, var_i))
                if (var_i, var_j) in edges_dict['certain_edges']:
                    edges_dict['certain_edges'].remove((var_i, var_j))

    for main_node in  grouped_dict.keys():
        print(f'edges_dict for {main_node}: ')
        print('directed edges:',edges_dict['certain_edges'])
        print('undirected edges:',edges_dict['uncertain_edges'])
        check_node_relationship(main_node)    
    #########################
    return direct_dict, forbid_dict

def kci_pruning(data, revised_graph):
    from causallearn.utils.cit import CIT
    kci_forbid_dict = {}
    test = CIT(data.to_numpy(), 'kci') # construct a CIT instance with data and method name
    for idx_1 in range(len(data.columns)-1):
        for idx_2 in range(idx_1+1, len(data.columns)):
            p_value = test(idx_1, idx_2)
            if p_value > 0.05 and revised_graph[idx_1, idx_2]!=0:
                print(f'({data.columns[idx_1]}, {data.columns[idx_2]}): {p_value}')
                kci_forbid_dict[idx_1,idx_2] = p_value
    return kci_forbid_dict

def edges_to_relationship(data, edges_dict, boot_edges_prob=None):
    '''
    :param data: Pandas DataFrame format.
    :param graph: An adjacent matrix in Numpy Ndarray format - Matrix[i,j] = 1 indicates j->i
    :param boot_probability: A matrix in Numpy Ndarray format
                             recording bootstrap probability of directed edges,
                             e.g., Matrix[i,j] records probability of existence of edge i -> j.
    :param edges_dict: A dict containing lists of all types of relationships
    :param boot_prob: A dict containing probability matrix of all types of edges
    :return: A dictionary of lists describing different edge types' relationships for each node pairs and corresponding bootstrap probability.
    '''
    relation_dict = {
            'certain_edges': 'causes',
            'uncertain_edges': 'has undirected relationship with',
            'bi_edges': 'has hidden confounder with',
            'half_edges': 'is not a descendant of',
            'none_edges': 'has no D-seperation set with'
        }
    result_dict = {
            'certain_edges': [],
            'uncertain_edges': [],
            'bi_edges': [],
            'half_edges': [],
            'none_edges': []
        }
    summary_dict = {
            'certain_edges': 'These variable pairs have certain directed edge between them: \n',
            'uncertain_edges': 'These variable pairs have undirected relationship between them: \n',
            'bi_edges': 'These variable pairs have hidden confounders between them: \n',
            'half_edges': 'These variable pairs have non-descendant relationship between them: \n',
            'none_edges': 'These variable pairs have no D-seperation between them: \n'
        }
    for edge_type in relation_dict.keys():
        edges_list = edges_dict[edge_type]
        for edges in edges_list:
            if boot_edges_prob is not None:
                idx_j = data.columns.get_loc(edges[0])
                idx_i = data.columns.get_loc(edges[1])
                prob = boot_edges_prob[edge_type][idx_i, idx_j]
                result_dict[edge_type].append(f'{edges[0]} {relation_dict[edge_type]} {edges[1]} with bootstrap probability {prob}')
            else:
                result_dict[edge_type].append(f'{edges[0]} {relation_dict[edge_type]} {edges[1]}')
    
    filtered_result_dict = {key: value for key, value in result_dict.items() if value}

    relation_text = ""
    for key in filtered_result_dict:
        relation_text += f"{summary_dict[key]}"
        for pair_relation in filtered_result_dict[key]:
            relation_text += f'{pair_relation}, '
        relation_text += '\n'
    
    return filtered_result_dict, relation_text


