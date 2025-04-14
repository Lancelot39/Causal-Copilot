import numpy as np
from sympy.stats.rv import probability
import ast
import json 
from openai import OpenAI
from postprocess.visualization import Visualization
from collections import Counter
import networkx as nx
from pydantic import BaseModel
import networkx as nx
from pydantic import BaseModel


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
        boot_sample.index = pd.RangeIndex(start=0, stop=len(boot_sample))

    # Get the algorithm function from wrappers
    algo_func = getattr(wrappers, algorithm)

    # Execute the algorithm with data and hyperparameters
    converted_graph, info, raw_result = algo_func(hyperparameters).fit(boot_sample)

    boot_graph = converted_graph

    return boot_graph

def bootstrap_probability(boot_result, algorithm):

    m = boot_result.shape[1]

    certain_edges_prob = np.zeros((m,m))  # -> and use converted graph
    uncertain_edges_prob = np.zeros((m,m))  # -
    bi_edges_prob = np.zeros((m,m))  # <->
    half_certain_edges_prob = np.zeros((m,m))  # o->
    half_uncertain_edges_prob = np.zeros((m,m))  # o-
    none_edges_prob = np.zeros((m,m))  # o-o
    none_exist_prob = np.zeros((m,m)) # did not exist edge

    for i in range(m):
        for j in range(m):
            if i == j: continue
            else:
                elements_ij = boot_result[:, i, j]
                elements_ji = boot_result[:, j, i]

                # i x j
                none_exist_prob[i, j] = np.mean((elements_ij == 0) & (elements_ji == 0))
                # j -> i
                certain_edges_prob[i, j] = np.mean((elements_ij == 1) & (elements_ji == 0))
                # i - j
                uncertain_edges_prob[i, j] = np.mean((elements_ij == 2) | (elements_ji == 2))
                # i <-> j
                bi_edges_prob[i, j] = np.mean((elements_ij == 3) | (elements_ji == 3))
                # j o-> i
                half_certain_edges_prob[i, j] = np.mean((elements_ij == 4) & (elements_ji == 0))
                # j o- i
                half_uncertain_edges_prob[i, j] = np.mean((elements_ij == 5) & (elements_ji == 0))
                # i o-o j
                none_edges_prob[i, j] = np.mean((elements_ij == 6) | (elements_ji == 6))


    edges_prob = np.stack((certain_edges_prob, uncertain_edges_prob, bi_edges_prob, half_certain_edges_prob, half_uncertain_edges_prob, none_edges_prob, none_exist_prob), axis=0)

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
    # 0. none_exist_prob: x
    # 1. certain_edges_prob: ->
    # 2. uncertain_edges_prob: -
    # 3. bi_edges_prob: <->
    # 4. half_certain_edges_prob: o->
    # 5. half_uncertain_edges_prob: o-
    # 6. none_edges_prob: o-o

    edges_prob = bootstrap_probability(boot_effect_save_array, algorithm)

    recommend = ['->', '-', '<->', 'o->', 'o-', 'o-o','Forbid']

    boot_recommend = {}

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            else:
                element_ij = raw_graph[i, j]
                element_ji = raw_graph[j, i]
                prob_ij = edges_prob[:, i, j]

                non_exist_raw = (element_ij == 0 and element_ji == 0)
                certain_edge_raw = (element_ij == 1 and element_ji == 0)
                uncertain_edge_raw = (element_ij == 2 or element_ji == 2)
                bi_edge_raw = (element_ij == 3 or element_ji == 3)
                half_certain_edges_raw = (element_ij == 4 and element_ji == 0)
                half_uncertain_edges_raw = (element_ij == 5 and element_ji == 0)
                none_edge_raw = (element_ij == 6 or element_ji == 6)

                cond0 = certain_edge_raw and (edges_prob[0, i, j] < 0.05)  # j -> i
                cond1 = uncertain_edge_raw and (edges_prob[1, i, j] < 0.05)  # j - i
                cond2 = bi_edge_raw and (edges_prob[2, i, j] < 0.05)  # j <-> i
                cond3 = half_certain_edges_raw and (edges_prob[3, i, j] < 0.05)  # j o-> i
                cond4 = half_uncertain_edges_raw and (edges_prob[4, i, j] < 0.05)  # j o- i
                cond5 = none_edge_raw and (edges_prob[5, i, j] < 0.05)  # j o-o i
                cond6 = non_exist_raw and (edges_prob[6, i, j] < 0.05)  # j x i

                # Bootstrap probability is less than 0.05
                if cond0 or cond1 or cond2 or cond3 or cond4 or cond5 or cond6:
                    boot_recommend[str(j) + '-' + str(i)] = recommend[np.argmax(prob_ij)] + '(' + str(
                        np.max(prob_ij)) + ')'

                # Bootstrap probability is greater than 0.95
                if (not certain_edge_raw) and (edges_prob[0, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = '->' + '(' + str(edges_prob[0, i, j]) + ')'
                elif (not uncertain_edge_raw) and (edges_prob[1, i, j] > 0.95):
                        boot_recommend[str(j) + '-' + str(i)] = '-' + '(' + str(edges_prob[1, i, j]) + ')'
                elif (not bi_edge_raw) and (edges_prob[2, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = '<->' + '(' + str(edges_prob[2, i, j]) + ')'
                elif (not half_certain_edges_raw) and (edges_prob[3, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'o->' + '(' + str(edges_prob[3, i, j]) + ')'
                elif (not half_uncertain_edges_raw) and (edges_prob[4, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'o-' + '(' + str(edges_prob[4, i, j]) + ')'
                elif (not none_edge_raw) and (edges_prob[5, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'o-o' + '(' + str(edges_prob[5, i, j]) + ')'
                elif (not non_exist_raw) and (edges_prob[6, i, j] > 0.95):
                    boot_recommend[str(j) + '-' + str(i)] = 'Forbid' + '(' + str(edges_prob[6, i, j]) + ')'

    # Convert edges_prob to a dict
    boot_edges_prob = {'certain_edges': edges_prob[0,:,:],
                       'uncertain_edges': edges_prob[1,:,:],
                       'bi_edges': edges_prob[2,:,:],
                       'half_certain_edges': edges_prob[3,:,:], # o->
                       'half_uncertain_edges': edges_prob[4,:,:], # o-
                       'none_edges': edges_prob[5,:,:],
                       'none_existence':edges_prob[6, :, :]}


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
    
    middle_prob_edges = list(set(low_prob_edges_direct+high_prob_edges_undirect+\
                                 middle_prob_edges_direct+middle_prob_edges_undirect))
    
    #print('middle_prob_edges',middle_prob_edges)

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
            if raw_graph[pair[0],pair[1]] != 0:
                bootstrap_check_dict[dict_key]['exist'].append(pair)
            else:
                bootstrap_check_dict[dict_key]['non-exist'].append(pair)
    exist_check(high_prob_edges_direct, 'high_prob_edges')
    exist_check(low_prob_edges_direct, 'low_prob_edges')
    exist_check(middle_prob_edges, 'middle_prob_edges')

    return bootstrap_check_dict


def get_json(args, prompt):
        client = OpenAI(api_key=args.apikey)
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

def call_llm_new(args, prompt, prompt_type):
    with open('postprocess/context/COT_prompt.txt', 'r') as file:
        cot_context = file.read()
    
    # initiate a client
    client = OpenAI()
    client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in Causal Discovery."}])
    if 'cot' in prompt_type:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": cot_context}])

    # get response          
    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}])
    contents = response.choices[0].message.content
    # parse response
    llm_answer = {}
    lines = contents.split('\n')
    lines = [line for line in lines if line.startswith('(')]
    for line in lines:
        try:
            pair, result, explanation = line.split(':')[0].strip(), line.split(':')[1].strip().upper(), line.split(':')[2].strip()
            llm_answer[pair] = {'result': result,
                                'explanation': explanation}
        except Exception as e:
            print('The returned LLM evaluation response is wrong, try again')
            print(lines)
            print(e)
            llm_answer = call_llm_new(args, prompt, prompt_type)
    return llm_answer

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
    #print('grouped_dict',grouped_dict)

    direct_dict = {}
    forbid_dict = {}
    ##############iteration##################
    def check_node_relationship(main_node):
        # Relationships for main node
        relation_text_dict, relation_text = edges_to_relationship(data, edges_dict, boot_edges_prob)
        try:
            directed_exist_texts_mainnode = ', '.join([text for text in relation_text_dict['certain_edges'] if main_node in text])
        except:
            directed_exist_texts_mainnode = 'None'
        try:
            undirected_exist_texts_mainnode = ', '.join([text for text in relation_text_dict['uncertain_edges'] if main_node in text])
        except:
            undirected_exist_texts_mainnode = 'None'
        
        related_pairs = grouped_dict[main_node]
        # All Pairwise Relationships
        if 'all_relation' in prompt_type:
            relationship = f"""
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            {relation_text}
            According to the results shown above, it has been determined that {directed_exist_texts_mainnode} and {undirected_exist_texts_mainnode}, but it may not be correct. 
            """
        # Markov Blanket Context
        if 'markov_blanket' in prompt_type:
            relationship = f"""
            We have conducted the statistical causal discovery algorithm to find the following causal relationships from a statistical perspective:
            Edges of node {main_node}:
            {directed_exist_texts_mainnode} and {undirected_exist_texts_mainnode}
            """
            # Relationships for related node
            # Extract tuples containing main node
            certain_edges = edges_dict.get('certain_edges', [])
            uncertain_edges = edges_dict.get('uncertain_edges', [])
            tuples_with_mainnode = [t for t in certain_edges + uncertain_edges if main_node in t]
            related_nodes = [item for t in tuples_with_mainnode for item in t if item != main_node]
            for node in related_nodes:
                if 'certain_edges' in relation_text_dict:
                    directed_exist_texts_related = ', '.join([text for text in relation_text_dict['certain_edges'] if node in text])
                else:
                    directed_exist_texts_related = 'None'
                if 'uncertain_edges' in relation_text_dict:
                    undirected_exist_texts_related = ', '.join([text for text in relation_text_dict['uncertain_edges'] if node in text])
                else:
                    undirected_exist_texts_related = 'None'
                relationship += f"""
                Edges of node {node}:
                {directed_exist_texts_related} and {undirected_exist_texts_related}
                """
        # Basic Prompt: No infos and ask relationships directly
        else: 
            relationship = ''

        task = f"Firstly, determine the causal relationship between\n"
        for node_i, node_j in related_pairs:
            task += f" {node_i} and {node_j},"
       
        format =  ""
        for node_i, node_j in related_pairs:
            format += f"({node_i}, {node_j}): A or B or C or D: explanations ; \n"
        replacement = {              
            "[COLUMNS]": ', '.join([col for col in data.columns]),
            "[MAIN_NODE]": main_node,
            "[RELATIONSHIP]": relationship,
            "[TASK]": task
            }
        with open('postprocess/context/pruning_prompt.txt', 'r') as file:
            prompt_pruning = file.read()
            for placeholder, value in replacement.items():
                prompt_pruning = prompt_pruning.replace(placeholder, value)
          
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
                #print('result_list',result_list)
                explanation_list = [single_vote['explanation'] for single_vote in merged_dict[pair_i]]
                majority_result = Counter(result_list).most_common(1)[0][0]
                majority_explanation = explanation_list[result_list.index(majority_result)]
                llm_answer[pair_i]={'result': majority_result,
                                    'explanation': majority_explanation}
        ########### end of voting #################### 
        #print('response: ',llm_answer)
        # Update revised graph and edge dict
        for pair in llm_answer.keys():
            try:
                var_j, var_i = tuple(item.strip().strip('"').strip("'") for item in pair.strip('()').split(','))
                idx_j, idx_i = data.columns.str.lower().get_loc(var_j.lower()), data.columns.str.lower().get_loc(var_i.lower())
                if llm_answer[pair]['result'] == 'A':
                    #if (var_j, var_i) not in edges_dict['certain_edges']:
                    direct_dict[(idx_j, idx_i)] = ((var_j, var_i), llm_answer[pair]['explanation'])
                    edges_dict['certain_edges'].append((var_j, var_i))
                elif llm_answer[pair]['result'] == 'B':
                    #if (var_i, var_j) not in edges_dict['certain_edges']:
                    direct_dict[(idx_i, idx_j)] = ((var_i, var_j), llm_answer[pair]['explanation'])
                    edges_dict['certain_edges'].append((var_i, var_j))
                elif llm_answer[pair]['result'] == 'C':
                    forbid_dict[(idx_j, idx_i)] = ((var_j, var_i), llm_answer[pair]['explanation'])
                    if (var_j, var_i) in edges_dict['certain_edges']:
                        edges_dict['certain_edges'].remove((var_j, var_i))
                    if (var_i, var_j) in edges_dict['certain_edges']:
                        edges_dict['certain_edges'].remove((var_i, var_j))
            except:
                continue

    for main_node in  grouped_dict.keys():
        # print(f'edges_dict for {main_node}: ')
        # print('directed edges:',edges_dict['certain_edges'])
        # print('undirected edges:',edges_dict['uncertain_edges'])
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
            'half_certain_edges': 'is not a descendant of',
            'half_uncertain_edges': 'is not a descendant of',
            'none_edges': 'has no D-seperation set with'
        }
    result_dict = {
            'certain_edges': [],
            'uncertain_edges': [],
            'bi_edges': [],
            'half_certain_edges': [],
            'half_uncertain_edges': [],
            'none_edges': []
        }
    summary_dict = {
            'certain_edges': 'These variable pairs have certain directed edge between them: \n',
            'uncertain_edges': 'These variable pairs have undirected relationship between them: \n',
            'bi_edges': 'These variable pairs have hidden confounders between them: \n',
            'half_certain_edges': 'These variable pairs have non-descendant directed relationship between them: \n',
            'half_uncertain_edges': 'These variable pairs have non-descendant undirected relationship between them: \n',
            'none_edges': 'These variable pairs have no D-seperation between them: \n'
        }
    
    for edge_type in relation_dict.keys():
        edges_list = edges_dict[edge_type]
        for edges in edges_list:
            if boot_edges_prob is not None:
                try:
                    idx_j = data.columns.str.lower().get_loc(edges[0].lower())
                    idx_i = data.columns.str.lower().get_loc(edges[1].lower())
                except:
                    try:
                        idx_j = data.columns.str.lower().get_loc(edges[0].lower().replace('_', ' '))
                        idx_i = data.columns.str.lower().get_loc(edges[1].lower().replace('_', ' '))
                    except:
                        continue  
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


def LLM_remove_cycles(args, message):
    client = OpenAI(api_key=args.apikey)
    class VarList(BaseModel):
        nodes: list[str]
    
    completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini-2024-07-18",
    messages=[
        {"role": "system", "content": f"You are a helpful assistant, the following relationships form a cycle, please choose an egde to remove this cycle, and save nodes of this edge in a list. "},
        {"role": "user", "content": message},
    ],
    response_format=VarList,
    )

    node_list = completion.choices[0].message.parsed.nodes
    return node_list

def check_cycle(args, data, graph):
    columns = data.columns
    graph_copy = graph.copy()
    # Set symmetric positions to 0
    ones_indices = np.where(graph == 1)
    symmetric_indices = (ones_indices[1], ones_indices[0])  # swap row and column indices
    graph_copy[symmetric_indices] = 0
    G = nx.from_numpy_array(graph_copy, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, dict(enumerate(columns)))

    # Check for cycles in the directed graph
    acyclic = nx.is_directed_acyclic_graph(G)

    # Output the result
    if acyclic:
        print("The graph is acyclic (no cycles).")
    else:
        print("The graph has cycles.")
        cycles = list(nx.simple_cycles(G))
        for cycle in cycles:
            print(cycle)
            prompt = " -> ".join(f"{n}" for n in cycle)
            prompt +=  f" -> {cycle[0]}"
            print(prompt)
            remove_nodes = LLM_remove_cycles(args, prompt)
            print('remove_nodes',remove_nodes)
            ind_i = columns.str.lower().get_loc(remove_nodes[0].lower())
            ind_j = columns.str.lower().get_loc(remove_nodes[1].lower())
            graph[ind_i, ind_j] = graph[ind_j, ind_i] = 0
    
    return graph
