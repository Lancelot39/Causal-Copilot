import sys
sys.path.insert(0, 'causal-learn')

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag

def array2cpdag(adj_array, node_names):
    # for methods return cpdag
    g = GeneralGraph([])
    node_map = {}
    num_nodes = adj_array.shape[0]
    
    # Create nodes
    for i in range(num_nodes):
        node_name = node_names[i]
        node_map[node_name] = GraphNode(node_name)
        g.add_node(node_map[node_name])
    
    # Create edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_array[i, j] == 1:
                node1 = node_map[node_names[i]]
                node2 = node_map[node_names[j]]
                edge = Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW)
                g.add_edge(edge)
    
    truth_cpdag = dag2cpdag(g)
    return truth_cpdag


class Judge(object):
    def __init__(self, global_state, args):
        self.global_state = global_state
        self.args = args

    def quality_judge(self, data, full_graph, algorithm, hyperparameters, knowledge_docs, boot_num, prompt_type, voting_num):
        '''
        :param data: Given Tabular Data in Pandas DataFrame format
        :param full_graph: An adjacent matrix in Numpy Ndarray format -
                           causal graph using the full dataset - Matrix[i,j] = 1 indicates j->i
        :param algorithm: String representing the algorithm name
        :param hyperparameters: Dictionary of hyperparameter names and values
        :param knowledge_docs: A doc containing all necessary domain knowledge information from GPT-4.
        :param boot_num: Number of bootstrap iterations..
        :return: obvious errors in causal analysis results,
                 bootstrap probability of directed edges,
                 revised causal graph based on errors.
        '''

        from postprocess.judge_functions import bootstrap, bootstrap_recommend, llm_evaluation_new, kci_pruning, check_cycle

        # Statistics Perspective: Bootstrapping to get probability of edges using the selected algorithm.
        edge_recom, boot_probability = bootstrap(data=data, full_graph=full_graph, algorithm=algorithm, hyperparameters=hyperparameters,
                                                  boot_num=boot_num, ts=False, parallel=self.args.parallel)
        #print("Edge Recommendations from Bootstrap method: ", edge_recom)
        #print("Bootstrap Probability: ", boot_probability)

        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        bk = BackgroundKnowledge()
        ############Edge Pruning with Bootstrap############
        print('Bootstrap Pruning Decisioning...')
        revised_graph = full_graph.copy()
        bootstrap_check_dict = bootstrap_recommend(full_graph, boot_probability)
        #print('bootstrap_check_dict: ',bootstrap_check_dict)
        # add non-exist edges with high prob
        if bootstrap_check_dict['high_prob_edges']['non-exist'] != []:
            for idx_i, idx_j in bootstrap_check_dict['high_prob_edges']['non-exist']:
                  revised_graph[idx_i, idx_j] = 1
                  revised_graph[idx_j, idx_i] = 0
                  node_pattern1 = data.columns[idx_j]
                  node_pattern2 = data.columns[idx_i]
                  bk.add_required_by_pattern(node_pattern1, node_pattern2)
        # delete exist edges with low prob
        if bootstrap_check_dict['low_prob_edges']['exist'] != []:
            for idx_i, idx_j in bootstrap_check_dict['low_prob_edges']['exist']:
                  revised_graph[idx_i, idx_j] = 0
                  revised_graph[idx_j, idx_i] = 0
                  node_pattern1 = data.columns[idx_j]
                  node_pattern2 = data.columns[idx_i]
                  bk.add_forbidden_by_pattern(node_pattern1, node_pattern2)
        #####old version#####
        fixed_pairs = []
        bootstrap_pruning_record = []
        # for k in edge_recom.keys():
        #     (i, j) = tuple(map(int, k.split('-')))
        #     edge = edge_recom[k].split('(')[0]
        #     prob = float(edge_recom[k].split('(')[1].strip(')'))
        #     # Change edges according to recommendation if bootstrap probability>0.95
        #     if prob > 0.95:
        #         fixed_pairs.append((i,j))
        #         fixed_pairs.append((j,i))
        #         text = f'{data.columns[j]} {edge} {data.columns[i]}'
        #         bootstrap_pruning_record.append(text)
        #         if edge == '->':
        #             revised_graph[i, j] = 1
        #             revised_graph[j, i] = -1
        #         elif edge == '-':
        #             revised_graph[i, j] = revised_graph[j, i] = -1
        #         elif edge == '<->':
        #             revised_graph[i, j] = revised_graph[j, i] = 1
        #         elif edge == 'o->':
        #             revised_graph[i, j] = 1
        #             revised_graph[j, i] = 2
        #         elif edge == 'o-o':
        #             revised_graph[i, j] = revised_graph[j, i] = 2
        #         else:
        #             revised_graph[i, j] = revised_graph[j, i] = 0
        ########################

        ############ Edge Pruning with LLM ############
        from postprocess.visualization import convert_to_edges
        revised_edges_dict = convert_to_edges(self.global_state.algorithm.selected_algorithm, self.global_state.user_data.processed_data.columns, revised_graph)
        direct_dict, forbid_dict = llm_evaluation_new(data, self.args, revised_edges_dict, 
                                                      self.global_state.results.bootstrap_probability, bootstrap_check_dict, 
                                                      prompt_type, voting_num)
        llm_pruning_record={
            'direct_record': direct_dict,
            'forbid_record': forbid_dict
        }
        
        ####construct prior knowledge and revise graph according to LLM#########
        for direct_pair in direct_dict:
            j, i = direct_pair[0], direct_pair[1]
            if (revised_graph[i, j]!=1 or revised_graph[j, i]!=0) and (i, j) not in fixed_pairs:
                revised_graph[i, j] = 1
                revised_graph[j, i] = 0
                node_pattern1 = data.columns[j]
                node_pattern2 = data.columns[i]
                bk.add_required_by_pattern(node_pattern1, node_pattern2)
        for forbid_pair in forbid_dict:
            j, i = forbid_pair[0], forbid_pair[1]
            node_pattern1 = data.columns[j]
            node_pattern2 = data.columns[i]
            if (revised_graph[i, j]!=0 or revised_graph[j, i]!=0) and (i, j) not in fixed_pairs:
                bk.add_forbidden_by_pattern(node_pattern1, node_pattern2)
                revised_graph[i, j] = revised_graph[j, i] = 0
        ########################

        ############ Edge Pruning with KCI ############
        kci_forbid_dict = kci_pruning(self.global_state.user_data.processed_data, revised_graph)
        print('kci_forbid_dict', kci_forbid_dict)
        for idx_j, idx_i in kci_forbid_dict.keys():
            revised_graph[idx_i, idx_j] = revised_graph[idx_j, idx_i] = 0
        
        ########### Check Cycles ##########
        revised_graph = check_cycle(self.args, data, revised_graph)

        return {}, bootstrap_check_dict, boot_probability, llm_pruning_record, revised_graph, bk


    def forward(self, global_state, prompt_type, voting_num):
        adj_matrix = global_state.results.converted_graph

        (conversation,
         global_state.results.bootstrap_check_dict, 
         global_state.results.bootstrap_probability,
         global_state.results.llm_errors, # llm_pruning_record
         global_state.results.revised_graph,
         global_state.results.prior_knowledge
        )= self.quality_judge(
        data=global_state.user_data.processed_data,
        full_graph=adj_matrix,
        algorithm=global_state.algorithm.selected_algorithm,
        hyperparameters=global_state.algorithm.algorithm_arguments,
        knowledge_docs=global_state.user_data.knowledge_docs,
        boot_num=global_state.statistics.boot_num,
        prompt_type=prompt_type, 
        voting_num=voting_num
        )
        global_state.logging.knowledge_conversation.append(conversation)
        return global_state

    def user_postprocess(self, user_revise_dict):
        from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
        from postprocess.visualization import convert_to_edges
        bk = BackgroundKnowledge()
        revised_graph = self.global_state.results.revised_graph.copy()
        variables = self.global_state.user_data.processed_data.columns
        # Add and Orientation
        for add_pair in user_revise_dict['add_edges']+user_revise_dict['orient_edges']:
            bk.add_required_by_pattern(add_pair[0], add_pair[1])
            idx_j = variables.str.lower().get_loc(add_pair[0].lower())
            idx_i = variables.str.lower().get_loc(add_pair[1].lower())
            revised_graph[idx_i, idx_j] = 1
            revised_graph[idx_j, idx_i] = 0
        # Forbid
        for forbid_pair in user_revise_dict['forbid_edges']:
            bk.add_forbidden_by_pattern(forbid_pair[0], forbid_pair[1])
            idx_j = variables.str.lower().get_loc(forbid_pair[0].lower())
            idx_i = variables.str.lower().get_loc(forbid_pair[1].lower())
            revised_graph[idx_i, idx_j] = revised_graph[idx_j, idx_i] = 0
        # Update Revised Graph
        self.global_state.results.revised_graph = revised_graph 
        self.global_state.results.revised_edges = convert_to_edges(self.global_state.algorithm.selected_algorithm, 
                                                                   variables, revised_graph)
        return self.global_state

    def graph_refutation(self, global_state):
        import networkx as nx
        from openai import OpenAI
        data = global_state.user_data.processed_data
        revised_graph = global_state.results.revised_graph
        savepath = global_state.user_data.output_graph_dir
        revised_G = nx.from_numpy_array((revised_graph.T == 1).astype(int), create_using=nx.DiGraph)
        node_map = {i: data.columns[i] for i in range(data.shape[1])}
        revised_G = nx.relabel_nodes(revised_G, node_map)
        from dowhy.gcm.falsify import falsify_graph
        # Run evaluation and plot the result using `plot=True`
        result = falsify_graph(revised_G, data,  n_permutations=10, plot_histogram=True, suggestions=False,
                            plot_kwargs={'savepath':f'{savepath}/refutation_graph.jpg',
                                            'display': False})
        import re
        matches = re.findall(r'\|([^|]*)\|', str(result))[1:]
        # Clean up the matches to remove extra whitespace
        cleaned_matches = [match.strip() for match in matches]
        # Concatenate all matches with a space
        result = ' '.join(cleaned_matches)
        # Generate an analysis paragraph
        prompt = f"""
**Context**
To analyze the reliability of the causal graph, we conduct a graph refutation test, and we need a brief analysis for it.
**Your Task**
Write a brief 1 paragraph analysis for the causal graph refutation test based on the provided test result and test introduction.
**Test Result**
{result}
**Test Introduction**
The results of falsify_graph show the output of two tests. The first measures whether the LMCs implied by the graph are satisfied by the data. It compares the number of LMCs violated by the given graph to the number of LMCs violated by random graphs. For a significance value of 0.05, if the number of LMC violations by the given graph is lower than the 5% best random graphs, then we do not reject the graph. The second test (tPa) checks whether the graph is falsifiable. That is, assuming that the given graph is correct, how many other graphs share the same number of LMC violations? Since the graph is assumed to be correct, the correct LMCs are those that are implied by the graph and hence the reference number of violations is zero. For a significance value of 0.05, if less than 5% of random graphs have zero LMC violations, then it indicates that the LMCs implied by the graph can falsify (or refute) the graph.
"""
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content

        return response_doc  

    def evaluation(self, global_state, revise=False):
        '''
        :param est_graph: estimated adjacent matrix of causal graph in Panda Ndarray format
        :param ground_truth: ground truth, represented by adjacent matrix in Panda Ndarray format - Matrix[i,j] indicates j->i
        :return: Structural Hamming Distance, precision, recall, F1 score.
        '''

        import numpy as np
        from sklearn.metrics import precision_score, recall_score, f1_score
        from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
        from causallearn.graph.SHD import SHD

        if global_state.algorithm.selected_algorithm in ['PC', 'FCI', 'GES', 'CDNOD'] and not revise:
            print('Selected Algorithm: ', global_state.algorithm.selected_algorithm)
            if global_state.algorithm.selected_algorithm == 'PC':
                est_graph = global_state.results.raw_result.G
            elif global_state.algorithm.selected_algorithm == 'CDNOD':
                est_graph = global_state.results.raw_result.G
            elif global_state.algorithm.selected_algorithm == 'GES':
                est_graph = global_state.results.raw_result['G']
            elif global_state.algorithm.selected_algorithm == 'FCI':
                # TODO: improve for better handling edge o-o, o->, o-, currently ignore this part
                est_graph = global_state.results.raw_result[0]

            if global_state.statistics.domain_index is not None:
                est_graph.remove_node(list(est_graph.node_map.keys())[-1])
            ground_truth = array2cpdag(global_state.user_data.ground_truth.transpose(), 
                                       node_names=global_state.user_data.processed_data.columns)
            shd = SHD(ground_truth, est_graph).get_shd()
            adj = AdjacencyConfusion(ground_truth, est_graph)
            precision = adj.get_adj_precision()
            recall = adj.get_adj_recall()
        else:
            if global_state.statistics.domain_index is not None:
                # TODO: handle the weird case where the domain index is not the last column
                global_state.results.converted_graph = global_state.results.converted_graph[:-1, :-1]
            ground_truth_flat = global_state.user_data.ground_truth.flatten() 
            if revise:
                 est_graph_flat = np.where(global_state.results.revised_graph==1, 1, 0).flatten()
            else:
                est_graph_flat = global_state.results.converted_graph.flatten()
            shd = np.sum(np.abs(ground_truth_flat - est_graph_flat))

            adj_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

            for truth, est in zip(ground_truth_flat, est_graph_flat):
                if truth == 1 and est == 0:
                    adj_metrics['fn'] += 1
                elif truth == 0 and est == 1:
                    adj_metrics['fp'] += 1
                elif truth == 1 and est == 1:
                    adj_metrics['tp'] += 1
                else:
                    adj_metrics['tn'] += 1

            precision = adj_metrics['tp'] / (adj_metrics['tp'] + adj_metrics['fp']) if (adj_metrics['tp'] + adj_metrics['fp']) > 0 else 0
            recall = adj_metrics['tp'] / (adj_metrics['tp'] + adj_metrics['fn']) if (adj_metrics['tp'] + adj_metrics['fn']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'shd': shd, 'precision': precision, 'recall': recall, 'f1': f1}




