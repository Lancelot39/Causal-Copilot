import numpy as np
import pandas as pd
import networkx as nx
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.utils.DAG2CPDAG import dag2cpdag
import os
from test_result.New_accuracy_PC import count_skeleton_accuracy
import networkx as nx

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
    return g
    truth_cpdag = dag2cpdag(g)
    print(truth_cpdag)
    return truth_cpdag


def manual_metrics(true_mat, est_mat):
    shd = np.sum(np.abs(true_mat - est_mat))/2
    confuse_dict = {'TP':0, 'FP':0, 'TN':0, 'FN':0}
    true_edges = convert_to_edges(true_mat)
    est_edges = convert_to_edges(est_mat)
    
    certain_intersect = list(set(true_edges['certain_edges']) & set(est_edges['certain_edges']))
    sorted_true_edges = list({tuple(sorted(t)) for t in true_edges['certain_edges']})
    uncertain_intersect = list(set(sorted_true_edges) & set(est_edges['uncertain_edges']))

    est_all_edges = len(np.where(est_mat != 0)[0])/2

    # confuse_dict['TP'] = len(certain_intersect) + len(uncertain_intersect)/2
    # confuse_dict['FP'] = len(est_edges['certain_edges'])+len(est_edges['uncertain_edges'])/2-confuse_dict['TP']
    indices_0 = np.where(true_mat == 0)

    for i, j in zip(indices_0[0], indices_0[1]):
        if (i != j) and (est_mat[i, j] == 0):
            confuse_dict['TN'] += 1/2
    indices_1 = np.where(true_mat == 1)
    for i, j in zip(indices_1[0], indices_1[1]):
        if est_mat[i, j] == 1:
            confuse_dict['TP'] += 1
        elif est_mat[i, j] == -1:
            if est_mat[j, i] == -1:
                confuse_dict['TP'] += 1/2
    confuse_dict['FP'] = est_all_edges - confuse_dict['TP']
    confuse_dict['FN'] = len(indices_1[0])-confuse_dict['TP']
    precision = confuse_dict['TP'] / (confuse_dict['TP']+confuse_dict['FP'])
    recall = confuse_dict['TP'] / (confuse_dict['TP']+confuse_dict['FN'])

    results = {
    "TP": confuse_dict['TP'],
    "FP": confuse_dict['FP'],
    "FN": confuse_dict['FN'],
    "TN": confuse_dict['TN'],
    "precision": precision,
    "recall": recall,
    "structural_hamming_distance": shd
}
    return results


def causallearn_metrics(true_mat, est_mat, node_names):
    from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
    from causallearn.graph.SHD import SHD
    # For adjacency matrices
    truth_cpdag = array2cpdag(true_mat, node_names)
    est_cpdag = array2cpdag(est_mat, node_names)
    adj = AdjacencyConfusion(truth_cpdag, est_cpdag)

    adjTp = adj.get_adj_tp()
    adjFp = adj.get_adj_fp()
    adjFn = adj.get_adj_fn()
    adjTn = adj.get_adj_tn()

    adjPrec = adj.get_adj_precision()
    adjRec = adj.get_adj_recall()

    # Structural Hamming Distance
    shd = SHD(truth_cpdag, est_cpdag).get_shd()

    results = {
    "TP": adjTp,
    "FP": adjFp,
    "FN": adjFn,
    "TN": adjTn,
    "precision": adjPrec,
    "recall": adjRec,
    "structural_hamming_distance": shd
}
    return results

def skeleton_metrics(true_mat, est_mat):
    skeleton_result = count_skeleton_accuracy(true_mat, est_mat)
    results = {
    "TP": skeleton_result['TP'],
    "FP": skeleton_result['FP'],
    "FN": skeleton_result['FN'],
    "TN": skeleton_result['TN'],
    "precision": skeleton_result['precision_skeleton'],
    "recall": skeleton_result['recall_skeleton'],
    "structural_hamming_distance": skeleton_result['shd_skeleton']
}
    return results

def convert_to_edges(adj_matrix):
    certain_edges = [] # ->
    uncertain_edges = []  # -
    bi_edges = []    #<->
    half_edges = []  # o->
    none_edges = []  # o-o

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1]):
        # save the determined edges (j -> i)
        if adj_matrix[j, i] == -1:
            certain_edges.append((j, i))
        # save the bidirected edges (j <-> i)
        elif adj_matrix[j, i] == 1:
            bi_edges.append((j, i))
        # save the half determined edges (j o-> i)
        elif adj_matrix[j, i] == 2:
            half_edges.append((j, i))
    indices = np.where(adj_matrix == 2)
    for i, j in zip(indices[0], indices[1]):
        # save the non determined edges (j o-o i)
        if adj_matrix[j, i] == 2:
            none_edges.append((j, i))
    indices = np.where(adj_matrix == -1)
    for i, j in zip(indices[0], indices[1]):
        # save the uncertain edges (j - i)
        if adj_matrix[j, i] == -1:
            uncertain_edges.append((j, i))

    uncertain_edges = list({tuple(sorted(t)) for t in uncertain_edges})
    none_edges = list({tuple(sorted(t)) for t in none_edges})
    all_edges = certain_edges.copy() + uncertain_edges.copy() + bi_edges.copy() + half_edges.copy() + none_edges.copy()

    edges_dict = {
        'all_edges': all_edges,
        'certain_edges': certain_edges,
        'uncertain_edges': uncertain_edges,
        'bi_edges': bi_edges,
        'half_edges': half_edges,
        'none_edges': none_edges
    }
    return edges_dict

def calculate_all_metrics(data_path='dataset/sachs/sachs.csv',
                          true_graph_path='postprocess/test_result/sachs/base_graph.npy',
                          est_graph_path='postprocess/test_result/sachs/',
                          origin_graph='origin_graph.npy',
                          revise_graph='revised_graph.npy',
                          save_path='postprocess/test_result/metrics.csv'
                          ):
    node_names = pd.read_csv(data_path).columns
    print(node_names)
    true_mat = np.load(true_graph_path)
    all_results = []

    prompt_folders = ['base', 'cot_base', 'markov_blanket', 'cot_markov_blanket', 'all_relation', 'cot_all_relation']
    voting_folders = ['1_voting', '3_voting', '10_voting', '20_voting']
    for prompt_folder in prompt_folders:
        for voting_folder in voting_folders:
            folder_path = est_graph_path + f'{prompt_folder}/{voting_folder}/'
            if os.path.exists(folder_path):
                initial_mat = np.load(folder_path+origin_graph)
                initial_G = nx.from_numpy_array(initial_mat.T)
                revised_mat = np.load(folder_path+revise_graph)
                revised_G = nx.from_numpy_array((revised_mat.T == 1).astype(int), create_using=nx.DiGraph)

                initial_manual_result = manual_metrics(true_mat, initial_mat)
                initial_causallearn_result = causallearn_metrics(true_mat, initial_mat, node_names)
                initial_skeleton_result = skeleton_metrics(true_mat, initial_mat)
                print(list(nx.simple_cycles(initial_G)))
                initial_cycles = len(list(nx.simple_cycles(initial_G)))
            
                revised_manual_result = manual_metrics(true_mat, revised_mat)
                revised_causallearn_result = causallearn_metrics(true_mat, revised_mat, node_names)
                revised_skeleton_result = skeleton_metrics(true_mat, revised_mat)
                revised_cycles = len(list(nx.recursive_simple_cycles(revised_G)))
                print(list(nx.recursive_simple_cycles(revised_G)))

                initial_manual_result['cycle'] = initial_causallearn_result['cycle'] = initial_skeleton_result['cycle'] = initial_cycles
                revised_manual_result['cycle'] = revised_causallearn_result['cycle'] = revised_skeleton_result['cycle'] = revised_cycles
                initial_manual_result['method'] = revised_manual_result['method'] = 'manual'
                initial_causallearn_result['method'] = revised_causallearn_result['method'] = 'package'
                initial_skeleton_result['method'] = revised_skeleton_result['method'] = 'skeleton'
                initial_manual_result['status'] = initial_causallearn_result['status'] = initial_skeleton_result['status'] = 'initial'
                revised_manual_result['status'] = revised_causallearn_result['status'] = revised_skeleton_result['status'] = 'revised'
                initial_manual_result['prompt'] = revised_manual_result['prompt'] \
                = initial_causallearn_result['prompt'] = revised_causallearn_result['prompt'] \
                = initial_skeleton_result['prompt'] = revised_skeleton_result['prompt'] \
                = prompt_folder
                initial_manual_result['voting'] = revised_manual_result['voting']\
                = initial_causallearn_result['voting'] = revised_causallearn_result['voting'] \
                = initial_skeleton_result['voting'] = revised_skeleton_result['voting'] \
                = voting_folder
                
                all_results.append(initial_manual_result)
                all_results.append(initial_causallearn_result)
                all_results.append(initial_skeleton_result)
                all_results.append(revised_manual_result)
                all_results.append(revised_causallearn_result)
                all_results.append(revised_skeleton_result)
            else:
                continue
        #     break
        # break

    df = pd.DataFrame(all_results)
    df.to_csv(save_path, index=False)



def independent_test_pruning(data_path='dataset/sachs/sachs.csv',
                            est_graph_path='postprocess/test_result/sachs/',
                            revise_graph='revised_graph.npy',
                            ind_revise_graph='ind_revised_graph.npy'):
    from causallearn.utils.cit import CIT
    def visualize_mat(mat, file_name):
        from pywhy_graphs import PAG
        from pywhy_graphs.viz import draw
        pag = PAG()
        non_zero_indices = np.where(mat.T==1)
        non_zero_positions = list(zip(non_zero_indices[0], non_zero_indices[1]))
        for edge in non_zero_positions:
            print(edge)
            pag.add_edge(data.columns[edge[0]], data.columns[edge[1]], pag.directed_edge_name)
        dot_graph = draw(pag, shape='circle')
        dot_graph.render(outfile=folder_path+file_name, cleanup=True)

    data = pd.read_csv(data_path)
    prompt_folders = ['base', 'cot_base', 'markov_blanket', 'cot_markov_blanket', 'all_relation', 'cot_all_relation'
                      ]
    voting_folders = [#'1_voting', 
        '3_voting', '10_voting', '20_voting'
        ]
    for prompt_folder in prompt_folders:
        for voting_folder in voting_folders:
            folder_path = est_graph_path + f'{prompt_folder}/{voting_folder}/'
            if os.path.exists(folder_path):
                revised_mat = np.load(folder_path+revise_graph)
                visualize_mat(revised_mat, 'revised_graph.pdf')

                test = CIT(data.to_numpy(), 'kci') # construct a CIT instance with data and method name
                for idx_1 in range(len(data.columns)-1):
                    for idx_2 in range(idx_1+1, len(data.columns)):
                        p_value = test(idx_1, idx_2)
                        if p_value > 0.05 and revised_mat[idx_1, idx_2]!=0:
                            print(f'({data.columns[idx_1]}, {data.columns[idx_2]}): {p_value}')
                            revised_mat[idx_1, idx_2] = revised_mat[idx_2, idx_1] = 0
                np.save(folder_path+ind_revise_graph, revised_mat)
                visualize_mat(revised_mat, 'ind_revised_graph.pdf')

                


if __name__ == "__main__":
    calculate_all_metrics(data_path='dataset/sachs/sachs.csv',
                          true_graph_path='postprocess/test_result/sachs/base_graph.npy',
                          est_graph_path='postprocess/test_result/sachs_new/',
                          origin_graph='origin_graph.npy',
                          revise_graph='ind_revised_graph.npy',
                          save_path='postprocess/test_result/sachs_new/ind_metrics_sachs.csv')
    # independent_test_pruning(data_path='dataset/sachs/sachs.csv',
    #                         est_graph_path='postprocess/test_result/sachs_new/',
    #                         revise_graph='revised_graph.npy',
    #                         ind_revise_graph='ind_revised_graph.npy')
    
    # ########################


    # # residual independence test
    # X = data[['Plcg']]
    # Y = data[['PIP2']]
    # from sklearn.linear_model import LinearRegression
    # from hyppo.independence import Hsic

    # # Fit Gaussian regression models
    # model_xy = LinearRegression().fit(X, Y)
    # model_yx = LinearRegression().fit(Y, X)

    # # Compute the estimated noise terms
    # y_pred = model_xy.predict(X)
    # e_y = Y - y_pred
    # x_pred = model_yx.predict(Y)
    # e_x = X - x_pred
    # # Perform HSIC test
    # hsic = Hsic()
    # stat_xy, pvalue_xy = hsic.test(X.to_numpy(), e_y.to_numpy())
    # stat_yx, pvalue_yx = hsic.test(Y.to_numpy(), e_x.to_numpy())

    # print(f"Regression from X to Y:")
    # print(f"Regression result: Y = {model_xy.coef_[0][0]:.2f}X + {model_xy.intercept_[0]:.2f}")
    # print(f"HSIC p-value: {pvalue_xy:.6f}")

    # print("Regression from Y to X:")
    # print(f"Regression result: X = {model_yx.coef_[0][0]:.2f}Y + {model_yx.intercept_[0]:.2f}")
    # print(f"HSIC p-value: {pvalue_yx:.6f}")

    # # Determine the causal direction
    # if pvalue_xy < pvalue_yx:
    #     print("The causal direction is X -> Y.")
    # else:
    #     print("The causal direction is Y -> X.")
    # #####################
    # # combined_array = np.hstack((X.values, Y.values, e_x, e_y))
    # # from causallearn.utils.cit import CIT
    # # kci_obj = CIT(data.to_numpy(), "fisherz") # construct a CIT instance with data and method name
    # # print(kci_obj(5,7,[6]))
    # # print(data.columns)




