import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List

from causalnex.structure.data_generators import gen_stationary_dyn_net_and_df

def dict_to_adjacency_matrix(result_dict, num_nodes, lookback_period):
    # adj_matrix = np.zeros((lookback_period, num_nodes, num_nodes))
    lagged_adj_matrix = np.zeros((lookback_period + 1, num_nodes, num_nodes))
    
    nodes = list(result_dict.keys())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    for target, causes in result_dict.items():
        target_idx = node_to_idx[target]
        for cause, lag in causes:
            cause_idx = node_to_idx[cause]
            lag_index = -lag
            # if 0 <= lag_index <= lookback_period:
            # adj_matrix[lag_index, target_idx, cause_idx] = 1
            lagged_adj_matrix[lag_index, target_idx, cause_idx] = 1
                
    summary_adj_matrix = np.any(lagged_adj_matrix, axis=0).astype(int)
    
    return summary_adj_matrix, lagged_adj_matrix

def get_graph(sm, data):
    graph_dict = dict()
    for name in data:
        node = int(name.split('_')[0])
        graph_dict[node] = []

    for c, e in sm.edges:
        node_e = int(e.split('_')[0])
        node_c = int(c.split('_')[0])
        tc = int(c.partition("lag")[2])
        te = int(e.partition("lag")[2])
        if te !=0:
            print(c, e)
        lag = -1 * tc
        graph_dict[node_e].append((node_c, lag))
            
    return graph_dict

def generate_stationary_linear(
        n_nodes,
        n_samples,
        lag,
        degree_intra=0,
        degree_inter=2,
        w_min_intra=0.04,
        w_max_intra=0.2,
        w_min_inter=0.06,
        w_max_inter=0.3,
        w_decay=1.0,
        noise='linear-gauss'
    ):
    graph_net, df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df(
        num_nodes=n_nodes,
        n_samples=n_samples,
        p=lag,
        degree_intra=degree_intra,
        degree_inter=degree_inter,
        w_min_intra=w_min_intra,
        w_max_intra=w_max_intra,
        w_min_inter=w_min_inter,
        w_max_inter=w_max_inter,
        w_decay=1.0,
        sem_type=noise
    )
    print("Sample data generated", inter_nodes, intra_nodes)
    graph_true = get_graph(graph_net, intra_nodes)
    summary, adj = dict_to_adjacency_matrix(graph_true, n_nodes, lag)
    gt_graph = np.column_stack(adj)
    df = df[intra_nodes]
    df.columns = [el.split('_')[0] for el in df.columns]
    
    return df, gt_graph, summary, graph_net

def is_discrete(series, threshold=0.05):
    if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
        return 1
    unique_ratio = series.nunique() / len(series)
    if pd.api.types.is_integer_dtype(series):
        return int(unique_ratio < threshold)
    if pd.api.types.is_float_dtype(series):
        return int(unique_ratio < threshold)
    return 1

def column_type(df, threshold=0.05):
    data_types = np.zeros(df.shape, dtype='int')
    type_array = np.array([is_discrete(df[col], threshold) for col in df.columns])
    for d in range(len(type_array)):
        data_types[:,d] = type_array[d]
    return data_types