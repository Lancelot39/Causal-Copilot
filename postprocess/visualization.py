import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pywhy_graphs import PAG
# from pywhy_graphs.viz import draw
import sys
sys.path.append('causal-learn')
from causallearn.search.FCMBased.lingam.utils import make_dot
from .draw import draw

class Visualization(object):
    def __init__(self, global_state, threshold: float=0.95):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.global_state = global_state
        self.data = global_state.user_data.processed_data[global_state.user_data.visual_selected_features]
        self.data_idx = [global_state.user_data.processed_data.columns.get_loc(var) for var in global_state.user_data.visual_selected_features]
        self.bootstrap_prob = global_state.results.bootstrap_probability
        self.save_dir = global_state.user_data.output_graph_dir
        self.threshold = threshold

    def get_pos(self, mat):
        adj_matrix = (mat != 0).astype(int).T
        g = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i] for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        pos = nx.spring_layout(g)
        return pos

    def plot_pdag(self, mat, save_path, pos=None, relation=False):
        #print(mat)
        algo = self.global_state.algorithm.selected_algorithm
        path = os.path.join(self.save_dir, save_path)
        edges_dict = convert_to_edges(algo, self.data.columns, mat)
        pag = PAG()
        for edge in edges_dict['certain_edges']:
            pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
        for edge in edges_dict['uncertain_edges']:
            pag.add_edge(edge[0], edge[1], pag.undirected_edge_name)
        for edge in edges_dict['bi_edges']:
            pag.add_edge(edge[0], edge[1], pag.bidirected_edge_name)
        for edge in edges_dict['half_certain_edges']:
            pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)
        for edge in edges_dict['half_uncertain_edges']:
            pag.add_edge(edge[0], edge[1], pag.undirected_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)
        for edge in edges_dict['none_edges']:
            pag.add_edge(edge[0], edge[1], pag.circle_edge_name)
            pag.add_edge(edge[1], edge[0], pag.circle_edge_name)

        if pos is not None:
            dot_graph = draw(pag, full_node_names=list(pos.keys()), pos=pos, shape='circle')  
            pos_G = pos              
        else:
            dot_graph = draw(pag, shape='circle')
            pos_G = None
        dot_graph.render(outfile=path, cleanup=True)
        
        return pos_G
       
    def boot_heatmap_plot(self):
        
        boot_prob_dict = {k: v for k, v in self.bootstrap_prob.items() if v is not None and sum(v.flatten())>0}
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_certain_edges': 'Directed Non-Ancestor Edge', #(o->)
                    'half_uncertain_edges': 'Undirected Non-Ancestor Edge', #(o-)
                    'none_edges': 'No D-Seperation Edge', #(o-o)
                    'none_existence':'No Edge'}
        paths = []
        for key in boot_prob_dict.keys():
            prob_mat = boot_prob_dict[key]
            name = name_map[key]
            # Create a heatmap
            plt.figure(figsize=(8, 6))
            #plt.rcParams['font.family'] = 'Times New Roman'
            sns.heatmap(prob_mat[self.data_idx, :][:, self.data_idx], annot=True, cmap='Reds', fmt=".2f", square=True, cbar_kws={"shrink": .8},
                        xticklabels=self.global_state.user_data.visual_selected_features,
                        yticklabels=self.global_state.user_data.visual_selected_features)            
            plt.title(f'Confidence Heatmap for {name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            # Save the plot
            save_path_conf = os.path.join(self.save_dir, f'{key}_confidence_heatmap.jpg')
            plt.savefig(fname=save_path_conf, dpi=1000)
            paths.append(save_path_conf)

        return paths

    def metrics_plot(self, original_metrics, revised_metrics):
        # Sample data
        shd_revised = revised_metrics.pop('shd')
        shd_original = original_metrics.pop('shd')

        # Extract keys and values
        metrics = list(revised_metrics.keys())
        revised_values = list(revised_metrics.values())
        original_values = list(original_metrics.values())

        # Set the bar width
        bar_width = 0.35

        # Set position of bars on the X axis
        r1 = np.arange(len(metrics))
        r2 = [x + bar_width for x in r1]

        # Create the figure and the first axis
        fig, ax1 = plt.subplots()

        # Create bars for revised metrics on the first y-axis
        ax1.bar(r1, original_values, color='#FFB266', width=bar_width, label='Original Metrics')
        ax1.bar(r2, revised_values, color='#0080FF', width=bar_width, label='Revised Metrics')

        # Add labels and title for the first y-axis
        ax1.set_xlabel('Metrics', fontweight='bold')
        ax1.set_ylabel('Values (Precision, Recall, F1)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Set y-axis limits for the first y-axis
        ax1.set_ylim(0, 1)

        # Create a second y-axis
        ax2 = ax1.twinx()

        # Plot the shd metric on the second y-axis
        ax2.bar(r1[-1] + 1, shd_original, color='#FFB266', width=bar_width, label='Original SHD')
        ax2.bar(r2[-1] + 1, shd_revised, color='#0080FF', width=bar_width, label='Revised SHD')

        # Set the second y-axis label
        ax2.set_ylabel('SHD Values', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Set x-ticks
        ax1.set_xticks([r + bar_width / 2 for r in range(len(metrics) + 1)])
        metrics.append('SHD')
        ax1.set_xticklabels(metrics)

        # Add legends
        ax1.legend(loc='upper left')

        # Set the title
        plt.title('Metrics Comparison of Original and Revised Graphs')

        # Show the plot
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, 'metrics.jpg')
        plt.savefig(fname=save_path, dpi=1000)

        return save_path

def convert_to_edges(algo, variables, mat):
    labels = {i: variables[i] for i in range(len(variables))}

    certain_edges = [] # ->
    uncertain_edges = []  # -
    bi_edges = []    #<->
    half_certain_edges = []  # o->
    half_uncertain_edges = []  # o-
    none_edges = []  # o-o

    for ind_i in range(mat.shape[0]):
        for ind_j in range(mat.shape[0]):
            if ind_i == ind_j: continue
            elif mat[ind_i, ind_j] == 1: 
                certain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 2: 
                uncertain_edges.append((ind_i, ind_j))
            elif mat[ind_i, ind_j] == 3: 
                bi_edges.append((ind_i, ind_j))
            elif mat[ind_i, ind_j] == 4: 
                half_certain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 5: 
                half_uncertain_edges.append((ind_j, ind_i))
            elif mat[ind_i, ind_j] == 6: 
                none_edges.append((ind_j, ind_i))

    uncertain_edges = list({tuple(sorted(t)) for t in uncertain_edges})
    none_edges = list({tuple(sorted(t)) for t in none_edges})
    all_edges = certain_edges.copy() + uncertain_edges.copy() + bi_edges.copy() + half_certain_edges.copy() + half_uncertain_edges.copy() + none_edges.copy()

    all_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in all_edges]
    certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in certain_edges]
    uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in uncertain_edges]
    bi_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in bi_edges]
    half_certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_certain_edges]
    half_uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_uncertain_edges]
    none_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in none_edges]
    edges_dict = {
        'all_edges': all_edges_names,
        'certain_edges': certain_edges_names,
        'uncertain_edges': uncertain_edges_names,
        'bi_edges': bi_edges_names,
        'half_certain_edges': half_certain_edges_names,
        'half_uncertain_edges': half_uncertain_edges_names,
        'none_edges': none_edges_names
    }
    return edges_dict
   

def test_fixed_pos():
    # Create a fully connected graph
    n_nodes = 5
    fully_connected = np.ones((n_nodes, n_nodes))
    np.fill_diagonal(fully_connected, 0)

    # Create a 50% connected graph by randomly setting half the edges to 0
    fifty_percent = fully_connected.copy()
    n_edges = (n_nodes * (n_nodes-1)) // 2  # Number of edges in upper triangle
    edges_to_remove = n_edges // 2  # Remove half the edges
    
    # Get indices of upper triangle (excluding diagonal)
    upper_indices = np.triu_indices(n_nodes, k=1)
    
    # Randomly select edges to remove
    remove_idx = np.random.choice(n_edges, edges_to_remove, replace=False)
    for idx in remove_idx:
        i, j = upper_indices[0][idx], upper_indices[1][idx]
        fifty_percent[i,j] = fifty_percent[j,i] = 0

    # Use spring layout to get positions
    import networkx as nx
    G_full = nx.from_numpy_array(fully_connected, create_using=nx.DiGraph)
    G_half = nx.from_numpy_array(fifty_percent, create_using=nx.DiGraph)
    
    pos_full = nx.spring_layout(G_full, seed=42)


    # Plot using the visualization functions
    draw(G_full, pos=pos_full, full_node_names=list(pos_full.keys()), shape='circle').render(outfile='fully_connected.pdf', cleanup=True)
    draw(G_half, pos=pos_full, full_node_names=list(pos_full.keys()), shape='circle').render(outfile='fifty_percent.pdf', cleanup=True)
    

if __name__ == '__main__':
    test_fixed_pos()