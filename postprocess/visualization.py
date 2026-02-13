import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from tigramite.plotting import plot_time_series_graph

from types import SimpleNamespace

# from pywhy_graphs import PAG
import sys
# sys.path.append('causal-learn')
# from causallearn.search.FCMBased.lingam.utils import make_dot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from postprocess.draw import draw
from postprocess.pag import PAG_custom
from utils.logger import logger



def get_layout(g):
    """Generate layout positions for a graph using a two-stage approach.
    
    First applies ForceAtlas2 for global structure, then refines with Fruchterman-Reingold.
    Falls back to spring_layout if fa2 is not available.
    
    Args:
        g: A networkx graph object
        
    Returns:
        dict: Node positions mapping
    """
    try:
        # First stage: ForceAtlas2 layout for initial global structure
        import fa2_modified as fa2
        forceatlas2 = fa2.ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT Lin-log mode
            adjustSizes=False,  # Prevent overlap
            edgeWeightInfluence=1.0,
            
            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            
            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,
            
            # Log
            verbose=False
        )
        
        # Initial positions with ForceAtlas2
        initial_pos = forceatlas2.forceatlas2_networkx_layout(g, pos=None, iterations=500)
        
        # Second stage: Fruchterman-Reingold for fine-tuning
        pos = nx.fruchterman_reingold_layout(g, pos=initial_pos, iterations=50, seed=42)
    except ImportError:
        # Fallback if fa2 is not available
        pos = nx.spring_layout(g, seed=42)
        logger.warning("fa2 package not found, using spring_layout")
    return pos

class Visualization(object):
    def __init__(self, global_state, threshold: float=0.95):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        :param threshold: threshold for the bootstrap probability to accept an edge.
        """
        self.global_state = global_state

        intersection_features = list(global_state.user_data.visual_selected_features)
        self.data = global_state.user_data.processed_data[intersection_features]
        self.data_idx = [global_state.user_data.processed_data.columns.get_loc(var) for var in intersection_features]
        self.bootstrap_prob = global_state.results.bootstrap_probability
        self.save_dir = global_state.user_data.output_graph_dir
        self.threshold = threshold

    def get_pos(self, mat):
        """Get node positions for graph visualization."""
        adj_matrix = (mat != 0).astype(int).T
        g = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i].replace('_', ' ') for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        
        # Get layout positions
        pos = get_layout(g)
        return pos

    def plot_pdag(self, mat, save_path, pos=None, relation=False):
        logger.debug(f"Adjacency matrix shape: {mat.shape}", "Visualization")
        algo = self.global_state.algorithm.selected_algorithm
        path = os.path.join(self.save_dir, save_path)
        data_idx = [self.global_state.user_data.processed_data.columns.get_loc(var) for var in self.global_state.user_data.visual_selected_features]
        data_labels = [self.global_state.user_data.processed_data.columns[i].replace('_', ' ') for i in data_idx]
        mat = mat[data_idx, :][:, data_idx]
        edges_dict = convert_to_edges(algo, data_labels, mat)
        logger.debug(f"Edges dictionary: {len(edges_dict)} entries", "Visualization")
        pag = PAG_custom()
        for edge in edges_dict['certain_edges']:
            try:
                pag.add_edge(edge[0], edge[1], pag.directed_edge_name)
            except:
                pag.remove_edge(edge[1], edge[0], pag.directed_edge_name)
                pag.add_edge(edge[0], edge[1], pag.bidirected_edge_name)
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
        for edge in edges_dict['associated_edges']:
            pag.add_edge(edge[0], edge[1], pag.associated_edge_name)
            pag.add_edge(edge[1], edge[0], pag.associated_edge_name)

        # Determine data type for color coding
        data_type = "time-series" if self.global_state.statistics.time_series else "tabular"

        if pos is not None:
            dot_graph = draw(pag, full_node_names=list(pos.keys()), pos=pos, shape='circle', data_type=data_type)  
            pos_G = pos              
        else:
            dot_graph = draw(pag, shape='circle', data_type=data_type)
            pos_G = None
        dot_graph.render(outfile=path, cleanup=True)
        
        return pos_G
    
    def plot_lagged_causal_graph(lag_matrix, var_names=None, save_path=None):
        """
        Standalone function to visualize a lagged causal graph.
        
        Args:
            lag_matrix: A lagged causal adjacency matrix of shape (L, N, N) where L is the number of lags,
                    and N is the number of nodes. lag_matrix[l, i, j] != 0 means j causes i with lag l.
            save_path: Path to save the plot.
            var_names: Optional list of variable names. If None, default names X0, X1, etc. will be used.
                        
        Returns:
            tigramite_mat: A tigramite-compatible graph array of shape (N, N, L)
        """
        # Convert matrix to tigramite format
        tigramite_mat = convert_lagged_mat_to_tigramite(lag_matrix)
        
        # Handle the case where the input is a 2D matrix (single lag)
        if len(lag_matrix.shape) == 2:
            L = 1
            N = lag_matrix.shape[0]
        else:
            L, N, _ = lag_matrix.shape
        
        # Create default variable names if not provided
        if var_names is None:
            var_names = [f"X{i}" for i in range(N)]
            
        
        # Call tigramite's plot function
        fig, ax = plot_time_series_graph(
            graph=tigramite_mat,
            # val_matrix=val_matrix,
            var_names=var_names,
            # link_colorbar_label="Effect strength" if val_matrix is not None else None,
            # figsize=(10, 6),
            save_name=save_path,
            # fig_ax=(fig, ax),
            # arrow_linewidth=4,
            # node_size=0.1,
            # arrowhead_size=20,
            curved_radius=0.1,
            # label_fontsize=10,
        )
        plt.close(fig)
            
        
        return tigramite_mat
       
    def boot_heatmap_plot(self):
        
        # Ensure save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.debug(f"Created save directory: {self.save_dir}", "Visualization")
        
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
            try:
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
                plt.close()  # Close the figure to free memory
                paths.append(save_path_conf)
                logger.debug(f"Successfully generated confidence heatmap: {save_path_conf}", "Visualization")
            except Exception as e:
                logger.error(f"Failed to generate confidence heatmap for {key}: {str(e)}", "Visualization")
                plt.close()  # Ensure figure is closed even on error
                continue

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

    def plot_lag_pdag(self, mat, val_matrix=None, save_path=None, **kwargs):
        """
        Plot lagged causal graph and return a tigramite-compatible graph array.
        
        Args:
            mat: A lagged causal adjacency matrix of shape (L, N, N) where L is the number of lags,
                 and N is the number of nodes. mat[l, i, j] != 0 means j causes i with lag l.
            save_path: Path to save the plot.
            val_matrix: Optional value matrix for edge colors in tigramite's plot.
                        Should be of shape (N, N, L) matching tigramite_mat structure.
            
        Returns:
            tigramite_mat: A tigramite-compatible graph array of shape (N, N, L)
        """
        # Convert matrix to tigramite format
        tigramite_mat = convert_lagged_mat_to_tigramite(mat)
        
        # Handle the case where the input is a 2D matrix (single lag)
        if len(mat.shape) == 2:
            L = 1
            N = mat.shape[0]
        else:
            L, N, _ = mat.shape
        
        # Create graph representation
        algo = self.global_state.algorithm.selected_algorithm
        path = os.path.join(self.save_dir, save_path)
        data_idx = [self.global_state.user_data.processed_data.columns.get_loc(var) for var in self.global_state.user_data.visual_selected_features]
        
        # Use a subset of the matrix if data_idx is a subset
        if len(data_idx) < N:
            tigramite_mat = tigramite_mat[data_idx, :][:, data_idx]
            if val_matrix is not None:
                val_matrix = val_matrix[data_idx, :][:, data_idx]
            N = len(data_idx)
        
        # Prepare variable names
        var_names = [self.data.columns[i] for i in range(N)]
        
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Call tigramite's plot function
        plot_time_series_graph(
            graph=tigramite_mat,
            # val_matrix=val_matrix,
            var_names=var_names,
            # link_colorbar_label="Effect strength" if val_matrix is not None else None,
            # figsize=(10, 6),
            save_name=path,
            # link_width=link_width,
            # fig_ax=(fig, ax),
            arrow_linewidth=1,
            # node_size=0.1,
            # arrowhead_size=20,
            # curved_radius=0.2,
            # label_fontsize=10,
            **kwargs
        )
            
        return tigramite_mat


def convert_to_edges(algo, variables, mat):
    labels = {i: variables[i] for i in range(len(variables))}

    certain_edges = [] # ->
    uncertain_edges = []  # -
    bi_edges = []    #<->
    half_certain_edges = []  # o->
    half_uncertain_edges = []  # o-
    none_edges = []  # o-o
    associated_edges = []  # --
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
            elif mat[ind_i, ind_j] == 7:
                associated_edges.append((ind_j, ind_i))
                
    uncertain_edges = list({tuple(sorted(t)) for t in uncertain_edges})
    none_edges = list({tuple(sorted(t)) for t in none_edges})
    all_edges = certain_edges.copy() + uncertain_edges.copy() + bi_edges.copy() + half_certain_edges.copy() + half_uncertain_edges.copy() + none_edges.copy() + associated_edges.copy()

    all_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in all_edges]
    certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in certain_edges]
    uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in uncertain_edges]
    bi_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in bi_edges]
    half_certain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_certain_edges]
    half_uncertain_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in half_uncertain_edges]
    none_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in none_edges]
    associated_edges_names = [(labels[edge[0]], labels[edge[1]]) for edge in associated_edges]
    edges_dict = {
        'all_edges': all_edges_names,
        'certain_edges': certain_edges_names,
        'uncertain_edges': uncertain_edges_names,
        'bi_edges': bi_edges_names,
        'half_certain_edges': half_certain_edges_names,
        'half_uncertain_edges': half_uncertain_edges_names,
        'none_edges': none_edges_names,
        'associated_edges': associated_edges_names
    }
    return edges_dict
   
def convert_lagged_mat_to_tigramite(mat):
    """
    Convert a lagged causal matrix from format (L, N, N) to tigramite format (N, N, L).
    
    In the input format, mat[l, i, j] != 0 means j causes i with lag l.
    In tigramite format, mat[i, j, l] = "string" represents the edge type from i to j with lag l.
    
    Args:
        mat: A lagged causal adjacency matrix of shape (L, N, N)
            
    Returns:
        tigramite_mat: A matrix of shape (N, N, L) with string edge representations
    """
    # Handle the case where the input is a 2D matrix (single lag)
    if len(mat.shape) == 2:
        mat = mat.reshape(1, *mat.shape)
    
    L, N, _ = mat.shape
    tigramite_mat = np.zeros((N, N, L), dtype=object)
    
    # Initialize tigramite_mat with empty strings
    for i in range(N):
        for j in range(N):
            for l in range(L):
                tigramite_mat[i, j, l] = ""
    
    # Edge type mapping from numerical to tigramite string representation
    edge_type_map = {
        1: "-->",  # Directed causal link
        2: "o-o",  # Undirected contemporaneous link
        3: "<->",  # Bidirected edge
        4: "o?>",  # Partial directed edge (directed non-ancestor)
        5: "o?o",  # Partial undirected edge (undirected non-ancestor)
        6: "<?>",   # No d-separation edge
        7: "o-o"   # Associated edge, temporarily use undirected edge to represent
    }
    
    # Convert numerical matrix to tigramite's string edge representation
    for l in range(L):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                    
                # Get value from original matrix (where j causes i)
                val = mat[l, i, j]
                
                if val != 0:
                    # Map numerical value to string edge type
                    if val in edge_type_map:
                        edge_str = edge_type_map[val]
                    else:
                        # Default to directed edge if value is not in mapping
                        edge_str = "-->"
                        
                    # In tigramite, edge representation is from i to j (opposite of input)
                    # So we're reversing the direction here
                    tigramite_mat[j, i, l] = edge_str
                    
                    # Handle contemporaneous links (lag=0) specially
                    if l == 0:
                        # For bidirectional contemporaneous links
                        if mat[l, j, i] != 0:
                            # If both i→j and j→i exist at lag 0
                            # Represents them as an undirected link
                            if val == 1 and mat[l, j, i] == 1:
                                tigramite_mat[i, j, l] = "o-o"
                                tigramite_mat[j, i, l] = "o-o"
                            # If they have conflicting edge types
                            else:
                                tigramite_mat[i, j, l] = "x-x"
                                tigramite_mat[j, i, l] = "x-x"
                                
    return tigramite_mat

def test_fixed_pos():
    # Create a DAG (Directed Acyclic Graph)
    n_nodes = 50
    # Initialize with zeros
    dag = np.zeros((n_nodes, n_nodes))
    
    # Create a simple chain structure: 0->1->2->3->4
    for i in range(n_nodes-1):
        dag[i, i+1] = 1
    
    # Add some additional edges to make it more complex
    # But ensure it remains acyclic (only connect from lower to higher indices)
    dag[0, 2] = 1  # 0->2
    dag[1, 3] = 1  # 1->3
    dag[0, 4] = 1  # 0->4
    
    # Create a sparser version of the DAG by removing some edges
    sparse_dag = dag.copy()
    # Remove edge 0->2
    sparse_dag[0, 2] = 0
    # Remove edge 1->3
    sparse_dag[1, 3] = 0

    # Use spring layout to get positions
    G_full = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    G_sparse = nx.from_numpy_array(sparse_dag, create_using=nx.DiGraph)
    
    pos_full = get_layout(G_full)
    pos_sparse = get_layout(G_sparse)

    # Plot using the visualization functions
    draw(G_full, pos=pos_full, full_node_names=list(pos_full.keys()), shape='circle').render(outfile='fully_connected.pdf', cleanup=True)
    draw(G_sparse, pos=pos_sparse, full_node_names=list(pos_sparse.keys()), shape='circle').render(outfile='sparse_dag.pdf', cleanup=True)


def test_lagged_visualization():
    """
    Test function for the lagged causal graph visualization.
    Creates a simple lagged causal graph and visualizes it.
    """
    # Create a simple lagged causal graph
    # 3 variables, 2 lags
    n_vars = 3
    n_lags = 2
    
    # Initialize with zeros
    lag_matrix = np.zeros((n_lags, n_vars, n_vars))
    
    # Add some causal relationships
    # Lag 1: X0 -> X1, X1 -> X2
    lag_matrix[0, 1, 0] = 1  # X0 causes X1 with lag 1
    lag_matrix[0, 2, 1] = 1  # X1 causes X2 with lag 1
    
    # Lag 2: X0 -> X2
    lag_matrix[1, 2, 0] = 1  # X0 causes X2 with lag 2
    
    # Define variable names
    var_names = ["Temperature", "Humidity", "Pressure"]

    # Create a simple global state for testing plot_lag_pdag using SimpleNamespace
    from types import SimpleNamespace
    
    # Create nested namespaces for the global state structure
    user_data = SimpleNamespace(
        processed_data=pd.DataFrame({
            'Temperature': np.random.randn(100),
            'Humidity': np.random.randn(100),
            'Pressure': np.random.randn(100)
        }),
        visual_selected_features=['Temperature', 'Humidity', 'Pressure'],
        output_graph_dir='./'
    )
    
    algorithm = SimpleNamespace(
        selected_algorithm='TestAlgorithm'
    )
    
    results = SimpleNamespace(
        bootstrap_probability=None
    )
    
    # Combine into the main global state
    fake_global_state = SimpleNamespace(
        user_data=user_data,
        algorithm=algorithm,
        results=results
    )
        
    # Create a visualization instance with the fake global state
    test_visualizer = Visualization(fake_global_state, threshold=0.5)

    test_visualizer.plot_lag_pdag(lag_matrix, val_matrix=None, save_path="temp_lagged_graph.pdf", )
    
    logger.success("Lagged causal graph visualization completed")

if __name__ == '__main__':
    # test_fixed_pos()
    # # my_visual_initial = Visualization(global_state)
    # # if global_state.results.raw_pos is None:
    # #     data_idx = [global_state.user_data.processed_data.columns.
    # get_loc(var) for var in global_state.user_data.
    # visual_selected_features]
    # #     pos = my_visual_initial.get_pos(global_state.results.
    # converted_graph[data_idx, :][:, data_idx])
    # #     global_state.results.raw_pos = pos
    # # if global_state.user_data.ground_truth is not None:
    # #     my_visual_initial.plot_pdag(global_state.user_data.
    # ground_truth, f'{global_state.algorithm.selected_algorithm}
    # _true_graph.jpg', global_state.results.raw_pos)
    # #     my_visual_initial.plot_pdag(global_state.user_data.
    # ground_truth, f'{global_state.algorithm.selected_algorithm}
    # _true_graph.pdf', global_state.results.raw_pos)
    # #     chat_history.append((None, (f'{global_state.user_data.
    # output_graph_dir}/{global_state.algorithm.selected_algorithm}
    # _true_graph.jpg',)))
    # #     yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, 
    # chat_history, download_btn
    # # if global_state.results.converted_graph is not None:
    # #     my_visual_initial.plot_pdag(global_state.results.
    # converted_graph, f'{global_state.algorithm.selected_algorithm}
    # _initial_graph.jpg', global_state.results.raw_pos)
    # #     my_visual_initial.plot_pdag(global_state.results.
    # converted_graph, f'{global_state.algorithm.selected_algorithm}
    # _initial_graph.pdf', global_state.results.raw_pos)
    test_lagged_visualization()