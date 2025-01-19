import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx


class GraphEvaluator:
    """Unified evaluator for causal graphs that handles various edge types and graph structures."""
    
    EDGE_TYPES = {
        'no_edge': 0,
        'directed': 1,  # → (Note: (i,j)=1 means j->i)
        'undirected': 2,  # -
        'bidirected': 3,  # ↔
        'partial_directed': 4,  # o→
        'partial_undirected': 5,  # o-
        'partial_unknown': 6,  # o-o
    }

    def __init__(self, has_hidden_confounders: bool = False, n_samples: int = 10000, sample: bool = False, seed: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            has_hidden_confounders: Whether the dataset contains hidden confounders
            n_samples: Number of DAGs/MAGs to sample for evaluation
            sample: Whether to sample multiple graphs for evaluation
        """
        self.has_hidden_confounders = has_hidden_confounders
        self.n_samples = n_samples
        self.sample = sample
        self.seed = seed
        
    def _ensure_asymmetric(self, graph: np.ndarray) -> np.ndarray:
        """
        Check if the graph is asymmetric and raise error if symmetry is found.
        
        Args:
            graph: Input adjacency matrix
            
        Raises:
            ValueError: If symmetric entries are found in the graph
            
        Returns:
            The input graph if it is asymmetric
        """
        n = graph.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if graph[i,j] == graph[j,i] and graph[i,j] != self.EDGE_TYPES['no_edge']:
                    raise ValueError(f"Symmetric entries found at positions ({i},{j}) and ({j},{i})")
        
        return graph
    
    def _sample_no_edge(self, sample, i, j):
        pass

    def _sample_directed_edge(self, sample, i, j):
        sample[i, j] = self.EDGE_TYPES['directed']  # Keep j->i
        return sample
    
    def _sample_undirected_edge(self, sample, i, j):
        choice = np.random.choice([0, 1])
        if choice == 1:
            sample[i, j] = self.EDGE_TYPES['directed']  # j->i
        else:
            sample[j, i] = self.EDGE_TYPES['directed']  # i->j
        return sample
    
    def _sample_bidirected_edge(self, sample, i, j):
        if self.has_hidden_confounders:
            sample[i, j] = self.EDGE_TYPES['bidirected']
        else:
            choice = np.random.choice([0, 1, 2])
            if choice == 1:
                sample[i, j] = self.EDGE_TYPES['directed']  # j->i
            elif choice == 2:
                sample[j, i] = self.EDGE_TYPES['directed']  # i->j
        return sample

    def _sample_partial_directed_edge(self, sample, i, j):
        if self.has_hidden_confounders:
            choice = np.random.choice([0, 1])
            if choice == 0:
                sample[i, j] = self.EDGE_TYPES['directed']
            else:
                sample[i, j] = self.EDGE_TYPES['bidirected']
        else:
            choice = np.random.choice([0, 1, 2])
            if choice == 1:
                sample[i, j] = self.EDGE_TYPES['directed']  # j->i
            elif choice == 2:
                sample[j, i] = self.EDGE_TYPES['directed']  # i->j
        return sample

    def _sample_partial_unknown_edge(self, sample, i, j):
        if self.has_hidden_confounders:
            choice = np.random.choice([0, 1, 2])
            if choice == 0:
                sample[i, j] = self.EDGE_TYPES['directed']
            elif choice == 1:
                sample[j, i] = self.EDGE_TYPES['directed']
            else:
                sample[i, j] = self.EDGE_TYPES['bidirected']
        else:
            choice = np.random.choice([0, 1, 2])
            if choice == 1:
                sample[i, j] = self.EDGE_TYPES['directed']  # j->i
            elif choice == 2:
                sample[j, i] = self.EDGE_TYPES['directed']  # i->j
        return sample

    def _sample_possible_graphs(self, graph: np.ndarray) -> List[np.ndarray]:
        """
        Sample possible DAGs/MAGs from the input graph based on edge types.
        
        Args:
            graph: Input adjacency matrix where (i,j)=1 means j->i
            
        Returns:
            List of sampled graphs
        """
        n = graph.shape[0]
        samples = []

        best_shd = float('inf')
        best_graph = None
        
        for _ in range(self.n_samples):
            sample = np.zeros_like(graph)
            
            for i in range(n):
                for j in range(n):
                    edge_type = graph[i, j]
                    if edge_type == self.EDGE_TYPES['directed']:
                        sample = self._sample_directed_edge(sample, i, j)
                    elif edge_type == self.EDGE_TYPES['undirected']:
                        sample = self._sample_undirected_edge(sample, i, j)
                    elif edge_type == self.EDGE_TYPES['bidirected']:
                        sample = self._sample_bidirected_edge(sample, i, j)
                    elif edge_type == self.EDGE_TYPES['partial_directed']:
                        sample = self._sample_partial_directed_edge(sample, i, j)
                    elif edge_type == self.EDGE_TYPES['partial_undirected']:
                        raise ValueError("Partial undirected edges are not found in the existed algorithms")
                    elif edge_type == self.EDGE_TYPES['partial_unknown']:
                        sample = self._sample_partial_unknown_edge(sample, i, j)
            
            if self._check_graph_validity(graph, sample):
                samples.append(sample)
            
        return samples
    
    def _sample_best_graph_for_pred(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> np.ndarray:
        """
        Sample the best graph for the pred_graph by using the edge in the true_graph instead of sampling.
        
        Args:
            true_graph: Ground truth adjacency matrix where (i,j)=1 means j->i
            pred_graph: Predicted adjacency matrix where (i,j)=1 means j->i
            
        Returns:
            The best sampled graph for the pred_graph
        """
        n = true_graph.shape[0]
        best_graph = np.zeros_like(pred_graph)
        
        for i in range(n):
            for j in range(n):
                edge_type = pred_graph[i, j]
                if edge_type == self.EDGE_TYPES['no_edge']:
                    continue
                elif edge_type == self.EDGE_TYPES['directed']:
                    best_graph[i, j] = self.EDGE_TYPES['directed']
                elif edge_type == self.EDGE_TYPES['undirected']:
                    if true_graph[i, j] == self.EDGE_TYPES['directed']:
                        best_graph[i, j] = self.EDGE_TYPES['directed']
                    elif true_graph[j, i] == self.EDGE_TYPES['directed']:
                        best_graph[j, i] = self.EDGE_TYPES['directed']
                elif edge_type == self.EDGE_TYPES['bidirected']:
                    if true_graph[i, j] == self.EDGE_TYPES['directed']:
                        best_graph[i, j] = self.EDGE_TYPES['directed']
                    elif true_graph[j, i] == self.EDGE_TYPES['directed']:
                        best_graph[j, i] = self.EDGE_TYPES['directed']
                elif edge_type == self.EDGE_TYPES['partial_directed']:
                    if true_graph[i, j] == self.EDGE_TYPES['directed']:
                        best_graph[i, j] = self.EDGE_TYPES['directed']
                    elif true_graph[j, i] == self.EDGE_TYPES['directed']:
                        best_graph[j, i] = self.EDGE_TYPES['directed']
                    elif true_graph[i, j] == self.EDGE_TYPES['bidirected']:
                        best_graph[i, j] = self.EDGE_TYPES['bidirected']
                elif edge_type == self.EDGE_TYPES['partial_undirected']:
                    raise ValueError("Partial undirected edges are not found in the existed algorithms")
                elif edge_type == self.EDGE_TYPES['partial_unknown']:
                    if true_graph[i, j] == self.EDGE_TYPES['directed']:
                        best_graph[i, j] = self.EDGE_TYPES['directed']
                    elif true_graph[j, i] == self.EDGE_TYPES['directed']:
                        best_graph[j, i] = self.EDGE_TYPES['directed']
                    elif true_graph[i, j] == self.EDGE_TYPES['bidirected']:
                        best_graph[i, j] = self.EDGE_TYPES['bidirected']
                      
        
        return best_graph
    

    def _check_graph_validity(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> bool:
        """
        Check if a predicted graph is valid compared to the true graph.
        A graph is valid if it does not have additional colliders or cycles.
        
        Args:
            true_graph: Ground truth adjacency matrix where (i,j)=1 means j->i
            pred_graph: Predicted adjacency matrix where (i,j)=1 means j->i
            
        Returns:
            bool: True if the predicted graph is valid, False otherwise
        """
        n = true_graph.shape[0]

        # Check for cycles in predicted graph
        def detect_cycles(adj_matrix):
            graph = nx.DiGraph((adj_matrix > 0).astype(int))  # Only consider entries > 0 as edges
            return not nx.is_directed_acyclic_graph(graph)

        # Detect unshielded colliders in a graph
        def detect_unshielded_colliders(adj_matrix):
            # (i, j) == 1 means j -> i
            unshielded_colliders = set()
            for z in range(n):
                parents = np.where(adj_matrix[z, :] == 1)[0]  # Find all parents of node z
                for i in range(len(parents)):
                    for j in range(i + 1, len(parents)):
                        x, y = parents[i], parents[j]
                        # (x -> z <- y) is an unshielded collider if x and y are not adjacent
                        if adj_matrix[x, y] == 0 and adj_matrix[y, x] == 0:  # x and y are not adjacent
                            unshielded_colliders.add((x, z, y))
            return unshielded_colliders

        # Check for cycles
        if detect_cycles(pred_graph):
            return False

        # Compare colliders
        true_colliders = detect_unshielded_colliders(true_graph)
        pred_colliders = detect_unshielded_colliders(pred_graph)
        
        # Check for additional colliders
        if len(pred_colliders - true_colliders) > 0:
            return False

        return True
   

    def compute_metrics(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics between true and predicted graphs.
        
        Args:
            true_graph: Ground truth adjacency matrix where (i,j)=1 means j->i
            pred_graph: Predicted adjacency matrix where (i,j)=1 means j->i
            
        Returns:
            Dictionary containing precision, recall, F1, and SHD metrics
        """
        # Ensure graphs are asymmetric
        true_graph = self._ensure_asymmetric(true_graph)
        pred_graph = self._ensure_asymmetric(pred_graph)
        
        # Sample possible graphs
        if self.sample:
            pred_samples = self._sample_possible_graphs(pred_graph)
        else:
            pred_samples = [self._sample_best_graph_for_pred(true_graph, pred_graph)]
        
        # Initialize metric accumulators
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_shd = 0.0
        
        # Compute metrics for each sampled pair
        best_graph = None
        best_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'shd': float('inf')}
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_shd = 0.0
        
        for pred_sample in pred_samples:
            metrics = self._compute_single_metrics(true_graph, pred_sample)
            
            # Track best metrics
            if metrics['shd'] < best_metrics['shd']:
                best_metrics = metrics.copy()
                best_graph = pred_sample
            
            # Also track averages
            total_precision += metrics['precision']
            total_recall += metrics['recall'] 
            total_f1 += metrics['f1']
            total_shd += metrics['shd']
        
        # Take the best metrics as the final result
        n = len(pred_samples)
        # print(f"Total samples: {n}, Total precision: {total_precision}, total recall: {total_recall}, total f1: {total_f1}, total shd: {total_shd}")
        return {
            'precision': best_metrics['precision'],
            'recall': best_metrics['recall'],
            'f1': best_metrics['f1'],
            'shd': best_metrics['shd'],
            'best_graph': best_graph,
            # 'precision': total_precision / n,
            # 'recall': total_recall / n,
            # 'f1': total_f1 / n,
            # 'shd': total_shd / n
        }

    def _compute_single_metrics(self, true_graph: np.ndarray, pred_graph: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for a single pair of graphs.
        
        Args:
            true_graph: Ground truth adjacency matrix where (i,j)=1 means j->i
            pred_graph: Predicted adjacency matrix where (i,j)=1 means j->i
            
        Returns:
            Dictionary containing precision, recall, F1, and SHD metrics
        """
        # Convert to binary adjacency matrices
        true_edges = (true_graph != self.EDGE_TYPES['no_edge']).astype(int)
        pred_edges = (pred_graph != self.EDGE_TYPES['no_edge']).astype(int)
        
        # Compute true positives, false positives, false negatives
        tp = np.sum((true_edges == 1) & (pred_edges == 1))
        fp = np.sum((true_edges == 0) & (pred_edges == 1))
        fn = np.sum((true_edges == 1) & (pred_edges == 0))
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute SHD
        shd = np.sum(true_edges != pred_edges)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'shd': shd
        }

if __name__ == "__main__":
    # Create a test case
    # Initialize evaluator
    # NOTE: Set sample=False to use the best graph for evaluation
    # The input pred_graph should be converted to a asymmetric matrix before input, where (i,j) represents edge_type
    # For the symmetric edges, only one side (i, j) should be set to the edge_type, the other side should be set to 0
    evaluator = GraphEvaluator(has_hidden_confounders=False, sample=False)
    
    # Create a simple true graph (10 nodes)
    true_graph = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 0 -> Node 1
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 1 - Node 2 (undirected)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 2 -> Node 3
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 3 -> Node 4
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Node 4 -> Node 5
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Node 5 -> Node 6
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Node 6 -> Node 7
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Node 7 -> Node 8
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   # Node 8 -> Node 9
    ])
    
    # Create a predicted graph
    pred_graph = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 0 -> Node 1 (correct)
        [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # Node 1 -> Node 2 (partially correct, got direction)
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 2 -> Node 3 (correct)
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Node 3 -> Node 4 (correct)
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Node 4 -> Node 5 (correct)
        [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],  # Node 5 -> Node 6 (correct)
        [0, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Node 6 -> Node 7 (correct)
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Node 7 -> Node 8 (correct)
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]   # Node 8 -> Node 9 (correct)
    ])
    
    # Compute metrics
    metrics = evaluator.compute_metrics(true_graph, pred_graph)
    
    # Print results
    print("Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"Structural Hamming Distance: {metrics['shd']:.3f}")
    print(f"Best Graph: {metrics['best_graph']}")