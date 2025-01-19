import numpy as np
import pandas as pd
from typing import Dict, Tuple
import cupy as cp
from gpucsl.pc.pc import GaussianPC, DiscretePC
from causallearn.graph.GraphClass import CausalGraph

def kci_test_gpu(X: int, Y: int, Z: list, data: np.ndarray, gamma: float, alpha: float) -> bool:
            """
            GPU-accelerated KCI test using CuPy for kernel computations.
            """
            n_samples = data.shape[0]

            if len(Z) == 0:
                X_data = cp.asarray(data[:, X]).reshape(-1, 1)
                Y_data = cp.asarray(data[:, Y]).reshape(-1, 1)
            else:
                Z_data = cp.asarray(data[:, Z])
                X_data = cp.asarray(data[:, X])
                Y_data = cp.asarray(data[:, Y])

                # Regress X on Z
                beta_X = cp.linalg.lstsq(Z_data, X_data, rcond=None)[0]
                X_residual = X_data - Z_data @ beta_X
                X_data = X_residual.reshape(-1, 1)

                # Regress Y on Z
                beta_Y = cp.linalg.lstsq(Z_data, Y_data, rcond=None)[0]
                Y_residual = Y_data - Z_data @ beta_Y
                Y_data = Y_residual.reshape(-1, 1)

            # Compute kernel matrices
            K_X = cp.exp(-gamma * cp.linalg.norm(X_data[:, None] - X_data, axis=2) ** 2)
            K_Y = cp.exp(-gamma * cp.linalg.norm(Y_data[:, None] - Y_data, axis=2) ** 2)

            # Centering
            H = cp.eye(n_samples) - cp.ones((n_samples, n_samples)) / n_samples
            Kc_X = H @ K_X @ H
            Kc_Y = H @ K_Y @ H

            # Compute HSIC statistic
            hsic_stat = cp.trace(Kc_X @ Kc_Y) / ((n_samples - 1) ** 2)
            
            # Simple threshold test
            return hsic_stat < alpha

def accelerated_pc(data: np.ndarray, alpha: float = 0.05, indep_test: str = 'fisherz', 
                  depth: int = -1, stable: bool = True, uc_rule: int = 0,
                  uc_priority: int = -1, mvpc: bool = False, 
                  correction_name: str = 'MV_Crtn_Fisher_Z',
                  background_knowledge = None, verbose: bool = False,
                  show_progress: bool = False, node_names = None,
                  gamma: float = 0.5) -> Tuple[np.ndarray, Dict, CausalGraph]:
    """
    Accelerated PC algorithm that chooses between GPU-accelerated implementations
    based on the independence test.
    
    Args:
        Same interface as causal-learn's PC algorithm
        Additional arg:
            gamma: float, parameter for KCI test RBF kernel
            
    Returns:
        Tuple containing:
        - adjacency matrix
        - info dictionary
        - CausalGraph object
    """
    
    max_level = depth if depth > 0 else data.shape[1] - 1
    
    if indep_test == 'fisherz':
        # Use GaussianPC for continuous data
        if depth == -1:
             max_level = data.shape[1]
        else:
             max_level = depth
        ((directed_graph, separation_sets, pmax, discover_skeleton_runtime,
          edge_orientation_runtime, discover_skeleton_kernel_runtime),
         pc_runtime) = GaussianPC(data, max_level, alpha).set_distribution_specific_options().execute()
            
        # Convert to CausalGraph format
        cg = CausalGraph(directed_graph, node_names)
        cg.sepset = separation_sets
        
        info = {
            'sepset': separation_sets,
            'definite_UC': [], # Not implemented in GPU version
            'definite_non_UC': [], 
            'PC_elapsed': pc_runtime
        }
        
        return directed_graph, info, cg
        
    elif indep_test == 'chi_square':
        if depth == -1:
             max_level = data.shape[1]
        else:
             max_level = depth
        # Use DiscretePC for discrete data
        ((directed_graph, separation_sets, pmax, discover_skeleton_runtime,
          edge_orientation_runtime, discover_skeleton_kernel_runtime),
         pc_runtime) = DiscretePC(data, max_level, alpha).set_distribution_specific_options().execute()
            
        cg = CausalGraph(directed_graph, node_names)
        cg.sepset = separation_sets
        
        info = {
            'sepset': separation_sets,
            'definite_UC': [],
            'definite_non_UC': [],
            'PC_elapsed': pc_runtime
        }
        
        return directed_graph, info, cg
        
    elif indep_test == 'kci':
        # Use GPU-accelerated KCI test
        # Run PC with KCI test
        from causallearn.search.ConstraintBased.PC import pc
        return pc(data, alpha=alpha, indep_test=kci_test_gpu, 
                 depth=depth, stable=stable, uc_rule=uc_rule,
                 uc_priority=uc_priority, mvpc=mvpc,
                 correction_name=correction_name,
                 background_knowledge=background_knowledge,
                 verbose=verbose, show_progress=show_progress,
                 node_names=node_names)
    
    else:
        raise ValueError(f"Independence test {indep_test} not supported. " 
                       "Use 'fisherz', 'chi_square', or 'kci'")
