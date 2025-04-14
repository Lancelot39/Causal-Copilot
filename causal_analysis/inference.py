import numpy as np
import pandas as pd
import networkx as nx
import pickle

# REMOVE or comment out the direct import of CausalModel from DoWhy:
# from dowhy import gcm, CausalModel

# Keep gcm if you still want anomaly attribution and interventions:
from dowhy import gcm, CausalModel
# Import econml classes instead of using DoWhy's linear_regression or DML
from econml.dml import DML, LinearDML, SparseLinearDML, CausalForestDML

import shap
import sklearn
import matplotlib.pyplot as plt
import seaborn as snscl
from openai import OpenAI
from pydantic import BaseModel
import os
import sys
import warnings
from CEM_LinearInf.cem import cem
from CEM_LinearInf.balance import balance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from global_setting.state import Inference
from sklearn.preprocessing import LabelEncoder


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_analysis.DML.hte_filter import HTE_Filter as DML_HTE_Filter
from causal_analysis.DML.hte_params import HTE_Param_Selector as DML_HTE_Param_Selector
from causal_analysis.DML.hte_program import HTE_Programming as DML_HTE_Programming
from causal_analysis.DRL.hte_filter import HTE_Filter as DRL_HTE_Filter
from causal_analysis.DRL.hte_params import HTE_Param_Selector as DRL_HTE_Param_Selector
from causal_analysis.DRL.hte_program import HTE_Programming as DRL_HTE_Programming
from causal_analysis.IV.hte_filter import HTE_Filter as IV_HTE_Filter
from causal_analysis.IV.hte_params import HTE_Param_Selector as IV_HTE_Param_Selector
from causal_analysis.IV.hte_program import HTE_Programming as IV_HTE_Programming
from causal_analysis.MetaLearners.hte_filter import HTE_Filter as MetaLearners_HTE_Filter
from causal_analysis.MetaLearners.hte_params import HTE_Param_Selector as MetaLearners_HTE_Param_Selector
from causal_analysis.MetaLearners.hte_program import HTE_Programming as MetaLearners_HTE_Programming
from causal_analysis.help_functions import *
from causal_analysis.analysis import *
from global_setting.Initialize_state import global_state_initialization
from postprocess.judge_functions import *
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class Analysis(object):
    def __init__(self, global_state, args):
        """
        Hardcoded to test the Auto MPG dataset and adjacency separately,
        while preserving references to global_state and args (which can be None).
        """
        self.global_state = global_state
        self.args = args
        self.data = global_state.user_data.processed_data
        self.graph = convert_adj_mat(global_state.results.revised_graph)
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph) # convert adj matrix into DiGraph
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})
        self.dot_graph = self._generate_dowhy_graph()
        # Construct Causal Model via dowhy/gcm
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        # gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        # gcm.fit(self.causal_model, self.data)

    def _print_data_disclaimer(self):
        """
        Analyze the dataset's columns and print disclaimers about their nature (continuous/discrete/categorical).

        Previously, we had a hard-coded disclaimer. Now we dynamically check column types and counts.
        """
        dtypes = self.data.dtypes
        numeric_cols = dtypes[dtypes != 'object'].index.tolist()
        object_cols = dtypes[dtypes == 'object'].index.tolist()

        continuous_cols = []
        discrete_cols = []

        # Simple heuristic: if a numeric column has more than 10 unique values, consider it continuous; otherwise discrete
        for col in numeric_cols:
            unique_vals = self.data[col].nunique()
            if unique_vals > 10:
                continuous_cols.append(col)
            else:
                discrete_cols.append(col)

        print("\n" + "="*60)
        print("DATASET ANALYSIS:")
        if object_cols:
            print(f"Categorical (non-numeric) columns detected: {object_cols}")
        else:
            print("No categorical columns detected.")

        if continuous_cols:
            print(f"Continuous numeric columns detected: {continuous_cols}")
        else:
            print("No continuous numeric columns detected.")

        if discrete_cols:
            print(f"Discrete/low-cardinality numeric columns detected: {discrete_cols}")
        else:
            print("No discrete/low-cardinality numeric columns detected.")

        print("Please ensure your causal assumptions align with these column types.")
        print("="*60 + "\n")

    def feature_importance(self, target_node, linearity, visualize=True):
        print('start feature importance analysis')
        parent_nodes = list(self.G.predecessors(target_node))

        X = self.data.drop(columns=[target_node])
        y = self.data[[target_node]]
        X_background = shap.utils.sample(X, int(len(X)*0.2))  # background distribution for SHAP
        if linearity:
            model_linear = sklearn.linear_model.LinearRegression()
            model_linear.fit(X, y)
            explainer_linear = shap.Explainer(model_linear.predict, X_background)
            shap_values = explainer_linear(X)
        else:
            # Xd = xgboost.DMatrix(X, label=y)
            # model_xgb = xgboost.train({"eta": 1, "max_depth": 3, "base_score": 0, "lambda": 0}, Xd, 1)
            # pred = model_xgb.predict(Xd, output_margin=True)
            model_rf = RandomForestRegressor()
            model_rf.fit(X, y)
            explainer_rf = shap.TreeExplainer(model_rf)
            shap_values = explainer_rf(X)
        # Mean absolute SHAP values
        shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X.columns)
        mean_shap_values = shap_df.mean()
        shap_df.to_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv', index=False)
        
        figs = []
        if visualize == True:
            # 1st SHAP Plot beeswarm
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.beeswarm(shap_values, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            figs.append(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png')

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.bar(shap_values, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png', bbox_inches='tight')  # Save as PNG
            figs.append(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png')
            plt.close()
        return parent_nodes, mean_shap_values, figs

    def _propensity_score_matching(self, confounders, treatment, outcome):
        """
        Perform Propensity Score Matching (PSM) using sklearn.
        :param treatment: The treatment variable.
        :param outcome: The outcome variable.
        :return: Matched data after PSM.
        """
        # Debug: Print treatment variable
        print(f"Treatment variable ({treatment}):\n{self.data[treatment].head()}")
        print(f"Unique values in treatment variable: {self.data[treatment].unique()}")

        # Ensure the treatment variable is binary
        if len(self.data[treatment].unique()) > 2:
            # Binarize the treatment variable using a threshold (e.g., median)
            threshold = self.data[treatment].median()
            self.data[treatment] = (self.data[treatment] > threshold).astype(int)
            print(f"Treatment variable binarized using threshold {threshold}.")

        # Debug: Print binarized treatment variable
        print(f"Binarized treatment variable:\n{self.data[treatment].head()}")

        # Ensure the treatment variable is a 1-dimensional array
        treatment_values = self.data[treatment].values
        print(f"Shape of treatment values: {treatment_values.shape}")
        if treatment_values.ndim != 1:
            raise ValueError(f"Treatment variable must be 1-dimensional. Got shape {treatment_values.shape}.")

        # Step 1: Calculate propensity scores using LogisticRegression
        #confounders = self.data[confounders].drop(columns=[treatment, outcome])  # Features (confounders)
        confounders = self.data[confounders]

        # Debug: Print confounders
        print(f"Confounders:\n{confounders.head()}")
        print(f"Shape of confounders: {confounders.shape}")

        # Ensure confounders are not empty
        if confounders.empty:
            raise ValueError("No confounders found. Ensure the dataset contains valid confounders.")

        # Fit logistic regression model
        propensity_model = LogisticRegression(solver='liblinear', max_iter=1000)
        propensity_model.fit(confounders, treatment_values)
        
        # Add propensity scores to the data
        self.data['propensity_score'] = propensity_model.predict_proba(confounders)[:, 1]

        # Step 2: Separate treated and control groups
        treated = self.data[self.data[treatment] == 1]
        control = self.data[self.data[treatment] == 0]

        # Step 3: Perform nearest neighbor matching using sklearn
        nbrs = NearestNeighbors(n_neighbors=1).fit(control[['propensity_score']])
        distances, indices = nbrs.kneighbors(treated[['propensity_score']])

        # Step 4: Create matched data
        matched_control = control.iloc[indices.flatten()]
        matched_data = pd.concat([treated, matched_control])

        return matched_data
                
    def _coarsened_exact_matching(self,confounders,cont_confounder, treatment, outcome):
        """
        Perform Coarsened Exact Matching (CEM).
        """
        # Initialize CEM
        my_cem = cem(self.data, confounder_cols = confounders, cont_confounder_cols = cont_confounder,  col_t = treatment, col_y = outcome)
        matched_data = my_cem.match()
        return matched_data

    def _identify_confounders(self, treatment, outcome):
        """
        Identify confounders for the treatment and outcome using the causal graph.
        If the graph is not available, use the LLM to suggest confounders.
        """
        # Validate that treatment and outcome are valid nodes in the graph
        if treatment not in self.G.nodes or outcome not in self.G.nodes:
            raise ValueError(f"Invalid treatment or outcome variable in the graph: {treatment}, {outcome}")

        # Use the causal graph to identify confounders
        confounders = list(set(self.G.predecessors(treatment)) & set(self.G.predecessors(outcome)))
        return confounders 
    
    def _check_balance(self, data, matched_data, treatment, outcome, confounders, cont_confounders, title):
        """
        Check balance of confounders between treated and control groups using density, KS scatter, and ECDF plots.
        """
        # Ensure the confounders and continuous confounders are valid columns
        valid_confounders = [col for col in confounders if col in matched_data.columns]
        valid_cont_confounders = [col for col in cont_confounders if col in matched_data.columns]

        if not valid_confounders:
            raise ValueError("No valid confounders found in the matched data.")
        if not valid_cont_confounders:
            print("Warning: No valid continuous confounders found in the matched data.")

        # Generate density plot
        figs = generate_density_plot(self.global_state, data, matched_data, treatment, valid_confounders, title)
        return figs

    def _matched_treatment_effect(self, matched_data, treatment, outcome, treat, control, X_col):
        treated_mean = matched_data[matched_data[treatment] == treat][outcome].mean()
        control_mean = matched_data[matched_data[treatment] == control][outcome].mean()
        ate = treated_mean - control_mean
        cont_X_col = [var for var in X_col if self.global_state.statistics.data_type_column[var]=='Continuous']
        coarsen_data = coarsen_continuous_variables(matched_data, cont_X_col)
        cate_results = {}
        for label in X_col:
            if label in cont_X_col:
                group_col = f'coarsen_{label}'
            else:
                group_col = label
            cate = coarsen_data.groupby(group_col).apply(lambda x: x[x[treatment] == treat][outcome].mean() - x[x[treatment] == 0][outcome].mean())
            cate_results[label] = cate
        
        return ate, cate_results
    
    def estimate_causal_effect_matching(self, treatment, outcome, confounders, X_col, treat, control, cont_confounders, method="propensity_score", visualize=True):
        """
        Estimate the causal effect using matching methods.
        
        :param treatment: The treatment variable.
        :param outcome: The outcome variable.
        :param confounders: List of confounder column names.
        :param cont_confounders: List of continuous confounder column names.
        :param method: Matching method ("propensity_score" or "cem").
        :param visualize: Whether to visualize balance before and after matching.
        :return: Causal estimate, matched data, and balance plots.
        """
        print("\n" + "#" * 60)
        print(f"Estimating Causal Effect using Matching: {method}")
        print(f"Treatment: {treatment} | Outcome: {outcome}")
        print("#" * 60)

        # Step 1: Perform Matching
        if method == "propensity_score":
            matched_data = self._propensity_score_matching(confounders, treatment, outcome)
        elif method == "cem":
            # Pass confounders and continuous confounders to CEM
            matched_data = self._coarsened_exact_matching(confounders, cont_confounders, treatment, outcome)
        else:
            raise ValueError(f"Unsupported matching method: {method}")
        
        # Step 2: Coarsen continuous variables for balance checking
        if method == "cem":
            matched_data = coarsen_continuous_variables(matched_data, cont_confounders)

        # Step 3: Balance Checking (After Matching)
        if visualize:
            figs = self._check_balance(
                data=self.data,  # Original dataframe (before matching)
                matched_data=matched_data,  # Matched dataframe (after matching)
                treatment=treatment,
                outcome=outcome,
                confounders=confounders, 
                cont_confounders=cont_confounders,
                title="Balance Checking"
            )
        else:
            figs = []

        # Step 4: Estimate Treatment Effect
        ate, cate_results = self._matched_treatment_effect(matched_data, treatment, outcome, treat, control, X_col)
        cate_plot_path = os.path.join(self.global_state.user_data.output_graph_dir, 'cate_plot.png')
        plot_cate_bars_by_group(cate_results, cate_plot_path)
        figs.append(cate_plot_path)
        print(f"\nAverage Treatment Effect (ATE) using {method}: {ate}")
        return ate, cate_results, matched_data, figs

    def _generate_dowhy_graph(self):
        """
        Generate a causal graph in DOT format for DoWhy.
        """
        edges = nx.edges(self.G)
        dot_format = "digraph { "
        for u, v in edges:
            dot_format += f"{u} -> {v}; "
        dot_format += "}"
        return dot_format
        
    def attribute_anomalies(self, target_node, anomaly_samples, confidence_level=0.95):
        """
        Performing anomaly attribution with:
        - Limiting selected nodes to a maximum of 10 nodes including the Ancestors + Graph Nodes in the causal graph 
        - Detecting and resolving cycles.
        - Using GCM for causal model fitting.
        - Computing variance for the target node.
        - Plotting causal graph with arrow strengths.
        - Computing and visualizing anomaly attribution scores.
        """
        print("\n" + "#" * 60)
        print(f"Performing Anomaly Attribution for {target_node}")
        print("#" * 60)

        # Initialize directories
        output_dir = self.global_state.user_data.output_graph_dir
        column_names = list(self.data.columns)
        figs = []

        # Detect and resolve cycles
        graph_matrix = self.causal_model.graph  

        if isinstance(graph_matrix, nx.DiGraph):
            graph_matrix = nx.to_numpy_array(graph_matrix)

        resolved_graph = check_cycle(self.args, self.data, graph_matrix)  

        # Ensure the graph is fully acyclic
        G = nx.from_numpy_array(resolved_graph, create_using=nx.DiGraph)

        if not nx.is_directed_acyclic_graph(G):
            print("❌ Cycle removal failed! The graph is still cyclic.")
            raise ValueError("Cycle removal was unsuccessful. Graph must be a DAG.")
        
        # Store cycle detection results in global state
        self.global_state.inference.cycle_detection_result = {
            "detected_cycles": list(nx.simple_cycles(G)),  # Store any remaining detected cycles
            "final_graph_after_resolution": nx.to_dict_of_lists(G)  # Store resolved graph
        }

        # Store editing history
        self.global_state.inference.editing_history.append({
        "removed_edges": "Stored inside check_cycle function",  # Can be extracted from `check_cycle` if needed
        "final_resolved_graph": nx.to_dict_of_lists(G)})


        # Fix Node Naming Issue: Relabel Graph Nodes
        column_names = list(self.data.columns)
        mapping = {i: column_names[i] for i in range(len(column_names))}
        G = nx.relabel_nodes(G, mapping)

        # Ensure Ancestors + Graph Nodes Are Included (Limit to 10 nodes)
        ancestors = list(nx.ancestors(G, target_node))  # Get ancestors of target node
        graph_nodes = list(G.nodes)  # Get all nodes in the causal graph

        if not ancestors:
            print(f"❌ Target node {target_node} has no ancestors. Not possible to calculate anomaly attribution.")
            return None, []

        # Combine and limit nodes to max 10
        selected_columns = list(set(ancestors + graph_nodes + [target_node]))

        # Ensure selected columns exist in the dataset
        selected_columns = [col for col in selected_columns if col in column_names]

        max_nodes = 10
        if len(selected_columns) > max_nodes:
            print(f"⚠️ More than {max_nodes} nodes selected. Reducing selection.")

            # Keep target node and prioritize ancestors
            selected_columns = [target_node] + ancestors[:max_nodes - 1]

            # Fill remaining slots with graph nodes
            remaining_slots = max_nodes - len(selected_columns)
            extra_nodes = [node for node in graph_nodes if node not in selected_columns][:remaining_slots]

            selected_columns.extend(extra_nodes)

        print("✅ Final selected columns (max 10):", selected_columns)

        # Filter dataset
        self.data = self.data[selected_columns].copy()
        self.G = self.G.subgraph(selected_columns)

        # Assign Causal Mechanisms using GCM
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        
        try:
            gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
            gcm.fit(self.causal_model, self.data)
        except KeyError as e:
            print(f"❌ KeyError: {e}. Possible mismatch between graph nodes and DataFrame columns.")
            raise

        # Variance Plot for Target Node
        if self.data[target_node].isna().all():
            print(f"⚠️ No data available for {target_node}, skipping variance plot.")
        else:
            plt.figure(figsize=(8, 5))
            self.data[target_node].plot(ylabel=target_node, title=f"{target_node} Value Distribution", rot=45)
            plt.grid(True)
            var_plot_path = f"{output_dir}/variance_plot_{target_node}.png"
            plt.savefig(var_plot_path, bbox_inches='tight')
            plt.show()
            plt.close()
            figs.append(var_plot_path)
            print(f"✅ Variance plot saved: {var_plot_path}")

        # Causal Graph with Arrow Strengths
        if len(list(self.G.predecessors(target_node))) == 0:
            print(f"❌ Target node {target_node} has no ancestors. Not possible to compute anomaly attribution.")
            return None, []

        arrow_strengths = gcm.arrow_strength(self.causal_model, target_node=target_node)

        plt.figure(figsize=(10, 7))
        pos = nx.kamada_kawai_layout(self.G)  # Better node spacing
        nx.draw(self.G, pos, with_labels=True, node_size=2500, node_color="lightblue", edge_color="gray", alpha=0.8)

        # Add edge labels only for arrows that point to the target node
        edge_labels = {
            (u, v): f"{arrow_strengths.get((u, v), 0):.2f}"
            for u, v in self.G.edges if v == target_node
        }

        # Draw edges with varying thickness based on strength
        for (u, v) in self.G.edges:
            strength = arrow_strengths.get((u, v), 0)
            nx.draw_networkx_edges(self.G, pos, edgelist=[(u, v)], arrowstyle="->",
                                arrowsize=20, edge_color="black", width=2 + 6 * strength, alpha=0.8)

        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color="darkred", font_size=12, font_weight="bold")

        causal_graph_path = f"{output_dir}/causal_graph_{target_node}.png"
        plt.savefig(causal_graph_path, bbox_inches='tight')
        plt.show()
        plt.close()
        figs.append(causal_graph_path)
        print(f"✅ Causal graph saved: {causal_graph_path}")

        #  Anomaly Attribution Scores
        attribution_results = gcm.attribute_anomalies(
            causal_model=self.causal_model,
            target_node=target_node,
            anomaly_samples=anomaly_samples,
            anomaly_scorer=None,
            attribute_mean_deviation=False,
            num_distribution_samples=1000,
            shapley_config=None
        )

        # Convert results to DataFrame
        df = pd.DataFrame([
            {
                "Node": node,
                "MeanAttributionScore": np.mean(scores),
                "LowerCI": np.percentile(scores, (1 - confidence_level) / 2 * 100),
                "UpperCI": np.percentile(scores, (1 + confidence_level) / 2 * 100)
            }
            for node, scores in attribution_results.items()
        ]).sort_values("MeanAttributionScore", ascending=False)

        if df.empty:
            print("⚠️ No anomaly attribution results found, skipping plot generation.")
        else:
            plt.figure(figsize=(10, 6))
            plt.barh(df["Node"], df["MeanAttributionScore"], 
                    xerr=[df["MeanAttributionScore"] - df["LowerCI"], df["UpperCI"] - df["MeanAttributionScore"]],
                    color='skyblue', height=0.7)
            plt.title(f"Anomaly Attribution Scores for {target_node}")
            plt.xlabel("Attribution Score")
            plt.gca().invert_yaxis()

            attribution_path = f"{output_dir}/attribution_scores_{target_node}.png"
            plt.savefig(attribution_path, bbox_inches='tight')
            plt.show()
            plt.close()
            figs.append(attribution_path)
            print(f"✅ Attribution scores plot saved: {attribution_path}")

        return df, figs
    
    def attribute_distributional_changes(self, target_node, data_old, data_new, method='distribution_change', confidence_level=0.95):
        """
        Performing distributional change attribution to identify which causal mechanisms changed between two datasets.
        - Limiting selected nodes to a maximum of 10 nodes including the Ancestors + Graph Nodes in the causal graph 
        - Detecting and resolving cycles.
        - Using GCM for causal model fitting.

        Parameters:
        - target_node: The target node whose distribution change we want to explain.
        - data_old: DataFrame representing the system before the change.
        - data_new: DataFrame representing the system after the change.
        - method: 'distribution_change' (default) or 'distribution_change_robust'.
        - confidence_level: Confidence level for confidence intervals (default: 0.95).

        Returns:
        - DataFrame with attribution scores for each node.
        - List of file paths to the saved plots.
        """
        print("\n" + "#"*60)
        print(f"Performing Distributional Change Attribution for target node: {target_node}")
        print(f"Method: {method}")
        print("#"*60)

        # Define the path where plots will be saved
        path = self.global_state.user_data.output_graph_dir

        # Detect and resolve cycles
        graph_matrix = self.G  

        if isinstance(graph_matrix, nx.DiGraph):
            graph_matrix = nx.to_numpy_array(graph_matrix)

        resolved_graph = check_cycle(self.args, self.data, graph_matrix)  

        # Ensure the graph is fully acyclic
        G = nx.from_numpy_array(resolved_graph, create_using=nx.DiGraph)

        if not nx.is_directed_acyclic_graph(G):
            print("❌ Cycle removal failed! The graph is still cyclic.")
            raise ValueError("Cycle removal was unsuccessful. Graph must be a DAG.")
        
        # Store cycle detection results in global state
        self.global_state.inference.cycle_detection_result = {
            "detected_cycles": list(nx.simple_cycles(G)),  
            "final_graph_after_resolution": nx.to_dict_of_lists(G)  
        }

        # Store editing history
        self.global_state.inference.editing_history.append({
        "removed_edges": "Stored inside check_cycle function",
        "final_resolved_graph": nx.to_dict_of_lists(G)})

        # Fix Node Naming Issue: Relabel Graph Nodes
        column_names = list(self.data.columns)
        mapping = {i: column_names[i] for i in range(len(column_names))}
        G = nx.relabel_nodes(G, mapping)

        # Ensure all nodes have assigned causal mechanisms
        self.causal_model = gcm.InvertibleStructuralCausalModel(G)

            
        # Fix Node Naming Issue: Relabel Graph Nodes
        column_names = list(self.data.columns)
        mapping = {i: column_names[i] for i in range(len(column_names))}
        G = nx.relabel_nodes(G, mapping)

        # Limit selected nodes to a max of 10 (Ancestors + Graph Nodes)
        ancestors = list(nx.ancestors(G, target_node))  
        graph_nodes = list(G.nodes)  

        if not ancestors:
            print(f"❌ Target node {target_node} has no ancestors. Not possible to calculate distributional change attribution.")
            return None, []

        # Combine and limit nodes to max 10
        selected_columns = list(set(ancestors + graph_nodes + [target_node]))
        selected_columns = [col for col in selected_columns if col in column_names]

        max_nodes = 10
        if len(selected_columns) > max_nodes:
            print(f"⚠️ More than {max_nodes} nodes selected. Reducing selection.")
            selected_columns = [target_node] + ancestors[:max_nodes - 1]
            remaining_slots = max_nodes - len(selected_columns)
            extra_nodes = [node for node in graph_nodes if node not in selected_columns][:remaining_slots]
            selected_columns.extend(extra_nodes)

        print("✅ Final selected columns (max 10):", selected_columns)

        # Save task info
        self.global_state.inference.task_info.append({
            "target_node": target_node,
            "selected_columns": selected_columns
        })

        # Filter dataset
        data_old = data_old[selected_columns].copy()
        data_new = data_new[selected_columns].copy()
        self.data = self.data[selected_columns].copy()
        self.G = self.G.subgraph(selected_columns)

        # Ensure all nodes have assigned causal mechanisms
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)

        try:
            gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
            gcm.fit(self.causal_model, self.data)
        except ValueError as e:
            print(f"❌ Error assigning causal mechanisms: {e}")
            missing_nodes = [node for node in self.causal_model.graph.nodes if node not in self.causal_model.causal_mechanism]
            print(f"⚠️ Nodes missing causal mechanisms: {missing_nodes}")
            raise

        try:
            gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
            gcm.fit(self.causal_model, self.data)
        except ValueError as e:
            print(f"❌ Error assigning causal mechanisms: {e}")
            missing_nodes = [node for node in self.causal_model.graph.nodes if node not in self.causal_model.causal_mechanism]
            print(f"⚠️ Nodes missing causal mechanisms: {missing_nodes}")
            raise

        # Perform distributional change attribution
        try:
            if method == 'distribution_change':
                attributions = gcm.distribution_change(
                    causal_model=self.causal_model,
                    old_data=data_old,
                    new_data=data_new,
                    target_node=target_node
                )
            elif method == 'distribution_change_robust':
                attributions = gcm.distribution_change_robust(
                    causal_model=self.causal_model,
                    data_old=data_old,
                    data_new=data_new,
                    target_node=target_node
                )
            else:
                raise ValueError("Invalid method. Choose 'distribution_change' or 'distribution_change_robust'.")
        except Exception as e:
            print(f"❌ Error during distribution change computation: {e}")
            raise

        # Convert results to DataFrame
        df = pd.DataFrame([
            {"Node": node, "AttributionScore": score}
            for node, score in attributions.items()
        ]).sort_values("AttributionScore", ascending=False)

        # Extract data for plotting
        nodes = df["Node"]
        scores = df["AttributionScore"]

        # Create bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=nodes, y=scores, palette="coolwarm")
        plt.xlabel("Nodes")
        plt.ylabel("Attribution Score")
        plt.title(f"Distributional Change Attribution for {target_node}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        bar_plot_path = os.path.join(path, 'distribution_change_attribution_bar_plot.png')
        plt.savefig(bar_plot_path, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"✅ Saved bar plot: {bar_plot_path}")

        # Create violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=nodes, y=scores, inner="quartile", palette="coolwarm")
        plt.xlabel("Nodes")
        plt.ylabel("Attribution Score")
        plt.title(f"Distributional Change Attribution for {target_node}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        violin_plot_path = os.path.join(path, 'distribution_change_attribution_violin_plot.png')
        plt.savefig(violin_plot_path, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"✅ Saved violin plot: {violin_plot_path}")

        # Return DataFrame and plot paths
        figs = [bar_plot_path, violin_plot_path]
        return df, figs
    
    # def prepare_treatment_column(self, data, treatment_col, T0=None, T1=None, model_name=None):
    #     treatment = data[treatment_col]

    #     # Case 1: Categorical string
    #     if treatment.dtype == 'object' or treatment.dtype.name == 'category':
    #         print(f"[INFO] Categorical string treatment detected")
    #         if T0 is None or T1 is None:
    #             unique_vals = sorted(treatment.unique().tolist())
    #             T0, T1 = unique_vals[0], unique_vals[1]
    #         return treatment, T0, T1

    #     # Case 2: Binary numeric
    #     if treatment.nunique() == 2:
    #         sorted_vals = sorted(treatment.unique())
    #         print(f"[INFO] Binary treatment detected: {sorted_vals}")
    #         if T0 is None or T1 is None:
    #             T0, T1 = sorted_vals[0], sorted_vals[1]
    #         return treatment, T0, T1

    #     # Case 3: Numeric, low-cardinality, but non-integer
    #     if pd.api.types.is_numeric_dtype(treatment):
    #         unique_vals = sorted(treatment.unique())
    #         if len(unique_vals) <= 10:
    #             print(f"[INFO] Low-cardinality numeric treatment detected: {unique_vals}")
    #             if T0 is None or T1 is None:
    #                 T0, T1 = unique_vals[0], unique_vals[1]
    #             return treatment, T0, T1

    #         # Case 4: Continuous treatment
    #         if model_name == 'CausalForestDML':
    #             print(f"[WARN] CausalForestDML doesn't support continuous treatment. Switching to LinearDML.")
    #             self.global_state.inference.hte_algo_json['name'] = 'LinearDML'

    #         if T0 is None or T1 is None:
    #             T0 = treatment.quantile(0.1)
    #             T1 = treatment.quantile(0.9)
    #         return treatment, T0, T1

    #     raise ValueError(f"Unsupported treatment type: {treatment.dtype}")

    def prepare_treatment_column(
        self,
        data,
        treatment_col,
        T0=None,
        T1=None,
        model_name=None,
        discretize=False
    ):
        treatment = data[treatment_col]

        # Optional quantile-based discretization
        if discretize:
            print(f"[INFO] Discretizing treatment column '{treatment_col}' into quantile bins")
            try:
                # 3 bins: low, medium, high
                treatment_binned = pd.qcut(treatment, q=3, labels=[0, 1, 2])
            except ValueError as e:
                print(f"[WARN] Could not discretize '{treatment_col}': {e}")
                return treatment, T0, T1  # fallback to original treatment

            unique_vals = sorted(treatment_binned.unique())
            if T0 is None or T1 is None:
                T0, T1 = unique_vals[0], unique_vals[-1]  # e.g., 0 vs 2
            return treatment_binned, T0, T1

        # Case 1: String / object-type (leave as-is)
        if treatment.dtype == 'object' or treatment.dtype.name == 'category':
            print(f"[INFO] Categorical string treatment detected")
            unique_vals = sorted(treatment.unique().tolist())
            if T0 is None or T1 is None:
                T0, T1 = unique_vals[0], unique_vals[1]
            return treatment, T0, T1

        # Case 2: Binary numeric
        if treatment.nunique() == 2:
            sorted_vals = sorted(treatment.unique())
            print(f"[INFO] Binary treatment detected: {sorted_vals}")
            if T0 is None or T1 is None:
                T0, T1 = sorted_vals[0], sorted_vals[1]
            return treatment, T0, T1

        # Case 3: Discrete numeric (≤ 10 unique values)
        unique_vals = sorted(treatment.unique())
        if pd.api.types.is_numeric_dtype(treatment) and len(unique_vals) <= 10:
            print(f"[INFO] Low-cardinality numeric treatment detected: {unique_vals}")
            if T0 is None or T1 is None:
                T0, T1 = unique_vals[0], unique_vals[1]
            return treatment, T0, T1

        # Case 4: Continuous treatment → switch model if needed
        if model_name == 'CausalForestDML':
            print("[WARN] CausalForestDML doesn't support continuous treatment. Switching to LinearDML.")
            self.global_state.inference.hte_algo_json['name'] = 'LinearDML'

        if T0 is None or T1 is None:
            T0 = treatment.quantile(0.1)
            T1 = treatment.quantile(0.9)
            print(f"[INFO] Using 10th and 90th percentiles as T0/T1: {T0:.3f}, {T1:.3f}")

        return treatment, T0, T1
    
    def estimate_effect_dml(self, outcome, treatment, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data

        # Step 1: Algorithm selection
        filter = DML_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)

        # Step 2: Determine model and treatment preparation mode
        model_name = self.global_state.inference.hte_algo_json['name']
        discretize = True if model_name == 'CausalForestDML' else False

        # Step 3: Prepare treatment column
        encoded_treatment, T0, T1 = self.prepare_treatment_column(
            self.data, treatment, T0, T1, model_name, discretize=discretize
        )
        self.data[treatment] = encoded_treatment
        self.global_state.user_data.processed_data = self.data

        # Step 4: Re-check model (in case treatment preprocessing switched model)
        model_name = self.global_state.inference.hte_algo_json['name']

        # Step 5: Reranker
        reranker = DML_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)

        # Step 6: Create and fit model
        print(f"[DEBUG] treatment column name: {treatment}, T0: {T0}, T1: {T1}")
        assert isinstance(treatment, str), f"❌ treatment must be str, got {type(treatment)}"
        assert treatment in self.data.columns, f"❌ Column '{treatment}' not found in data columns"
        print(f"[DEBUG] Final T_col = {treatment}")
        print(f"[DEBUG] Columns in data: {self.data.columns.tolist()}")

        programmer = DML_HTE_Programming(self.args, y_col=outcome, T_col=treatment, T0=T0, T1=T1, X_col=X_col, W_col=W_col)
        programmer.fit_model(self.global_state)
        self.global_state.inference.dml_programmer = programmer

        # Step 7: Handle T0/T1 override for discrete models like CausalForestDML
        if model_name == 'CausalForestDML':
            treatment_values = sorted(self.data[treatment].unique())
            T0, T1 = treatment_values[0], treatment_values[1]
            print(f"[INFO] Overriding T0/T1 for CausalForestDML: T0={T0}, T1={T1}")
            programmer.T0 = T0
            programmer.T1 = T1
            if programmer.model:
                programmer.model.T0 = T0
                programmer.model.T1 = T1

        # Step 8: Estimate effects
        print(f"Final T0: {T0}, T1: {T1}")
        print(f"Treatment values seen during fit: {self.data[treatment].unique()}")
        ate, ate_lower, ate_upper = programmer.forward(self.global_state, task='ate')
        att, att_lower, att_upper = programmer.forward(self.global_state, task='att')
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')

        hte = pd.DataFrame({'hte': hte.flatten()})
        output_dir = self.global_state.user_data.output_graph_dir
        os.makedirs(output_dir, exist_ok=True)
        hte.to_csv(f'{output_dir}/hte.csv', index=False)

        result = {
            'ate': [ate, ate_lower, ate_upper],
            'att': [att, att_lower, att_upper],
            'hte': [hte, hte_lower, hte_upper]
        }

        return result

    def estimate_effect_drl(self, outcome, treatment, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data

        # Step 1: Model selection
        filter = DRL_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)

        # Step 2: Get model name
        model_name = self.global_state.inference.hte_algo_json['name']
        discretize = True if model_name in ['DRL','ForestDRL', 'LinearDRL', 'SparseDRL'] else False  # Optional binning strategy

        # Step 3: Prepare treatment
        encoded_treatment, T0, T1 = self.prepare_treatment_column(
            self.data, treatment, T0, T1, model_name, discretize=discretize
        )
        self.data[treatment] = encoded_treatment
        self.global_state.user_data.processed_data = self.data

        # Step 4: Re-check model name if updated during treatment handling
        model_name = self.global_state.inference.hte_algo_json['name']

        # Step 5: Param Selection
        reranker = DRL_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)

        # Step 6: Initialize and fit model
        print(f"[DEBUG] treatment column name: {treatment}, T0: {T0}, T1: {T1}")
        assert isinstance(treatment, str), f"❌ treatment must be str, got {type(treatment)}"
        assert treatment in self.data.columns, f"❌ Column '{treatment}' not found in data columns"
        print(f"[DEBUG] Final T_col = {treatment}")
        print(f"[DEBUG] Columns in data: {self.data.columns.tolist()}")

        programmer = DRL_HTE_Programming(self.args, y_col=outcome, T_col=treatment, T0=T0, T1=T1, X_col=X_col, W_col=W_col)
        programmer.fit_model(self.global_state)
        self.global_state.inference.drl_programmer = programmer

        # Step 7: Ensure T0 and T1 are valid for effect calculation
        treatment_values = sorted(self.data[treatment].unique())
        if T0 not in treatment_values or T1 not in treatment_values:
            T0, T1 = treatment_values[0], treatment_values[-1]
            print(f"[INFO] Overriding T0/T1 to valid values: T0={T0}, T1={T1}")
            programmer.T0 = T0
            programmer.T1 = T1
            if programmer.model:
                programmer.model.T0 = T0
                programmer.model.T1 = T1

        # Step 8: Estimate effects
        print(f"Final T0: {T0}, T1: {T1}")
        print(f"Treatment values seen during fit: {self.data[treatment].unique()}")

        ate, ate_lower, ate_upper = programmer.forward(self.global_state, task='ate')
        att, att_lower, att_upper = programmer.forward(self.global_state, task='att')
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')

        hte = pd.DataFrame({'hte': hte.flatten()})
        output_dir = self.global_state.user_data.output_graph_dir
        os.makedirs(output_dir, exist_ok=True)
        hte.to_csv(f'{output_dir}/hte.csv', index=False)

        return {
            'ate': [ate, ate_lower, ate_upper],
            'att': [att, att_lower, att_upper],
            'hte': [hte, hte_lower, hte_upper]
        }

    
    def contains_iv(self, treatment, outcome):
        """
        Determines if the causal graph contains a valid instrumental variable (IV) for the
        given treatment and outcome pair.
        
        An instrumental variable Z must satisfy three conditions:
        1. Z is associated with the treatment variable (relevance)
        2. Z affects the outcome only through the treatment (exclusion restriction)
        3. Z has no common causes with the outcome (independence)
        
        Args:
            treatment (str): The treatment variable name
            outcome (str): The outcome variable name
            
        Returns:
            tuple: (exists_iv (bool), iv_variable (str or None))
        """
        # Get all nodes in the graph
        all_nodes = list(self.G.nodes())
        
        # Check each node as a potential IV
        for node in all_nodes:
            # Skip if node is the treatment or outcome
            if node == treatment or node == outcome:
                continue
                
            # 1. Check if Z affects T (relevance)
            affects_treatment = False
            for path in nx.all_simple_paths(self.G, node, treatment, cutoff=1):
                if len(path) == 2:  # Direct path from node to treatment
                    affects_treatment = True
                    break
                    
            if not affects_treatment:
                continue
                
            # 2. Check if Z affects Y only through T (exclusion restriction)
            direct_to_outcome = False
            for path in nx.all_simple_paths(self.G, node, outcome, cutoff=1):
                if len(path) == 2:  # Direct path from node to outcome
                    direct_to_outcome = True
                    break
                    
            if direct_to_outcome:
                continue
                
            # 3. Check if Z has no common causes with Y (independence)
            # For simplicity, we check if there are no incoming edges to Z
            # (This is a simplification - a more thorough check would verify Z's parents don't affect Y)
            if len(list(self.G.predecessors(node))) == 0:
                return True, node
                
        return False, None
    
    def estimate_effect_iv(self, outcome, treatment, instrument_variable, T0, T1, X_col, W_col, query):
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data

        # Step 1: Apply Filtering (if necessary)
        filter = MetaLearners_HTE_Filter(self.args)
        self.global_state = filter.forward(self.global_state, query)

        # Step 2: Select the best hyperparameters for the Metalearner
        reranker = MetaLearners_HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)
        
        
        print("W_col being passed:", W_col)
        # Step 3: Choose and initialize the appropriate Metalearner
        programmer = MetaLearners_HTE_Programming(
            self.args, y_col=outcome, T_col=treatment, T0=T0, T1=T1, X_col=X_col
        )
        programmer.fit_model(self.global_state)

        # Step 4: Estimate ATE, ATT, HTE
        ate, ate_lower, ate_upper = programmer.forward(self.global_state, task='ate')
        att, att_lower, att_upper = programmer.forward(self.global_state, task='att')
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')

        # Convert HTE results to a DataFrame
        hte_df = pd.DataFrame({'hte': hte.flatten()})
        print(f"Saving HTE results to: {self.global_state.user_data.output_graph_dir}/hte.csv")
        hte_df.to_csv(f'{self.global_state.user_data.output_graph_dir}/hte.csv', index=False)

        result = {
            'ate': [ate, ate_lower, ate_upper],
            'att': [att, att_lower, att_upper],
            'hte': [hte, hte_lower, hte_upper]
        }
        return result
    


    def estimate_effect_linear(self, 
                               treatment, 
                               outcome, 
                               control_value=0, 
                               treatment_value=1):
        """
        Estimate the causal effect of a treatment on an outcome using
        DoWhy (backdoor.linear_regression).

        :param treatment: str, name of the treatment variable
        :param outcome: str, name of the outcome variable
        :param control_value: value representing 'control'
        :param treatment_value: value representing 'treated'
        :param target_units: 'ate' (default), 'treated', or a callable (lambda df: df[...] for subgroups)

        :return: (causal_estimate, p_value)
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect of Treatment: {treatment} on Outcome: {outcome}")
        print(f"Method: backdoor.linear_regression | target_units='ate'")
        print("#"*60)

        print("\nCreating Causal Model...")
        print(self.dot_graph)
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph  # The DOT-format graph
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        # print("\nIdentified Estimand:")
        # print(identified_estimand)
        # Backdoor linear_regression
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=control_value,
            treatment_value=treatment_value,
            target_units='ate'
        )
        print("\nCausal Estimate:")
        print(causal_estimate)
        # Significance Test
        significance_results = causal_estimate.estimator.test_significance(
            self.data, 
            causal_estimate.value
        )
        p_value = significance_results['p_value'][0]
        print("Significance Test Results:", p_value)

        return causal_estimate, p_value

    def counterfactual_estimation(self, treatment_name, response_name, observed_val = None, intervened_treatment = None):
        # observed_val should be a df as the processed data
        if observed_val is None:
            min_index = self.data[treatment_name].idxmin()
            observed_val = pd.DataFrame([self.data.loc[min_index].to_dict()])
        else:
            column_type = self.global_state.statistics.data_type_column

            for column in self.data.columns:
                if column_type[column] == "Category":
                    observed_cat = observed_val[column].iloc[0]
                    category_mapping = dict(enumerate(self.data[column].cat.categories))
                    observed_cat_code = {v: k for k, v in category_mapping.items()}.get(observed_cat)
                    observed_val.loc[0, column] = observed_cat_code

        if intervened_treatment is None:
            intervened_treatment = self.data[treatment_name].min() + 1.0

        est_sample = gcm.counterfactual_samples(
            self.causal_model,
            {treatment_name: lambda x: intervened_treatment},
            observed_data=observed_val)

        categories = ['Observed', 'Intervened']
        treatment_values = [observed_val[treatment_name].iloc[0], est_sample[treatment_name].iloc[0]]
        response_values = [observed_val[response_name].iloc[0], est_sample[response_name].iloc[0]]

        bar_width = 0.35

        x = np.arange(len(categories))  # [0, 1]

        plt.bar(x - bar_width / 2, treatment_values, width=bar_width, label= treatment_name, color='blue')
        plt.bar(x + bar_width / 2, response_values, width=bar_width, label=response_name, color='orange')

        plt.ylabel('Values')
        plt.title('Counterfactual Estimation')
        plt.xticks(x, categories)
        plt.legend()

        for i, v in enumerate(treatment_values):
            plt.text(i - bar_width / 2, v, f"{v:.1f}", ha='center')
        for i, v in enumerate(response_values):
            plt.text(i + bar_width / 2, v, f"{v:.1f}", ha='center')

        path = self.global_state.user_data.output_graph_dir

        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving counterfactual estimation plot to {os.path.join(path, 'counterfactual_est_fig.jpg')}")
        plt.savefig(os.path.join(path, 'counterfactual_est_fig.jpg'))

    def simulate_intervention(self, treatment_name, response_name,
                              shift_intervention_val = None):
    # def simulate_intervention(self, treatment_name, response_name,
    #                             atomic_intervention_org_val=None, atomic_intervention_new_val=None,
    #                             shift_intervention_val=None):
        '''
            shift_intervention: a numeric value indicating how much shift of interventions have on original observations.
            atomic_intervention_org: the initial observed value of treatment for atomic intervention
            atomic_intervention_new: the value of new intervention on the treatment for atomic intervention
        '''
        # Atomic Intervention
        # if atomic_intervention_org_val is None:
        #     atomic_intervention_org_val = self.data[treatment_name].min()
        #
        # if atomic_intervention_new_val is None:
        #     atomic_intervention_new_val = atomic_intervention_org_val + 1
        #
        # atomic_samples_org = gcm.interventional_samples(self.causal_model,
        #                                             {treatment_name: lambda x: atomic_intervention_org_val},
        #                                             num_samples_to_draw=1000)
        # atomic_samples_new = gcm.interventional_samples(self.causal_model,
        #                                             {treatment_name: lambda x: atomic_intervention_new_val},
        #                                             num_samples_to_draw=1000)
        #
        # plt.figure(figsize=(12, 6))
        #
        # # Histogram for atomic_samples_org
        # plt.subplot(1, 2, 1)
        # plt.hist(atomic_samples_org[response_name], bins=30, color='blue', alpha=0.7)
        # plt.title(f'Original Atomic Samples: {response_name}')
        # plt.xlabel(response_name)
        # plt.ylabel('Frequency')
        #
        # # Histogram for atomic_samples_new
        # plt.subplot(1, 2, 2)
        # plt.hist(atomic_samples_new[response_name], bins=30, color='orange', alpha=0.7)
        # plt.title(f'New Atomic Samples: {response_name}')
        # plt.xlabel(response_name)
        # plt.ylabel('Frequency')
        #
        # plt.tight_layout()
        # # plt.show()
        #
        # path = self.global_state.user_data.output_graph_dir
        # if not os.path.exists(path):
        #     os.makedirs(path)
        #
        # print(f"Saving atomic intervention comparison plot to {os.path.join(path, 'atomic_intervention.jpg')}")
        # plt.savefig(os.path.join(path, 'atomic_intervention.jpg'))

        # Shift Intervention
        if shift_intervention_val is None:
            shift_intervention_val = 1
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)
        shift_samples = gcm.interventional_samples(self.causal_model,
                                                    {treatment_name: lambda x: x + shift_intervention_val},
                                                    num_samples_to_draw=len(self.data))

        plt.figure(figsize=(8, 6))
        # Histogram for atomic_samples_org
        plt.subplot(1, 2, 1)
        plt.hist(self.data[response_name], bins=30, color='#409fde', alpha=0.7)
        plt.title(f'Observed Data: {response_name}')
        plt.xlabel(response_name)
        plt.ylabel('Frequency')

        # Histogram for atomic_samples_new
        plt.subplot(1, 2, 2)
        plt.hist(shift_samples[response_name], bins=30, color='#f77f3e', alpha=0.7)
        plt.title(f'Shift Intervention: {response_name}')
        plt.xlabel(response_name)
        plt.ylabel('Frequency')

        plt.tight_layout()
        # plt.show()

        path = self.global_state.user_data.output_graph_dir
        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Saving shift intervention comparison plot to {os.path.join(path, 'shift_intervention.jpg')}")
        plt.savefig(os.path.join(path, 'shift_intervention.jpg'))
        # figs = [os.path.join(path, 'shift_intervention.jpg')]

        # Save dataset
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_org.csv')}")
        # atomic_samples_org.to_csv(os.path.join(path, 'simulated_atomic_intervention_org.csv'), index=False)
        #
        # print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention_new.csv')}")
        # atomic_samples_new.to_csv(os.path.join(path, 'simulated_atomic_intervention_new.csv'), index=False)

        print(f"Saving simulated dataset {os.path.join(path, 'simulated_shift_intervention.csv')}")
        shift_samples.to_csv(os.path.join(path, 'simulated_shift_intervention.csv'), index=False)
        # Boxplot Comparison to summarize distributions
        plt.figure(figsize=(8, 6))
        data_to_plot = [self.data[response_name], shift_samples[response_name]]
        plt.boxplot(data_to_plot, labels=["Observed Data", "Shift Intervention"], patch_artist=True)
        plt.title(f'Shift Intervention Comparison: {response_name}')
        plt.ylabel(response_name)
        print(f"Saving box-plot comparison plot to {os.path.join(path, 'shift_intervention_boxplot.jpg')}")
        plt.savefig(os.path.join(path, 'shift_intervention_boxplot.jpg'))
        figs_boxplot = os.path.join(path, 'shift_intervention_boxplot.jpg')
        # figs.append(figs_boxplot)

        # Violin plot comparison
        plt.figure(figsize=(8, 6))
        plt.violinplot([self.data[response_name], shift_samples[response_name]], showmeans=True)
        plt.xticks([1, 2], ["Observed Data", "Shift Intervention"])
        plt.title(f'Shift Intervention Violin Plot: {response_name}')
        plt.ylabel(response_name)
        print(f"Saving violin plot comparison to {os.path.join(path, 'shift_intervention_violinplot.jpg')}")
        plt.savefig(os.path.join(path, 'shift_intervention_violinplot.jpg'))
        figs_violin = os.path.join(path, 'shift_intervention_violinplot.jpg')
        figs = [figs_violin]
             
        return figs, shift_samples


    def sensitivity_analysis(self, target_node, model, estimand, estimate, treatment, outcome):
        # if self.global_state.statistics.linearity:
        #      simulation_method = "linear-partial-R2"
        # else:
        #      simulation_method = "non-parametric-partial-R2",

        # Use the most important factor as the benchmark_common_causes
        file_exists = os.path.exists(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        if not file_exists:
            self.feature_importance(target_node, visualize=False)
        shap_df = pd.read_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        max_col = shap_df.mean().idxmax()

        print('model.get_common_causes',model.get_common_causes())
        if model.get_common_causes() == [] or model.get_common_causes() is None:
            refute = model.refute_estimate(estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9)
            figs = []
        else:
            refute = model.refute_estimate(estimand, estimate,
                                method_name = "add_unobserved_common_cause",
                                simulation_method = "non-parametric-partial-R2",
                                benchmark_common_causes = [max_col],
                                effect_fraction_on_outcome = [1,2,3]
                                )
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/ate_refutation.png')
            plt.close()
            figs = ['ate_refutation.png']
        print(refute)

        return refute, figs 

    
    def forward(self, task, desc, key_node, chat_history):
        if task == 'Feature Importance':
            linearity = self.global_state.statistics.linearity
            if linearity:
                chat_history.append((None, "💡 We use the linear model to calculate SHAP Value because your dataset is linear."))
            else:
                chat_history.append((None, "💡 We use the non-linear model to calculate SHAP Value because your dataset is non-linear."))
            parent_nodes, mean_shap_values, figs = self.feature_importance(key_node, linearity, visualize=True)
            response = generate_analysis_feature_importance(self.args, key_node, parent_nodes, mean_shap_values, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history 
        
        elif task == 'Treatment Effect Estimation':
            task_info = self.global_state.inference.task_info[self.global_state.inference.task_index]
            confounders = task_info['confounders']
            cont_confounders = task_info['cont_confounders']
            treatment = task_info['treatment']
            treat = task_info['treat']
            control = task_info['control']
            hte_variables = task_info['X_col']
            parent_nodes = list(self.G.predecessors(key_node))
            method = task_info['hte_method']

            ### Run algorithm
            if method in ["dml", "drl"]:
                if method == "dml":
                    result = self.estimate_effect_dml(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                            X_col=hte_variables, W_col=confounders, query=desc)
                else:
                    result = self.estimate_effect_drl(outcome=key_node, treatment=treatment, T0=control, T1=treat,
                                                            X_col=hte_variables, W_col=confounders, query=desc)
                response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variables, confounders, result, desc)
                chat_history.append(("📝 Analyze for ATE and ATT...", None))
                chat_history.append((None, response[0]))
                chat_history.append(("📝 Analyze for HTE...", None))
                for fig in figs:
                    chat_history.append((None, (f'{fig}',)))
                chat_history.append((None, response[1]))

            # TODO: Add MetaLearner Estimation
                
            elif method == "iv":
                exist_IV, iv_variable = self.contains_iv(treatment, key_node)
                result = self.estimate_effect_iv(outcome=key_node, treatment=treatment, instrument_variable=iv_variable, T0=control, T1=treat,
                                                        X_col=hte_variables, W_col=confounders, query=desc)
                response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variables, confounders, result, desc)
                chat_history.append(("📝 Analyze for ATE and ATT...", None))
                chat_history.append((None, response[0]))
                chat_history.append(("📝 Analyze for HTE...", None))
                for fig in figs:
                    chat_history.append((None, (f'{fig}',)))
                chat_history.append((None, response[1]))
                
            # elif method == "uplift":
            #     result = self.estimate_effect_uplift(outcome=key_node, treatment=treatment, T0=control, T1=treat,
            #                                             X_col=hte_variables, W_col=confounders, query=desc)
            #     response, figs = generate_analysis_econml(self.args, self.global_state, key_node, treatment, parent_nodes, hte_variables, confounders, result, desc)
                
            #     # Add special uplift-specific formatting for chat history
            #     chat_history.append(("📝 Analyzing Individual Treatment Effects with Uplift Modeling...", None))
            #     chat_history.append((None, response[0]))
            #     chat_history.append(("📝 Analyzing Heterogeneous Treatment Effects...", None))
                
            #     # Add uplift-specific visualizations
            #     for fig in figs:
            #         chat_history.append((None, (f'{fig}',)))
            #     chat_history.append((None, response[1]))
                
            elif method in ["cem", "propensity_score"]:
                # Perform matching-based estimation
                ate, cate_result, matched_data, figs = self.estimate_causal_effect_matching(
                    treatment=treatment,
                    outcome=key_node,
                    confounders=confounders,
                    X_col=hte_variables,
                    treat=treat,
                    control=control,
                    cont_confounders=cont_confounders,
                    method=method,
                    visualize=True
                )
                response = generate_analysis_matching(self.args, treatment, key_node, method, confounders, ate, cate_result, desc)
                chat_history.append(("📝 Matching Balance Checking...", None))
                chat_history.append((None, (f'{figs[0]}',)))
                chat_history.append((None, response[0]))
                chat_history.append(("📝 Analyze for ATE...", None))
                chat_history.append((None, response[1]))
                chat_history.append(("📝 Analyze for CATE...", None))
                chat_history.append((None, (f'{figs[1]}',)))
                chat_history.append((None, response[2]))

            elif method == 'linear_regression':
                ate, p_value = self.estimate_effect_linear(
                               treatment=treatment,
                               outcome=key_node,
                               control_value=control, 
                               treatment_value=treat)
                response = generate_analysis_linear_regression(args, treatment, key_node, ate, p_value, desc)
            
            return response, figs, chat_history
        
        elif task == 'Anormaly Attribution':
            df, figs = self.attribute_anomalies(target_node=key_node, anomaly_samples=self.data, confidence_level=0.95)
            parent_nodes = list(self.G.predecessors(key_node))
            if len(parent_nodes) == 0:
                chat_history.append((None, "⚠️ No parent nodes found for the target node."))
                return None, None, chat_history
            response = generate_analysis_anormaly(self.args, df, key_node, parent_nodes, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history
                       
        elif task == 'Distributional Change Attribution':
            # Split the dataset into two subsets (data_old and data_new)
            # For demonstration, we split the dataset into two halves
            data_old = self.data.iloc[:len(self.data)//2]  # First half as "old" data
            data_new = self.data.iloc[len(self.data)//2:]  # Second half as "new" data

            # Perform distributional change attribution
            df, figs = self.attribute_distributional_changes(target_node=key_node, data_old=data_old, data_new=data_new)
            response = generate_analysis_anormaly_dist(self.args, df, key_node, desc)
            return response, figs

        elif task == 'Counterfactual Estimation':
            treatment = self.global_state.inference.task_info[self.global_state.inference.task_index]['treatment']
            shift_intervention_val = self.global_state.inference.task_info[self.global_state.inference.task_index]['shift_value']
            figs, shift_df = self.simulate_intervention(treatment_name = treatment, response_name = key_node, shift_intervention_val = shift_intervention_val)
            response = generate_conterfactual_estimation(self.args, self.global_state, shift_intervention_val, shift_df, treatment, key_node, desc)
            for fig in figs:
                chat_history.append((None, (f'{fig}',)))
            chat_history.append((None, response))
            return response, figs, chat_history
        else:
            return None, None, chat_history


if __name__ == '__main__':
    import argparse
    import pickle
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="./demo_data/20250130_130622/hotel_booking_cancellation.csv",
            help='Path to the input dataset file (e.g., CSV format or directory location)'
        )

        # Output file for results
        parser.add_argument(
            '--output-report-dir',
            type=str,
            default='causal_analysis/test_result',
            help='Directory to save the output report'
        )

        # Output directory for graphs
        parser.add_argument(
            '--output-graph-dir',
            type=str,
            default='causal_analysis/test_result',
            help='Directory to save the output graph'
        )

        parser.add_argument(
            '--apikey',
            type=str,
            default=None,
            help='API Key'
        )

        parser.add_argument(
            '--simulation_mode',
            type=str,
            default="offline",
            help='Simulation mode: online or offline'
        )

        parser.add_argument(
            '--data_mode',
            type=str,
            default="real",
            help='Data mode: real or simulated'
        )

        parser.add_argument(
            '--debug',
            action='store_true',
            default=False,
            help='Enable debugging mode'
        )

        parser.add_argument(
            '--initial_query',
            type=str,
            default="selected algorithm: PC",
            help='Initial query for the algorithm'
        )
        args = parser.parse_args()
        return args
    args = parse_args()

    # Load global state
    with open('./demo_data/20250130_130622/hotel_booking_cancellation/output_graph/PC_global_state.pkl', 'rb') as file:
        global_state = pickle.load(file)
        analysis = Analysis(global_state, args)

        result = analysis.estimate_effect_dml(outcome ='is_canceled', 
                                              treatment = 'previous_cancellations', 
                                              T0=0, T1=1, 
                                              X_col =['is_repeated_guest', 'previous_bookings_not_canceled', 'booking_changes', 'deposit_type','customer_type'],
                                              W_col=[],
                                              query='Find the treatment effect')
        

    

        def _to_scalar(x):
            return x.item() if isinstance(x, np.ndarray) and x.size == 1 else x

        ate, ate_lower, ate_upper = map(_to_scalar, result['ate'])
        att, att_lower, att_upper = map(_to_scalar, result['att'])

        print(f"\n=== ATE ===")
        print(f"ATE: {ate:.4f} (95% CI: [{ate_lower:.4f}, {ate_upper:.4f}])")

        print(f"\n=== ATT ===")
        print(f"ATT: {att:.4f} (95% CI: [{att_lower:.4f}, {att_upper:.4f}])")


        # result = analysis.estimate_effect_dml(outcome ='is_canceled', 
        #                                       treatment = 'previous_cancellations', 
        #                                       T0=0, T1=1, 
        #                                       X_col =['is_repeated_guest', 'previous_bookings_not_canceled', 'booking_changes', 'deposit_type','customer_type'],
        #                                       W_col=[],
        #                                       query='Find the treatment effect')
        
        #  result = analysis.estimate_effect_dml(outcome ='deaths', 
        #                                       treatment = 'richter', 
        #                                       T0=0, T1=1, 
        #                                       X_col =['year', 'month', 'day', 'region','area'],
        #                                       W_col=[],
        #                                       query='Find the treatment effect')   

        # result = analysis.estimate_effect_dml(outcome ='SalePrice', 
        #                                       treatment = 'HouseStyle', 
        #                                       T0=0, T1=1, 
        #                                       X_col =['OverallQual', 'OverallCond', 'YearBuilt', 'GarageArea'],
        #                                       W_col=[],
        #                                       query='Find the treatment effect')

        




    # # Ensure inference attributes exist in global_state
    # if not hasattr(global_state, "inference"):
    #     global_state.inference = Inference()

    # if not hasattr(global_state.inference, "editing_history"):
    #     global_state.inference.editing_history = []

    # if not hasattr(global_state.inference, "cycle_detection_result"):
    #     global_state.inference.cycle_detection_result = {}

    # if not hasattr(global_state.inference, "task_info"):
    #     global_state.inference.task_info = []

    # print(global_state.user_data.processed_data["CentralAir"].value_counts(normalize=True))
    # print("Missing values in dataset:\n", global_state.user_data.processed_data.isnull().sum())
    # print(global_state.user_data.processed_data.head())

    # Initialize analysis
    # my_analysis = Analysis(global_state, args)

    # # Ensure processed_data exists
    # if not hasattr(global_state, "user_data") or not hasattr(global_state.user_data, "processed_data"):
    #     raise ValueError("Processed data is missing from global_state. Ensure data is properly loaded.")

    # treatment_col = "CentralAir"  # Adjust if necessary

    # # Check if treatment column exists
    # if treatment_col in global_state.user_data.processed_data.columns:
    #     unique_treatment_values = global_state.user_data.processed_data[treatment_col].unique()
    #     print(f"Unique values in {treatment_col}:", unique_treatment_values)

    #     if len(unique_treatment_values) < 2:
    #         raise ValueError(f"Treatment column '{treatment_col}' has insufficient variation: {unique_treatment_values}")
    # else:
    #     raise ValueError(f"Treatment column '{treatment_col}' not found in dataset.")
    
    # global_state.user_data.processed_data["CentralAir"] = (
    # global_state.user_data.processed_data["CentralAir"] > 0).astype(int)  # Converts to 0 or 1
    # print("Fixed Unique values in CentralAir:", global_state.user_data.processed_data["CentralAir"].unique())
    


    # # Run the MetaLearner estimation
    # result = my_analysis.estimate_effect_MetaLearners(
    #     outcome='SalePrice', 
    #     treatment='CentralAir', 
    #     T0=0, T1=1, 
    #     X_col=['LotArea', 'OverallQual'], 
    #     W_col=['YearBuilt', 'GrLivArea', 'GarageCars'], 
    #     query='What is the treatment effect of CentralAir on SalePrice'
    # )

    # print("Unique Treatment Values in Data:", global_state.user_data.processed_data['CentralAir'].unique())
    # print("Available keys in global_state.statistics.data_type_column:", global_state.statistics.data_type_column.keys())
    # print("\nFinal MetaLearner Estimation Results:", result)

    
    # args = parse_args()
    # # with open('demo_data/20250121_223113/lalonde/output_graph/PC_global_state.pkl', 'rb') as file:
    # #     global_state = pickle.load(file)
    
    # # my_analysis = Analysis(global_state, args)
    # # # my_analysis.estimate_effect_dml(outcome='re78', treatment='treat', 
    # # #                                 T0=0, T1=1, 
    # # #                                 X_col=['age', 'nodegr'], 
    # # #                                 W_col=['educ', 'age', 'married', 'nodegr'], 
    # # #                                 query='What is the treatment effect of treat on re78')
    # # my_analysis.feature_importance(target_node='re78', linearity=False, visualize=True)
    # # #my_analysis.simulate_intervention(treatment_name = 'married', response_name = 're78', shift_intervention_val =1)
    # with open('./demo_data/20250130_130622/house_price/output_graph/PC_global_state.pkl', 'rb') as file:
    #     global_state = pickle.load(file)

    # # Dynamically update `inference`
    # if not hasattr(global_state, "inference"):
    #     global_state.inference = Inference()

    # if not hasattr(global_state.inference, "editing_history"):
    #     global_state.inference.editing_history = []

    # if not hasattr(global_state.inference, "cycle_detection_result"):
    #     global_state.inference.cycle_detection_result = {}

    # if not hasattr(global_state.inference, "task_info"):
    #     global_state.inference.task_info = []

    # my_analysis = Analysis(global_state, args)

    # treatment_col = "CentralAir"  # Adjust if necessary
    # if treatment_col in global_state.user_data.processed_data:
    #     unique_treatment_values = global_state.user_data.processed_data[treatment_col].unique()
    #     print(f"Unique values in {treatment_col}:", unique_treatment_values)

    # if len(unique_treatment_values) < 2:
    #     raise ValueError(f"Treatment column '{treatment_col}' has insufficient variation: {unique_treatment_values}")
    # else:
    #     raise ValueError(f"Treatment column '{treatment_col}' not found in dataset.")
    
    # result = my_analysis.estimate_effect_MetaLearners(outcome='SalePrice', treatment='CentralAir', 
    #                      T0=0, T1=1, 
    #                      X_col=['LotArea', 'OverallQual'], 
    #                      W_col=['YearBuilt', 'GrLivArea', 'GarageCars'], 
    #                      query='What is the treatment effect of CentralAir on SalePrice')
    # print("Unique Treatment Values in Data:", global_state.user_data.processed_data['CentralAir'].unique())
    # print("Available keys in global_state.statistics.data_type_column:", global_state.statistics.data_type_column.keys())
    # print("\nFinal MetaLearner Estimation Results:", result)


    # # result = my_analysis.estimate_effect_MetaLearners(outcome='re78', treatment='treat', 
    # #                                 T0=0, T1=1, 
    # #                                 X_col=['age', 'nodegr'], 
    # #                                 W_col=['educ', 'age', 'married', 'nodegr'], 
    # #                                 query='What is the treatment effect of treat on re78')
    # # my_analysis.feature_importance(target_node='SalePrice', linearity=False, visualize=True)
    # # anomaly_samples = my_analysis.data.head(10)  # Replace with your anomaly DataFrame
    # # anomaly_df, anomaly_figs = my_analysis.attribute_anomalies(
    # #     target_node='SalePrice', 
    # #     anomaly_samples=anomaly_samples, 
    # #     confidence_level=0.95
    # # )
    # # print("Anomaly Attribution Results:", anomaly_df)
    # #my_analysis.simulate_intervention(treatment_name = 'married', response_name = 're78', shift_intervention_val =1)

    # # # Initialize analysis
    # # my_analysis = Analysis(global_state, args)

    # # # Load dataset and split into `data_old` and `data_new`
    # # full_data = my_analysis.data  # Load the full dataset
    # # split_point = len(full_data) // 2  # Split into two halves
    # # data_old = full_data.iloc[:split_point]  # First half as "old" data
    # # data_new = full_data.iloc[split_point:]  # Second half as "new" data

    # # # Test `attribute_distributional_changes`
    # # target_node = 'SalePrice'  # Change to your variable of interest
    # # method = 'distribution_change'  # or 'distribution_change_robust'
    
    # # dist_change_df, dist_change_figs = my_analysis.attribute_distributional_changes(
    # #     target_node=target_node,
    # #     data_old=data_old,
    # #     data_new=data_new,
    # #     method=method,
    # #     confidence_level=0.95
    # # )

    # # print("Distributional Change Attribution Results:\n", dist_change_df)
    # # Print results
    
