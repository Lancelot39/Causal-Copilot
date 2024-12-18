import os
import numpy as np
import pandas as pd
import networkx as nx
from dowhy import CausalModel
from dowhy import gcm
import shap
import sklearn
import matplotlib.pyplot as plt



def convert_adj_mat(mat):
    # In downstream analysis, we only keep direct edges and ignore all undirect edges
    mat = (mat==1).astype(int)
    G = mat.T
    return G

class Analysis(object):
    def __init__(self, global_state, args):
        """
        Hardcoded to test the Auto MPG dataset and adjacency separately,
        while preserving references to global_state and args (which can be None).
        """
        self.global_state = global_state
        self.args = args
        # self.data = global_state.user_data.processed_data
        self.data = pd.read_csv('dataset/Auto_mpg/Auto_mpg.csv')
        # self.graph = convert_adj_mat(global_state.result.revised_graph)
        self.graph = convert_adj_mat(np.load('dataset/Auto_mpg/base_graph.npy'))

        # convert adj matrix into DiGraph
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph)
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})
        
        # Construct Causal Model
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)

    def feature_importance(self, target_node, visualize=True):
        parent_relevance, noise_relevance = gcm.parent_relevance(self.causal_model, target_node=target_node)
        # parent_relevance, noise_relevance
        parent_nodes = list(self.G.predecessors(target_node))

        X = self.data.drop(columns=[target_node])
        y = self.data[[target_node]]
        X100 = shap.utils.sample(X, 100)  # 100 instances for use as the background distribution

        # SHAP value for a simple linear model
        model_linear = sklearn.linear_model.LinearRegression()
        model_linear.fit(X, y)
        explainer_linear = shap.Explainer(model_linear.predict, X100)
        shap_values_linear = explainer_linear(X)
        
        # Calculate the mean SHAP value for each feature
        shap_df = pd.DataFrame(np.abs(shap_values_linear.values), columns=X.columns)
        mean_shap_values = shap_df.mean()
        
        if visualize:
            # Output directory
            if self.global_state and self.global_state.user_data.output_graph_dir:
                output_dir = self.global_state.user_data.output_graph_dir
            else:
                output_dir = "./auto_mpg_output"  # local folder for standalone testing
            
            os.makedirs(output_dir, exist_ok=True)

            # 1st SHAP Plot beeswarm
            ax = shap.plots.beeswarm(shap_values_linear, plot_size=(8,6), show=False)
            plt.savefig(f'{output_dir}/shap_beeswarm_plot.png', bbox_inches='tight')
            # plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            # plt.show()

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8,6))
            shap.plots.bar(shap_values_linear, ax=ax, show=False)
            plt.savefig(f'{output_dir}/shap_bar_plot.png', bbox_inches='tight')
            # plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png', bbox_inches='tight')  # Save as PNG
            # plt.show()
            plt.close()

        return parent_nodes, mean_shap_values

    def estimate_causal_effect(self, treatment, outcome, control_value=0, treatment_value=1):
        """
        Estimate the causal effect of a treatment on an outcome using DoWhy (backdoor.linear_regression).
        """
        print("Creating Causal Model...")
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self._generate_dowhy_graph()
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("Identified Estimand:", identified_estimand)

        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=control_value,
            treatment_value=treatment_value
        )
        print("Causal Estimate:", causal_estimate)
        model.test_significance(causal_estimate)

        return causal_estimate

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


def main(global_state, args):
    analysis = Analysis(global_state, args)
    # analysis.feature_importance('mpg', visualize=False)
    # Feature Importance
    target_node = 'mpg'  # Example outcome variable
    parents, shap_values = analysis.feature_importance(target_node, visualize=True)
    print(f"Parent nodes of {target_node}:", parents)
    print("Mean SHAP values:\n", shap_values)

    # Causal Effect Estimation
    treatment = 'horsepower'  # Example treatment variable
    outcome = 'mpg'           # Example outcome variable
    causal_estimate = analysis.estimate_causal_effect(treatment, outcome)
    print("Estimated Causal Effect:", causal_estimate)


if __name__ == "__main__":
    main(None, None)