import numpy as np
import pandas as pd
import networkx as nx
from dowhy import CausalModel
from dowhy import gcm
import shap
import sklearn
import matplotlib.pyplot as plt
import os

def convert_adj_mat(mat):
    # In downstream analysis, we only keep direct edges and ignore all undirected edges
    mat = (mat == 1).astype(int)
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

        # Hardcoded dataset
        self.data = pd.read_csv('dataset/Auto_mpg/Auto_mpg.csv')
        # Originally, we planned to use processed_data from global_state:
        # self.data = global_state.user_data.processed_data

        # If global_state.result.revised_graph existed, it’s overridden by the next line:
        # self.graph = convert_adj_mat(global_state.result.revised_graph)
        
        # Now, we hardcode the graph:
        self.graph = convert_adj_mat(np.load('dataset/Auto_mpg/base_graph.npy'))

        # Dynamically analyze column types and print disclaimers
        # (Previously we had a hard-coded disclaimer message, now removed)
        # print("\n" + "="*60)
        # print("DISCLAIMER: The Auto MPG dataset primarily contains continuous numeric features.")
        # print("Make sure this aligns with your causal assumptions and modeling approaches.")
        # print("="*60 + "\n")
        # Instead of the above hard-coded disclaimer, we now analyze the columns:
        self._print_data_disclaimer()

        # Create DiGraph
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph)
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})

        # Construct Causal Model via dowhy/gcm
        self.causal_model = gcm.InvertibleStructuralCausalModel(self.G)
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)

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

    def feature_importance(self, target_node, visualize=True):
        """
        Calculate parent relevance (gcm.parent_relevance) and SHAP values for a linear model.
        """
        print("\n" + "#"*60)
        print(f"Calculating Feature Importance for Target Node: {target_node}")
        print("#"*60)

        parent_relevance, noise_relevance = gcm.parent_relevance(self.causal_model, target_node=target_node)
        parent_nodes = list(self.G.predecessors(target_node))

        X = self.data.drop(columns=[target_node])
        y = self.data[[target_node]]
        X100 = shap.utils.sample(X, 100)  # background distribution for SHAP

        model_linear = sklearn.linear_model.LinearRegression()
        model_linear.fit(X, y)
        explainer_linear = shap.Explainer(model_linear.predict, X100)
        shap_values_linear = explainer_linear(X)
        
        # Mean absolute SHAP values
        shap_df = pd.DataFrame(np.abs(shap_values_linear.values), columns=X.columns)
        mean_shap_values = shap_df.mean()

        print("\n=== Feature Importance Results ===")
        print(f"Parent nodes of {target_node}: {parent_nodes}")
        print("\nMean SHAP values:\n", mean_shap_values)
        print("=================================\n")

        if visualize:
            # Output directory fallback
            if self.global_state and self.global_state.user_data.output_graph_dir:
                output_dir = self.global_state.user_data.output_graph_dir
            else:
                output_dir = "./auto_mpg_output"  # local folder for standalone testing

            os.makedirs(output_dir, exist_ok=True)

            print("Generating SHAP plots...\n")
            # Beeswarm
            shap.plots.beeswarm(shap_values_linear, plot_size=(8,6), show=False)
            plt.savefig(f'{output_dir}/shap_beeswarm_plot.png', bbox_inches='tight')
            plt.show()

            # Bar
            fig, ax = plt.subplots(figsize=(8,6))
            shap.plots.bar(shap_values_linear, ax=ax, show=False)
            plt.savefig(f'{output_dir}/shap_bar_plot.png', bbox_inches='tight')
            plt.show()
            plt.close()

        return parent_nodes, mean_shap_values

    def estimate_causal_effect(self, treatment, outcome, control_value=0, treatment_value=1):
        """
        Estimate the causal effect of a treatment on an outcome using DoWhy (backdoor.linear_regression).
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect of Treatment: {treatment} on Outcome: {outcome}")
        print("#"*60)

        print("\nCreating Causal Model...")
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self._generate_dowhy_graph()
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("\nIdentified Estimand:")
        print(identified_estimand)

        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            control_value=control_value,
            treatment_value=treatment_value
        )

        print("\nCausal Estimate:")
        print(causal_estimate)
        # Call test_significance with the estimate value
        significance_results = causal_estimate.estimator.test_significance(self.data,causal_estimate.value)
        print("Significance Test Results:", significance_results['p_value'][0])

        print("\n=== Interpretation Hint ===")
        print("A negative causal estimate indicates that increasing the treatment variable (e.g., horsepower)")
        print("tends to decrease the outcome variable (e.g., mpg), assuming the model and assumptions hold.")
        print("============================\n")

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
    
    def attribute_anomalies(self, target_node, anomaly_samples, confidence_level=0.95):
        """
        Perform anomaly attribution and save the bar chart with confidence intervals to ./auto_mpg_output.
        """
        print("\n" + "#"*60)
        print(f"Performing Anomaly Attribution for target node: {target_node}")
        print("#"*60)

        # Call the attribute_anomalies function
        attribution_results = gcm.attribute_anomalies(
            causal_model=self.causal_model,  # Your fitted InvertibleStructuralCausalModel
            target_node=target_node,        # The target node for anomaly attribution
            anomaly_samples=anomaly_samples,  # DataFrame of anomalous samples
            anomaly_scorer=None,            # Use default anomaly scorer
            attribute_mean_deviation=False, # Attribute anomaly score (not mean deviation)
            num_distribution_samples=3000,  # Number of samples for marginal distribution
            shapley_config=None             # Use default Shapley config
        )

        # Convert results to a DataFrame
        rows = []
        for node, contributions in attribution_results.items():
            rows.append({
                "Node": node,
                "MeanAttributionScore": np.mean(contributions),
                "LowerCI": np.percentile(contributions, (1 - confidence_level) / 2 * 100),
                "UpperCI": np.percentile(contributions, (1 + confidence_level) / 2 * 100)
            })

        df = pd.DataFrame(rows).sort_values("MeanAttributionScore", ascending=False)

        # Prepare output directory
        output_dir = "./auto_mpg_output"
        os.makedirs(output_dir, exist_ok=True)

        # Extract info for plotting
        nodes = df["Node"]
        scores = df["MeanAttributionScore"]
        lower_bounds = df["LowerCI"]
        upper_bounds = df["UpperCI"]
        error = np.array([scores - lower_bounds, upper_bounds - scores])

        # Create figure
        plt.figure(figsize=(10, 6))  # Adjusted figure size for better readability
        plt.bar(nodes, scores, yerr=error, align='center', ecolor='black', capsize=5, color='skyblue')
        plt.xlabel("Nodes")
        plt.ylabel("Mean Attribution Score")
        plt.title(f"Anomaly Attribution for {target_node}")
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, 'attribution_plot.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.show()  # Show the plot in the notebook or script
        plt.close()

        return df
    
def main(global_state, args):
    """
    Modify the main function to call attribute_anomalies and save the results in ./auto_mpg_output.
    """
    print("Welcome to the Causal Analysis Demo using the Auto MPG dataset.\n")
    
    analysis = Analysis(global_state, args)
    
    # Feature Importance
    target_node = 'mpg'
    parents, shap_values = analysis.feature_importance(target_node, visualize=True)

    # Causal Effect Estimation
    treatment = 'horsepower'
    outcome = 'mpg'
    causal_estimate = analysis.estimate_causal_effect(treatment, outcome)

    # Ask the user for anomaly values for all columns
    print("\nPlease provide the anomaly values for the following variables:")
    anomaly_values = {}
    for node in analysis.data.columns:  # Use all columns from the dataset
        value = float(input(f"Enter the anomaly value for '{node}': "))
        anomaly_values[node] = value
    anomaly_samples = pd.DataFrame([anomaly_values])

    # Perform anomaly attribution
    df = analysis.attribute_anomalies(target_node=target_node, anomaly_samples=anomaly_samples)

    # Save the DataFrame results as well
    output_dir = "./auto_mpg_output"
    os.makedirs(output_dir, exist_ok=True)
    df_path = os.path.join(output_dir, 'attribution_results.csv')
    df.to_csv(df_path, index=False)

    print("All steps completed. The analysis, plots, and anomaly attribution results are saved in './auto_mpg_output'.")

if __name__ == "__main__":
    main(None, None)