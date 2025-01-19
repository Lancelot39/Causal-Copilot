import numpy as np
import pandas as pd
import networkx as nx
from dowhy import gcm, CausalModel
from dowhy import gcm, CausalModel
import shap
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from pydantic import BaseModel
import os 
import sys 
import warnings

from causal_analysis.hte.hte_filter import HTE_Filter
from causal_analysis.hte.hte_params import HTE_Param_Selector
from causal_analysis.hte.hte_program import HTE_Programming
#from hte import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from global_setting.Initialize_state import global_state_initialization

def convert_adj_mat(mat):
    # In downstream analysis, we only keep direct edges and ignore all undirected edges
    mat = np.array(mat)
    mat = (mat == 1).astype(int)
    G = mat.T
    return G

def plot_hte_dist(hte, fig_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(hte['hte'], bins=30, kde=True, color='skyblue', alpha=0.7)
    plt.axvline(hte['hte'].mean(), color='firebrick', linestyle='--', label='Mean HTE')
    plt.xlabel("Heterogeneous Treatment Effect (HTE)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Heterogeneous Treatment Effects")
    # Save figure
    plt.savefig(fig_path)

def plot_cate_violin(data, hte, group_cols, fig_path):
    data = pd.concat([data, hte], axis=1)
    num_groups = len(group_cols)
    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=False)
    if num_groups == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for ax, group_col in zip(axes, group_cols):
        sns.violinplot(
            x=group_col, y='hte', data=data, ax=ax, inner="quartile", scale="width"
        )
        # Customize the subplot
        ax.set_title(f"CATE Distribution by {group_col.capitalize()}")
        ax.set_xlabel(group_col.capitalize())
        ax.set_ylabel("CATE")
        ax.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(fig_path)

class Analysis(object):
    def __init__(self, global_state, args):
        """
        Hardcoded to test the Auto MPG dataset and adjacency separately,
        while preserving references to global_state and args (which can be None).
        """
        self.global_state = global_state
        self.args = args
        self.data = global_state.user_data.processed_data
        #TODO: graph format
        self.graph = convert_adj_mat(global_state.results.revised_graph)
        self.G = nx.from_numpy_array(self.graph, create_using=nx.DiGraph) # convert adj matrix into DiGraph
        self.G = nx.relabel_nodes(self.G, {i: name for i, name in enumerate(self.data.columns)})
        print(self.G)
        self.dot_graph = self._generate_dowhy_graph()
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
        print('start feature importance analysis')
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
        shap_df.to_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv', index=False)
        
        figs = []
        if visualize == True:
            # 1st SHAP Plot beeswarm
            ax = shap.plots.beeswarm(shap_values_linear, plot_size=(8,6), show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_beeswarm_plot.png', bbox_inches='tight')  # Save as PNG
            #plt.savefig(f'shap_beeswarm_plot.png', bbox_inches='tight') 
            figs.append("shap_beeswarm_plot.png")
            # plt.show()

            # 2nd SHAP Plot Bar
            fig, ax = plt.subplots(figsize=(8, 6))
            ax = shap.plots.bar(shap_values_linear, ax=ax, show=False)
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/shap_bar_plot.png', bbox_inches='tight')  # Save as PNG
            #plt.savefig(f'shap_bar_plot.png', bbox_inches='tight') 
            figs.append("shap_bar_plot.png")
            #plt.show()
            plt.close()
        return parent_nodes, mean_shap_values, figs
    
    def estimate_causal_effect(self, treatment, outcome, 
                           control_value=0, treatment_value=1,
                           target_units='ate'):
        """
        Estimate the causal effect of a treatment on an outcome 
        using DoWhy (backdoor.linear_regression).
    
        :param target_units: 'ate', 'treated', or a callable lambda for a subset (CATE).
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect of Treatment: {treatment} on Outcome: {outcome}")
        print(f"Method: backdoor.linear_regression | target_units={target_units}")
        print("#"*60)
    
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph
        )
        try:
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            print("\nIdentified Estimand:")
            print(identified_estimand)
        
            causal_estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                control_value=control_value,
                treatment_value=treatment_value,
                target_units=target_units  # <- new parameter
            )
        
            print("\nCausal Estimate:")
            print(causal_estimate)
            # Call test_significance with the estimate value
            significance_results = causal_estimate.estimator.test_significance(self.data,causal_estimate.value)
            p_value = significance_results['p_value'][0]
            print("Significance Test Results:", p_value)

            # # Sensitivity Analysis for the estimation
            # refutation, figs = self.sensitivity_analysis(outcome, model, identified_estimand, causal_estimate, treatment, outcome)
            refutation, figs = None, []
        except Exception as e:
            import traceback
            traceback.print_exc()
            causal_estimate, p_value, refutation, figs = None, None, None, []
    
        print("\n=== Interpretation Hint ===")
        print("Negative estimate => increasing treatment tends to decrease outcome.")
        print("============================\n")

        return causal_estimate, p_value, refutation, figs

    def estimate_causal_effect_dml(self, treatment, outcome,
                               control_value=0, treatment_value=1,
                               target_units='ate'):
        """
        Estimate the causal effect (DML) of a treatment on an outcome,
        specifying target_units for ATE, ATT, or a custom subset (CATE).
        """
        print("\n" + "#"*60)
        print(f"Estimating Causal Effect (DML) of Treatment: {treatment} on Outcome: {outcome}")
        print(f"Method: backdoor.dml | target_units={target_units}")
        print("#"*60)
    
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph
        )
    
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        print("\nIdentified Estimand (DML):")
        print(identified_estimand)

    
        # Example default regressors I hard-coded for now.
        #
        from sklearn.ensemble import RandomForestRegressor
        outcome_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        treatment_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        final_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)

        warnings.filterwarnings('ignore')
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.DML",
            method_params={
                "init_params":{
                "model_y": outcome_model,
                "model_t": treatment_model,
                "model_final": final_model
                },
                "fit_params":{}
            },
            control_value=control_value,
            treatment_value=treatment_value,
            target_units=target_units  # <- here
        )
    
        # significance test (similar to linear_regression approach)
        p_value = None
    
        print("\n=== Interpretation Hint ===")
        print("Uses DML with RandomForestRegressor as default for outcome & treatment models.")
        print("============================\n")

        # # Sensitivity Analysis for the estimation
        # refutation, figs = self.sensitivity_analysis(outcome, model, identified_estimand, causal_estimate, treatment, outcome)
        refutation, figs = None, []
        return causal_estimate, p_value, refutation, figs

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
        fig_path = f'{self.global_state.user_data.output_graph_dir}/attribution_plot.png'
        plt.savefig(fig_path, bbox_inches='tight')
        figs = ['attribution_plot.png']
        # plt.show()  # Show the plot in the notebook or script
        # plt.close()
        return df, figs

    def estimate_hte_effect(self, outcome, treatment, X_col, query):
        print("\nCreating Causal Model...")
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            graph=self.dot_graph
        )
        identified_estimand = model.identify_effect()
        W_col = identified_estimand.get_backdoor_variables()
        if len(W_col) == 0:
            W_col = ['W']
            W = pd.DataFrame(np.zeros((len(self.data), 1)), columns=W_col)
            self.data = pd.concat([self.data, W], axis=1)
            self.global_state.user_data.processed_data = self.data
        # Algorithm selection and deliberation
        filter = HTE_Filter(args)
        self.global_state = filter.forward(self.global_state, query)

        reranker = HTE_Param_Selector(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        self.global_state = reranker.forward(self.global_state)

        programmer = HTE_Programming(self.args, y_col=outcome, T_col=treatment, X_col=X_col, W_col=W_col)
        hte, hte_lower, hte_upper = programmer.forward(self.global_state, task='hte')
        hte = pd.DataFrame({'hte': hte.flatten()})
        hte.to_csv(f'{self.global_state.user_data.output_graph_dir}/hte.csv', index=False)

        dist_fig_path = f'{self.global_state.user_data.output_graph_dir}/hte_dist.png'
        plot_hte_dist(hte, dist_fig_path)
        figs = ['hte_dist.png']
        cate_fig_path = f'{self.global_state.user_data.output_graph_dir}/cate_dist.png'
        visual_X_col = [col for col in X_col if self.global_state.statistics.data_type_column[col]!='continuous']
        if visual_X_col != []:
            plot_cate_violin(self.data, hte, X_col, cate_fig_path)
            figs.append('cate_dist.png')

        return hte, hte_lower, hte_upper, figs

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

        plt.bar(x - bar_width / 2, treatment_values, width=bar_width, label='Treatment', color='blue')
        plt.bar(x + bar_width / 2, response_values, width=bar_width, label='Response', color='orange')

        plt.ylabel('Values')
        plt.title('Counterfactual Estimation')
        plt.xticks(x, categories)
        plt.legend()

        for i, v in enumerate(treatment_values):
            plt.text(i - bar_width / 2, v + 0.5, f"{v:.1f}", ha='center')
        for i, v in enumerate(response_values):
            plt.text(i + bar_width / 2, v + 0.5, f"{v:.1f}", ha='center')

        path = self.global_state.user_data.output_graph_dir

        if not os.path.exists(path):
            os.makedirs(path)
        print(f"Saving residuals plot to {os.path.join(path, 'counterfactual_est_fig.jpg')}")
        plt.savefig(os.path.join(path, 'counterfactual_est_fig.jpg'))


    def simulate_intervention(self, treatment_name, shift_intervention = None, atomic_intervention = None):
        if atomic_intervention is None:
            atomic_intervention_val = self.data[treatment_name].min() + 1.0

        if shift_intervention is None:
            shift_intervention_val = 1

        atomic_samples = gcm.interventional_samples(self.causal_model,
                                             {treatment_name: lambda x: atomic_intervention_val},
                                             num_samples_to_draw=1000)

        shift_samples = gcm.interventional_samples(self.causal_model,
                                                    {treatment_name: lambda x: x + shift_intervention_val},
                                                    num_samples_to_draw=1000)

        path = self.global_state.user_data.output_graph_dir

        if not os.path.exists(path):
            os.makedirs(path)

        print(f"Saving simulated dataset {os.path.join(path, 'simulated_atomic_intervention.csv')}")
        atomic_samples.to_csv(os.path.join(path, 'simulated_atomic_intervention.csv'), index=False)

        print(f"Saving simulated dataset {os.path.join(path, 'simulated_shift_intervention.csv')}")
        shift_samples.to_csv(os.path.join(path, 'simulated_shift_intervention.csv'), index=False)



    def sensitivity_analysis(self, target_node, model, estimand, estimate, treatment, outcome):
        # if self.global_state.statistics.linearity:
        #      simulation_method = "linear-partial-R2"
        # else:
        #      simulation_method = "non-parametric-partial-R2",

        # # Use the most important factor as the benchmark_common_causes
        # file_exists = os.path.exists(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        # if not file_exists:
        #     self.feature_importance(target_node, visualize=False)
        # shap_df = pd.read_csv(f'{self.global_state.user_data.output_graph_dir}/shap_df.csv')
        # max_col = shap_df.mean().idxmax()

        #print('model.get_common_causes',model.get_common_causes())
        common_causes = model.get_common_causes()
        if common_causes == [] or common_causes is None:
            refute = model.refute_estimate(estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.9)
            figs = []
        else:
            refute = model.refute_estimate(estimand, estimate,
                                method_name = "add_unobserved_common_cause",
                                simulation_method = "non-parametric-partial-R2",
                                benchmark_common_causes = common_causes,
                                effect_fraction_on_outcome = [1,2,3]
                                )
            plt.savefig(f'{self.global_state.user_data.output_graph_dir}/ate_refutation.png')
            plt.close()
            figs = ['ate_refutation.png']
        print(refute)

        return refute, figs 

    def call_LLM(self, format, prompt, message):
        client = OpenAI(organization=self.args.organization, project=self.args.project, api_key=self.args.apikey)
        if format:
            completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            response_format=format,
            )
            parsed_response = completion.choices[0].message.parsed
        else: 
            completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ],
            )
            parsed_response = completion.choices[0].message.content
        return parsed_response

    def forward(self, task, desc, key_node):
        if task == 'Feature Importance':
            parent_nodes, mean_shap_values, figs = self.feature_importance(key_node, visualize=True)
            prompt = f"""
            I'm doing the feature importance analysis and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Result Variable we care about**: {key_node}
            **Parent Nodes of the Result Variable**: {parent_nodes}
            **Mean of Shapley Values**: {mean_shap_values}
            ""Description from User**: {desc}
            """
            response = self.call_LLM(None, 'You are an expert in Causal Discovery.', prompt)
            return response, figs 
        
        elif task == 'Average Treatment Effect Estimation':
            prompt = f"""
            I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
            {desc}
            The variable name must be among these variables: {self.data.columns}
            Only return me with the variable name, do not include anything else.
            """
            treatment = self.call_LLM(None, 'You are an expert in Causal Discovery.', prompt)
            causal_estimate, p_value, refutation, figs = self.estimate_causal_effect(treatment=treatment, outcome=key_node, control_value=0, treatment_value=1)
            parent_nodes = list(self.G.predecessors(key_node))
            if causal_estimate is not None:
                # Analysis for Effect Estimation
                prompt = f"""
                I'm doing the Treatment Effect Estimation analysis and please help me to write a brief analysis in bullet points.
                Here are some informations:
                **Result Variable we care about**: {key_node}
                **Treatment Variable**: {treatment}
                **Parent Nodes of the Result Variable**: {parent_nodes}
                **Causal Estimate Result**: {causal_estimate.value}
                **P-value of Significance Test for Causal Estimate**: {p_value}
                **Description from User**: {desc}
                """
                response1 = self.call_LLM(None, 'You are an expert in Causal Discovery.', prompt)
                # Analysis for Refutation Analysis
                prompt = f"""
                I'm doing the refutation analysis for my treatment effect estimation, please help me to write a brief analysis in bullet points.
                **Contents you need to incode**
                1. Brief Introduction of the refutation analysis method we use
                2. Summary of the refutation analysis result
                3. Brief Interpretation of the plot
                4. Conclude whether the treatment effect estimation is reliable or not based on the refutation analysis result
                Here are some informations:
                **Result Variable we care about**: {key_node}
                **Causal Estimate Result**: {causal_estimate}
                """
                if figs == []:
                    prompt += f"""
                **Refutation Result**: 
                {str(refutation)}
                **Method We Use**: Use Data Subsampling to answer Does the estimated effect change significantly when we replace the given dataset with a randomly selected subset?
                """
                else:
                    prompt += f"""
                **Sensitivity Analysis Result**: 
                {str(refutation)}
                **Method We Use**:
                Sensitivity analysis helps us study how robust an estimated effect is when the assumption of no unobserved confounding is violated. That is, how much bias does our estimate have due to omitting an (unobserved) confounder? Known as the omitted variable bias (OVB), it gives us a measure of how the inclusion of an omitted common cause (confounder) would have changed the estimated effect.
                **Information in the plot**: 
                a. The x-axis shows hypothetical partial R2 values of unobserved confounder(s) with the treatment. The y-axis shows hypothetical partial R2 of unobserved confounder(s) with the outcome. 
                b. At <x=0,y=0>, the black diamond shows the original estimate (theta_s) without considering the unobserved confounders.
                c. The contour levels represent adjusted estimate of the effect, which would be obtained if the unobserved confounder(s) had been included in the estimation model. 
                d. The red contour line is the critical threshold where the adjusted effect goes to zero. Thus, confounders with such strength or stronger are sufficient to reverse the sign of the estimated effect and invalidate the estimate’s conclusions. 
                e. The red triangle shows the estimated effect when the unobserved covariate has 1 or 2 or 3 times partial-R^2 of a chosen benchmark observed covariate with the outcome.
                **Description from User**: {desc}
                """
                response2 = self.call_LLM(None, 'You are an expert in Causal Discovery.', prompt)
                response = response1 + '\n' + response2
            else:
                response = "We cannot identify a valid treatment effect estimand for your query, please adjust your query based on the causal graph and retry."
                figs = []

            return response, figs
        
        elif task == 'Heterogeneous Treatment Effect Estimation':
            message = desc
            class VarList(BaseModel):
                treatment: str
                confounders: list[str]
            prompt = f"""You are a helpful assistant, please do the following tasks:
            Firstly, identify the Treatment Variable in user's query and save it in treatment as a string
            Secondly, identify a list of heterogeneous confounders and save it in confounders as a list of string, the list SHOULD NOT be empty
            The variable name must be among these variables: {self.data.columns}
            The outcome Y is {key_node}
            """
            parsed_response = self.call_LLM(VarList, prompt, message)
            treatment = parsed_response.treatment
            confounders = parsed_response.confounders
            hte, hte_lower, hte_upper, figs = self.estimate_hte_effect(outcome=key_node, treatment=treatment, X_col=confounders, query=desc)
            parent_nodes = list(self.G.predecessors(key_node))

            prompt = f"""
            I'm doing the Heterogeneous Treatment Effect Estimation and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Result Variable we care about**: {key_node}
            **Treatment Variable**: {treatment}
            **Heterogeneous Confounders we coutrol**: {confounders}
            **Method we use**: 
            Double Machine Learning, algorithm {self.global_state.inference.hte_algo_json['name']} with model_y={self.global_state.inference.hte_model_param['model_y']}, model_t={self.global_state.inference.hte_model_param['model_t']}
            **Upper and Lower Bound of Confidence Inferval with P-value=0.05**: {hte_upper}, {hte_lower}
            **Information in the plot**: Distribution of the HTE; Violin plot of the CATE grouped by: {confounders}
            **Description from User**: {desc}
            """
            response = self.call_LLM(None, 'You are an expert in Causal Discovery.',prompt)
            return response, figs
        
        elif task == 'Anormaly Attribution':
            df, figs = self.attribute_anomalies(target_node=key_node, anomaly_samples=self.data, confidence_level=0.95)
            parent_nodes = list(self.G.predecessors(key_node))
            prompt = f"""
            I'm doing the Anormaly Attribution analysis and please help me to write a brief analysis in bullet points.
            Here are some informations:
            **Abnormal Variable we care about**: {key_node}
            **Parent Nodes of the Abnormal Variable**: {parent_nodes}
            **Anormaly Attribution Result Table**: 
            {df.to_markdown()}
            **Description from User**: {desc}
            **Methods to calculate Anormaly Attribution Score**
            We estimated the contribution of the ancestors of {key_node}, including {key_node} itself, to the observed anomaly.
            In this method, we use invertible causal mechanisms to reconstruct and modify the noise leading to a certain observation. We then ask, “If the noise value of a specific node was from its ‘normal’ distribution, would we still have observed an anomalous value in the target node?”. The change in the severity of the anomaly in the target node after altering an upstream noise variable’s value, based on its learned distribution, indicates the node’s contribution to the anomaly. The advantage of using the noise value over the actual node value is that we measure only the influence originating from the node and not inherited from its parents.
            """
            response = self.call_LLM(None, 'You are an expert in Causal Discovery.', prompt)
            return response, figs
        
        else:
            return None, None


    

##############
def LLM_parse_query(args, format, prompt, message):
    client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
    if format:
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        response_format=format,
        )
        parsed_response = completion.choices[0].message.parsed
    else: 
        completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        )
        parsed_response = completion.choices[0].message.content
    return parsed_response

def main(global_state, args):
    """
    Modify the main function to call attribute_anomalies and save the results in ./auto_mpg_output.
    """
    
    analysis = Analysis(global_state, args)
    message = "What is the Average Treatment Effect of PIP2 on PIP3"

    # Example Test HTE Estimation with DML
    hte, hte_lower, hte_upper, figs = analysis.estimate_hte_effect(outcome='PIP2', treatment='Akt', X_col=['Erk', 'Plcg'], query=message)
    # EXAMPLE: Test the new DML method
    dml_estimate, dml_p_value, refutaion, figs = analysis.estimate_causal_effect_dml(
        treatment='PIP2',
        outcome='PIP3',
        target_units='treated'  # e.g., ATT
    )
    print("DML Estimate:", dml_estimate.value)
    print("p-value:", dml_p_value)

    # EXAMPLE: Compare with linear approach, specifying a different target_units
    lin_estimate, lin_p_value, refutaion, figs = analysis.estimate_causal_effect(
        treatment='Plcg',
        outcome='PIP3',
        #target_units=lambda df: df[df['PIP2'] > 100].index  # example subgroup
        target_units='ate'
    )
    print("Linear Subgroup Estimate (CATE approach):", lin_estimate.value)
    print("p-value:", lin_p_value)
    # Testing code to check the above functions replace it appropriately
    return None

    
    class InfList(BaseModel):
                tasks: list[str]
                descriptions: list[str]
                key_node: list[str]
    prompt = f"""You are a helpful assistant, please do the following tasks:
            **Tasks*
            Firstly please identify what tasks the user want to do and save them as a list in tasks.
            Please choose among the following causal tasks, if there's no matched task just return an empty list 
            You can only choose from the following tasks: 
            1. Average Treatment Effect Estimation; 2. Heterogeneous Treatment Effect Estimation 3. Anormaly Attribution; 4. Feature Importance
            Secondly, save user's description for their tasks as a list in descriptions, the length of description list must be the same with task list
            Thirdly, save the key result variable user care about as a list, each task must have a key result variable and they can be the same, the length of result variable list must be the same with task list
            key result variable must be among this list!
            {global_state.user_data.processed_data.columns}
            **Question Examples**
            1. Average Treatment Effect Estimation:
            What is the causal effect of introducing coding classes in schools on students' future career prospects?
            What is the average treatment effect of a minimum wage increase on employment rates?
            How much does the availability of free internet in rural areas improve educational outcomes?
            How does access to affordable childcare affect women’s labor force participation?
            What is the impact of reforestation programs on air quality in urban areas?
            2. Heterogeneous Treatment Effect Estimation:
            What is the heterogeneity in the impact of reforestation programs on air quality across neighborhoods with varying traffic density?
            How does the introduction of mental health support programs in schools impact academic performance differently for students with varying levels of pre-existing stress?
            Which demographic groups benefit most from telemedicine adoption in terms of reduced healthcare costs and improved health outcomes?
            How does the effectiveness of renewable energy subsidies vary for households with different income levels or geographic locations?
            3. Anormaly Attribution
            How can we attribute a sudden increase in stock market volatility to specific economic events or market sectors?
            Which variables (e.g., transaction amount, location, time) explain anomalies in loan repayment behavior?
            What factors explain unexpected delays in surgery schedules or patient discharge times?
            What are the root causes of deviations in supply chain delivery times?
            What factors contribute most to unexpected drops in product sales during a specific period?
            4. Feature Importance
            What are the most influential factors driving credit score predictions?
            What are the key factors influencing the effectiveness of a specific treatment or medication?
            Which product attributes (e.g., price, brand, reviews) are the most influential in predicting online sales?
            Which environmental variables (e.g., humidity, temperature, CO2 levels) are most important for predicting weather patterns?
            What customer behaviors (e.g., browsing time, cart size) contribute most to predicting cart abandonment?
            """
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    parsed_response = LLM_parse_query(args, InfList, prompt, message)
    tasks_list, descs_list, key_node_list = parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
    print(tasks_list, descs_list, key_node_list)
    #tasks_list, descs_list, key_node_list = ['Treatment Effect Estimation'], ['Analyze the treatment effect of PIP2 to PIP3.'], ['PIP3']
    import matplotlib.image as mpimg
    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
        print(task, desc, key_node)
        response, figs = analysis.forward(task, desc, key_node)
        print(response)
        for file_name in figs:
            img = mpimg.imread(f'{global_state.user_data.output_graph_dir}/{file_name}')  # Read the image
            plt.imshow(img)  

if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

        # Input data file
        parser.add_argument(
            '--data-file',
            type=str,
            default="dataset/sachs/sachs.csv",
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

        # OpenAI Settings
        parser.add_argument(
            '--organization',
            type=str,
            default="org-gw7mBMydjDsOnDlTvNQWXqPL",
            help='Organization ID'
        )

        parser.add_argument(
            '--project',
            type=str,
            default="proj_SIDtemBJMHUWG7CPdU7yRjsn",
            help='Project ID'
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

        parser.add_argument(
            '--parallel',
            type=bool,
            default=False,
            help='Parallel computing for bootstrapping.'
        )

        parser.add_argument(
            '--demo_mode',
            type=bool,
            default=False,
            help='Demo mode'
        )

        parser.add_argument(
            '--revised_graph',
            type=str,
            default='postprocess/test_data/sachs/base_graph.npy',
            help='Demo mode'
        )

        args = parser.parse_args()
        return args
    args = parse_args()
    global_state = global_state_initialization(args)
    global_state.user_data.raw_data = pd.read_csv(args.data_file)
    global_state.user_data.processed_data = global_state.user_data.raw_data
    global_state.results.revised_graph = np.load(args.revised_graph)
    global_state.user_data.output_graph_dir = args.output_graph_dir
    global_state.statistics.description = 'Continuous Nonlinear dataset with 11 columns'
    global_state.statistics.data_type_column = {key: 'continuous' for key in global_state.user_data.processed_data.columns}

    main(global_state, args)
