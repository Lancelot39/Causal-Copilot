from causal_analysis.help_functions import *

def generate_analysis_econml(args, global_state, key_node, treatment, parent_nodes, X_col, W_col, result, query):
    # Analysis for Effect Estimation
    ate, ate_lower, ate_upper = result['ate']
    att, att_lower, att_upper = result['att']
    hte, hte_lower, hte_upper = result['hte']
    prompt_ate = f"""
    I'm doing the Treatment Effect Estimation analysis and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    **Focus on the result and DO NOT list out result variable, treatment variable, parent nodes, confounders, etc.**
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Treatment Variable**: {treatment}
    **Parent Nodes of the Result Variable**: {parent_nodes}
    **Confounders**: {W_col}
    
    **Causal Estimate Result**: 
    Average Treatment Effect: {ate}, Confidence Interval: ({ate_lower}, {ate_upper})
    Average Treatment Effect on Treated: {att}, Confidence Interval: ({att_lower}, {att_upper})
    """
    response_ate = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt_ate)

    prompt_hte = f"""
    I'm doing the Heterogeneous Treatment Effect Estimation and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    **Focus on the result and DO NOT list out result variable, treatment variable, Heterogeneous Variables, confounders, etc.**
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Treatment Variable**: {treatment}
    **Heterogeneous Variables we coutrol**: {X_col}
    **Description from User**: {query}
    **Information in the plot**: 
    Plot 1: Distribution of the HTE; 
    Plot 2: Violin plot of the CATE grouped by: {X_col}
    """
    response_hte = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt_hte)
    
    figs = []
    dist_fig_path = f'{global_state.user_data.output_graph_dir}/hte_dist.png'
    plot_hte_dist(hte, dist_fig_path)
    figs.append(dist_fig_path)
    cate_fig_path = f'{global_state.user_data.output_graph_dir}/cate_dist.png'
    plot_cate_violin(global_state, hte, X_col, cate_fig_path)
    figs.append(cate_fig_path)

    return [response_ate,response_hte], figs

def generate_analysis_matching(args, treatment, outcome, method, W_col, ate, cate_result, query):
    # Generate a response using the LLM 
    response_balance = """ 
📊 The figure above is for balance checking. It compares heterogeneous variables before and after matching.
Matching is a technique used in causal analysis to create comparable groups (e.g., treated vs. control) by aligning them on key variables.
To reduce bias from confounding variables and make the groups more similar, so any differences in outcomes are more likely due to the treatment—not pre-existing differences.
"""
    prompt = f"""
    I'm doing the Matching-Based Effect Treatment Estimation analysis and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    **Focus on the result and DO NOT list out result variable, treatment variable, parent nodes, confounders, etc.**
    Here are some informations:
    **Treatment Variable**: {treatment}
    **Outcome Variable**: {outcome}
    **Matching Method**: {method}
    **Confounders**: {W_col}
    **Description from User**: {query}
    **Average Treatment Effect (ATE)**: {ate}
    """
    response_ate = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    prompt = f"""
    I'm doing the Matching-Based Effect Treatment Estimation analysis and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    **Focus on the result and DO NOT list out result variable, treatment variable, parent nodes, confounders, etc.**
    Here are some informations:
    **Treatment Variable**: {treatment}
    **Outcome Variable**: {outcome}
    **Matching Method**: {method}
    **Confounders**: {W_col}
    **Description from User**: {query}
    **Conditional Average Treatment Effect**:
    """
    for key, value in cate_result.items():
        prompt += f"Conditional on {key}: {value}\n"
    response_cate = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return [response_balance, response_ate, response_cate]

def generate_analysis_linear_regression(args, treatment, outcome, ate, p_value, query):
    # Generate a response using the LLM
    prompt = f"""
    I'm doing the Treatment Effect Estimation analysis and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    Here are some informations:
    **Treatment Variable**: {treatment}
    **Outcome Variable**: {outcome}
    **Method**: Linear Regression
    **Average Treatment Effect (ATE)**: {ate}
    **P-value**: {p_value}
    **Description from User**: {query}
    """

    response = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return response
        
    # # Analysis for Refutation Analysis
    # prompt = f"""
    # I'm doing the refutation analysis for my treatment effect estimation, please help me to write a brief analysis in bullet points.
    # **Contents you need to incode**
    # 1. Brief Introduction of the refutation analysis method we use
    # 2. Summary of the refutation analysis result
    # 3. Brief Interpretation of the plot
    # 4. Conclude whether the treatment effect estimation is reliable or not based on the refutation analysis result
    # Here are some informations:
    # **Result Variable we care about**: {key_node}
    # **Causal Estimate Result**: {causal_estimate}
    # """
    # if figs == []:
    #     prompt += f"""
    # **Refutation Result**: 
    # {str(refutation)}
    # **Method We Use**: Use Data Subsampling to answer Does the estimated effect change significantly when we replace the given dataset with a randomly selected subset?
    # """
    # else:
    #     prompt += f"""
    # **Sensitivity Analysis Result**: 
    # {str(refutation)}
    # **Method We Use**:
    # Sensitivity analysis helps us study how robust an estimated effect is when the assumption of no unobserved confounding is violated. That is, how much bias does our estimate have due to omitting an (unobserved) confounder? Known as the omitted variable bias (OVB), it gives us a measure of how the inclusion of an omitted common cause (confounder) would have changed the estimated effect.
    # **Information in the plot**: 
    # a. The x-axis shows hypothetical partial R2 values of unobserved confounder(s) with the treatment. The y-axis shows hypothetical partial R2 of unobserved confounder(s) with the outcome. 
    # b. At <x=0,y=0>, the black diamond shows the original estimate (theta_s) without considering the unobserved confounders.
    # c. The contour levels represent adjusted estimate of the effect, which would be obtained if the unobserved confounder(s) had been included in the estimation model. 
    # d. The red contour line is the critical threshold where the adjusted effect goes to zero. Thus, confounders with such strength or stronger are sufficient to reverse the sign of the estimated effect and invalidate the estimate’s conclusions. 
    # e. The red triangle shows the estimated effect when the unobserved covariate has 1 or 2 or 3 times partial-R^2 of a chosen benchmark observed covariate with the outcome.
    # **Description from User**: {desc}
    # """

    # response2 = LLM_parse_query(self.args, None, 'You are an expert in Causal Discovery.', prompt)
    # response = response1 + '\n' + response2

def generate_analysis_feature_importance(args, key_node, parent_nodes, mean_shap_values, desc):
    prompt = f"""
    I'm doing the feature importance analysis and please help me to write a brief analysis.
    You can use some bullet points to summarize the result, but not too much.
    Here are some informations:
    **Result Variable we care about**: {key_node}
    **Parent Nodes of the Result Variable**: {parent_nodes}
    ""Description from User**: {desc}
    **Mean of Shapley Values**: {mean_shap_values}
    """
    
    response = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return response

def generate_analysis_anormaly(args, df, key_node, parent_nodes, desc):
    prompt = f"""
    I'm doing the Anormaly Attribution analysis and please help me to write a brief analysis.
    Don't use bullet point.
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

    response = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return response

def generate_analysis_anormaly_dist(args, df, key_node, desc):
    # Generate a response using the LLM
    prompt = f"""
    I'm doing the Distributional Change Attribution analysis and please help me to write a brief analysis.
    Don't use bullet point.
    Here are some informations:
    **Target Variable we care about**: {key_node}
    **Attribution Scores**: 
    {df.to_markdown()}
    **Description from User**: {desc}
    **Methods to calculate Distributional Change Attribution**
    We compared two datasets (old and new) to identify which nodes in the causal graph contributed most to the change in the distribution of the target variable.
    """

    response = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return response

def generate_conterfactual_estimation(args, global_state, shift_intervention_val, shift_df, treatment, key_node, desc):
    original_df = global_state.user_data.processed_data
    prompt = f"""
    I'm doing the Counterfactual Estimation analysis and please help me to write a brief analysis.
    Don't use bullet point.
    Here are some informations:
    **Target Variable we care about**: {key_node}
    **Treatment Variable**: {treatment}
    **Treatment Shift Values**: {shift_intervention_val}
    **Original Distribution of Target Variable**: {original_df[key_node].describe().to_string()}. 
    Please use some descriptive words to describe the distribution. DO NOT only list summary statistics, e.g., quantiles, maximum, minium, etc.
    **Counterfactual Estimation Result of Target Variable**: {shift_df[key_node].describe().to_string()}. 
    Please use some descriptive words to describe the distribution. DO NOT only list summary statistics, e.g., quantiles, maximum, minium, etc. 
    You need to compare the counterfactual estimation result with the original distribution of target variables.
    **Description from User**: {desc}
    **Methods to calculate Counterfactual Estimation**
    We estimated the counterfactual distribution of the target variable by shifting the observed samples in the direction of the identified confounders.
    """

    response = LLM_parse_query(args, None, 'You are an expert in Causal Discovery.', prompt)
    return response