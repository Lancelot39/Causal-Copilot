import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
import re
import numpy as np 
from plumbum.cmd import latexmk
from plumbum import local
from report.help_functions import *
import json 
#from report.report_generation import compile_tex_to_pdf_with_refs

class Inference_Report_generation(object):
    def __init__(self, global_state, args, task_index=0):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        """
        self.global_state = global_state
        self.args = args 
        self.client = OpenAI()
        self.statistics_desc = global_state.statistics.description
        self.knowledge_docs = global_state.user_data.knowledge_docs[0]
        # Data info
        self.data = global_state.user_data.processed_data.copy()
        self.data.columns = [var.replace('_', ' ') for var in self.data.columns]
        self.statistics = global_state.statistics
        # Inference info
        self.task_info = global_state.inference.task_info[task_index]

        # Path to find the visualization graph
        self.visual_dir = global_state.user_data.output_graph_dir
        self.report_dir = global_state.user_data.output_report_dir

    def generate_proposal(self):
        proposal = self.task_info['result']['proposal']
        prompt = f"""I'm writing a report on the causal inference task. Here is the brief proposal I have generated based on the data and the task. 
        Please write a 1-2 paragraphs proposal, illustraing why we choose these tasks to address the causal inference query.
        Only include the proposal, do not include any title or section number, etc.
        **Proposal Information:**
        {proposal}
        **Query:**
        {self.task_info['desc'][0]}
        """
        proposal = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
        proposal = bold_conversion(proposal)
        proposal = list_conversion(proposal)
        proposal = fix_latex_itemize(proposal)
        proposal = fix_latex_itemize_LLM(self.client, proposal)
        proposal = proposal.replace('_', ' ')
        proposal = proposal.replace('\documentclass{article}', '').replace('\\begin{document}', '').replace('\\end{document}', '')
        return proposal
    
    def generate_treatment_effect(self):
        te_template = load_context("report/context/inference/treatment_effect.tex")
        treatment = self.task_info['treatment']
        idx = self.task_info['task'].index('Treatment Effect Estimation')
        outcome = self.task_info['key_node'][idx]
        confounder = self.task_info['confounders']
        hte_var = self.task_info['X_col']
        method = self.task_info['hte_method']
        with open("report/context/inference/hte_method_reason.json", 'r') as f:
            method_reason = json.load(f)[method]
        response = self.task_info['result']['Treatment Effect Estimation']['response']
        figs = self.task_info['result']['Treatment Effect Estimation']['figs']
        for idx, r in enumerate(response):
            print(r)
            r = bold_conversion(r)
            r = list_conversion(r)
            r = fix_latex_itemize(r)
            r = fix_latex_itemize_LLM(self.client, r)
            #r = remove_redundant_point(r)
            r = remove_redundant_title(r)
            r = r.replace('_', ' ')
            response[idx] = r

        if method in ['cem', 'propensity_score']:
            analysis = f"""
\\textbf{{Balance Check}}

\\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.7\\textwidth]{{{figs[0]}}}
    \caption{{Distribution of Confounders before and after matching.}}
\end{{figure}}

{response[0]}

\\textbf{{Average Treatment Effect (ATE)}}

{response[1]}

\subsubsection{{Conditional Average Treatment Effect (CATE)}}

\\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.8\\textwidth]{{{figs[1]}}}
    \caption{{CATE Bar Plots grouped by different confounders.}}
\end{{figure}}

{response[2]}
"""
        elif method in ['dml', 'drl']:
            analysis = f"""
\\textbf{{Average Treatment Effect (ATE) and Average Treatment Effect on the Treated (ATT)}}

{response[0]}

\\textbf{{Heterogeneous Treatment Effect (HTE)}}

\\begin{{figure}}[H]
    \centering
    \includegraphics[height=0.3\\textheight]{{{figs[0]}}}
    \caption{{Distribution of HTE.}}
\end{{figure}}

\\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.8\\textwidth]{{{figs[1]}}}
    \caption{{Violin plot of HTE by Heterogeneous Variables.}}
\end{{figure}}

{response[1]}
"""
        else:
            analysis = f"""
"""     
        replacement = {'[TREATMENT_VAR]': treatment.replace('_', ' '), 
                       '[OUTCOME_VAR]': outcome.replace('_', ' '), 
                       '[CONFOUNDER_VAR]': (','.join(confounder)).replace('_', ' '), 
                       '[HTE_VAR]': (','.join(hte_var)).replace('_', ' '),
                       '[METHOD]': method.replace('_', ' '),
                       '[METHOD_REASON]': method_reason.replace('_', ' '),
                       '[ANALYSIS]': analysis}
        for placeholder, value in replacement.items():
            te_template = te_template.replace(placeholder, value)
        return te_template
    
    def generate_feature_importance(self):
        if self.global_state.statistics.linearity:
            model = 'Linear Regression'
        else:
            model = 'Random Forest'
        prompt = f"""I'm writing a report on the causal inference task. I use the SHAP method to explain the feature importance of the model.
        I use the {model} model to calculate the SHAP value. 
        Help me to write a **brief** explanation of why we use this method with 2-3 sentences.
        **Query:**
        {self.task_info['desc'][0]}
        """
        reason = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
        response = self.task_info['result']['Feature Importance']['response']
        figs = self.task_info['result']['Feature Importance']['figs']
        response = bold_conversion(response)
        response = list_conversion(response)
        response = fix_latex_itemize(response)
        response = fix_latex_itemize_LLM(self.client, response)
        #response = remove_redundant_point(response)
        response = remove_redundant_title(response)
        response = response.replace('_', ' ')

        fi_template = load_context("report/context/inference/feature_importance.tex")
        replacement = {'[MODEL]': model, 
                       '[MODEL_REASON]': reason, 
                       '[RESULTS]': response.replace('_', ' '), 
                       '[GRAPH1]': figs[0], 
                       '[GRAPH2]': figs[1]}
        for placeholder, value in replacement.items():
            fi_template = fi_template.replace(placeholder, value)
        
        return fi_template
    
    def generate_attribution(self):
        prompt = f"""I'm writing a report on the causal inference task. I conduct Abnormality Attribution analysis to answer the following user's query:
        **Query:**
        {self.task_info['desc'][0]}
        Write a 1-2 paragraphs explanation of the Abnormality Attribution method we use.
        **Mehtodology:**
        In this method, we use invertible causal mechanisms to reconstruct and modify the noise leading to a certain observation. We then ask, “If the noise value of a specific node was from its ‘normal’ distribution, would we still have observed an anomalous value in the target node?”. The change in the severity of the anomaly in the target node after altering an upstream noise variable’s value, based on its learned distribution, indicates the node’s contribution to the anomaly. The advantage of using the noise value over the actual node value is that we measure only the influence originating from the node and not inherited from its parents.
        """
        method = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
        response = self.task_info['result']['Feature Importance']['response']
        figs = self.task_info['result']['Feature Importance']['figs']
        response = bold_conversion(response)
        response = list_conversion(response)
        response = fix_latex_itemize(response)
        response = fix_latex_itemize_LLM(self.client, response)
        #response = remove_redundant_point(response)
        response = remove_redundant_title(response)

        at_template = load_context("report/context/inference/attribution.tex")
        replacement = {'[METHOD]': method.replace('_', ' '), 
                       '[RESULTS]': response.replace('_', ' '), 
                       '[GRAPH]': figs[0]}
        for placeholder, value in replacement.items():
            at_template = at_template.replace(placeholder, value)
        
        return at_template
    
    def generate_counterfactual_estimation(self):
        prompt = f"""I'm writing a report on the causal inference task. I conduct Counterfactual Estimation analysis to answer the following user's query:
        **Query:**
        {self.task_info['desc'][0]}
        Write a 1-2 paragraphs explanation of the Counterfactual Estimation method we use.
        **Mehtodology:**
        Counterfactuals are very similar to Simulating the Impact of Interventions, with an important difference: when performing interventions, we look into the future, for counterfactuals we look into an alternative past. To reflect this in the computation, when performing interventions, we generate all noise using our causal models. For counterfactuals, we use the noise from actual observed data.
        """
        method = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
        response = self.task_info['result']['Counterfactual Estimation']['response']
        print('original response', response)
        figs = self.task_info['result']['Counterfactual Estimation']['figs']
        response = bold_conversion(response)
        response = list_conversion(response)
        response = fix_latex_itemize(response)
        response = fix_latex_itemize_LLM(self.client, response)
        #response = remove_redundant_point(response)
        response = remove_redundant_title(response)
        print('2nd response', response)

        cf_template = load_context("report/context/inference/counterfact.tex")
        replacement = {'[METHOD]': method.replace('_', ' '), 
                       '[RESULTS]': response.replace('\$','$').replace('_', ' '), 
                       '[GRAPH1]': figs[0]}
        for placeholder, value in replacement.items():
            cf_template = cf_template.replace(placeholder, value)
        print('3rd response', replacement)
        
        return cf_template


    def generate_discussion(self):
        if self.task_info['result']['discussion']!={}:
            discussion_content = self.task_info['result']['discussion']
            prompt = f"""I'm writing a report on the causal inference task. I have generated some discussion based on the results. 
            Please help me to write a 1-2 paragraphs discussion, summarizing the questions and answers.
            You can bold some important sentences with **.
            DO NOT include any title or section number.
            **Discussion History:**
            """
            for q, a in discussion_content.items():
                prompt += f"""
            **Question**
            {q}
            **Answer**
            {a}
            """
            discussion = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
            discussion = bold_conversion(discussion)
            discussion = list_conversion(discussion)
            discussion = fix_latex_itemize(discussion)
            discussion = fix_latex_itemize_LLM(self.client, discussion)
            discussion = discussion.replace('_', ' ')
        else:
            discussion = 'You did not conduct any discussion with causal copilot.'
        
        return discussion

    def generate_next_step(self):
        tasks = self.task_info['task']
        prompt = f"""I'm writing a report on the causal inference task. I have generated some discussion based on the results. 
        Please help me to write a 1-2 paragraphs next step todo, including
        Potential Improvements: Some any refinements or additional analyses needed.
        Future Research Directions: Suggest how the findings can be expanded or validated further.
        You can bold some important sentences with **.
        DO NOT include any title or section number.
        The following are the results of the tasks:
        """
        for task in tasks:
            prompt += f"""
        **{task}**
        {self.task_info['result'][task]['response']}
        """
        next_step = LLM_parse_query(self.client, None, 'You are an expert in Causal Discovery.', prompt)
        next_step = bold_conversion(next_step)
        next_step = list_conversion(next_step)
        next_step = fix_latex_itemize(next_step)
        next_step = fix_latex_itemize_LLM(self.client, next_step)
        next_step = next_step.replace('_', ' ')
        return next_step
    
    def generation(self):
        # Load the context
        context = load_context("report/context/inference/inference_template.tex")
        treatment_effect = None
        feature_importance = None
        attribution = None
        counterfactual = None
        # Generate the proposal
        proposal = self.generate_proposal()
        #Generate the treatment effect estimation
        if 'Treatment Effect Estimation' in self.task_info['task']:
            treatment_effect = self.generate_treatment_effect()
        if 'Feature Importance' in self.task_info['task']:
            feature_importance = self.generate_feature_importance()
        if 'Abnormality Attribution' in self.task_info['task']:
            attribution = self.generate_attribution()
        if 'Counterfactual Estimation' in self.task_info['task']:
            counterfactual = self.generate_counterfactual_estimation()
        discussion = self.generate_discussion()
        #print(discussion)
        next_step = self.generate_next_step()
        #print(next_step)
        # Replace the placeholders
        replacement = {'[PROPOSAL]': proposal.replace('$', '\$').replace('`', '').replace('%', '\%'), 
                       '[TE_RESULT]': treatment_effect.replace('$', '\$').replace('`', '').replace('%', '\%') if treatment_effect is not None else '',
                       '[FI_RESULT]': feature_importance.replace('$', '\$').replace('`', '').replace('%', '\%') if feature_importance is not None else '',
                       '[AT_RESULT]': attribution.replace('$', '\$').replace('`', '').replace('%', '\%') if attribution is not None else '',
                       '[CF_RESULT]': counterfactual.replace('$', '\$').replace('`', '').replace('%', '\%') if counterfactual is not None else '',
                       '[DISCUSSION]': discussion.replace('$', '\$').replace('$_', '\_').replace('%', '\%'),
                       '[NEXT_STEP]': next_step.replace('$', '\$').replace('$_', '\_').replace('%', '\%')}
        for placeholder, value in replacement.items():
            context = context.replace(placeholder, value)
        # Save the context to a file
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)
        with open(f'{self.report_dir}/inference_report.tex', 'w') as f:
            f.write(context)
        # print('start compilation')
        # compile_tex_to_pdf_with_refs(f'{self.report_dir}/report.tex', self.report_dir)
        # return f'{self.report_dir}/report.pdf'
        return context



def test(args, global_state):
    my_report = Inference_Report_generation(global_state, args)
    report = my_report.generation()



import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')

    # Input data file
    parser.add_argument(
        '--data-file',
        type=str,
        default="demo_data/20250130_183915/2021online_shop/2021online_shop.csv",
        help='Path to the input dataset file (e.g., CSV format or directory location)'
    )

    # Output file for results
    parser.add_argument(
        '--output-report-dir',
        type=str,
        default='demo_data/20250130_183915/2021online_shop/output_report',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        default='demo_data/20250130_183915/2021online_shop/output_graph',
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
        default=True,
        help='Enable debugging mode'
    )

    parser.add_argument(
        '--initial_query',
        type=str,
        default="selected algorithm: FGES",
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

    args = parser.parse_args()
    return args

import pickle  
if __name__ == '__main__':
    args = parse_args()
    with open('demo_data/20250121_223113/lalonde/output_graph/inference_global_state.pkl', 'rb') as file:
        global_state = pickle.load(file)
    test(args, global_state)
    # save_path = 'demo_data/20250121_223113/lalonde/output_report'
    # compile_tex_to_pdf_with_refs(f'{save_path}/report.tex', save_path)
    


