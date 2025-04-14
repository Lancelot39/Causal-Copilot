import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openai import OpenAI
import re
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import numpy as np 
from plumbum.cmd import latexmk
from plumbum import local
import networkx as nx
from postprocess.visualization import Visualization, convert_to_edges
from postprocess.judge_functions import edges_to_relationship
from report.help_functions import *
from report.inference_report_generation import Inference_Report_generation
import glob
import ast
import json 

def compile_tex_to_pdf_with_refs(tex_file, output_dir=None, clean=True):
    """
    Silently compile a TeX file to PDF with multiple passes for references
    
    Args:
        tex_file (str): Path to the .tex file
        output_dir (str, optional): Output directory for the PDF
        clean (bool): Whether to clean auxiliary files after compilation
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    try:
        tex_dir = os.path.dirname(tex_file)
        if output_dir is None:
            output_dir = tex_dir

        # Multiple passes for references 
        try:
            # Build latexmk arguments
            args = [
                '-pdf',                     # Generate PDF output
                '-interaction=nonstopmode', # Don't stop for errors
                '-halt-on-error',           # Stop on errors
                '-f',
                '-bibtex',                   # Use bibtex for references
                f'-output-directory={output_dir}'
            ]
                
            # Add input file
            args.append(tex_file)
            
            # Run latexmk
            with local.env(TEXINPUTS=":./"):
                latexmk[args]()
        except Exception as e:
            print(f"Error in compilation pass")
            print(str(e))
            return False
        
        print(f"Successfully compiled {tex_file} to {output_dir}")
        return True
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False

class Report_generation(object):
    def __init__(self, global_state, args):
        """
        :param global_state: a dict containing global variables and information
        :param args: arguments for the report generation
        """
        ######## Load the chosen global state and record all global states history ########
        inference_global_state = global_state
        if global_state.results.report_selected_index is not None:
            self.global_state_list = []
            for algo in global_state.logging.global_state_logging:
                with open(f'{global_state.user_data.output_graph_dir}/{algo}_global_state.pkl', 'rb') as f:
                    self.global_state_list.append(pickle.load(f))
            global_state = self.global_state_list[global_state.results.report_selected_index]
        else:
            self.global_state_list = [global_state]
            global_state.logging.global_state_logging = [global_state.algorithm.selected_algorithm]
        # ###
        # self.global_state_list = []
        # for algo in ['PCMCI', 'VARLiNGAM']:
        #     with open(f'{global_state.user_data.output_graph_dir}/{algo}_global_state.pkl', 'rb') as f:
        #         self.global_state_list.append(pickle.load(f))
        # global_state = self.global_state_list[0]
        # global_state.logging.global_state_logging
        # inference_global_state = None
        # for state in self.global_state_list:
        #     if state.inference.task_index > -1:
        #         inference_global_state = state
        #         break

        self.client = OpenAI()
        if global_state.user_data.meaningful_feature:
            self.data_mode = 'real'
        else:
            self.data_mode = 'simulated'
        self.data_file = global_state.user_data.output_graph_dir.split('/')[-2]
        self.global_state = global_state
        self.inference_global_state = inference_global_state
        self.args = args 
        self.statistics_desc = global_state.statistics.description
        self.knowledge_docs = global_state.user_data.knowledge_docs_for_user[0]
        # Data info
        self.data = global_state.user_data.processed_data.copy()
        self.data.columns = [var.replace('_', ' ') for var in self.data.columns]
        self.statistics = global_state.statistics
        # EDA info
        self.eda_result = global_state.results.eda
        # Result graph matrix
        self.raw_graph = global_state.results.raw_result
        self.lag_graph = global_state.results.lagged_graph
        self.graph = global_state.results.converted_graph
        self.revised_graph = global_state.results.revised_graph
        self.bootstrap_probability = global_state.results.bootstrap_probability
        self.original_metrics = global_state.results.metrics
        self.revised_metrics = global_state.results.revised_metrics
        # algo&hp selection prompts
        self.algo = global_state.algorithm.selected_algorithm
        self.algo_can = global_state.algorithm.algorithm_candidates
        self.algo_param = global_state.algorithm.algorithm_arguments_json
        self.prompt = global_state.logging.select_conversation[0]['response']
        self.hp_prompt = global_state.logging.argument_conversation[0]['response']
        # Path to find the visualization graph
        self.visual_dir = global_state.user_data.output_graph_dir

    def get_title(self):
        response_title = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant, please give me the name of the given dataset {self.data_file}\n"
                                                "For example, if the dataset is Sachs.csv, then return me with 'Sachs'. If the dataset is a directory called Abalone, then return me with 'Abalone'.\n"
                                                "Only give me the string of name, do not include anything else."},
            ]
        )
        dataset = response_title.choices[0].message.content
        dataset = dataset.replace('_', ' ')
        title = f'Causal Discovery Report on {dataset.capitalize()}'
        return title, dataset
    
    def intro_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery and inference on a dataset and write a report. There are some background knowledge about this dataset.
        1. Please write a brief introduction paragraph. I only need the paragraph, don't include any title.
        2. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
        
        Background about this dataset: {self.knowledge_docs}
        """
    
        print("Start to find Introduction")
        response_dist = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_intro = response_dist.choices[0].message.content
        response_intro = response_intro.replace("_", r"\_")
        return response_intro
    
    def background_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.There are three sections:
### 1. Detailed Explanation about the Variables
### 2. Possible Causal Relations among These Variables
### 3. Other Background Domain Knowledge that may be Helpful for Experts
        **Your Tasks**
        1. Summarize contents in <Section 1. Detailed Explanation about the Variables and Section 3. Other Background Domain Knowledge that may be Helpful for Experts> in 1-2 paragraphs.
        2. I only need the text, do not include title
        3. If you want to use bollet points, make sure it's in latex {{itemize}} format. 
        4. If there are contents like **content** which means the bold fonts, change it into latex bold format \\textbf{{content}}
        Background about this dataset: {self.knowledge_docs}
        """
    
        print("Start to find Background")
        response_background = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        section1 = response_background.choices[0].message.content
        section1 = re.sub(r'.*\*\*(.*?)\*\*', r'\\textbf{\1}', section1)       
        section1 = list_conversion(section1)
        section1 = fix_latex_itemize(section1)
        section1 = bold_conversion(section1)
        section1 = section1.replace("_", " ")

        class Relationship(BaseModel):
            cause: str
            result: str
            explanation: str

        class CausalRelation(BaseModel):
            """Model for storing causal relations between variables."""
            relationships: List[Relationship]
        col_names = '\t'.join(self.data.columns)
        prompt = f"""
I want to conduct a causal discovery on a dataset and write a report. There is some background knowledge about this dataset.
There are three sections:
### 1. Detailed Explanation about the Variables
### 2. Possible Causal Relations among These Variables (Do not too much, only include important ones)
### 3. Other Background Domain Knowledge that may be Helpful for Experts
Please extract all relationships in the second section ### 2. Possible Causal Relations among These Variables, and return as the specified format

**Thinking Steps**
1. Extract all pairwise relationships, for example A causes B because ....; C causes D because ....; Only include relationships between two variables!
2. Check whether these variables are among {col_names}, please delete contents that include any other variables!
3. For each relationship, create an object with "cause", "result", and "explanation" fields.

**Example Format**
If you have: "A causes B because explanation1; C causes D because explanation2"
The output should be:
[
  {{
    "cause": "A",
    "result": "B",
    "explanation": "explanation1"
  }},
  {{
    "cause": "C",
    "result": "D",
    "explanation": "explanation2"
  }}
]

**You Must**
1. Only pairwise relationships can be included
2. All variables should be among {col_names}, please delete contents that include any other variables!
3. Make sure each cause-result pair is properly linked with its explanation in a single object.

**Background Knowledge**
{self.knowledge_docs}
"""
        result_parsed = {}
        response = LLM_parse_query(self.client, CausalRelation, "You are an expert in the causal discovery field and helpful assistant.", prompt)
        for rel in response.relationships:
            tup = (rel.cause, rel.result)
            result_parsed[tup] = rel.explanation
        
        variables = self.data.columns
        print("variables: ", variables)
        zero_matrix = np.zeros((len(variables), len(variables)))
        if result_parsed != {}:
            section2 = """
            \\begin{itemize}
            """
            # Potential Relationship Visualization
            valid_pairs = 0
            for pair in result_parsed.keys():
                explanation = result_parsed[pair]
                #pair = ast.literal_eval(pair)
                if pair[0].lower() in variables.str.lower() and pair[1].lower() in variables.str.lower():
                    ind1 = variables.str.lower().get_loc(pair[0].lower())
                    ind2 = variables.str.lower().get_loc(pair[1].lower())
                    zero_matrix[ind2, ind1] = 1
                    section2 += f"\item \\textbf{{{pair[0].replace('_', ' ')} $\\rightarrow$ {pair[1].replace('_', ' ')}}}: {explanation.replace('_', ' ')} \n"
                    valid_pairs += 1
            section2 += "\end{itemize}"
            if valid_pairs == 0:
                section2 = """
                The LLM doesn't find any causal relationship among variables.
                """
        else:
            section2 = """
                The LLM doesn't find any causal relationship among variables.
                """

        my_visual = Visualization(self.global_state)
        g = nx.from_numpy_array(zero_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i] for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        pos = nx.spring_layout(g)
        _ = my_visual.plot_pdag(zero_matrix, 'potential_relation.pdf', pos=self.global_state.results.raw_pos, relation=True)
        relation_path = f'{self.visual_dir}/potential_relation.pdf'
            
        if sum(zero_matrix.flatten())!=0:
            relation_prompt = f"""
            {section2}
            \\begin{{figure}}[H]
            \centering
            \includegraphics[width=0.5\linewidth]{{{relation_path}}}
            \caption{{\label{{fig:relation}}A Causal Graph Suggested by LLM.}}
            \end{{figure}}
            """
        else:
            relation_prompt = f"""
                {section2}
                """
        return section1, relation_prompt

    def data_prop_prompt(self):
        n, m = self.data.shape
        shape = f'({n}, {m})'
        missingness = 'True' if self.statistics.missingness else 'False'
        data_type = self.statistics.data_type
        linearity = 'True' if self.statistics.linearity else 'False'
        gaussian_error = 'True' if self.statistics.gaussian_error else 'False'
        stationary = 'True' if self.statistics.data_type == 'Time-series' else 'False'
        heterogeneous = 'True' if self.statistics.heterogeneous else 'False'

        prop_table = f"""
        \\begin{{tabular}}{{rrrrrrr}}
            \\toprule
            Shape ($n$ x $d$) & Data Type & Missing Value & Linearity & Gaussian Errors & Time-Series & Heterogeneity \\\\
            \midrule
            {shape}   & {data_type} & {missingness} & {linearity} & {gaussian_error} & {stationary} & {heterogeneous} \\\\
            \\bottomrule
        \end{{tabular}}
        """
        return prop_table
    
    def preprocess_plot_prompt(self):
        if os.path.isfile(f'{self.visual_dir}/residuals_plot.jpg'):
            preprocess_plot = f"""
            The following are Residual Plots and Q-Q Plots for seleted pair of vairables.
            \\begin{{figure}}[H]
                \centering
                \\begin{{subfigure}}{{0.45\\textwidth}}
                    \centering
                    \includegraphics[width=\linewidth]{{{self.visual_dir}/residuals_plot.jpg}}
                    \\vfill
                    \caption{{Residual Plot.}}
                \end{{subfigure}}
                \\begin{{subfigure}}{{0.45\\textwidth}}
                    \centering
                    \includegraphics[width=\linewidth]{{{self.visual_dir}/qq_plot.jpg}}
                    \\vfill
                    \caption{{Q-Q Plot.}}
                \end{{subfigure}}
            \caption{{Plots for Data Properties Checking.}}
            \end{{figure}}   
            """
        else:
            preprocess_plot = ""
        return preprocess_plot

    def ts_eda_prompt(self):
        ts_eda_corr_summary = self.eda_result['lag_corr_summary']
        corr_summary_graph = self.eda_result['plot_path_lag_corr']
        corr_summary_text = granger_causality_to_latex(ts_eda_corr_summary['potential_granger_causality'])
        
        response = f"""
\subsection{{Correlation Analysis}}

\\begin{{figure}}[H]
        \centering
        \includegraphics[width=\linewidth]{{{self.visual_dir}/eda_lag_correlation.jpg}}
        \caption{{\label{{fig:corr}}Heatmap of Time-Lagged Correlations Among Variables.}}
\end{{figure}}

{corr_summary_text}
        """

        diagnostics_summary_graph = glob.glob(f"{self.visual_dir}/eda_ts_diagnostics_*")[:2]
        diagnostics_summary = self.eda_result['diagnostics_summary']
        diagnostics_summary_text = stationarity_summary_to_latex(diagnostics_summary)
        
        response += rf"""
\subsection{{Time Series Stationarity Analysis}}

\begin{{figure}}[H]
\centering
    """
        graph_width = str(1/len(diagnostics_summary_graph)-0.02)
        for idx, path in enumerate(diagnostics_summary_graph):
            response += rf"""
    \begin{{subfigure}}{{{graph_width}\textwidth}}
        \includegraphics[width=\textwidth]{{{path}}}
    \end{{subfigure}}
    """
            if idx != len(diagnostics_summary_graph) - 1:
                response += "\hfill"

            else:
                response += rf"""
            \caption{{Time Series Diagnostics for two selected variables. Each plot shows the variable with moving average, first difference transform, and ACF/PACF plots.}}
        \end{{figure}} 
        """
        # response += diagnostics_summary_text
        
        return response
    
    def eda_prompt(self):
        dist_doc, corr_doc = eda_summary_to_latex(self.eda_result)
        plot_path_dist = self.eda_result['plot_path_dist'][0].strip("'") or "",
        plot_path_corr = self.eda_result['plot_path_corr'][0].strip("'") or "", 
        response = rf"""
        \subsection{{Distribution Analysis}}
The following figure presents distributions of various variables. The orange dashed line indicates the mean, while the black solid line denotes the median. Variables are categorized into three types based on their distributional characteristics.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{self.visual_dir}/eda_dist.jpg}}
\caption{{Distribution Plots of Variables.}}
\end{{figure}}

{dist_doc}

\subsection{{Correlation Analysis}}

\begin{{minipage}}[t]{{0.5\linewidth}}
    {corr_doc}
\vfill
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.5\linewidth}}
    \begin{{figure}}[H]
        \centering
        \vspace{{-1.5cm}}
        \includegraphics[width=\linewidth]{{{self.visual_dir}/eda_corr.jpg}}
        \caption{{Correlation Heatmap of Variables.}}
    \end{{figure}}
\end{{minipage}}
        """
        return response
             
    def algo_selection_prompt(self):
        algo_candidates = self.algo_can
        response = """
        \\begin{itemize}
        """
 
        for algo in algo_candidates:
            sub_block = f"""
            \item \\textbf{{{algo}}}:
            \\begin{{itemize}}
                \item \\textbf{{Description}}: {algo_candidates[algo]['description']}
                \item \\textbf{{Justification}}: {algo_candidates[algo]['justification']}
            \end{{itemize}}
                         """
            response += sub_block

        response += """
                    \end{itemize}
                    """
        return response
    
    def param_selection_prompt(self):
        params = self.algo_param['hyperparameters']
        response = """
        \\begin{itemize}
        """
 
        for param in params:
            sub_block = f"""
            \item 
            \\textbf{{{params[param]['full_name']}}}:
            \\begin{{itemize}}
                \item \\textbf{{Value}}: {params[param]['value']}
                \item \\textbf{{Explanation}}: {params[param]['explanation']}
            \end{{itemize}}
                         """
            response += sub_block

        response += """
                    \end{itemize}
                    """
        response = response.replace('_', '\_')
        return response

    def procedure_prompt(self):
        algo_list = self.algo_selection_prompt()
        param_list = self.param_selection_prompt()

        response = f"""
        In this section, we provide a detailed description of the causal discovery process implemented by Causal Copilot. 
        We also provide the chosen algorithms and hyperparameters, along with the justifications for these selections.
        \subsection{{Data Preprocessing}}
        In this initial step, we preprocessed the data and examined its statistical characteristics. 
        This process involved data cleaning, handling missing values, and performing exploratory data analysis to examine variable distributions and inter-variable relationships.
                
        \subsection{{Algorithm Recommendation assisted with LLM}}
        Following preprocessing, we employed a large language model (LLM) to assist in 
        selecting appropriate algorithms for causal discovery based on the statistical characteristics of the dataset and relevant background knowledge. 
        The top three chosen algorithms, listed in order of suitability, are as follows:   
        {algo_list}
        Considering data properties, algorithm capability and user's instruction, the final algorithm we choose is [ALGO].
        \subsection{{Hyperparameter Values Proposal assisted with LLM}}
        Once the algorithms were selected, the LLM aided in proposing hyperparameters 
        for the chosen algorithm, which are specified below:
        {param_list}
        """

        if self.data_mode == 'real':
            response += rf"""
            \subsection{{Graph Tuning with Bootstrap and LLM Suggestion}}
            In the final step, we performed graph tuning with suggestions provided by the Bootstrap and LLM.
            
            We first applied the Bootstrap method to estimate the confidence level associated with each edge in the initial graph. Specifically:
            \begin{{itemize}}
                \item If an edge not present in the initial graph exhibited a Bootstrap confidence greater than 90\%, we added it to the graph.
                \item Conversely, if an existing edge had a confidence lower than 10\%, we removed it.
                \item For edges with moderate confidence (between 10\% and 90\%), we consulted the LLM to assess their validity and directionality, drawing on its extensive background knowledge.
            \end{{itemize}}
            The LLM contributed by:
            \begin{{itemize}}
                \item Reintroducing plausible edges that may have been overlooked by statistical methods;
                \item Removing or redirecting edges that appeared statistically valid but were conceptually implausible.
            \end{{itemize}}
    
            To improve the robustness of LLM-generated suggestions, we employed a voting mechanism. Importantly, LLM recommendations were not allowed to override high-confidence decisions made by the Bootstrap procedure.
            By integrating insights from both of Bootsratp and LLM to refine the causal graph, we can achieve improvements in graph's accuracy and robustness.
            """
        return response
    
    def graph_generate_prompts(self):
        if self.global_state.statistics.time_series:
            graph_response = rf"""
    \section{{Causal Graph Estimation Summary}}

    \subsection{{Causal Graph Discovered by the Algorithm}}

    \begin{{figure}}[H]
    \centering
    \begin{{subfigure}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{self.visual_dir}/{self.algo}_timelag_graph.pdf}}
        \caption{{Time Lag Graph Discovered by the Algorithm.}}
        \label{{fig:timelag}}
    \end{{subfigure}}
    \hfill
    \begin{{subfigure}}[b]{{0.48\textwidth}}
        \centering
        \includegraphics[width=\textwidth]{{{self.visual_dir}/{self.algo}_initial_graph.pdf}}
        \caption{{Summary Graph Discovered by the Algorithm. Solid lines represent causal edges identified by the algorithm, while dashed lines indicate strong correlations without inferred causality.}}
        \label{{fig:summary}}
    \end{{subfigure}}
    \caption{{Graphs Discovered by the Algorithm.}}
    \label{{fig:both_graphs}}
\end{{figure}}
            """
        else:
            graph_response = rf"""
    \section{{Causal Graph Estimation Summary}}

    \subsection{{Causal Graph Discovered by the Algorithm}}

    \begin{{figure}}[H]
        \centering
        \includegraphics[width=0.5\textwidth]{{{self.visual_dir}/{self.algo}_initial_graph.pdf}}
        \caption{{Causal Graph Discovered by the Algorithm. Solid lines represent causal edges identified by the algorithm, while dashed lines indicate strong correlations without inferred causality.}}
    \end{{figure}}

    The above is the original causal graph produced by our algorithm.
            """
        return graph_response
    
    def graph_effect_prompts(self):
        """
        Prompts for Initial Graph Analysis integrated with background knowledge
        Provide following infos:
        1. Relationship of the initial graph that has been converted into natural language
        2. Variable names
        3. Don't include Bootstrap infos here
        """
        if self.graph.sum() == 0:
            response_doc =  "⚠️ According to the given dataset, we cannot find any causal relationship among variables. Please provide more samples or check the data quality.\n"
        else:
            variables = self.data.columns
            edges_dict = convert_to_edges(self.algo, variables, self.graph)
            relation_text_dict, relation_text = edges_to_relationship(self.data, edges_dict)

            prompt = f"""
            The following text describes the causal relationship among variables:
            {relation_text}
            You are an expert in the causal discovery field and are familiar with background knowledge of these variables: {variables.tolist()}
            1. Please write one paragraph to describe the causal relationship, list your analysis as bullet points clearly.
            2. If variable names have meanings, please integrate background knowledge of these variables in the causal relationship analysis.
            Please use variable names {variables[0]}, {variables[1]}, ... in your description.
            3. Do not simply list out all relationships! You should summarize them and give some conclusions.
            4. Do not include any Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
            
            For example:
            The result graph shows the causal relationship among variables clearly. The {variables[1]} causes the {variables[0]}, ...
            """
            print("Start to find graph effect")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_doc = response.choices[0].message.content
            # response_doc = response_doc.replace('_', ' ')
        return response_doc

    def graph_revise_prompts(self):
        file_path = f'{self.visual_dir}/{self.algo}_revised_graph.pdf'
        if self.bootstrap_probability is not None and os.path.exists(file_path):
            response = f"""
            By using the method mentioned in the Section 4.4, we provide a revise graph pruned with Bootstrap and LLM suggestion.
            Pruning results are as follows.
            """
            if self.global_state.results.bootstrap_errors != []:
                    response += f"""
                    The following are force and forbidden results given by Bootstrap:
                    
                    {', '.join(self.global_state.results.bootstrap_errors)}
                    """
            else:
                response += f"""
                Bootstrap doesn't force or forbid any edges.
                """
            llm_evaluation_json = self.global_state.results.llm_errors
            direct_record = llm_evaluation_json['direct_record']
            forbid_record = llm_evaluation_json['forbid_record']
            
            if  forbid_record != {} and forbid_record is not None:
                response += f"""
                The following relationships are forbidden by LLM:
                
                \\begin{{itemize}}
                """
                for item in forbid_record.values():
                    response += f"""
                    \item \\textbf{{{item[0][0].replace('_', ' ')} $\\rightarrow$ {item[0][1].replace('_', ' ')}}}: {item[1]}
                    """
                response += f"""
                \end{{itemize}}
                """     
            else:
                response += f"""
                LLM doesn't forbid any edges.
                """
                
            llm_direction_reason = dict(list(direct_record.items())[:10])
            if llm_direction_reason!=[] and llm_direction_reason is not None:
                if len(llm_direction_reason) > 10:
                    response += f"""
                    The following are directions added by the LLM (We only include a part of edges):
                    \\begin{{itemize}}
                    """
                else:
                    response += f"""
                        The following are directions added by the LLM:
                        \\begin{{itemize}}
                        """
                for item in llm_direction_reason.values():
                    response += f"""
                    \item \\textbf{{{item[0][0].replace('_', ' ')} $\\rightarrow$ {item[0][1].replace('_', ' ')}}}: {item[1]}
                    """
                response += f"""
                \end{{itemize}}
                """ 
            else:
                response += f"""
                LLM doesn't decide any direction of edges.
                """
            # response = response.replace('_', ' ')
            response += """
            This structured approach ensures a comprehensive and methodical analysis of the causal relationships within the dataset.
            """

            response += fr"""
            \begin{{figure}}[H]
            \centering
            \includegraphics[width=0.5\textwidth]{{{self.visual_dir}/{self.algo}_revised_graph.pdf}}
            \caption{{Revised Graph by LLM.}}
            \end{{figure}}
            """

        else:
            response = 'You have skipped the Pruning and Reliability Analysis.'
        #print('graph revise prompt: ', response)
        return response
    
    def confidence_graph_prompts(self):
        ### generate graph layout ###
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_certain_edges': 'Undirected Directed Non-Ancestor Edge', #(o->)
                    'half_uncertain_edges': 'Non-Ancestor Edge', #(o-)
                    'none_edges': 'No D-Seperation Edge', #(o-o)
                    'none_existence':'No Edge'}
        graph_text = """
        \\begin{figure}[H]
            \centering
        """
        if self.bootstrap_probability is not None:
            bootstrap_dict = {k: v for k, v in self.bootstrap_probability.items() if v is not None and sum(v.flatten())>0}
            zero_graphs = [k for k, v in self.bootstrap_probability.items() if  v is not None and sum(v.flatten())==0]
            if len(zero_graphs) < len(bootstrap_dict):
                width = 0.3  # Approximately 1/3 of textwidth with some spacing
                # Counter to track position in the row
                counter = 0
                for key in bootstrap_dict.keys():
                    graph_path = f'{self.visual_dir}/{key}_confidence_heatmap.jpg'
                    caption = f'{name_map[key]}'
                    # Add subfigure
                    graph_text += f"""
                    \\begin{{subfigure}}{{{width}\\textwidth}}
                            \centering
                            \includegraphics[width=\linewidth]{{{graph_path}}}
                            \\vfill
                            \caption{{{caption}}}
                    \end{{subfigure}}"""
                    # Increment counter
                    counter += 1
                    # Start a new row after every 3 plots or at the end
                    if counter % 3 == 0 and counter < len(bootstrap_dict):
                        graph_text += "\n        \\\\[10pt]"  # Line break and some vertical space
                graph_text += """
                \caption{Confidence Heatmap of Different Edges.}
                \end{figure}    
                """
                ### Generate text illustration
                text_map = {'certain_edges': rf'directed edge ($\rightarrow$)', #(->)
                            'uncertain_edges': 'undirected edge ($-$)', #(-)
                            'bi_edges': 'edge with hidden confounders ($\leftrightarrow$)', #(<->)
                            'half_certain_edges': rf'edge of non-ancestor (o$\rightarrow$)', #(o->)
                            'half_uncertain_edges': 'edge of non-ancestor (o$-$)', #(o-)
                            'none_edges': 'egde of no D-Seperation set (o$-$o)', #(o-o)
                            'none_existence':'No Edge'}
                graph_text += "The above heatmaps show the confidence probability we have on different kinds of edges, including "
                for k in bootstrap_dict.keys():
                    graph_text += f"{text_map[k]}, "
                zero_graphs = [k.replace("_", "-") for k in zero_graphs]
                graph_text += "The heatmap of " + ', '.join(zero_graphs) + " is not shown because probabilities of all edges are 0. "
            else:
                return ""
        else:
            graph_text = 'You have skipped the Pruning and Reliability Analysis.'
        return graph_text
        

    def confidence_analysis_prompts(self):
        if self.bootstrap_probability is not None:
            edges_dict = self.global_state.results.raw_edges
            relation_text_dict, relation_text = edges_to_relationship(self.data, edges_dict, self.bootstrap_probability)
            variables = '\t'.join(self.data.columns)
            prompt = f"""
            The following text describes the causal relationship among variables from a statisical perspective:
            {relation_text}
            We use traditional causal discovery algorithm to find this relationship, and the probability is calculated with bootstrapping.
            This result is solely from statistical perspective, so it is not reliable enough.
            You are an expert in the causal discovery field and are familiar with background knowledge of these variables: {variables}
            Based on this statistical confidence result, and background knowledge about these variables,
            Please write a paragraph to analyze the reliability of this causal relationship graph. 
            
            **Your Task**
            Firstly, briefly describe how we get these probability with 1-2 sentences.
            Secondly, categorize and these relationships into 3 types and list them out: 
            High Confidence Level (probability>=0.9), 
            Moderate Confidence Level (probability between 0.1 and 0.9), 
            Low Confidence Level (probability<=0.1)
            Do Not include Numerical Numbers in your text!
            
            **Template**
            To evaluate the confidence associated with each edge in the causal graph, we employed a bootstrapping procedure to estimate the probability of existence for each edge. From a statistical perspective, we categorize these probabilities into three levels:
            \\begin{{itemize}}
            \item \\textbf{{High Confidence Edges}}: ...
            \item \\textbf{{Moderate Confidence Edges}}: ...
            \item \\textbf{{Low Confidence Edges}}: ...
            \end{{itemize}}
            **You Must**
            1. Follow the template above, do not include anything else like  ```
            2. Write in a professional and concise way, and include all relationships provided.
            3. The list must be in latex format
            """
            print("Start to analyze graph reliability")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_doc = response.choices[0].message.content
            response_doc = response_doc.replace('%', '\%')
            response_doc = response_doc.replace('_', ' ')
            #print('reliability analysis:',response_doc)
        else:
            response_doc = ''
        return response_doc

    def refutation_analysis_prompts(self):
        if self.global_state.results.refutation_analysis is not None:
            text = f"""
                    \\begin{{figure}}[H]
                        \centering
                        \includegraphics[width=0.7\\textwidth]{{{self.visual_dir}/refutation_graph.jpg}}
                        \caption{{Refutation Graph.}}
                    \end{{figure}} \n
                    """
            text += self.global_state.results.refutation_analysis.replace('_', '\_')
            text = text.replace('%', '\%')
        else:
            text = 'You have skipped the refutation analysis step.'
        return text 
    
    def comparision_prompt(self):
        if len(self.global_state_list) < 2:
            return "", ""
        else:
            graph_text = """
            \subsection{Result Graph Comparision}
            \\begin{figure}[H]
                \centering
            """
            algos = [state.algorithm.selected_algorithm for state in self.global_state_list]
            length = round(1/len(self.global_state_list), 2)-0.01
            for algo in algos:
                graph_path = f'{self.visual_dir}/{algo}_initial_graph.pdf'
                caption = f'Initial Result Graph of {algo}'
                graph_text += f"""
                \\begin{{subfigure}}{{{length}\\textwidth}}
                        \centering
                        \includegraphics[width=\linewidth]{{{graph_path}}}
                        \\vfill
                        \caption{{{caption}}}
                    \end{{subfigure}}"""            
            graph_text += """
            \caption{Result Graph Comparision of Different Algorithms.}
            \end{figure}    
            """

            prompt = f"""
Help me to write a comparison of the following causal discovery results of different algorithms.
**You should cover the following points**:
1. The different edges of different algorithms' causal graphs.
2. The common edges of different algorithms' causal graphs.
3. Which edges are more reliable and why.
**Note**:
1. Only include your comparison text content in the response. Don't include any other things!
2. Do not include any titles or subtitles. I only need the comparison text.
**causal discovery results of different algorithms**:
"""
            for state in self.global_state_list:
                prompt += f"Result of Algorithm {state.algorithm.selected_algorithm}:\n"            
                relation_text_dict, relation_text = edges_to_relationship(self.data, state.results.raw_edges)
                prompt += f"{relation_text}\n"
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     f"""
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": prompt}
                ]
            )
            response_doc = response.choices[0].message.content
            response_doc = response_doc.replace("_", r"\_")
            response_doc = bold_conversion(response_doc)
            return graph_text, response_doc
        
    def abstract_prompt(self):
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     f"""
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": 
                     f"""
                     Help me to write a short abstract according to the given information. 
                     You should cover what data is analyzed (find it in title and introduction), what methodology we used, what result we got, and what is our contribution.
                     Only include your abstract text content in the response. Don't include any other things like title, etc.
                     Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
                     0. Title: {self.title}
                     1. Introduction: {self.intro_info}
                     2. Selected Algorithm: {self.algo}
                     3. Discovery Procedure: {self.discover_process},
                     4. Graph Result Analysis: {self.graph_prompt},
                     5. Reliability Analysis: {self.reliability_prompt}
                     6. Comparision of different algorithms: {self.result_comparison}
                     7. Inference Analysis: {self.inf_report}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        response_doc = response_doc.replace("_", r"\_")
        return response_doc
    
    def keyword_prompt(self):
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     """
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": 
                     f"""
                     Give me some keywords according to the given information, and these keywords are for an academic report.
                    You should seperate each keywords with a comma, like 'keyword1, keyword2, keyword3'.
                     Only include your keywords text in the response. Don't include any other things.
                     Only include 5 most important key words.
                     0. Title: {self.title}
                     1. Abstract: {self.abstract}
                     1. Introduction: {self.intro_info}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        return response_doc
    
    def conclusion_prompt(self):
        response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": 
                     f"""
                     You are an expert in the causal discovery field and helpful assistant.                    
                     """},
                    {"role": "user", "content": 
                     f"""
                     Help me to write a 1-2 paragraphs conclusion according to the given information. 
                     You should cover:
                      1. what data is analyzed (find it in title and introduction)
                      2. what methodology we used (find it in Discovery Procedure)
                      3. what result we got, this point is important (find it in Graph Result Analysis and Graph Revise Procesure)
                      4. what is our contribution, this point is important (summarize it by yourself)
                     Only include your conclusion text content in the response. Don't include any other things like title, etc.
                     0. Title: {self.title}
                     1. Introduction: {self.intro_info}
                     2. Discovery Procedure: {self.discover_process},
                     3. Graph Result Analysis: {self.graph_prompt},
                     4. Graph Revise Procesure: {self.revise_process}
                     5. Reliability Analysis: {self.reliability_prompt}
                     6. Comparision of different algorithms: {self.result_comparison}
                     7. Inference Analysis: {self.inf_report}
                     """}
                ]
            )

        response_doc = response.choices[0].message.content
        response_doc = response_doc.replace("_", r"\_")
        return response_doc
    
    def latex_convert(self, text):
        prompt = f"""
        Please convert this markdown format text into latex format.
        For example, 
        1. for subheadings '## heading', convert it into '\subsection{{heading}}; for subsubheadings '### heading', convert it into '\subsubsection{{heading}}'
        2. for list of items '-item1 -item2 -item3', convert them into 
        '\\begin{{itemize}}
        \item item1
        \item item2
        \item item3
        \end{{itemize}}'
        3. For bolding notation '**text**', convert it into \\textbf{{text}}.
        4. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
        This is the text you need to convert: {text}
        Only response converted text, please do not include other things like 'Here is the converted text in LaTeX format:'
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        return response_doc

    def generation(self, debug=False):
            '''
            generate and save the report
            :param debug: bool, if True, use template directly without LLM
            :return: Str: A technique report explaining all the results for readers
            '''
            if debug:
                # Load appropriate template based on data mode and ground truth
                prompt_template = load_context("report/context/template_debug.tex")
                return prompt_template

            # Non-debug path continues with full report generation
            print("Start to generate the report")
            # Data info
            df = self.data.copy()
            if self.data.shape[1] > 20:  
                # choose 10 columns randomly
                random_columns = np.random.choice(df.columns, size=10, replace=False)
                df = df[random_columns]
            # df.columns = [var.replace('_', ' ') for var in df.columns]
            data_preview = df.head().to_latex(index=False)
            if len(self.data.columns) >= 8:
                data_preview = f"""
                \\resizebox{{\\textwidth}}{{!}}{{
                {data_preview}
                }}
                """
            data_prop_table = self.data_prop_prompt()
            # Intro info
            self.title, dataset = self.get_title()
            self.intro_info = self.intro_prompt()
            # Background info
            if self.data_mode == 'real':
                self.background_info1, self.background_info2 = self.background_prompt()
            else:
                self.background_info1, self.background_info2 = '', ''
            # EDA info
            if self.global_state.statistics.time_series:
                self.eda_info = self.ts_eda_prompt()
            else:
                self.eda_info = self.eda_prompt()
            # Procedure info
            self.discover_process = self.procedure_prompt()
            self.preprocess_plot = self.preprocess_plot_prompt()
            self.graph_block = self.graph_generate_prompts()
            # Graph effect info
            self.graph_prompt = bold_conversion(self.global_state.logging.graph_conversion['initial_graph_analysis'])
            self.graph_prompt = list_conversion(self.graph_prompt)
            self.graph_prompt = fix_latex_itemize(self.graph_prompt)
            # Graph Revise info
            if self.data_mode == 'real':
                self.revise_process = self.graph_revise_prompts()
            else:
                self.revise_process = '' 
            # Graph Reliability info
            self.reliability_prompt = self.confidence_analysis_prompts()
            self.confidence_graph_prompt = self.confidence_graph_prompts()
            self.refutation_analysis = self.refutation_analysis_prompts()
            self.result_comparison_graph_text, self.result_comparison = self.comparision_prompt()
            
            # Causal Inference info
            if self.inference_global_state.inference.task_index != -1:
                print('self.inference_global_state.inference.task_index: ', self.inference_global_state.inference.task_index)
                self.inf_report = ""
                for task_index in range(self.inference_global_state.inference.task_index+1):
                    print('task_index: ', task_index)
                    self.inf_report += f"\section{{Causal Inference Results for Task {task_index+1}}}\n"
                    inf_report_generator = Inference_Report_generation(self.inference_global_state, self.args, task_index)
                    task_result = inf_report_generator.generation()
                    self.inf_report += f"{task_result}\n"
            else:
                self.inf_report = ''

            self.abstract = self.abstract_prompt()
            self.conclusion = self.conclusion_prompt()
            

            if self.data_mode == 'simulated':
                prompt_template = load_context("report/context/template_simulated_notruth.tex")
            else:
                prompt_template = load_context("report/context/template_real_notruth.tex")

            replacement1 = {
                "[ABSTRACT]": self.abstract.replace("&", r"\&") or "",
                "[INTRO_INFO]": self.intro_info.replace("&", r"\&") or "",
                "[BACKGROUND_INFO1]": self.background_info1.replace("&", r"\&") or "",
                "[BACKGROUND_INFO2]": self.background_info2.replace("&", r"\&") or "",
                "[DATA_PREVIEW]": data_preview or "",
                "[DATA_PROP_TABLE]": data_prop_table or "",
                "[EDA_INFO]": self.eda_info or "",
                "[CAUSAL_GRAPH]": self.graph_block.replace("&", r"\&") or "",
                "[RESULT_ANALYSIS]": self.graph_prompt.replace("&", r"\&") or "",
                "[DISCOVER_PROCESS]": self.discover_process.replace("&", r"\&") or "",
                "[PREPROCESS_GRAPH]": self.preprocess_plot or "",
                "[REVISE_PROCESS]": self.revise_process.replace("&", r"\&") or "",
                "[RELIABILITY_ANALYSIS]": self.reliability_prompt.replace("&", r"\&") or "",
                "[CONFIDENCE_GRAPH]": self.confidence_graph_prompt or "",
                "[REFUTATION_GRAPH]": self.refutation_analysis or "",
                "[RESULT_COMPARISION]": self.result_comparison.replace("&", r"\&") or "",
                "[CONCLUSION]": self.conclusion.replace("&", r"\&") or "",
                "[INFERENCE]": self.inf_report or ""
            }
            replacement2 = {
                "[TITLE]": self.title or "",
                "[DATASET]": dataset or "",
                "[POTENTIAL_GRAPH]": f'{self.visual_dir}/potential_relation.pdf',
                "[ALGO]": self.algo or "",
                "[RESULT_GRAPH0]": f'{self.visual_dir}/true_graph.pdf',
                "[RESULT_GRAPH3]": f'{self.visual_dir}/metrics.jpg',
                "[RESULT_GRAPH_COMPARISION]": self.result_comparison_graph_text
            }

            for placeholder, value in replacement1.items():
                prompt_template = prompt_template.replace(placeholder, value)
            for placeholder, value in replacement2.items():
                prompt_template = prompt_template.replace(placeholder, value)
            prompt_template = replace_unicode_with_latex(prompt_template)
            #print(prompt_template)
            return prompt_template
    
    def latex_bug_checking(self, tex_path, num_error_corrections=2):
        save_path = self.global_state.user_data.output_report_dir

        # Iteratively fix any LaTeX bugs
        for i in range(num_error_corrections):
            # Filter trivial bugs in chktex
            check_output = os.popen(f"lacheck {tex_path} -q -n2 -n24 -n13 -n1").read()
            with open(tex_path, 'r', encoding='utf-8') as file:
                tex_content = file.read()
            if check_output:
                response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": f"""
                     You are a helpful debugging assistant, help me to fix bugs in the latex report I will give you. 
                     1. Please fix the LaTeX errors guided by the output of `chktek`:
                        {check_output}.
                    ** YOU SHOULD **
                     1. Make the minimal fix required and do not change any other contents!
                     2. Only include your latex content in the response which can be rendered to pdf directly. Don't include other things like '''latex '''
                     """},
                    {"role": "user", "content": tex_content}
                ]
                )
                output = response.choices[0].message.content
                with open(f'{save_path}/report_revised.tex', 'w', encoding='utf-8') as file:
                    file.write(output)
            else:
                break

    def save_report(self, report):
        save_path = self.global_state.user_data.output_report_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f'{save_path}/report.txt', 'w', encoding='utf-8') as file:
            # Write some text to the file
            file.write(report)
        with open(f'{save_path}/report.tex', 'w', encoding='utf-8') as file:
            file.write(report)
        # # fix latex bugs before rendering
        # print('check latex bug')
        # self.latex_bug_checking(f'{save_path}/report.tex')
        # Compile the .tex file to PDF using pdflatex
        print('start compilation')
        compile_tex_to_pdf_with_refs(f'{save_path}/report.tex', save_path)

def test(args, global_state):
    my_report = Report_generation(global_state, args)
    report = my_report.generation()
    my_report.save_report(report)

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
    # args = parse_args()
    # with open('/Users/wwy/Documents/Project/Causal-Copilot/demo_data/20250408_105532/DailyDelhiClimate/output_graph/PCMCI_global_state.pkl', 'rb') as file:
    #     global_state = pickle.load(file)
    # test(args, global_state)
    save_path = '/Users/wwy/Documents/Project/Causal-Copilot/demo_data/20250408_145536/sachs/output_report'
    compile_tex_to_pdf_with_refs(f'{save_path}/report.tex', save_path)
    

