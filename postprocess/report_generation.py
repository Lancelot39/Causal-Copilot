from openai import OpenAI
import os
import re
import numpy as np 
from plumbum.cmd import latexmk
from plumbum import local
import networkx as nx
from postprocess.visualization import Visualization, convert_to_edges
from postprocess.judge_functions import edges_to_relationship
import PyPDF2
import ast

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

        self.client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        self.data_mode = args.data_mode
        self.data_file = args.data_file
        self.global_state = global_state
        self.args = args 
        self.statistics_desc = global_state.statistics.description
        self.knowledge_docs = global_state.user_data.knowledge_docs[0]
        # Data info
        self.data = global_state.user_data.processed_data
        self.statistics = global_state.statistics
        # EDA info
        self.eda_result = global_state.results.eda
        # Result graph matrix
        self.raw_graph = global_state.results.raw_result
        self.graph = global_state.results.converted_graph
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
        if os.path.isdir(self.data_file):
            for file in os.listdir(self.data_file):
                if file.endswith(".csv"):
                    data_path = file
                    filename = os.path.splitext(os.path.basename(data_path))[0]
                    filename = filename.capitalize()
                    filename = filename.capitalize().replace("_", r"\_")
                    title = f'Causal Discovery Report on {filename}'
                    break
        elif self.data_file.endswith(".csv"):
            data_path = self.data_file
            filename = os.path.splitext(os.path.basename(data_path))[0]
            filename = filename.capitalize()
            filename = filename.capitalize().replace("_", r"\_")
            title = f'Causal Discovery Report on {filename}'
        else:
            title = 'Causal Discovery Report on Given Dataset'
        return title, filename
    
    def intro_prompt(self):
        prompt = f"""
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
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
        I want to conduct a causal discovery on a dataset and write a report. There are some background knowledge about this dataset.
        There are three sections:
        ### 1. Detailed Explanation about the Variables
        ### 2. Possible Causal Relations among These Variables
        ### 3. Other Background Domain Knowledge that may be Helpful for Experts
        Please give me text in the first section ### 1. Detailed Explanation about the Variables
        I only need the text, do not include title
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
        section1 = section1.replace(r'\textbf', r'\\newline \\textbf')

        col_names = '\t'.join(self.data.columns)
        prompt = f"""
I want to conduct a causal discovery on a dataset and write a report. There is some background knowledge about this dataset.
There are three sections:
### 1. Detailed Explanation about the Variables
### 2. Possible Causal Relations among These Variables
### 3. Other Background Domain Knowledge that may be Helpful for Experts

Please give me text in the second section ### 2. Possible Causal Relations among These Variables:
1. In this part, all relationships should be listed in this format: **A -> B**: explanation. 
2. Only one variable can appear at each side of ->; for example, **A -> B** is ok but **A -> B/C** is wrong, and **A -> B and C** is also wrong.
3. All variables should be among {col_names}, please delete contents that include any other variables!
4. Do not include any Greek Letters! Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$

For example: 
- **'Raf' -> 'Mek'**: Raf activates Mek through phosphorylation, initiating the MAPK signaling cascade.

I only need the text; do not include the title.
Background about this dataset: {self.knowledge_docs}
"""
        response_background = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        section2 = response_background.choices[0].message.content
        section2 = re.sub(r'.*\*\*(.*?)\*\*', r'\\textbf{\1}', section2)
        section2 = section2.replace(r'\textbf', r'\\newline \\textbf')

        variables = self.data.columns
        pattern = r'\*\*(.*?)\s*(→|↔|->|<->)\s*(.*?)\*\*'
        relations = []
        # Find all matches
        matches = re.findall(pattern, section2)
        for match in matches:
            left_part = match[0]
            right_part = match[2]

            elements = [left_part]
            split_elements = re.split(r'\s*(→|↔|->|<->)\s*', right_part)
            # Iterate over the split elements
            for i in range(0, len(split_elements), 2):
                element = split_elements[i]
                if element:  # Avoid empty strings
                    elements.append(element)
            # Create pairs for the elements
            for i in range(len(elements) - 1):
                # deal with case like 'Length and Diameter'->'Viscera Weight' or 'Length\Diameter'->'Viscera Weight'
                if '/' in elements[i + 1] or 'and' in elements[i + 1]:
                    targets = [t for t in elements[i + 1].split('/|and')]
                    for target in targets:
                        relations.append((elements[i], target))
                else:
                    relations.append((elements[i], elements[i + 1]))
        # deal with case like ('Length, Diameter, Height', 'Viscera Weight')
        result = []
        for relation in relations:
            left = relation[0].split(',')
            right = relation[1].split(',')
            for l in left:
                for r in right:
                    result.append((l.strip(), r.strip()))
        # Potential Relationship Visualization
        zero_matrix = np.zeros((len(variables), len(variables)))
        for tuple in result:
            if tuple[0].lower() in variables.str.lower() and tuple[1].lower() in variables.str.lower():
                ind1 = variables.str.lower().get_loc(tuple[0].lower())
                ind2 = variables.str.lower().get_loc(tuple[1].lower())
                zero_matrix[ind2, ind1] = 1

        my_visual = Visualization(self.global_state)
        g = nx.from_numpy_array(zero_matrix, create_using=nx.DiGraph)
        # Relabel nodes with variable names from data columns
        mapping = {i: self.data.columns[i] for i in range(len(self.data.columns))}
        g = nx.relabel_nodes(g, mapping)
        pos = nx.spring_layout(g)
        _ = my_visual.plot_pdag(zero_matrix, 'potential_relation.pdf', pos=pos, relation=True)
        relation_path = f'{self.visual_dir}/potential_relation.pdf'
        #TODO: fix graph bug here
        # section2 = re.sub(r'.*\*\*(.*?)\*\*', r'\\textbf{\1}', section2)
        # section2 = section2.replace(r'\textbf', r'\\newline \\textbf')

        def get_pdf_page_size(pdf_path):
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page = reader.pages[0]
                width = page.mediabox.width
                height = page.mediabox.height
                return height>(0.7*width)
            
        if sum(zero_matrix.flatten())!=0:
            figure_tall =  get_pdf_page_size(relation_path)
            if figure_tall:
                relation_prompt = f"""
                \\begin{{minipage}}[t]{{0.6\linewidth}}
                {section2}
                \\vfill
                \end{{minipage}}
                %\hspace{{0.05\\textwidth}}
                \\begin{{minipage}}[t]{{0.4\linewidth}}
                    \\begin{{figure}}[H]
                        \centering
                        \\resizebox{{\linewidth}}{{!}}{{\includegraphics[height=0.3\\textheight]{relation_path}}}
                        \caption{{\label{{fig:relation}}Possible Causal Relation Graph}}
                    \end{{figure}}
                \end{{minipage}}
                """
            else:
                relation_prompt = f"""
                {section2}

                \\begin{{figure}}[H]
                \centering
                \includegraphics[width=0.5\linewidth]{relation_path}
                \caption{{\label{{fig:relation}}Possible Causal Relation Graph}}
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
            Shape ($n$ x $d$) & Data Type & Missing Value & Linearity & Gaussian Errors & Time-Series & Heterogeneity \\
            \midrule
            {shape}   & {data_type} & {missingness} & {linearity} & {gaussian_error} & {stationary} & {heterogeneous} \\
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
                    \caption{{Residual Plot}}
                \end{{subfigure}}
                \\begin{{subfigure}}{{0.45\\textwidth}}
                    \centering
                    \includegraphics[width=\linewidth]{{{self.visual_dir}/qq_plot.jpg}}
                    \\vfill
                    \caption{{Q-Q Plot}}
                \end{{subfigure}}
            \caption{{Plots for Data Properties Checking}}
            \end{{figure}}   
            """
        else:
            preprocess_plot = ""
        return preprocess_plot

    def eda_prompt(self):
        dist_input_num = self.eda_result['dist_analysis_num']
        dist_input_cat = self.eda_result['dist_analysis_cat']
        corr_input = self.eda_result['corr_analysis']
        
        # Description of distribution
        response_dist_doc = ""
        if dist_input_num != {}:
            response_dist_doc += "Numerical Variables \n \\begin{itemize} \n"
            left_skew_list = []
            right_skew_list = []
            symmetric_list = []
            for feature in dist_input_num.keys():
                if dist_input_num[feature]['mean']<dist_input_num[feature]['median']:
                    left_skew_list.append(feature)
                elif dist_input_num[feature]['mean']>dist_input_num[feature]['median']:
                    right_skew_list.append(feature)
                else:
                    symmetric_list.append(feature)
            response_dist_doc += f"\item Slight left skew distributed variables: {', '.join(left_skew_list) if left_skew_list != [] else 'None'} \n"
            response_dist_doc += f"\item Slight right skew distributed variables: {', '.join(right_skew_list) if right_skew_list != [] else 'None'} \n"
            response_dist_doc += f"\item Symmetric distributed variables: {', '.join(symmetric_list) if symmetric_list != [] else 'None'} \n"
            response_dist_doc += "\end{itemize} \n"
        if dist_input_cat != {}:
            response_dist_doc += "Categorical Variables \n"
            response_dist_doc += "\\begin{itemize} \n"
            for feature in dist_input_cat.keys():
                response_dist_doc += f"\item {feature}: {dist_input_cat[feature]} \n"
            response_dist_doc += "\end{itemize} \n"
        #print('response_dist_doc: ', response_dist_doc)
        # Description of Correlation
        response_corr_doc = "\\begin{itemize} \n"
        high_corr_list  = [f'{key[0]} and {key[1]}' for key, value in corr_input.items() if abs(value) > 0.8]
        if len(high_corr_list)>10:
            response_corr_doc += f"\item Strong Correlated Variables: {', '.join(high_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item Strong Correlated Variables: {', '.join(high_corr_list) if high_corr_list != [] else 'None'} \n"
        med_corr_list = [f'{key[0]} and {key[1]}' for key, value in corr_input.items() if (abs(value) <= 0.8 and abs(value) > 0.5)]
        if len(med_corr_list)>10:
            response_corr_doc += f"\item Moderate Correlated Variables: {', '.join(med_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item Moderate Correlated Variables: {', '.join(med_corr_list) if med_corr_list != [] else 'None'} \n"
        low_corr_list = [f'{key[0]} and {key[1]}' for key, value in corr_input.items() if abs(value) <= 0.5]
        if len(low_corr_list)>10:
            response_corr_doc += f"\item Weak Correlated Variables: {', '.join(low_corr_list)}"
            response_corr_doc += ", etc. \n"
        else:
            response_corr_doc += f"\item Weak Correlated Variables: {', '.join(low_corr_list) if low_corr_list != [] else 'None'} \n"
        response_corr_doc += "\end{itemize} \n"
        print('response_corr_doc: ',response_corr_doc)
        return response_dist_doc, response_corr_doc
             

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
                        \\textbf{params[param]['full_name']}:
                        \\begin{{itemize}}
                            \item \\textbf{{Value}}: {params[param]['value']}
                            \item \\textbf{{Explanation}}: {params[param]['explanation']}
                        \end{{itemize}}

                         """
            response += sub_block

        response += """
                    \end{itemize}
                    """
        return response

    def procedure_prompt(self):
        algo_list = self.algo_selection_prompt()
        param_list = self.param_selection_prompt()

        repsonse = f"""
        In this section, we provide a detailed description of the causal discovery process implemented by Causal Copilot. 
        We also provide the chosen algorithms and hyperparameters, along with the justifications for these selections.

        \subsection{{Data Preprocessing}}
        In this initial step, we preprocessed the data and examined its statistical characteristics. 
        This involved cleaning the data, handling missing values, and performing exploratory data analysis to understand distributions and relationships between variables.
                
        \subsection{{Algorithm Selection assisted with LLM}}
        Following data preprocessing, we employed a large language model (LLM) to assist in 
        selecting appropriate algorithms for causal discovery based on the statistical characteristics of the dataset and relevant background knowledge. 
        The top three chosen algorithms, listed in order of suitability, are as follows:   
        {algo_list}

        Considering data properties, algorithm capability and LLM's suggestion, the final algorithm we choose is [ALGO].

        \subsection{{Hyperparameter Values Proposal assisted with LLM}}
        Once the algorithms were selected, the LLM aided in proposing hyperparameters 
        for the chosen algorithm, which are specified below:
        {param_list}

        """

        if self.args.data_mode == 'real':
            repsonse += f"""
            \subsection{{Graph Tuning with Bootstrap and LLM Suggestion}}
            In the final step, we performed graph tuning with suggestions provided by the Bootstrap and LLM.
            
            Firstly, we use the Bootstrap technique to get how much confidence we have on each edge in the initial graph.
            If the confidence probability of a certain edge is greater than 95% and it is not in the initial graph, we force it.
            Otherwise, if the confidence probability is smaller than 5% and it exists in the initial graph, we forbid it.
            For those moderate confidence edges, we utilize LLM to double check their existence and direction according to its knowledge repository.
            
            In this step LLM can use background knowledge to add some edges that are neglected by Statistical Methods, delete and redirect some unreasonable relationships.
            Voting techniques are used to enhance the robustness of results given by LLM, and the results given by LLM should not change results given by Bootstrap.
            Finally, we use Kernel-based Independence Test to remove redundant edges added by LLM hallucination.

            By integrating insights from both of Bootsratp and LLM to refine the causal graph, we can achieve improvements in graph's accuracy and robustness.
            """
        return repsonse
    
    def graph_effect_prompts(self):
        """
        Prompts for Initial Graph Analysis integrated with background knowledge
        Provide following infos:
        1. Relationship of the initial graph that has been converted into natural language
        2. Variable names
        3. Don't include Bootstrap infos here
        """
        variables = self.data.columns

        edges_dict = convert_to_edges(self.algo, variables, self.raw_graph)
        relation_text_dict, relation_text = edges_to_relationship(self.data, edges_dict)

        prompt = f"""
        The following text describes the causal relationship among variables:
        {relation_text}
        You are an expert in the causal discovery field and are familiar with background knowledge of these variables: {variables.tolist()}
        1. Please write one paragraph to describe the causal relationship, list your analysis as bullet points clearly.
        2. If variable names have meanings, please integrate background knowledge of these variables in the causal relationship analysis.
        Please use variable names {variables[0]}, {variables[1]}, ... in your description.
        3. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
        
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
        #print('graph effect: ',response_doc)
        return response_doc

    def list_conversion(self, text):
        # Split the text into lines
        lines = text.strip().split('\n')
        latex_lines = []
        
        # Process each line
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('+'):
                # Convert bullet points to LaTeX itemize
                if len(latex_lines) > 1:  # Not the first list item
                    latex_lines.append(r"  \item " + line[2:].strip())
                else:  # Starting a new itemize list
                    latex_lines.append(r"\begin{itemize}")
                    latex_lines.append(r"  \item " + line[2:].strip())
            else:
                # If it's a regular line, add it as is
                latex_lines.append(line)
        # Close any open itemize
        if len(latex_lines) > 1:
            latex_lines.append(r"\end{itemize}")
        
        return "\n".join(latex_lines)

    
    def graph_revise_prompts(self):
        repsonse = f"""
        By using the method mentioned in the Section 4.4, we provide a revise graph pruned with Bootstrap and LLM suggestion.
        Pruning results are as follows.
        """
        if self.global_state.results.bootstrap_errors != []:
                repsonse += f"""
                The following are force and forbidden results given by Bootstrap:
                
                {', '.join(self.global_state.results.bootstrap_errors)}
                """
        else:
            repsonse += f"""
            Bootstrap doesn't force or forbid any edges.
            """
        llm_evaluation_json = self.global_state.results.llm_errors
        direct_record = llm_evaluation_json['direct_record']
        forbid_record = llm_evaluation_json['forbid_record']
        
        if  forbid_record != {} and forbid_record is not None:
            repsonse += f"""
            The following relationships are forbidden by LLM:
            
            \\begin{{itemize}}
            """
            for item in forbid_record.values():
                repsonse += f"""
                \item \\textbf{{{item[0][0]} $\\rightarrow$ {item[0][1]}}}: {item[1]}
                """
            repsonse += f"""
            \end{{itemize}}
            """     
        else:
            repsonse += f"""
            LLM doesn't forbid any edges.
            """
            
        llm_direction_reason = direct_record  
        if llm_direction_reason!={} and llm_direction_reason is not None:
            repsonse += f"""
                The following are directions confirmed by the LLM:
                \\begin{{itemize}}
                """
            for item in llm_direction_reason.values():
                repsonse += f"""
                \item \\textbf{{{item[0][0]} $\\rightarrow$ {item[0][1]}}}: {item[1]}
                """
            repsonse += f"""
            \end{{itemize}}
            """ 
        else:
            repsonse += f"""
            LLM doesn't decide any direction of edges.
            """
        
        repsonse += """
        This structured approach ensures a comprehensive and methodical analysis of the causal relationships within the dataset.
        """
        #print('graph revise prompt: ', repsonse)
        return repsonse
    
    def confidence_graph_prompts(self):
        ### generate graph layout ###
        name_map = {'certain_edges': 'Directed Edge', #(->)
                    'uncertain_edges': 'Undirected Edge', #(-)
                    'bi_edges': 'Bi-Directed Edge', #(<->)
                    'half_edges': 'Non-Ancestor Edge', #(o->)
                    'non_edges': 'No D-Seperation Edge', #(o-o)
                    'non_existence':'No Edge'}
        graph_text = """
        \\begin{figure}[H]
            \centering

        """
        bootstrap_dict = {k: v for k, v in self.bootstrap_probability.items() if v is not None and sum(v.flatten())>0}
        zero_graphs = [k for k, v in self.bootstrap_probability.items() if  v is not None and sum(v.flatten())==0]
        length = round(1/len(bootstrap_dict), 2)-0.01
        for key in bootstrap_dict.keys():
            graph_path = f'{self.visual_dir}/{key}_confidence_heatmap.jpg'
            caption = f'{name_map[key]}'
            graph_text += f"""
            \\begin{{subfigure}}{{{length}\\textwidth}}
                    \centering
                    \includegraphics[width=\linewidth]{graph_path}
                    \\vfill
                    \caption{caption}
                \end{{subfigure}}"""
        
        graph_text += """
        \caption{Confidence Heatmap of Different Edges}
        \end{figure}    

        """
        ### Generate text illustration
        text_map = {'certain_edges': 'directed edge ($->$)', #(->)
                    'uncertain_edges': 'undirected edge ($-$)', #(-)
                    'bi_edges': 'edge with hidden confounders ($<->$)', #(<->)
                    'half_edges': 'edge of non-ancestor ($o->$)', #(o->)
                    'non_edges': 'egde of no D-Seperation set', #(o-o)
                    'non_existence':'No Edge'}
        graph_text = "The above heatmaps show the confidence probability we have on different kinds of edges, including "
        for k in bootstrap_dict.keys():
            graph_text += f"{text_map[k]}, "
        graph_text += 'and probability of no edge.'
        for k_zero in zero_graphs:
            k_zero= k_zero.replace("_", "-")
            graph_text += f"The heatmap of {k_zero} is not shown because probabilities of all edges are 0. "
    
        graph_text += """Based on the confidence probability heatmap, we have edges with high, moderate, and low edges.
        \\begin{{itemize}}"""
        high_prob_pairs = self.global_state.results.bootstrap_check_dict['high_prob_edges']['exist']+self.global_state.results.bootstrap_check_dict['high_prob_edges']['non-exist']
        middle_prob_pairs = self.global_state.results.bootstrap_check_dict['middle_prob_edges']['exist']+self.global_state.results.bootstrap_check_dict['middle_prob_edges']['non-exist']
        middle_prob_pairs = list(set(tuple(sorted((i, j))) for (i, j) in middle_prob_pairs))
        low_prob_pairs = self.global_state.results.bootstrap_check_dict['low_prob_edges']['exist']
        if high_prob_pairs != []:
            graph_text += "\n \item \\textbf{{High Confidence Edges}}: "
            graph_text += ', '.join(f'{self.data.columns[idx_j]} $\\rightarrow$ {self.data.columns[idx_i]}' for idx_i, idx_j in high_prob_pairs)
        if middle_prob_pairs != []:
            graph_text += "\n \item \\textbf{{Middle Confidence Edges}}: "
            graph_text += ', '.join(f'{self.data.columns[idx_j]} - {self.data.columns[idx_i]}' for idx_i, idx_j in middle_prob_pairs)
        if low_prob_pairs != []:
            graph_text += "\n \item \\textbf{{Low Confidence Edges}}: "
            graph_text += ', '.join(f'{self.data.columns[idx_j]} $\\rightarrow$ {self.data.columns[idx_i]}' for idx_i, idx_j in low_prob_pairs)
        graph_text += "\n \end{{itemize}}"
        return graph_text
        

    def confidence_analysis_prompts(self):
        edges_dict = self.global_state.results.revised_edges
        relation_text_dict, relation_text = edges_to_relationship(self.data, edges_dict, self.bootstrap_probability)
        #relation_prob = self.graph_effect_prompts()

        variables = '\t'.join(self.data.columns)
        prompt = f"""
        The following text describes the causal relationship among variables from a statisical perspective:
        {relation_text}
        We use traditional causal discovery algorithm to find this relationship, and the probability is calculated with bootstrapping.
        This result is solely from statistical perspective, so it is not reliable enough.
        You are an expert in the causal discovery field and are familiar with background knowledge of these variables: {variables}

        Based on this statistical confidence result, and background knowledge about these variables,
        Please write a paragraph to analyze the reliability of this causal relationship graph. 
        
        For example, you can write in the following way, and please analyze 1. the reliability and 2. give conclusion 
        base on both bootstrap probability and expert knowledge background.
        Template:
        From the Statistics perspective, we have high confidence to believe that these edges exist:..., and these edges don't exist:...
        
        However, based on the expert knowledge, we know that these edges exist:...., and these edges don't exist:... 
        
        Therefore, the result of this causal graph is reliable/not reliable.
        """
        #print('confidence_analysis_prompts:',prompt)
        print("Start to analyze graph reliability")
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in the causal discovery field and helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response_doc = response.choices[0].message.content
        #print('reliability analysis:',response_doc)
        return response_doc

    def refutation_analysis_prompts(self):
        text = f"""
                \\begin{{figure}}[H]
                    \centering
                    \includegraphics[height=0.3\\textheight]{{{self.global_state.user_data.output_graph_dir}/refutation_graph.jpg}}
                    \caption{{Refutation Graph}}
                \end{{figure}} \n
                """
        text += self.global_state.results.refutation_analysis
        return text 
    
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
                     2. Discovery Procedure: {self.discover_process},
                     3. Graph Result Analysis: {self.graph_prompt},
                     4. Reliability Analysis: {self.reliability_prompt}
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
    
    def load_context(self, filepath):
        with open(filepath, "r") as f:
            return f.read()

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


    # Function to replace Greek letters in text
    def replace_greek_with_latex(self, text):
        greek_to_latex = {
            "α": r"$\alpha$",
            "β": r"$\beta$",
            "γ": r"$\gamma$",
            "δ": r"$\delta$",
            "ε": r"$\epsilon$",
            "ζ": r"$\zeta$",
            "η": r"$\eta$",
            "θ": r"$\theta$",
            "ι": r"$\iota$",
            "κ": r"$\kappa$",
            "λ": r"$\lambda$",
            "μ": r"$\mu$",
            "ν": r"$\nu$",
            "ξ": r"$\xi$",
            "ο": r"$o$",
            "π": r"$\pi$",
            "ρ": r"$\rho$",
            "σ": r"$\sigma$",
            "τ": r"$\tau$",
            "υ": r"$\upsilon$",
            "φ": r"$\phi$",
            "χ": r"$\chi$",
            "ψ": r"$\psi$",
            "ω": r"$\omega$",
        }
        # Use regular expressions to find and replace Greek letters
        pattern = "|".join(map(re.escape, greek_to_latex.keys()))
        return re.sub(pattern, lambda match: greek_to_latex[match.group()], text)

    def generation(self, debug=False):
            '''
            generate and save the report
            :param debug: bool, if True, use template directly without LLM
            :return: Str: A technique report explaining all the results for readers
            '''
            if debug:
                # Load appropriate template based on data mode and ground truth
                prompt_template = self.load_context("postprocess/context/template_debug.tex")
                return prompt_template

            # Non-debug path continues with full report generation
            print("Start to generate the report")
            # Data info
            df = self.data.copy()
            if self.data.shape[1] > 20:  
                # choose 10 columns randomly
                random_columns = np.random.choice(df.columns, size=10, replace=False)
                df = df[random_columns]
            data_preview = df.head().to_latex(index=False)
            if len(self.data.columns) >= 9:
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
            dist_info, corr_info = self.eda_prompt()
            #dist_info = self.latex_convert(dist_info)
            #corr_info = self.latex_convert(corr_info)
            # Procedure info
            self.discover_process = self.procedure_prompt()
            self.preprocess_plot = self.preprocess_plot_prompt()
            # Graph effect info
            self.graph_prompt = self.list_conversion(self.global_state.logging.graph_conversion['initial_graph_analysis'])
            # Graph Revise info
            if self.data_mode == 'real':
                self.revise_process = self.graph_revise_prompts()
            else:
                self.revise_process = '' 
            # Graph Reliability info
            self.reliability_prompt = self.confidence_analysis_prompts()
            self.confidence_graph_prompt = self.confidence_graph_prompts()
            self.refutation_analysis = self.refutation_analysis_prompts()
            self.abstract = self.abstract_prompt()
            

            if self.data_mode == 'simulated':
                if self.global_state.user_data.ground_truth is not None:
                    prompt_template = self.load_context("postprocess/context/template_simulated.tex")
                else:
                    prompt_template = self.load_context("postprocess/context/template_simulated_notruth.tex")
            else:
                if self.global_state.user_data.ground_truth is not None:
                    prompt_template = self.load_context("postprocess/context/template_real.tex")
                else:
                    prompt_template = self.load_context("postprocess/context/template_real_notruth.tex")

            replacement1 = {
                "[ABSTRACT]": self.abstract.replace("&", r"\&") or "",
                "[INTRO_INFO]": self.intro_info.replace("&", r"\&") or "",
                "[BACKGROUND_INFO1]": self.background_info1.replace("&", r"\&") or "",
                "[BACKGROUND_INFO2]": self.background_info2.replace("&", r"\&") or "",
                "[DATA_PREVIEW]": data_preview or "",
                "[DATA_PROP_TABLE]": data_prop_table or "",
                "[DIST_INFO]": dist_info or "",
                "[CORR_INFO]": corr_info or "",
                "[RESULT_ANALYSIS]": self.graph_prompt.replace("&", r"\&") or "",
                "[DISCOVER_PROCESS]": self.discover_process.replace("&", r"\&") or "",
                "[PREPROCESS_GRAPH]": self.preprocess_plot or "",
                "[REVISE_PROCESS]": self.revise_process.replace("&", r"\&") or "",
                "[RELIABILITY_ANALYSIS]": self.reliability_prompt.replace("&", r"\&") or "",
                "[CONFIDENCE_GRAPH]": self.confidence_graph_prompt or "",
                "[REFUTATION_GRAPH]": self.resutation_analysis or ""
            }
            replacement2 = {
                "[TITLE]": self.title or "",
                "[DATASET]": dataset or "",
                "[POTENTIAL_GRAPH]": f'{self.visual_dir}/potential_relation.pdf',
                "[DIST_GRAPH]": self.eda_result['plot_path_dist'] or "",
                "[CORR_GRAPH]": self.eda_result['plot_path_corr'] or "", 
                "[ALGO]": self.algo or "",
                "[RESULT_GRAPH0]": f'{self.visual_dir}/true_graph.pdf',
                "[RESULT_GRAPH1]": f'{self.visual_dir}/initial_graph.pdf',
                "[RESULT_GRAPH2]": f'{self.visual_dir}/revised_graph.pdf',
                "[RESULT_GRAPH3]": f'{self.visual_dir}/metrics.jpg',
            }

            for placeholder, value in replacement1.items():
                prompt_template = prompt_template.replace(placeholder, value)
            for placeholder, value in replacement2.items():
                prompt_template = prompt_template.replace(placeholder, value)
            prompt_template = self.replace_greek_with_latex(prompt_template)
            #print(prompt_template)
            return prompt_template
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": """
                     You are a causal discovery expert, help me to revise this latex report templete. 
                     1. Provide your response in Latex format, and only include Latex content which can be rendered to pdf directly. Don't include other things
                     2. Pay attention to orders of section, subsection, subsubsection... based on the meaning of headings
                     3. If there is number in \subsection{{1. headings}}, please remove the number and only reserve the heading text \subsection{heading}
                     4. If there is step number in \subsection{{Step 1 headings}}, please remove the step number and only reserve the heading text \subsection{heading}
                     5. Replace ↔ with $\leftrightarrow$, -> with $\\rightarrow$
                     6. All **text** should be replaced with \\textbf{text}
                     7. Do not include any Greek Letters, Please change any Greek Letter into Math Mode, for example, you should change γ into $\gamma$
                     8. Do not change any parameters in figure settings
                     9. Only include your latex content in the response which can be rendered to pdf directly. Don't include other things like '''latex '''
                     10. ALL _ should be replaced by \_
                     """},
                    {"role": "user", "content": prompt_template}
                ]
            )

            output = response.choices[0].message.content
            return output
    
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
                with open(f'{save_path}/report.tex', 'w', encoding='utf-8') as file:
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
        # fix latex bugs before rendering
        print('check latex bug')
        self.latex_bug_checking(f'{save_path}/report.tex')
        # Compile the .tex file to PDF using pdflatex
        print('start compilation')
        compile_tex_to_pdf_with_refs(f'{save_path}/report.tex', save_path)
