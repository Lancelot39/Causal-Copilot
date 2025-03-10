# Kun Zhou Implemented
from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from algorithm.hyperparameter_selector import HyperparameterSelector
from postprocess.judge import Judge
from postprocess.visualization import Visualization, convert_to_edges
from preprocess.eda_generation import EDA
from report.report_generation import Report_generation
from global_setting.Initialize_state import global_state_initialization, load_data

import json
import argparse
import pandas as pd

import os

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
        default='dataset/sachs/output_report',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        default='dataset/sachs/output_graph',
        help='Directory to save the output graph'
    )

    # OpenAI Settings
    parser.add_argument(
        '--organization',
        type=str,
        default="org-Xa9VGT8plP28JBRKtqBRjG5l",
        help='Organization ID'
    )

    parser.add_argument(
        '--project',
        type=str,
        default="proj_t78xpJomilJJu5qtuOK4vWfR",
        help='Project ID'
    )

    parser.add_argument(
        '--apikey',
        type=str,
        default= None,
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
        default= "do causal discovery on the data", # "selected algorithm: PCMCI",
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

def load_real_world_data(file_path):
    #Baseline code
    # Checking file format and loading accordingly, right now it's for CSV only
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError(f"Unsupported file format for {file_path}")
    
    print("Real-world data loaded successfully.")
    return data

def process_user_query(query, data):
    #Baseline code
    query_dict = {}
    if ';' in query or ':' in query:
        for part in query.split(';'):
            key, value = part.strip().split(':')
            query_dict[key.strip()] = value.strip()

    if 'filter' in query_dict and query_dict['filter'] == 'continuous':
        # Filtering continuous columns, just for target practice right now
        data = data.select_dtypes(include=['float64', 'int64'])
    
    if 'selected_algorithm' in query_dict:
        selected_algorithm = query_dict['selected_algorithm']
        print(f"Algorithm selected: {selected_algorithm}")

    print("User query processed.")
    return data

def main(args):
    global_state = global_state_initialization(args)
    global_state = load_data(global_state, args)

    if args.data_mode == 'real':
        global_state.user_data.raw_data = load_real_world_data(args.data_file)
    
    global_state.user_data.processed_data = process_user_query(args.initial_query, global_state.user_data.raw_data)
    global_state.user_data.visual_selected_features = global_state.user_data.processed_data.columns.tolist()
    global_state.user_data.selected_features = global_state.user_data.processed_data.columns.tolist()

    print("-"*50, "Global State", "-"*50)
    print(global_state)
    print("-"*100)

    if args.debug:
        # Fake statistics for debugging
        global_state.statistics.sample_size = 853
        global_state.statistics.feature_number = 11
        global_state.statistics.missingness = False
        global_state.statistics.data_type = "Continuous"
        global_state.statistics.linearity = True
        global_state.statistics.gaussian_error = True
        global_state.statistics.stationary = "non time-series"
        global_state.user_data.processed_data = global_state.user_data.raw_data
        global_state.user_data.knowledge_docs = "This is fake domain knowledge for debugging purposes."
    else:
        global_state = stat_info_collection(global_state)
        global_state = knowledge_info(args, global_state)

    # Convert statistics to text
    global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

    print("Preprocessed Data: ", global_state.user_data.processed_data)
    print("Statistics Info: ", global_state.statistics.description)
    print("Knowledge Info: ", global_state.user_data.knowledge_docs)

    #############EDA###################
    my_eda = EDA(global_state)
    my_eda.generate_eda()
    
    # Algorithm selection and deliberation
    filter = Filter(args)
    global_state = filter.forward(global_state)

    reranker = Reranker(args)
    global_state = reranker.forward(global_state)

    hp_selector = HyperparameterSelector(args)
    global_state = hp_selector.forward(global_state)

    programmer = Programming(args)
    global_state = programmer.forward(global_state)

    #############Visualization for Initial Graph###################
    my_visual_initial = Visualization(global_state)
    if global_state.statistics.time_series:
        # print("Converted Graph: ", global_state.results.converted_graph.shape)
        converted_graph = global_state.results.converted_graph
        pos_est = my_visual_initial.get_pos(converted_graph[0])
        for i in range(converted_graph.shape[0]):
            _ = my_visual_initial.plot_pdag(converted_graph[i], f'{global_state.algorithm.selected_algorithm}_initial_graph_{i}.pdf', pos=pos_est)
        summary_graph = np.any(converted_graph, axis=0).astype(int)
        # pos_est = my_visual_initial.get_pos(summary_graph)
        _ = my_visual_initial.plot_pdag(summary_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph_summary.pdf', pos=pos_est)
        my_report = Report_generation(global_state, args)
    else:
        # Get the position of the nodes
        pos_est = my_visual_initial.get_pos(global_state.results.converted_graph)
        # Plot True Graph
        if global_state.user_data.ground_truth is not None:
            _ = my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos=pos_est)
        # Plot Initial Graph
        _ = my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.pdf', pos=pos_est)
        my_report = Report_generation(global_state, args)
        global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
        global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
        judge = Judge(global_state, args)
        if global_state.user_data.ground_truth is not None:
            print("Original Graph: ", global_state.results.converted_graph)
            print("Mat Ground Truth: ", global_state.user_data.ground_truth)
            global_state.results.metrics = judge.evaluation(global_state)
            print(global_state.results.metrics)
        
        global_state = judge.forward(global_state, 'cot_all_relation', 1)
    
    #############Visualization for Revised Graph###################
    # Plot Revised Graph
    my_visual_revise = Visualization(global_state)
    pos_new = my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.pdf', pos=pos_est)
    global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.raw_data.columns, global_state.results.revised_graph)
    # Plot Bootstrap Heatmap
    boot_heatmap_path = my_visual_revise.boot_heatmap_plot()
    global_state.results.refutation_analysis = judge.graph_refutation(global_state)

    # algorithm selection process
    '''
    round = 0
    flag = False

    while round < args.max_iterations and flag == False:
        code, results = programmer.forward(preprocessed_data, algorithm, hyper_suggest)
        flag, algorithm_setup = judge(preprocessed_data, code, results, statistics_dict, algorithm_setup, knowledge_docs)
    '''
    #############Report Generation###################
    import os 
    try_num = 1
    my_report = Report_generation(global_state, args)
    report = my_report.generation()
    my_report.save_report(report)
    report_path = os.path.join(global_state.user_data.output_report_dir, 'report.pdf')  
    while (not os.path.isfile(report_path)) and try_num<3:
        try_num += 1
        print('Error occur during the Report Generation, try again')
        report_gen = Report_generation(global_state, args)
        report = report_gen.generation(debug=False)
        report_gen.save_report(report)
    if not os.path.isfile(report_path) and try_num == 3:
        print('Error occur during the Report Generation three times, we stop.')
    ################################

    # User discussion part
    from user.discuss import Discussion
    discussion = Discussion(args, report)
    discussion.forward(global_state)

    return report, global_state

if __name__ == '__main__':
    args = parse_args()
    main(args)
            
