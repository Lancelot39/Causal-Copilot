# Kun Zhou Implemented
from data.simulation.simulation import SimulationManager
from preprocess.dataset import knowledge_info
from preprocess.stat_info_functions import stat_info_collection, convert_stat_info_to_text
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge
from postprocess.visualization import Visualization, convert_to_edges
from preprocess.eda_generation import EDA
from postprocess.report_generation import Report_generation
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
        default="dataset/Auto_mpg/Auto_mpg.csv",
        help='Path to the input dataset file (e.g., CSV format or directory location)'
    )

    # Output file for results
    parser.add_argument(
        '--output-report-dir',
        type=str,
        default='dataset/Auto_mpg/output_report',
        help='Directory to save the output report'
    )

    # Output directory for graphs
    parser.add_argument(
        '--output-graph-dir',
        type=str,
        default='dataset/Auto_mpg/output_graph',
        help='Directory to save the output graph'
    )

    # OpenAI Settings
    parser.add_argument(
        '--organization',
        type=str,
        default="org-5NION61XDUXh0ib0JZpcppqS",
        help='Organization ID'
    )

    parser.add_argument(
        '--project',
        type=str,
        default="proj_Ry1rvoznXAMj8R2bujIIkhQN",
        help='Project ID'
    )

    parser.add_argument(
        '--apikey',
        type=str,
        default="",
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

def main(args, prompt_type, voting_num):
    global_state = global_state_initialization(args)
    global_state = load_data(global_state, args)

    if args.data_mode == 'real':
        global_state.user_data.raw_data = load_real_world_data(args.data_file)
    
    global_state.user_data.processed_data = process_user_query(args.initial_query, global_state.user_data.raw_data)

    # Show the exacted global state
    print(global_state)

    # background info collection
    #print("Original Data: ", global_state.user_data.raw_data)

    if args.debug:
        # Fake statistics for debugging
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

    programmer = Programming(args)
    global_state = programmer.forward(global_state)

    #############Visualization for Initial Graph###################
    my_visual_initial = Visualization(global_state)
    # Get the position of the nodes
    pos_est = my_visual_initial.get_pos(global_state.results.raw_result)
    # Plot True Graph
    if global_state.user_data.ground_truth is not None:
        _ = my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', pos=pos_est)
    # Plot Initial Graph
    _ = my_visual_initial.plot_pdag(global_state.results.raw_result, 'initial_graph.pdf', pos=pos_est)
    my_report = Report_generation(global_state, args)
    global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.raw_result)
    global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()

    judge = Judge(global_state, args)
    if global_state.user_data.ground_truth is not None:
        print("Original Graph: ", global_state.results.converted_graph)
        print("Mat Ground Truth: ", global_state.user_data.ground_truth)
        global_state.results.metrics = judge.evaluation(global_state)
        print(global_state.results.metrics)
    import time 
    start_time = time.time()
    global_state = judge.forward(global_state, prompt_type, voting_num)
    end_time = time.time()
    duration = end_time-start_time
    with open('postprocess/test_result/sachs_pairwise/duration.txt', 'a') as file:
        # Write the text to the file
        file.write(f'prompt: {prompt}, voting_num: {voting_num}, duration: {duration} \n')

    # ##############################
    # if global_state.user_data.ground_truth is not None:
    #     print("Revised Graph: ", global_state.results.revised_graph)
    #     print("Mat Ground Truth: ", global_state.user_data.ground_truth)
    #     global_state.results.revised_metrics = judge.evaluation(global_state)
    #     print(global_state.results.revised_metrics)
    # ################################
    #############Visualization for Revised Graph###################
    # Plot Revised Graph
    my_visual_revise = Visualization(global_state)
    pos_new = my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.pdf', pos=pos_est)
    global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.raw_data.columns, global_state.results.revised_graph)
    # Plot Bootstrap Heatmap
    boot_heatmap_path = my_visual_revise.boot_heatmap_plot()

    ##### Save infos of Post-Processing ######
    import numpy as np
    import shutil
    import os 
    save_path = f'postprocess/test_result/Auto_mpg/{prompt_type}/{voting_num}_voting/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("Original Graph: ", global_state.results.raw_result.G.graph)
    np.save(save_path+'origin_graph', global_state.results.raw_result.G.graph)
    print("Revised Graph: ", global_state.results.revised_graph)
    np.save(save_path+'revised_graph', global_state.results.revised_graph)
    
    # from postprocess.test import manual_metrics, causallearn_metrics
    # true_mat = np.load('postprocess/test_result/ground_truth_sachs.npy')
    # initial_manual_result = manual_metrics(true_mat, global_state.results.raw_result.G.graph)
    # print('initial_manual_result',initial_manual_result)
    # revised_manual_result = causallearn_metrics(true_mat,  global_state.results.revised_graph)
    # print('revised_manual_result',revised_manual_result)
    # calculate time

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
    while not os.path.isfile(report_path) and try_num<=3:
        try_num = +1
        print('Error occur during the Report Generation, try again')
        report_gen = Report_generation(global_state, args)
        report = report_gen.generation(debug=False)
        report_gen.save_report(report)
        if not os.path.isfile(report_path) and try_num==3:
            print('Error occur during the Report Generation three times, we stop.')
    ################################
    destination_path = os.path.join(save_path, os.path.basename(report_path))
    shutil.copy(report_path, destination_path)
    
    return report, global_state


if __name__ == '__main__':
    args = parse_args()
    prompt_folders = ['base', 'markov_blanket', 'all_relation', 'cot_base', 'cot_markov_blanket', 'cot_all_relation']
    voting_folders = [3, 10, 20]
    for prompt in prompt_folders:
        for voting_num in voting_folders:
            main(args, prompt, voting_num)
            
