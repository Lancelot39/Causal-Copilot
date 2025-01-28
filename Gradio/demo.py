import os
import subprocess
from pathlib import Path

def init_graphviz():
    # Try apt-get (Debian/Ubuntu) first
    print("Attempting to install Graphviz using apt-get...")
    subprocess.run("apt-get update", shell=True, check=True)
    subprocess.run("apt-get install -y graphviz", shell=True, check=True)

    # Verify installation
    version = subprocess.run("dot -V", shell=True, check=True, 
                           capture_output=True, text=True)
    print(f"Graphviz installed successfully: {version.stderr}")

    # Add to PATH if needed
    graphviz_paths = ["/usr/local/bin", "/usr/bin"]
    for path in graphviz_paths:
        if os.path.exists(path) and path not in os.environ['PATH']:
            os.environ['PATH'] = f"{os.environ['PATH']}:{path}"

        
def init_latex():
    try:
        # Install TinyTeX
        subprocess.run("wget -qO- 'https://yihui.org/tinytex/install-bin-unix.sh' | sh", shell=True, check=True)
        
        # Add to PATH
        home = os.path.expanduser("~")
        os.environ['PATH'] = f"{os.environ['PATH']}:{home}/.TinyTeX/bin/x86_64-linux"
        
        # Install packages
        subprocess.run(f"{home}/.TinyTeX/bin/x86_64-linux/tlmgr update --self", shell=True, check=True)
        subprocess.run(f"{home}/.TinyTeX/bin/x86_64-linux/tlmgr install latexmk fancyhdr caption booktabs", shell=True, check=True)
        
        print("LaTeX setup completed successfully")
    except Exception as e:
        print(f"LaTeX setup failed: {e}")

def init_causallearn():
    subprocess.run("git submodule update --recursive", shell=True, check=True)
# Run initialization before importing plumbum
init_latex()
init_graphviz()
init_causallearn()

import gradio as gr
import pandas as pd
import io
import os
import shutil
from datetime import datetime
import sys
from queue import Queue
import json 
import time 
import traceback
import pickle
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Gradio.demo_config import get_demo_config
from global_setting.Initialize_state import global_state_initialization
from preprocess.stat_info_functions import *
from preprocess.dataset import knowledge_info
from preprocess.eda_generation import EDA
from algorithm.filter import Filter
from algorithm.program import Programming
from algorithm.rerank import Reranker
from postprocess.judge import Judge
from postprocess.visualization import Visualization, convert_to_edges
from causal_analysis.inference import Analysis
from report.report_generation import Report_generation
from user.discuss import Discussion
from openai import OpenAI
from pydantic import BaseModel
from help_functions import *

print('##########Initialize Global Variables##########')
# Global variables
UPLOAD_FOLDER = "./demo_data"

def update(info, key, value):
    new_info = info.copy()
    new_info[key] = value
    return new_info

#CURRENT_STAGE = 'initial_process'
MAX_CONCURRENT_REQUESTS = 5
MAX_QUEUE_SIZE = 10

# Demo dataset configs
DEMO_DATASETS = {
    "Abalone": {
        "name": "🐚 Real Dataset:Abalone",
        "path": "dataset/Abalone/Abalone.csv",
        "query": "YES. Find causal relationships between physical measurements and age of abalone. The dataset contains numerical measurements of physical characteristics.",
    },
    "Sachs": {
        "name": "🧬 Real Dataset: Sachs",
        "path": "dataset/sachs/sachs.csv", 
        "query": "YES. Discover causal relationships between protein signaling molecules. The data contains flow cytometry measurements of proteins and phospholipids."
    },
    "CCS Data": {
        "name": "📊 Real Dataset: CCS Data",
        "path": "dataset/CCS_Data/CCS_Data.csv",
        "query": "YES. Analyze causal relationships between variables in the CCS dataset. The data contains multiple continuous variables."
    },
    "Ozone": {
        "name": "🌫️ Real Dataset: Ozone", 
        "path": "dataset/Ozone/Ozone.csv",
        "query": "YES. This is a Time-Series dataset, investigate causal factors affecting ozone levels. The data contains atmospheric and weather measurements over time."
    },
    "Linear_Gaussian": {
        "name": "🟦 Simulated Data: Linear Gaussian",
        "path": "dataset/Linear_Gaussian/Linear_Gaussian_data.csv",
        "query": "NO. The data follows linear relationships with Gaussian noise. Please discover the causal structure."
    },
    "Linear_Nongaussian": {
        "name": "🟩 Simulated Data: Linear Non-Gaussian",
        "path": "dataset/Linear_Nongaussian/Linear_Nongaussian_data.csv", 
        "query": "NO. The data follows linear relationships with non-Gaussian noise. Please discover the causal structure."
    }
}


def upload_file(file, REQUIRED_INFO):
    # TODO: add more complicated file unique ID handling
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', '')), exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''),
                               os.path.basename(file.name))
    REQUIRED_INFO = update(REQUIRED_INFO, 'target_path', target_path)
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''))
    REQUIRED_INFO = update(REQUIRED_INFO, 'output_dir', output_dir)
    shutil.copy(file.name, target_path)
    return REQUIRED_INFO

def handle_file_upload(file, REQUIRED_INFO, CURRENT_STAGE, chatbot, file_upload_btn, download_btn):
    chatbot = chatbot.copy()
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            REQUIRED_INFO = upload_file(file, REQUIRED_INFO)
            REQUIRED_INFO = update(REQUIRED_INFO, 'data_uploaded', True)
            CURRENT_STAGE = 'initial_process'
            bot_message = (f"✅ Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns! \n"
                           "🤔 Please follow the guidances above for your initial query. \n"
                           "✨ It would be super helpful if you can include more relevant information, e.g., background/context/prior/statistical information!")
        else:
            bot_message = "❌ Please upload a CSV file."
        chatbot.append((None, bot_message))
        return REQUIRED_INFO, CURRENT_STAGE, chatbot, file_upload_btn, download_btn

    except Exception as e:
        error_message = f"❌ Error loading file: {str(e)}"
        chatbot.append((None, error_message))
        return chatbot, file_upload_btn, download_btn

def process_message(message, args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn):
    REQUIRED_INFO = update(REQUIRED_INFO, 'processing', True)
    # initial_process -> check sample size -> check missingness ratio and drop -> check correlation and drop -> check dimension and drop ->
    # stat analysis and algorithm -> user edit edges -> report generation
    try:
        if CURRENT_STAGE == 'initial_process':    
            print('check data upload')
            if not REQUIRED_INFO['data_uploaded']:
                chat_history.append((message, "Please upload your dataset first before proceeding."))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                # Initialize config
                config = get_demo_config()
                config.data_file = REQUIRED_INFO['target_path']
                for key, value in config.__dict__.items():
                    setattr(args, key, value)
                print('check initial query')

                config.initial_query = message
                chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, args = process_initial_query(message, chat_history, download_btn, args, REQUIRED_INFO, CURRENT_STAGE)
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
    
            # Initialize global state
            if REQUIRED_INFO['data_uploaded'] and REQUIRED_INFO['initial_query']:
                print('strart analysis')
                global_state = global_state_initialization(args)
                # Load data
                global_state.user_data.raw_data = pd.read_csv(REQUIRED_INFO['target_path'])
                global_state.user_data.processed_data = global_state.user_data.raw_data
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

                # Preprocessing - Step 1: Sample size checking
                n_row, n_col = global_state.user_data.raw_data.shape
                chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE = sample_size_check(n_row, n_col, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE)
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                print('stage1',CURRENT_STAGE)
                ### important feature selection query#####
                if CURRENT_STAGE == 'important_feature_selection':
                    chat_history.append((None, f"Do you have important features you care about? These are features in your provided dataset:\n"
                                            f"{', '.join(global_state.user_data.raw_data.columns)}"))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn            

        if CURRENT_STAGE == 'reupload_dataset':
            # chat_history, download_btn, REQUIRED_INFO = parse_reupload_query(message, chat_history, download_btn, REQUIRED_INFO)
            # yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            chat_history.append((None, f"🔄 Press Enter to confirm Reuploading the dataset..."))
            CURRENT_STAGE = 'initial_process'
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            return process_message(message, args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn)
        
        if CURRENT_STAGE == 'important_feature_selection':
            if (message == '') or (' no ' in message.lower()) or ('none' in message.lower()):
                chat_history.append((message, None))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn 
                var_list = []
                CURRENT_STAGE = 'mode_check'
            else:
                print('important_feature_selection')
                var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_var_selection_query(message, chat_history, download_btn, 'mode_check', args, global_state, REQUIRED_INFO, CURRENT_STAGE)
            global_state.user_data.important_features = var_list
        
        if CURRENT_STAGE == 'mode_check':
            print(CURRENT_STAGE)
            chat_history.append((None, "Do you want to use the interactive mode which allows the interaction with copilot in each step?\n"
                                    "If not, the whole procedure will be conducted automatically with suggested optimal settings.\n"
                                    "Please answer with 'Yes' or 'NO'."))
            CURRENT_STAGE = 'mode_setting'
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        
        if CURRENT_STAGE == 'mode_setting':
            print(CURRENT_STAGE)
            chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE = parse_mode_query(message, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        # Preprocess Step 2: Sparsity Checking
        if CURRENT_STAGE == 'sparsity_check':
            # missing value detection
            np_nan = np_nan_detect(global_state)
            if not np_nan:
                chat_history.append((None, "We do not detect NA values in your dataset, do you have the specific value that represents NA?\n"
                                            "If so, please provide here. Otherwise please input 'NO'."))
                CURRENT_STAGE = 'sparsity_check_1'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                CURRENT_STAGE = 'sparsity_check_2'
        
        if CURRENT_STAGE == 'sparsity_check_1':
            chat_history.append((message, None)) 
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = first_stage_sparsity_check(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        
        if CURRENT_STAGE == 'sparsity_check_2':
            global_state = missing_ratio_table(global_state) # Update missingness indicator in global state and generate missingness ratio table
            sparsity_dict = sparsity_check(df=global_state.user_data.processed_data)
            chat_history.append((None, "Missing Ratio Summary: \n"\
                                 f"1️⃣ High Missing Ratio Variables (>0.5): {', '.join(sparsity_dict['high']) if sparsity_dict['high']!=[] else 'None'} \n"\
                                 f"2️⃣ Moderate Missing Ratio Variables: {', '.join(sparsity_dict['moderate']) if sparsity_dict['moderate']!=[] else 'None'} \n"\
                                 f"3️⃣ Low Missing Ratio Variables (<0.3): {', '.join(sparsity_dict['low']) if sparsity_dict['low']!=[] else 'None'}"))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if sparsity_dict['moderate'] != []:
                CURRENT_STAGE = "sparsity_drop"
                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, "📍 The missing ratios of the following variables are greater than 0.3 and smaller than 0.5, please decide which variables you want to drop. \n"
                                                f"{', '.join(sparsity_dict['moderate'])}\n"
                                                "⚠️ Please note that variables you want to drop may be confounders, please be cautious in selection.\n"
                                                "Please seperate all variables with a semicolon ; and provide your answer following the template below: \n"
                                                "Templete: PKA; Jnk; PIP2; PIP3; Mek\n"
                                                "If you want LLM help you to decide, please enter 'LLM'."
                                                ))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else:
                    chat_history.append((None, f"📍 The missing ratios of the following variables are greater than 0.3 and smaller than 0.5, we will use LLM to decide which variables to drop. \n"
                                               f"{', '.join(sparsity_dict['moderate'])}"))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    message = 'LLM'
            else:
                CURRENT_STAGE = "sparsity_drop_done"
                print("sparsity_drop_done")
        if CURRENT_STAGE == 'sparsity_drop':
            chat_history, download_btn, global_state, REQUIRED_INFO = parse_sparsity_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if CURRENT_STAGE != "sparsity_drop_done":
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        
        if CURRENT_STAGE == "sparsity_drop_done":
            if sparsity_dict['high'] != []:
                chat_history.append((None, f"📍 The missing ratios of the following variables are greater than 0.5, we will drop them: \n"
                                        f"{', '.join(sparsity_dict['high'])}"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                ####### update variable list
                global_state.user_data.system_drop_features = [var for var in global_state.user_data.system_drop_features if var in global_state.user_data.raw_data.columns]
            if sparsity_dict['low'] != []:
                # impute variables with sparsity<0.3 in the following
                chat_history.append((None, f"📍 The missing ratios of the following variables are smaller than 0.3, we will impute them: \n" \
                                    f"{', '.join(sparsity_dict['low'])}"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            CURRENT_STAGE = 'reupload_dataset_done'
                       
        if CURRENT_STAGE == 'reupload_dataset_done':
            # Preprocess Step 3: correlation checking
            print('correlation_check')
            global_state = correlation_check(global_state)
            if global_state.user_data.high_corr_drop_features:
                chat_history.append((None, "Correlation Check Summary: \n"\
                                     f"We will drop {', '.join(global_state.user_data.high_corr_drop_features)} due to the fact that they are highly correlated with other features."))
            else:
                chat_history.append((None, "Correlation Check Summary: \n"\
                                        "No variables are highly correlated with each other."))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            CURRENT_STAGE = 'knowledge_generation'                

        if CURRENT_STAGE == 'knowledge_generation':
            CURRENT_STAGE = ''
            # Knowledge generation
            if args.data_mode == 'real':
                chat_history.append(("🌍 Generate background knowledge based on the dataset you provided...", None))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                #global_state = knowledge_info(args, global_state)
                global_state.user_data.knowledge_docs = "This is fake domain knowledge for debugging purposes."
                knowledge_clean = str(global_state.user_data.knowledge_docs).replace("[", "").replace("]", "").replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
                chat_history.append((None, knowledge_clean))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, 'If you have some more background information you want to add, please enter it here! Type No to skip this step.'))
                    CURRENT_STAGE = 'check_user_background'
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else:
                    CURRENT_STAGE = 'visual_dimension_check'
            else:
                global_state = knowledge_info(args, global_state)
                CURRENT_STAGE = 'visual_dimension_check'

        #checks the validity of the user's information
        if CURRENT_STAGE == 'check_user_background':
            message = message.strip()
            if message.lower() == 'no' or message == '':
                CURRENT_STAGE = 'visual_dimension_check'
            else:
                global_state.user_data.knowledge_docs += message
                chat_history.append((message, "✅ Successfully added your provided information!"))
                time.sleep(0.5)
                CURRENT_STAGE = 'visual_dimension_check'               

        if CURRENT_STAGE == 'visual_dimension_check':
            ## Preprocess Step 4: Choose Visualization Variables
            # High Dimensional Case: let user choose variables   highlight chosen variables
            if len(global_state.user_data.processed_data.columns) > 10:
                if len(global_state.user_data.important_features) > 10 or len(global_state.user_data.important_features) == 0:
                    CURRENT_STAGE = 'variable_selection'
                    if REQUIRED_INFO["interactive_mode"]:
                        chat_history.append((None, "Dimension Checking Summary:\n"\
                                            "💡 There are many variables in your dataset, please follow the template below to choose variables you care about for visualization: \n"
                                                "1. Please seperate each variables with a semicolon and restrict the number within 10; \n"
                                                "2. Please choose among the following variables: \n"
                                                f"{';'.join(global_state.user_data.processed_data.columns)} \n"
                                                "3. Templete: PKA; Jnk; PIP2; PIP3; Mek"))
                        yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                        return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    else: 
                        chat_history.append((None, "Dimension Checking Summary:\n"\
                                            "💡 There are many variables in your dataset, we will randomly choose 10 variables among selected important variables to visualize."))
                        yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else: # Only visualize variables user care about
                    chat_history.append((None, "Dimension Checking Summary:\n"\
                                         "💡 Because of the high dimensionality, We will only visualize 10 variables and include variables you care about."))
                    other_variables = list(set(global_state.user_data.processed_data.columns) - (set(global_state.user_data.processed_data.columns)&set(global_state.user_data.important_features)))
                    remaining_num = 10 - len(global_state.user_data.important_features)
                    global_state.user_data.visual_selected_features = global_state.user_data.important_features+other_variables[:remaining_num+1]
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    if REQUIRED_INFO["interactive_mode"]:
                        CURRENT_STAGE = 'stat_analysis'
                        yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    else:
                        CURRENT_STAGE = 'ts_check_done'
                        yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else: 
                global_state.user_data.visual_selected_features = global_state.user_data.selected_features
                chat_history.append((None, "Dimension Checking Summary:\n"\
                                     "💡 The dimension of your dataset is not too large, We will visualize all variables in the dataset."))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                if REQUIRED_INFO["interactive_mode"]:
                    CURRENT_STAGE = 'stat_analysis'
                    CURRENT_STAGE = 'stat_analysis'
                else:
                    CURRENT_STAGE = 'ts_check_done'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if CURRENT_STAGE == 'variable_selection':
            print('select variable')
            if REQUIRED_INFO["interactive_mode"]:
                var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_var_selection_query(message, chat_history, download_btn, 'stat_analysis', args, global_state, REQUIRED_INFO, CURRENT_STAGE)
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                # Update the selected variables
                global_state.user_data.visual_selected_features = var_list
            else: 
                CURRENT_STAGE = 'ts_check_done'
                try:
                    global_state.user_data.visual_selected_features = global_state.user_data.important_features[:10]
                except:
                    global_state.user_data.visual_selected_features = global_state.processed_data.columns[:10]             
        
        if CURRENT_STAGE == 'stat_analysis':
            # Statistical Analysis: Time Series
            chat_history.append((None, "Please indicate whether your dataset is Time-Series and set your time lag: \n"\
                                           "1️⃣ Input 'NO' if it is not a Time-Series dataset;\n"\
                                           "2️⃣ Input your time lag if you want to set it by yourself;\n"\
                                           "3️⃣ Input 'continue' if you want the time lag to be set automatically;\n"))
            CURRENT_STAGE = 'ts_check'
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        
        if CURRENT_STAGE == 'ts_check':
            chat_history.append((message, None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_ts_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if CURRENT_STAGE == 'ts_check_done':
            chat_history.append(
                (f"📈 Run statistical analysis on Dataset {REQUIRED_INFO['target_path'].split('/')[-1].replace('.csv', '')}...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

            user_linear = global_state.statistics.linearity
            user_gaussian = global_state.statistics.gaussian_error

            global_state = stat_info_collection(global_state)
            global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

            if global_state.statistics.data_type == "Continuous":
                if user_linear is None:
                    chat_history.append(("✍️ Generate residuals plots ...", None))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/residuals_plot.jpg',)))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                if user_gaussian is None:
                    chat_history.append(("✍️ Generate Q-Q plots ...", None))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/qq_plot.jpg',)))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

            chat_history.append((None, global_state.statistics.description))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if REQUIRED_INFO["interactive_mode"]:
                chat_history.append(('Here we just finished statistical Analysis for the dataset! Please enter anything you want to correct! Or type NO to skip this step.\n'
                                    "Template:\n"
                                    """
                                    linearity: True/False
                                    gaussian_error: True/False
                                    time_series: True/False
                                    data_type: Continuous/Category/Mixture
                                    heterogeneous: True/False
                                    domain_index: variable name of your domain index
                                    """,
                        None))
                CURRENT_STAGE = 'check_user_feedback'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                CURRENT_STAGE = 'eda_generation'
               
        #  process the user feedback
        if CURRENT_STAGE == 'check_user_feedback':
            chat_history.append((message, None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            message = message.strip()
            if message.lower() == 'no' or message == '':
                    CURRENT_STAGE = 'eda_generation'
            else:
                with open('global_setting/state.py', 'r') as file:
                    file_content = file.read()
                prompt = f"""
                I need to update variables in the Statistics class in the provided file according to user's input. 
                Help me to identify which variables need to be updated and save it in a json.
                If no changes are required or the input is not valid, return an empty dictionary.
                The keys in the json should be the variable names, and the values should be the new values. 
                Only return a json that can be parsed directly, do not include ```json
                message: {message}
                file: {file_content}
                """
                parsed_response = LLM_parse_query(None, prompt, message, args)
                try:
                    changes = json.loads(parsed_response)
                    global_state.statistics.update(changes)
                    print(global_state.statistics)
                except RuntimeError as e:
                    print(e)
                    chat_history.append(None, "That information may not be correct, please try again or type Quit to skip.")
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

                CURRENT_STAGE = 'eda_generation'
                chat_history.append((None, "✅ Successfully updated the settings according to your need!"))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if CURRENT_STAGE == 'eda_generation':
            # EDA Generation
            chat_history.append(("🔍 Run exploratory data analysis...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            my_eda = EDA(global_state)
            my_eda.generate_eda()
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_corr.jpg',)))
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_additional.jpg',)))
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_dist.jpg',)))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            CURRENT_STAGE = 'algo_selection'

        if CURRENT_STAGE == 'algo_selection':    
        # Algorithm Selection
            if global_state.algorithm.selected_algorithm is None:
                chat_history.append(("🤖 Select optimal causal discovery algorithm and its hyperparameter...", None))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                filter = Filter(args)
                global_state = filter.forward(global_state)
                reranker = Reranker(args)
                global_state = reranker.forward(global_state)
                chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
                
                alg_reason = global_state.algorithm.algorithm_candidates[global_state.algorithm.selected_algorithm]
                global_state.algorithm.selected_reason = \
                    (
                    f"\n\n{alg_reason['description']}\n\n"
                    f"\n{alg_reason['justification']}"
                )
                chat_history.append((None, f"🤔 Algorithm selection reasoning: {global_state.algorithm.selected_reason}"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

                if REQUIRED_INFO["interactive_mode"]:
                    CURRENT_STAGE = 'user_algo_selection'
                    if torch.cuda.is_available():
                        chat_history.append((None, "Do you want to specify an algorithm instead of the selected one? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES, AcceleratedDirectLiNGAM\n"
                                                "Otherwise please reply NO."))
                    else:
                        chat_history.append((None, "Do you want to specify an algorithm instead of the selected one? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES.\n"
                                                "Otherwise please reply NO."))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else:           
                    CURRENT_STAGE = 'hyperparameter_selection'     
            else:
                chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
                chat_history.append(
                    ("🤖 Select optimal hyperparameter for your selected causal discovery algorithm...", None))
                CURRENT_STAGE = 'hyperparameter_selection'  
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'user_algo_selection':  
            if REQUIRED_INFO["interactive_mode"]:
                chat_history.append((message, None))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if message.lower()=='no' or message=='':
                REQUIRED_INFO["current_stage"] = 'hyperparameter_selection'     
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            elif message in ['PC', 'FCI', 'CDNOD', 'GES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS', 'FGES', 'XGES', 'AcceleratedDirectLiNGAM']:
                global_state.algorithm.selected_algorithm = message
                global_state.algorithm.algorithm_arguments = None 
                CURRENT_STAGE = 'hyperparameter_selection'     
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else: 
                if torch.cuda.is_available():
                    chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                        "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                        "Fast Version: FGES, XGES, AcceleratedDirectLiNGAM\n"))
                else:
                    chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                        "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                        "Fast Version: FGES, XGES.\n"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if CURRENT_STAGE == 'hyperparameter_selection':  
            filter = Filter(args)
            global_state = filter.forward(global_state)
            reranker = Reranker(args)
            global_state = reranker.forward(global_state)
            hyperparameter_text, global_state = generate_hyperparameter_text(global_state)
            chat_history.append(
                (None,
                f"📖 Hyperparameters for the selected algorithm {global_state.algorithm.selected_algorithm}: \n\n {hyperparameter_text}"))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

            if REQUIRED_INFO["interactive_mode"]:
                CURRENT_STAGE = 'user_param_selection'
                with open('Gradio/param_context.json', 'r') as f:
                    param_hint = json.load(f)[global_state.algorithm.selected_algorithm]
                instruction = "Do you want to specify values for parameters instead of the selected one? If so, please specify your parameter following the template below: \n"
                for key in param_hint.keys():
                    instruction += f"{key}: value\n"
                instruction += "Otherwise please reply NO."
                chat_history.append((None, instruction))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                hint = "Here are instructions for hyper-parameter tuning:\n"
                for key, value in param_hint.items():
                    hint += f"- {key}: \n{value};\n "
                chat_history.append((None, hint))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:           
                REQUIRED_INFO["current_stage"] = 'algo_running'  

        if REQUIRED_INFO["current_stage"] == 'user_param_selection':  
            
            if REQUIRED_INFO["interactive_mode"]:
                chat_history.append((message, None))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_hyperparameter_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE)
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                if CURRENT_STAGE != 'algo_running':
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        
        # Causal Discovery
        if REQUIRED_INFO["current_stage"] == 'algo_running':   
            chat_history.append(("🔄 Run causal discovery algorithm...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            programmer = Programming(args)
            global_state = programmer.forward(global_state)
            REQUIRED_INFO["current_stage"] = 'initial_graph'
        # Visualization for Initial Graph
        if REQUIRED_INFO["current_stage"] == 'initial_graph':  
            chat_history.append(("📊 Generate causal graph visualization...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            my_visual_initial = Visualization(global_state)
            if global_state.results.raw_pos is None:
                data_idx = [global_state.user_data.processed_data.columns.get_loc(var) for var in global_state.user_data.visual_selected_features]
                pos = my_visual_initial.get_pos(global_state.results.converted_graph[data_idx, :][:, data_idx])
                global_state.results.raw_pos = pos
            if global_state.user_data.ground_truth is not None:
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, f'{global_state.algorithm.selected_algorithm}_true_graph.jpg', global_state.results.row_pos)
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, f'{global_state.algorithm.selected_algorithm}_true_graph.pdf', global_state.results.row_pos)
                chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_true_graph.jpg',)))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if global_state.results.converted_graph is not None:
                my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.jpg', global_state.results.row_pos)
                my_visual_initial.plot_pdag(global_state.results.converted_graph, f'{global_state.algorithm.selected_algorithm}_initial_graph.pdf', global_state.results.row_pos)
                chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_initial_graph.jpg',)))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                my_report = Report_generation(global_state, args)
                global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
                global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
                analysis_clean = global_state.logging.graph_conversion['initial_graph_analysis'].replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
                print(analysis_clean)
                chat_history.append((None, analysis_clean))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, "Do you want to further prune the initial graph with LLM and analyze the graph reliability?"))
                    CURRENT_STAGE = 'LLM_prune'
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else:
                    REQUIRED_INFO["current_stage"] = 'revise_graph'

        if REQUIRED_INFO["current_stage"] == 'LLM_prune':
            chat_history.append((message, None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            
            class Indicator(BaseModel):
                        indicator: bool
            prompt = """You are a helpful assistant, please identify whether user want to further continue the task and save the boolean result in indicator. """
            parsed_response = LLM_parse_query(Indicator, prompt, message, args)
            indicator = parsed_response.indicator
            if indicator:
                REQUIRED_INFO["current_stage"] = 'revise_graph'
            else: 
                global_state.results.revised_graph = global_state.results.converted_graph
                global_state.results.llm_errors = {'direct_record':None, 'forbid_record': None}
                REQUIRED_INFO["current_stage"] = 'user_prune'

        # Evaluation for Initial Graph
        if REQUIRED_INFO["current_stage"] == 'revise_graph':  
            chat_history.append(("📝 Evaluate and Revise the initial result...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            try:
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state, 'cot_all_relation', 3)
            except Exception as e:
                print('error during judging:', e)
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state, 'cot_all_relation', 3) 
            my_visual_revise = Visualization(global_state)
            global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.revised_graph)
            # Plot Bootstrap Heatmap
            paths = my_visual_revise.boot_heatmap_plot()
            chat_history.append(
                (None, f"The following heatmaps show the confidence probability we have on different kinds of edges in the initial graph"))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            for path in paths:
                chat_history.append((None, (path,)))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if args.data_mode=='real':
                # Plot Revised Graph
                if global_state.results.revised_graph is not None:
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.pdf', global_state.results.row_pos)
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.jpg', global_state.results.row_pos)
                    chat_history.append((None, f"This is the revised graph with Bootstrap and LLM techniques"))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_revised_graph.jpg',)))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    # Refutation Graph
                    chat_history.append(("📝 Evaluate the reliability of the revised result...", None))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    global_state.results.refutation_analysis = judge.graph_refutation(global_state)
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/refutation_graph.jpg',)))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    chat_history.append((None, global_state.results.refutation_analysis))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            
            chat_history.append((None, "✅ Causal discovery analysis completed"))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            CURRENT_STAGE = 'user_prune'
        
        if REQUIRED_INFO["current_stage"] == 'user_prune':
            if REQUIRED_INFO["interactive_mode"]:
                REQUIRED_INFO['current_stage'] = 'user_postprocess'
                chat_history.append((None, "If you are not satisfied with the causal graph, please tell us which edges you want to forbid or add, and we will revise the graph according to your instruction. \n"
                                            "Please follow the templete below, otherwise your input cannot be parsed. \n"
                                            "Add Edges: A1->A2; A3->A4; ... \n"
                                            "Forbid Edges: F1->F2; F3->F4; ... \n"
                                            "Orient Edges: O1->O2; O3->O4; ... \n"
                                            "Or Enter NO to move on to next step\n"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn                
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                with open(f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_global_state.pkl', 'wb') as f:
                    pickle.dump(global_state, f)
                #global_state.logging.global_state_logging.append(global_state.algorithm.selected_algorithm)
                if torch.cuda.is_available():
                    chat_history.append((None, "Do you want to retry other algorithms? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES, AcceleratedDirectLiNGAM\n"
                                                "Otherwise please reply NO."))
                else:
                    chat_history.append((None, "Do you want to retry other algorithms? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES.\n"
                                                "Otherwise please reply NO."))
                CURRENT_STAGE = 'retry_algo'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn                

        if CURRENT_STAGE == 'user_postprocess':
            chat_history.append((message, "📝 Start to process your Graph Revision Query..."))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            user_revise_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_user_postprocess(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE)
            print('user_revise_dict', user_revise_dict)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if CURRENT_STAGE == 'postprocess_parse_done':
                judge = Judge(global_state, args)
                global_state = judge.user_postprocess(user_revise_dict)
                my_visual_revise = Visualization(global_state)
                if global_state.results.revised_graph is not None:
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.pdf', global_state.results.row_pos)
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, f'{global_state.algorithm.selected_algorithm}_revised_graph.jpg', global_state.results.row_pos)
                    chat_history.append((None, f"This is the revised graph according to your instruction."))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_revised_graph.jpg',)))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                
                chat_history.append((None, "Rerun user prune process, press enter to begin."))
                CURRENT_STAGE = 'user_prune'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return process_message(message, args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn)
            elif CURRENT_STAGE == 'retry_algo':
                pass                
            else:
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn 
            
            if REQUIRED_INFO["current_stage"] == 'retry_algo':
                with open(f'{global_state.user_data.output_graph_dir}/{global_state.algorithm.selected_algorithm}_global_state.pkl', 'wb') as f:
                    pickle.dump(global_state, f)
                #global_state.logging.global_state_logging.append(global_state.algorithm.selected_algorithm)
                if torch.cuda.is_available():
                    chat_history.append((None, "Do you want to retry other algorithms? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES, AcceleratedDirectLiNGAM\n"
                                                "Otherwise please reply NO."))
                else:
                    chat_history.append((None, "Do you want to retry other algorithms? If so, please choose one from the following: \n"
                                                "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                                "Fast Version: FGES, XGES.\n"
                                                "Otherwise please reply NO."))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        if CURRENT_STAGE == 'retry_algo': # empty query or postprocess query parsed successfully
            message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_algo_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if CURRENT_STAGE == 'algo_selection':
                print(CURRENT_STAGE)
                print(global_state.algorithm.selected_algorithm)
                return process_message(message, args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn)
        
        if REQUIRED_INFO["current_stage"] == 'inference_analysis_check':
            chat_history.append((None, "Do you want to conduct downstream analysis based on the causal discovery result? You can descripbe your needs.\n"
                                        "Otherwise please input 'NO'.\n"
                                           "We support the following tasks: \n"
                                           "1️⃣ Average Treatment Effect Estimation\n"
                                           "2️⃣ Heterogeneous Treatment Effect Estimation\n"
                                           "3️⃣ Anormaly Attribution\n"
                                           "4️⃣ Feature Importance\n"
                                           "5️⃣ Conterfactual Simulation\n")) 
            CURRENT_STAGE = 'inference_analysis'
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn 
            return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn 
        if CURRENT_STAGE == 'inference_analysis': 
            print('inference_analysis')
            tasks_list, descs_list, key_node_list, reasons, chat_history, download_btn, global_state, REQUIRED_INFO = parse_inference_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            if tasks_list is not None:
                if tasks_list == []:
                    chat_history.append((None, "We cannot identify any supported task in your query, please retry or type 'NO' to skip this step."))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    #return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                else:
                    chat_history.append((None, f"📄 Proposal for your causal task:\n {reasons}"))
                    chat_history.append((None, f"Analyzing for your causal task..."))
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                    analysis = Analysis(global_state, args)
                    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
                        info, figs = analysis.forward(task, desc, key_node)
                        if info is None:
                            chat_history.append((None, 'Your query cannot be parsed, please ask again or reply NO to end this part.'))
                        else:
                            for fig in figs:
                                chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{fig}',)))
                            chat_history.append((None, info))
                            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                            global_state.logging.downstream_discuss.append({"role": "system", "content": info})
                    chat_history.append((None, "Do you have questions about this analysis? Or do you want to conduct other downstream analysis? \n"
                                                "Please reply NO if you want to end this part. Please describe your needs."))
                    CURRENT_STAGE = 'analysis_discussion'
                    yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn            
        
        if REQUIRED_INFO["current_stage"] == 'analysis_discussion':
            chat_history, download_btn, global_state, REQUIRED_INFO =  parse_inf_discuss_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO)
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        # Report Generation
        if REQUIRED_INFO["current_stage"] == 'report_generation_check': # empty query or postprocess query parsed successfully
            import glob
            global_state_files = glob.glob(f"{global_state.user_data.output_graph_dir}/*_global_state.pkl")
            global_state.logging.global_state_logging = []
            for file in global_state_files:
                with open(file, 'rb') as f:
                    temp_global_state = pickle.load(f)
                    global_state.logging.global_state_logging.append(temp_global_state.algorithm.selected_algorithm)
            if len(global_state.logging.global_state_logging) > 1:
                algos = global_state.logging.global_state_logging
                chat_history.append((None, "Detailed analysis of which algorithm do you want to be included in the report?\n"
                                     f"Please choose from the following: {', '.join(algos)}\n"
                                     "Note that a comparision of all algorithms'results will be included in the report."))
                CURRENT_STAGE = 'report_algo_selection'
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                REQUIRED_INFO["current_stage"] = 'report_generation'

        if CURRENT_STAGE == 'report_algo_selection':
            chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_report_algo_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE)
            if CURRENT_STAGE != 'report_generation':
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            
        if REQUIRED_INFO["current_stage"] == 'report_generation':    
            chat_history.append(("📝 Generate comprehensive report and it may take a few minutes, stay tuned...", None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            try_num = 1
            report_path = call_report_generation(global_state, args, REQUIRED_INFO['output_dir'])
            while not os.path.isfile(report_path) and try_num < 3:
                chat_history.append((None, "❌ An error occurred during the Report Generation, we are trying again and please wait for a few minutes."))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                try_num += 1
                report_path = call_report_generation(global_state, args, REQUIRED_INFO['output_dir'])
            chat_history.append((None, "🎉 Analysis complete!"))
            chat_history.append((None, "📥 You can now download your detailed report using the download button below."))
            download_btn = gr.DownloadButton(
                "📥 Download Exclusive Report",
                size="sm",
                elem_classes=["icon-button"],
                scale=1,
                value=os.path.join(REQUIRED_INFO['output_dir'], 'output_report', 'report.pdf'),
                interactive=True
            )
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            chat_history.append((None, "🧑‍💻 If you still have any questions, just say it and let me help you! If not, just say No"))
            CURRENT_STAGE = 'processing_discussion'
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            
        # User Discussion Rounds
        if REQUIRED_INFO["current_stage"] == 'processing_discussion':
            report = open(os.path.join(output_dir, 'output_report', 'report.tex')).read()
            discussion = Discussion(args, report)
            global_state.logging.final_discuss = [{"role": "system",
                                     "content": "You are a helpful assistant. Please always refer to the following Causal Analysis information to discuss with the user and answer the user's question\n\n%s" % discussion.report_content}]
            # Answer User Query based on Previous Info
            chat_history.append((message, None))
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            message = message.strip()
            if message.lower() == "no":
                chat_history.append((None, "Thank you for using Causal-Copilot! See you!"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
                # Re-initialize Status
                REQUIRED_INFO = update(REQUIRED_INFO, 'processing', True)
                REQUIRED_INFO = update(REQUIRED_INFO, 'data_uploaded', True)
                REQUIRED_INFO = update(REQUIRED_INFO, 'initial_query', True)
                CURRENT_STAGE = 'initial_process'
                return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            else:
                report = open(os.path.join(REQUIRED_INFO['output_dir'], 'output_report', 'report.tex')).read()
                discussion = Discussion(args, report)
                global_state.logging.final_discuss = [{"role": "system",
                                         "content": "You are a helpful assistant. Please always refer to the following Causal Analysis information to discuss with the user and answer the user's question\n\n%s" % discussion.report_content}]
                # Answer User Query based on Previous Info
                global_state.logging.final_discuss, output = discussion.interaction(global_state.logging.final_discuss, message)
                global_state.logging.final_discuss.append({"role": "system", "content": output})
                chat_history.append((None, output))
                chat_history.append((None, "Do you have any other questions?"))
                yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

        else: # postprocess query cannot be parsed
            yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
            return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
    
    except Exception as e:
        REQUIRED_INFO = update(REQUIRED_INFO, 'data_uploaded', False)
        REQUIRED_INFO = update(REQUIRED_INFO, 'initial_query', False)
        CURRENT_STAGE = 'initial_process'
        chat_history.append((None, f"❌ An error occurred during analysis: {str(e)}\n"
                             "Please click reset button and try again."))
        print('error:', e)
        traceback.print_exc()
        yield args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
        return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn
    
def call_report_generation(global_state, args, output_dir):
    report_gen = Report_generation(global_state, args)
    report = report_gen.generation(debug=False)
    report_gen.save_report(report)
    report_path = os.path.join(output_dir, 'output_report', 'report.pdf')
    return report_path

def clear_chat(REQUIRED_INFO, CURRENT_STAGE, global_state):
    # Reset global variables
    chat_history = []
    global_state = None
    # Reset required info flags
    REQUIRED_INFO = update(REQUIRED_INFO, 'data_uploaded', False)
    REQUIRED_INFO = update(REQUIRED_INFO, 'initial_query', False)
    REQUIRED_INFO = update(REQUIRED_INFO, 'target_path', None)
    REQUIRED_INFO = update(REQUIRED_INFO, 'output_dir', None)
    CURRENT_STAGE = 'initial_process'

    # Return initial welcome message
    chat_history =  [(None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today? \n"
                   "⏫ Some guidances before uploading your dataset: \n"
                   "1️⃣ The dataset should be tabular in .csv format, with each column representing a variable. \n "
                   "2️⃣ Ensure that the features are in numerical format or appropriately encoded if categorical. \n"
                   "3️⃣ For initial query, your dataset has meaningful feature names, please indicate it using 'YES' or 'NO'. \n"
                   "4️⃣ Please mention heterogeneity and its indicator's column name in your initial query if there is any. \n"
                   "💡 Example initial query: 'YES. Use PC algorithm to analyze causal relationships between variables. The dataset has heterogeneity with domain column named 'country'.' \n")]
    return REQUIRED_INFO, chat_history, CURRENT_STAGE, global_state

def load_demo_dataset(dataset_name, REQUIRED_INFO, CURRENT_STAGE, chatbot, demo_btn, download_btn):
    dataset = DEMO_DATASETS[dataset_name]
    source_path = dataset["path"]

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', '')),
                exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''),
                               os.path.basename(source_path))
    REQUIRED_INFO = update(REQUIRED_INFO, 'target_path', target_path)
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''))
    REQUIRED_INFO = update(REQUIRED_INFO, 'output_dir', output_dir)
    shutil.copy(source_path, target_path)

    REQUIRED_INFO['data_uploaded'] = True
    REQUIRED_INFO['initial_query'] = True

    df = pd.read_csv(target_path)
    #chatbot.append((f"{dataset['query']}", None))
    bot_message = f"✅ Loaded demo dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns."
    chatbot = chatbot.copy()
    chatbot.append((None, bot_message))
    return chatbot, demo_btn, download_btn, dataset['query']


js = """
function createGradioAnimation() {
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    var text = 'Welcome to Causal Copilot!';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];
                container.appendChild(letter);
                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }
    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);
    return 'Animation created';
}
"""

with gr.Blocks(js=js, theme=gr.themes.Soft(), css="""
    .input-buttons { 
        position: absolute !important; 
        right: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        display: flex !important;
        gap: 5px !important;
    }
    .icon-button { 
        padding: 0 !important;
        width: 32px !important;
        height: 32px !important;
        border-radius: 16px !important;
        background: transparent !important;
    }
    .icon-button:hover { 
        background: #f0f0f0 !important;
    }
    .icon {
        width: 20px;
        height: 20px;
        margin: 6px;
        display: inline-block;
        vertical-align: middle;
    }
    .message-wrap {
        display: flex !important;
        align-items: flex-start !important;
        gap: 10px !important;
        padding: 15px !important;
    }
    .avatar {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 20px !important;
    }
    .bot-avatar {
        background: #e3f2fd !important;
        color: #1976d2 !important;
    }
    .user-avatar {
        background: #f5f5f5 !important;
        color: #333 !important;
    }
    .message {
        padding: 12px 16px !important;
        border-radius: 12px !important;
        max-width: 100% !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        object-fit: contain !important;
    }
    .bot-message {
        background: #e3f2fd !important;
        margin-right: auto !important;
    }
    .user-message {
        background: #f5f5f5 !important;
        margin-left: auto !important;
    }
""") as demo:
    print('##########Initialize Global Variables##########')
    stage_state = gr.State('initial_process')
    state = gr.State(None)
    REQUIRED_INFO = gr.State({
            'data_uploaded': False,
            'initial_query': False,
            "interactive_mode": False, 
            'processing': False,
            'target_path': None,
            'output_dir': None
        })
    args = gr.State(type('Args', (), {})())
    chatbot = gr.Chatbot(
        value=[
            (None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today? \n"
                   "⏫ Some guidances before uploading your dataset: \n"
                   "1️⃣ The dataset should be tabular in .csv format, with each column representing a variable. \n "
                   "2️⃣ Ensure that the features are in numerical format or appropriately encoded if categorical. \n"
                   "3️⃣ For initial query, your dataset has meaningful feature names, please indicate it using 'YES' or 'NO'. \n"
                   "4️⃣ Please mention heterogeneity and its indicator's column name in your initial query if there is any. \n"
                   "💡 Example initial query: 'YES. Use PC algorithm to analyze causal relationships between variables. The dataset has heterogeneity with domain column named 'country'.' \n")],
        height=700,
        show_label=False,
        show_share_button=False,
        avatar_images=["https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f600.png",
                       "https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/1f916.png"],
        bubble_full_width=False,
        elem_classes=["message-wrap"],
        render_markdown=True
    )


    def disable_all_inputs(dataset_name, chatbot, clicked_btn, download_btn, msg, all_demo_buttons):
        """Disable all interactive elements"""
        updates = []
        for _ in range(int(all_demo_buttons)):
            updates.append(gr.update(interactive=False))
        updates.extend([
            gr.update(interactive=False),  # For download button
            #gr.update(value="", interactive=False),  # For textbox
            msg,
            gr.update(interactive=False),  # For file upload
            gr.update(interactive=False),  # For reset button
        ])
        return updates


    def enable_all_inputs(all_demo_buttons):
        """Re-enable all interactive elements"""
        updates = [
            gr.update(interactive=True) for _ in range(int(all_demo_buttons))  # For all demo buttons
        ]
        updates.extend([
            gr.update(interactive=True),  # For download button
            gr.update(value="", interactive=True),  # For textbox
            gr.update(interactive=True),  # For file upload
            gr.update(interactive=True),  # For reset button
        ])
        return updates


    with gr.Row():
        with gr.Column(scale=24):
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Enter text here",
                    elem_classes="input-box",
                    show_label=False,
                    container=False,
                    scale=12
                )
                file_upload = gr.UploadButton(
                    "📎 Upload Your Data (.csv)",
                    file_types=[".csv"],
                    size="sm",
                    elem_classes=["icon-button"],
                    scale=5,
                    file_count="single"
                )
                download_btn = gr.DownloadButton(
                    "📥 Download Exclusive Report",
                    size="sm",
                    elem_classes=["icon-button"],
                    scale=6,
                    interactive=False
                )
                reset_btn = gr.Button("🔄 Reset", scale=1, elem_classes=["icon-button"], size="sm")

    # Demo dataset buttons
    demo_btns = {}
    with gr.Row():
        for dataset_name in DEMO_DATASETS:
            demo_btn = gr.Button(f"{DEMO_DATASETS[dataset_name]['name']} Demo")
            demo_btns[dataset_name] = demo_btn

        for name, demo_btn in demo_btns.items():
            # Set up the event chain for each demo button
            print(name, demo_btn)
            demo_btn.click(
                fn=disable_all_inputs,  # First disable all inputs
                inputs=[
                    gr.Textbox(value=name, visible=False),
                    chatbot,
                    demo_btn,
                    download_btn,
                    msg,
                    gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
                ],
                outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
                queue=True
            ).then(
                fn=load_demo_dataset,
                inputs=[gr.Textbox(value=name, visible=False), REQUIRED_INFO, stage_state, chatbot, demo_btn, download_btn],
                outputs=[REQUIRED_INFO, stage_state, chatbot, demo_btn, download_btn, msg],
                queue=True,
                concurrency_limit=MAX_CONCURRENT_REQUESTS
            ).then(
                fn=process_message,
                inputs=[msg, args, state, REQUIRED_INFO, stage_state, chatbot, download_btn],
                outputs=[args, state, REQUIRED_INFO, stage_state, chatbot, download_btn],
                queue=True,
                concurrency_limit=MAX_CONCURRENT_REQUESTS
            ).then(
                fn=enable_all_inputs,
                inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
                outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
                queue=True
            ).then(
                fn=lambda: "",
                outputs=[msg]
            )

    # Event handlers with queue enabled
    msg.submit(
        fn=disable_all_inputs,  # First disable all inputs
        inputs=[
            gr.Textbox(value="", visible=False),
            chatbot,
            gr.Button(visible=False),
            download_btn,
            msg,
            gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
        ],
        outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
        queue=True
    ).then(
        fn=process_message,
        inputs=[msg, args, state, REQUIRED_INFO, stage_state, chatbot, download_btn],  ##########
        outputs=[args, state, REQUIRED_INFO, stage_state, chatbot, download_btn],
        concurrency_limit=MAX_CONCURRENT_REQUESTS,
        queue=True
    ).then(
        fn=enable_all_inputs,
        inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
        outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
        queue=True
    ).then(
        fn=lambda: "",
        outputs=[msg]
    )

    reset_btn.click(
        fn=clear_chat,
        inputs=[REQUIRED_INFO, stage_state, state],
        outputs=[REQUIRED_INFO,chatbot, stage_state, state],
        queue=False  # No need for queue on reset
    )
    ###########
    file_upload.upload(
        fn=disable_all_inputs,  # First disable all inputs
        inputs=[
            gr.Textbox(value="", visible=False),
            chatbot,
            gr.Button(visible=False),
            download_btn,
            msg,
            gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)  # Pass number of buttons instead
        ],
        outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
        queue=True
    ).then(
        fn=handle_file_upload,
        inputs=[file_upload, REQUIRED_INFO, stage_state, chatbot, file_upload, download_btn],
        outputs=[REQUIRED_INFO, stage_state, chatbot, file_upload, download_btn],
        concurrency_limit=MAX_CONCURRENT_REQUESTS,
        queue=True
    ).then(
        fn=enable_all_inputs,
        inputs=[gr.Textbox(value=str(len(DEMO_DATASETS)), visible=False)],
        outputs=[*list(demo_btns.values()), download_btn, msg, file_upload, reset_btn],
        queue=True
    )

    # Download report handler with updated visibility
    download_btn.click()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=MAX_CONCURRENT_REQUESTS,
               max_size=MAX_QUEUE_SIZE)  # Enable queuing at the app level
    demo.launch(share=True)


