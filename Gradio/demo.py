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
from causal_analysis.causal_analysis import Analysis
from postprocess.report_generation import Report_generation
from user.discuss import Discussion
from openai import OpenAI
from pydantic import BaseModel

# Global variables
UPLOAD_FOLDER = "./demo_data"
chat_history = []
target_path = None
output_dir = None
global_state = None
args = type('Args', (), {})()
REQUIRED_INFO = {
    'data_uploaded': False,
    'initial_query': False,
    'current_stage': 'initial_process',
    "interactive_mode": False
}
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


def upload_file(file):
    # TODO: add more complicated file unique ID handling
    global target_path, output_dir

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', '')), exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''),
                               os.path.basename(file.name))
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(file.name).replace('.csv', ''))
    shutil.copy(file.name, target_path)
    return target_path


def handle_file_upload(file, chatbot, file_upload_btn, download_btn):
    chatbot = chatbot.copy()
    try:
        global REQUIRED_INFO
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
            upload_file(file)
            REQUIRED_INFO['data_uploaded'] = True
            bot_message = (f"✅ Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns! \n"
                           "🤔 Please follow the guidances above for your initial query. \n"
                           "✨ It would be super helpful if you can include more relevant information, e.g., background/context/prior/statistical information!")
        else:
            bot_message = "❌ Please upload a CSV file."
        chatbot.append((None, bot_message))
        return chatbot, file_upload_btn, download_btn

    except Exception as e:
        error_message = f"❌ Error loading file: {str(e)}"
        chatbot.append((None, error_message))
        return chatbot, file_upload_btn, download_btn

def sample_size_check(n_row, n_col, chat_history, download_btn):
    global REQUIRED_INFO
    ## Few sample case: give warning
    if 1<= n_row/n_col < 5:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "⚠️ The dataset provided do not have enough sample size and may result in unreliable analysis. \n"
                                "Please upload a larger dataset if you mind that. Otherwise please enter 'continue'"))
        REQUIRED_INFO["current_stage"] = 'reupload_dataset'
    ## Not enough sample case: must reupload
    elif n_row/n_col < 1:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "⚠️ The sample size of dataset provided is less than its feature size. We are not able to conduct further analysis. Please provide more samples. \n"))
        REQUIRED_INFO["current_stage"] = 'reupload_dataset'
    ## Enough sample case
    else:
        chat_history.append((None, "Sample Size Check Summary: \n"\
                             "✅ The sample size is enough for the following analysis. \n"))
        REQUIRED_INFO["current_stage"] = 'mode_check'
    return chat_history, download_btn

def process_initial_query(message, chat_history, download_btn):
    global REQUIRED_INFO, args
    # TODO: check if the initial query is valid or satisfies the requirements
    print('initial query:', message)
    if 'YES' in message:
        args.data_mode = 'real'
        REQUIRED_INFO['initial_query'] = True
        chat_history.append((message, None))
    elif 'NO' in message:
        args.data_mode = 'simulated'
        REQUIRED_INFO['initial_query'] = True
        chat_history.append((message, None))
    else:
        print('not feature indicator')
        chat_history.append((message,
                                """Please enter your initial query first before proceeding. 
                             Please indicate if your dataset has meaningful feature names using 'YES' or 'NO', 
                             and you can also provide some information about the background/context/prior/statistical information about the dataset,
                             which would help us generate appropriate report for you.
                             """))
        #yield chat_history, download_btn
    return chat_history, download_btn    

def parse_reupload_query(message, chat_history, download_btn):
    if message == 'continue':
        chat_history.append((message, "📈 Continue the analysis..."))
        REQUIRED_INFO["current_stage"] = 'sparsity_check'
        return chat_history, download_btn
    else:
        REQUIRED_INFO['data_uploaded'] = False
        REQUIRED_INFO['current_stage'] == 'initial_process'
        #chat_history.append((message, None))
        process_message(message, chat_history, download_btn)
        #return chat_history, download_btn

def parse_var_selection_query(message, chat_history, download_btn, next_step):
    #var_list = [var.strip() for var in message.split(';') if var!='']
    class VarList(BaseModel):
        variables: list[str]
    prompt = "You are a helpful assistant, please extract variable names as a list. If you cannot find variable names, just return an empty list."
    parsed_vars = LLM_parse_query(VarList, prompt, message)
    var_list = parsed_vars.variables
    if var_list == []:
        chat_history.append((message, "Your variable selection query cannot be parsed, please follow the templete below and retry. \n"
                                        "Templete: PKA, Jnk, PIP2, PIP3, Mek"))
        return var_list, chat_history, download_btn
    else:
        missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns and var!='']
        if missing_vars != []:
            chat_history.append((message, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
            return var_list, chat_history, download_btn
        elif len(var_list) > 20:
            chat_history.append((message, "❌ Number of chosen Variables should be within 20, please check it and retry."))
            return var_list, chat_history, download_btn
        else:
            chat_history.append((message, "✅ Successfully parsed your provided variables."))
            REQUIRED_INFO["current_stage"] = next_step
            return var_list, chat_history, download_btn

def LLM_parse_query(format, prompt, message):
    global args, global_state
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

def parse_sparsity_query(message, chat_history, download_btn):
    global REQUIRED_INFO, global_state
    # Select features based on LLM
    if message == 'LLM' or '':
        try:
            global_state = llm_select_dropped_features(global_state=global_state, args=args)
        except:
            global_state = llm_select_dropped_features(global_state=global_state, args=args)
        if message == 'LLM':
            chat_history.append((message, "The following sparse variables suggested by LLM will be dropped: \n"
                                            ", ".join(global_state.user_data.llm_drop_features)))
        elif message == '':
            chat_history.append((message, "You do not choose any variables to drop, we will drop the following variables suggested by LLM: \n"
                                            ", ".join(global_state.user_data.llm_drop_features)))
        global_state = drop_greater_miss_between_30_50_feature(global_state)
        REQUIRED_INFO["current_stage"] = "reupload_dataset_done"
        #var_list = [var for var in global_state.user_data.llm_drop_features if var in global_state.user_data.raw_data.columns]
    # Select features based on user query
    else:
        class VarList(BaseModel):
            variables: list[str]
        prompt = "You are a helpful assistant, please extract variable names as a list. . If you cannot find variable names, just return an empty list."
        parsed_vars = LLM_parse_query(VarList, prompt, message)
        var_list = parsed_vars.variables
        if var_list == []:
            chat_history.append((message, "⚠️ Your sparse variable dropping query cannot be parsed, Please follow the templete below and retry. \n"
                                            "Templete: PKA, Jnk, PIP2, PIP3, Mek"))
        else:
            missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns]
            if missing_vars != []:
                chat_history.append((message, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
            else:
                chat_history.append((message, "✅ Successfully parsed your provided variables. These sparse variables you provided will be dropped."))
                global_state.user_data.user_drop_features = var_list
                global_state = drop_greater_miss_between_30_50_feature(global_state)
                REQUIRED_INFO["current_stage"] = "reupload_dataset_done"
    return chat_history, download_btn

def parse_ts_query(message, chat_history, download_btn):
    global global_state, REQUIRED_INFO
    if message.lower() == 'no':
        global_state.statistics.time_series = False
        REQUIRED_INFO["current_stage"] = 'ts_check_done'
    elif message == 'continue' or message == '':
        REQUIRED_INFO["current_stage"] = 'ts_check_done'
        global_state.statistics.time_series = True
    else:
        try:
            time_lag = int(message)
            global_state.statistics.time_lag = time_lag
            chat_history.append((None, f"✅ We successfully set your time lag to be {time_lag}."))
            REQUIRED_INFO["current_stage"] = 'ts_check_done'
            global_state.statistics.time_series = True
        except: 
            chat_history.append((None, f"❌ We cannot parse your query, please follow the template and retry."))
    return chat_history, download_btn

def parse_user_postprocess(message, chat_history, download_btn):
    global global_state
    import re 
    edges_dict = {
        "add_edges": [],
        "forbid_edges": [],
        "orient_edges": []
    }
    print('message:', message)
    # Define regex patterns for each type of edge
    add_pattern = r"Add Edges:\s*([^\s]+)\s*->\s*([^\s]+)"
    forbid_pattern = r"Forbid Edges:\s*([^\s]+)\s*->\s*([^\s]+)"
    orient_pattern = r"Orient Edges:\s*([^\s]+)\s*->\s*([^\s]+)"
    # Function to convert edge strings to tuples
    def parse_edges(edge_string):
        return [tuple(edge.strip().split('->')) for edge in edge_string.split(';') if edge.strip()]
    try:
        if message == '' or not ('Add Edges' in message or 'Forbid Edges' in message or 'Orient Edges' in message):
             REQUIRED_INFO['current_stage'] = 'retry_algo'
             chat_history.append((None, "💬 No valid query is provided, will go to the next step."))
             return edges_dict, chat_history, download_btn
        else:
             # Extract Add Edges
            edges_dict["add_edges"] = re.findall(add_pattern, message)
            # Extract Forbid Edges
            edges_dict["forbid_edges"] = re.findall(forbid_pattern, message)
            # Extract Orient Edges
            edges_dict["orient_edges"] = re.findall(orient_pattern, message)
            # Check whether all these variables exist
            variables = [item for sublist in edges_dict.values() for pair in sublist for item in pair]
            missing_vars = [var for var in variables if var not in global_state.user_data.raw_data.columns]
            if missing_vars != []:
                chat_history.append((None, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
                return edges_dict, chat_history, download_btn
            REQUIRED_INFO["current_stage"] = 'postprocess_parse_done'
            return edges_dict, chat_history, download_btn
    except Exception as e:
        chat_history.append((None, "❌ Your query cannot be parsed, please follow the templete and retry"))
        print(str(e))
        import traceback
        traceback.print_exc()
        return edges_dict, chat_history, download_btn

def parse_algo_query(message, chat_history, download_btn):
    global REQUIRED_INFO, global_state
    if message == '' or message.lower()=='no':
        chat_history.append((None, "💬 No algorithm is specified, will go to the next step..."))
        REQUIRED_INFO["current_stage"] = 'inference_analysis_check'      
    elif message not in ['PC', 'FCI', 'CDNOD', 'GES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']:
        chat_history.append((message, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                    "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS"))       
    else:  
        global_state.algorithm.selected_algorithm = message
        chat_history.append((message, f"✅ We will rerun the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"
                                       "Please press 'enter' in the chatbox to start the running..." ))
        REQUIRED_INFO["current_stage"] = 'algo_selection'
        #process_message(message, chat_history, download_btn)
    return message, chat_history, download_btn 

def parse_inference_query(message, chat_history, download_btn):
            chat_history.append((message, None))
            yield chat_history, download_btn
            if message.lower() == 'no' or message == '':
                chat_history.append((None, "✅ No need for downstream analysis, continue to the next section..."))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'report_generation'
            else:
                class InfList(BaseModel):
                            tasks: list[str]
                            descriptions: list[str]
                            key_node: list[str]
                prompt = """You are a helpful assistant, please do the following tasks:
                        **Tasks*
                        Firstly please identify what tasks the user want to do and save them as a list in tasks.
                        Please choose among the following causal tasks, if there's no matched task just return an empty list 
                        Tasks you can choose: 1. Treatment Effect Estimation; 2. Anormaly Attribution; 3. Feature Importance
                        Secondly, save user's description for their tasks as a list in descriptions, the length of description list must be the same with task list
                        Thirdly, save the key result variable user care about as a list, each task must have a key result variable and they can be the same, the length of result variable list must be the same with task list
                        **Question Examples**
                        1. Treatment Effect Estimation:
                        What is the causal effect of introducing coding classes in schools on students' future career prospects?
                        What is the average treatment effect of a minimum wage increase on employment rates?
                        How much does the availability of free internet in rural areas improve educational outcomes?
                        How does access to affordable childcare affect women’s labor force participation?
                        What is the impact of reforestation programs on air quality in urban areas?
                        2. Anormaly Attribution
                        How can we attribute a sudden increase in stock market volatility to specific economic events or market sectors?
                        Which variables (e.g., transaction amount, location, time) explain anomalies in loan repayment behavior?
                        What factors explain unexpected delays in surgery schedules or patient discharge times?
                        What are the root causes of deviations in supply chain delivery times?
                        What factors contribute most to unexpected drops in product sales during a specific period?
                        3. Feature Importance
                        What are the most influential factors driving credit score predictions?
                        What are the key factors influencing the effectiveness of a specific treatment or medication?
                        Which product attributes (e.g., price, brand, reviews) are the most influential in predicting online sales?
                        Which environmental variables (e.g., humidity, temperature, CO2 levels) are most important for predicting weather patterns?
                        What customer behaviors (e.g., browsing time, cart size) contribute most to predicting cart abandonment?
                        """
                global_state.logging.downstream_discuss.append({"role": "user", "content": message})
                parsed_response = LLM_parse_query(InfList, prompt, message)
                tasks_list, descs_list, key_node_list = parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
                print(tasks_list, descs_list, key_node_list)
                if tasks_list == []:
                    chat_history.append((None, "We cannot identify any supported task in your query, please retry or type 'NO' to skip this step."))
                    yield chat_history, download_btn
                    #return chat_history, download_btn
                else:
                    chat_history.append((None, f"Analyzing for your causal task..."))
                    yield chat_history, download_btn
                    analysis = Analysis(global_state, args)
                    for i, (task, desc, key_node) in enumerate(zip(tasks_list, descs_list, key_node_list)):
                        info, figs = analysis.forward(task, desc, key_node)
                        for fig in figs:
                            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/{fig}',)))
                        chat_history.append((None, info))
                        #yield chat_history, download_btn
                        global_state.logging.downstream_discuss.append({"role": "system", "content": info})
                    chat_history.append((None, "Do you have questions about this analysis? Or do you want to conduct other downstream analysis? \n"
                                                "Please reply NO if you want to end this part. Please describe your needs."))
                    REQUIRED_INFO["current_stage"] = 'analysis_discussion'
                    yield chat_history, download_btn
            return chat_history, download_btn


def process_message(message, chat_history, download_btn):
    global target_path, REQUIRED_INFO, global_state, args
    REQUIRED_INFO['processing'] = True
    # initial_process -> check sample size -> check missingness ratio and drop -> check correlation and drop -> check dimension and drop ->
    # stat analysis and algorithm -> user edit edges -> report generation
    try:
        if REQUIRED_INFO['current_stage'] == 'initial_process':    
            print('check data upload')
            if not REQUIRED_INFO['data_uploaded']:
                chat_history.append((message, "Please upload your dataset first before proceeding."))
                yield chat_history, download_btn
            else:
                # Initialize config
                config = get_demo_config()
                config.data_file = target_path
                for key, value in config.__dict__.items():
                    setattr(args, key, value)
                print('check initial query')
                config.initial_query = message
                chat_history, download_btn = process_initial_query(message, chat_history, download_btn)
                yield chat_history, download_btn
    
            # Initialize global state
            if REQUIRED_INFO['data_uploaded'] and REQUIRED_INFO['initial_query']:
                print('strart analysis')
                global_state = global_state_initialization(args)

                # Load data
                global_state.user_data.raw_data = pd.read_csv(target_path)
                global_state.user_data.processed_data = global_state.user_data.raw_data
                yield chat_history, download_btn
                ### important feature selection query#####
                chat_history.append((None, f"Do you have important features you care about? These are features in your provided dataset:\n"
                                           f"{', '.join(global_state.user_data.raw_data.columns)}"))
                REQUIRED_INFO["current_stage"] = 'important_feature_selection'
                yield chat_history, download_btn
                return chat_history, download_btn
        
        if REQUIRED_INFO["current_stage"] == 'important_feature_selection':
            var_list, chat_history, download_btn = parse_var_selection_query(message, chat_history, download_btn, 'sample_size_check')
            global_state.user_data.important_features = var_list
    
        if REQUIRED_INFO["current_stage"] == 'sample_size_check':
            # Preprocessing - Step 1: Sample size checking
            n_row, n_col = global_state.user_data.raw_data.shape
            chat_history, download_btn = sample_size_check(n_row, n_col, chat_history, download_btn)
            yield chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'reupload_dataset':
            if message == 'continue':
                chat_history.append((message, "📈 Continue the analysis..."))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'mode_check'
            else:
                print('recurrent message processing')
                REQUIRED_INFO['current_stage'] = 'initial_process'
                process_message(message, chat_history, download_btn)
        
        if REQUIRED_INFO["current_stage"] == 'mode_check':
                chat_history.append((None, "Do you want to use the interactive mode which allows the interaction with copilot in each step?\n"
                                        "If not, the whole procedure will be conducted automatically with suggested optimal settings.\n"
                                        "Please answer with 'Yes' or 'NO'."))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'mode_setting'
                return chat_history, download_btn
        
        if REQUIRED_INFO["current_stage"] == 'mode_setting':
            chat_history.append((message, None))
            yield chat_history, download_btn
            if message.lower() == 'yes':
                REQUIRED_INFO["interactive_mode"] = True
                REQUIRED_INFO["current_stage"] = 'sparsity_check'
                chat_history.append(("✅ Run with Interactive Mode...", None))
                yield chat_history, download_btn
            elif message.lower() == 'no' or message == '':
                REQUIRED_INFO["interactive_mode"] = False
                REQUIRED_INFO["current_stage"] = 'sparsity_check_2'
                chat_history.append(("✅ Run with Non-Interactive Mode...", None))
                yield chat_history, download_btn
            else: 
                chat_history.append((None, "❌ Invalid input, please try again!"))
                yield chat_history, download_btn
                return chat_history, download_btn
        
        # Preprocess Step 2: Sparsity Checking
        if REQUIRED_INFO["current_stage"] == 'sparsity_check':
            # missing value detection
            np_nan = np_nan_detect(global_state)
            if not np_nan:
                chat_history.append((None, "We do not detect NA values in your dataset, do you have the specific value that represents NA?\n"
                                            "If so, please provide here. Otherwise please input 'NO'."))
                REQUIRED_INFO["current_stage"] = 'sparsity_check_1'
                yield chat_history, download_btn
                return chat_history, download_btn
            else:
                REQUIRED_INFO["current_stage"] = 'sparsity_check_2'
        if REQUIRED_INFO["current_stage"] == 'sparsity_check_1':
            class NA_Indicator(BaseModel):
                        indicator: bool
                        na_indicator: str
            prompt = """You are a helpful assistant, please do the following tasks based on the provided context:
            **Context**
            We ask the user: We do not detect NA values in your dataset, do you have the specific value that represents NA? If so, please provide here. Otherwise please input 'NO'.
            Now we need to parse the user's input.
            **Task**
            Firstly, identify whether user answer 'no' or something like that, and save the boolean result in indicator. If user answers 'no' or something like that, the boolean should be True.
            Secondly if user provide the na_indicator, identify the indicator user specified in the query, and save the string result in na_indicator. """
            parsed_response = LLM_parse_query(NA_Indicator, prompt, message)
            indicator, na_indicator = parsed_response.indicator, parsed_response.na_indicator
            print(indicator, na_indicator)
            if indicator:
                global_state.user_data.nan_indicator = None
                REQUIRED_INFO["current_stage"] = 'sparsity_check_2'
            else:
                global_state.user_data.nan_indicator = na_indicator
                global_state, nan_detect = numeric_str_nan_detect(global_state)
                if nan_detect:
                    REQUIRED_INFO["current_stage"] = 'sparsity_check_2'
                else:
                    chat_history.append((None, "❌ We cannot find the NA value you specified in the dataset, please retry!"))
                    yield chat_history, download_btn
                    return chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'sparsity_check_2':
            global_state = missing_ratio_table(global_state) # Update missingness indicator in global state and generate missingness ratio table
            sparsity_dict = sparsity_check(df=global_state.user_data.processed_data)
            chat_history.append((None, "Missing Ratio Summary: \n"\
                                 f"1️⃣ High Missing Ratio Variables (>0.5): {', '.join(sparsity_dict['high']) if sparsity_dict['high']!=[] else 'None'} \n"\
                                 f"2️⃣ Moderate Missing Ratio Variables: {', '.join(sparsity_dict['moderate']) if sparsity_dict['moderate']!=[] else 'None'} \n"\
                                 f"3️⃣ Low Missing Ratio Variables (<0.3): {', '.join(sparsity_dict['low']) if sparsity_dict['low']!=[] else 'None'}"))
            yield chat_history, download_btn
            if sparsity_dict['moderate'] != []:
                REQUIRED_INFO["current_stage"] = 'sparsity_drop'
                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, "📍 The missing ratios of the following variables are greater than 0.3 and smaller than 0.5, please decide which variables you want to drop. \n"
                                                f"{', '.join(sparsity_dict['moderate'])}\n"
                                                "⚠️ Please note that variables you want to drop may be confounders, please be cautious in selection.\n"
                                                "Please seperate all variables with a semicolon ; and provide your answer following the template below: \n"
                                                "Templete: PKA; Jnk; PIP2; PIP3; Mek",
                                                "If you want LLM help you to decide, please enter 'LLM'."
                                                ))
                    yield chat_history, download_btn
                    return chat_history, download_btn
                else:
                    chat_history.append((None, f"📍 The missing ratios of the following variables are greater than 0.3 and smaller than 0.5, we will use LLM to decide which variables to drop. \n"
                                               f"{', '.join(sparsity_dict['moderate'])}"))
                    yield chat_history, download_btn
                    message = 'LLM'
            else:
                REQUIRED_INFO["current_stage"] = 'reupload_dataset_done'
                if sparsity_dict['high'] != []:
                    chat_history.append((None, f"📍 The missing ratios of the following variables are greater than 0.5, we will drop them: \n"
                                            f"{', '.join(sparsity_dict['high'])}"))
                    yield chat_history, download_btn
                    ####### update variable list
                    global_state.user_data.system_drop_features = [var for var in global_state.user_data.system_drop_features if var in global_state.user_data.raw_data.columns]
                if sparsity_dict['low'] != []:
                    # impute variables with sparsity<0.3 in the following
                    chat_history.append((None, f"📍 The missing ratios of the following variables are smaller than 0.3, we will impute them: \n" \
                                        f"{', '.join(sparsity_dict['low'])}"))
                    yield chat_history, download_btn
        
        if REQUIRED_INFO["current_stage"] == 'sparsity_drop':
            chat_history, download_btn = parse_sparsity_query(message, chat_history, download_btn)
            yield chat_history, download_btn
            
        if REQUIRED_INFO["current_stage"] == 'reupload_dataset_done':
            # Preprocess Step 3: correlation checking
            global_state = correlation_check(global_state)
            if global_state.user_data.high_corr_drop_features:
                chat_history.append((None, "Correlation Check Summary: \n"\
                                     f"We will drop {', '.join(global_state.user_data.high_corr_drop_features)} due to the fact that they are highly correlated with other features."))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'knowledge_generation'

        if REQUIRED_INFO["current_stage"] == 'knowledge_generation':
            REQUIRED_INFO["current_stage"] = ''
            # Knowledge generation
            if args.data_mode == 'real':
                chat_history.append(("🌍 Generate background knowledge based on the dataset you provided...", None))
                yield chat_history, download_btn
                global_state = knowledge_info(args, global_state)
                knowledge_clean = str(global_state.user_data.knowledge_docs).replace("[", "").replace("]", "").replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
                chat_history.append((None, knowledge_clean))
                yield chat_history, download_btn
                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, 'If you have some more background information you want to add, please enter it here! Type No to skip this step.'))
                    REQUIRED_INFO["current_stage"] = 'check_user_background'
                    yield chat_history, download_btn
                    return chat_history, download_btn
                else:
                    REQUIRED_INFO["current_stage"] = 'visual_dimension_check'
            else:
                global_state = knowledge_info(args, global_state)
                REQUIRED_INFO["current_stage"] = 'visual_dimension_check'

        #checks the validity of the user's information
        if REQUIRED_INFO["current_stage"] == 'check_user_background':
            if message.lower() == 'no' or message == '':
                REQUIRED_INFO["current_stage"] = 'visual_dimension_check'
            else:
                global_state.user_data.knowledge_docs += message
                chat_history.append((message, "✅ Successfully added your provided information!"))
                time.sleep(0.5)
                REQUIRED_INFO["current_stage"] = 'visual_dimension_check'               

        if REQUIRED_INFO["current_stage"] == 'visual_dimension_check':
            ## Preprocess Step 4: Choose Visualization Variables
            # High Dimensional Case: let user choose variables   highlight chosen variables
            if len(global_state.user_data.processed_data.columns) > 10:
                if len(global_state.user_data.important_features) > 10 or len(global_state.user_data.important_features) == 0:
                    REQUIRED_INFO["current_stage"] = 'variable_selection'
                    if REQUIRED_INFO["interactive_mode"]:
                        chat_history.append((None, "Dimension Checking Summary:\n"\
                                            "💡 There are many variables in your dataset, please follow the template below to choose variables you care about for visualization: \n"
                                                "1. Please seperate each variables with a semicolon and restrict the number within 10; \n"
                                                "2. Please choose among the following variables: \n"
                                                f"{';'.join(global_state.user_data.processed_data.columns)} \n"
                                                "3. Templete: PKA; Jnk; PIP2; PIP3; Mek"))
                        yield chat_history, download_btn
                        return chat_history, download_btn
                    else: 
                        chat_history.append((None, "Dimension Checking Summary:\n"\
                                            "💡 There are many variables in your dataset, we will randomly choose 10 variables among selected important variables to visualize."))
                        yield chat_history, download_btn
                else: # Only visualize variables user care about
                    chat_history.append((None, "Dimension Checking Summary:\n"\
                                         "💡 Because of the high dimensionality, We will only visualize variables you care about."))
                    yield chat_history, download_btn
                    global_state.user_data.visual_selected_features = global_state.user_data.important_features
                    if REQUIRED_INFO["interactive_mode"]:
                        REQUIRED_INFO["current_stage"] = 'stat_analysis'
                    else:
                        REQUIRED_INFO["current_stage"] = 'ts_check_done'
            else: 
                global_state.user_data.visual_selected_features = global_state.user_data.selected_features
                chat_history.append((None, "Dimension Checking Summary:\n"\
                                     "💡 The dimension of your dataset is not too large, We will visualize all variables in the dataset."))
                yield chat_history, download_btn
                if REQUIRED_INFO["interactive_mode"]:
                    REQUIRED_INFO["current_stage"] = 'stat_analysis'
                else:
                    REQUIRED_INFO["current_stage"] = 'ts_check_done'

        if REQUIRED_INFO["current_stage"] == 'variable_selection':
            if REQUIRED_INFO["interactive_mode"]:
                var_list, chat_history, download_btn = parse_var_selection_query(message, chat_history, download_btn, 'stat_analysis')
                yield chat_history, download_btn
                # Update the selected variables
                global_state.user_data.visual_selected_features = var_list
            else: 
                try:
                    global_state.user_data.visual_selected_features = global_state.user_data.important_features[:10]
                except:
                    global_state.user_data.visual_selected_features = global_state.processed_data.columns[:10]
                REQUIRED_INFO["current_stage"] = 'ts_check_done'

        
        if REQUIRED_INFO["current_stage"] == 'stat_analysis':
            # Statistical Analysis: Time Series
            chat_history.append((None, "Please indicate whether your dataset is Time-Series and set your time lag: \n"\
                                           "1️⃣ Input 'NO' if it is not a Time-Series dataset;\n"\
                                           "2️⃣ Input your time lag if you want to set it by yourself;\n"\
                                           "3️⃣ Input 'continue' if you want the time lag to be set automatically;\n"))
            REQUIRED_INFO["current_stage"] = 'ts_check'
            yield chat_history, download_btn
            return chat_history, download_btn
        
        if REQUIRED_INFO["current_stage"] == 'ts_check':
            chat_history.append((message, None))
            yield chat_history, download_btn
            chat_history, download_btn = parse_ts_query(message, chat_history, download_btn)
            yield chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'ts_check_done':
            chat_history.append(
                (f"📈 Run statistical analysis on Dataset {target_path.split('/')[-1].replace('.csv', '')}...", None))
            yield chat_history, download_btn

            user_linear = global_state.statistics.linearity
            user_gaussian = global_state.statistics.gaussian_error

            global_state = stat_info_collection(global_state)
            global_state.statistics.description = convert_stat_info_to_text(global_state.statistics)

            if global_state.statistics.data_type == "Continuous":
                if user_linear is None:
                    chat_history.append(("✍️ Generate residuals plots ...", None))
                    yield chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/residuals_plot.jpg',)))
                    yield chat_history, download_btn
                if user_gaussian is None:
                    chat_history.append(("✍️ Generate Q-Q plots ...", None))
                    yield chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/qq_plot.jpg',)))
                    yield chat_history, download_btn

            chat_history.append((None, global_state.statistics.description))
            yield chat_history, download_btn
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
                yield chat_history, download_btn
                if message.lower() == 'no' or message == '':
                    REQUIRED_INFO["current_stage"] = 'eda_generation'
                else:
                    REQUIRED_INFO["current_stage"] = 'check_user_feedback'
                return chat_history, download_btn
            else:
                REQUIRED_INFO["current_stage"] = 'eda_generation'
               
        #  process the user feedback
        if REQUIRED_INFO["current_stage"] == 'check_user_feedback':
            chat_history.append((message, None))
            yield chat_history, download_btn
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
            parsed_response = LLM_parse_query(None, prompt, message)
            try:
                changes = json.loads(parsed_response)
                global_state.statistics.update(changes)
                print(global_state.statistics)
            except RuntimeError as e:
                print(e)
                chat_history.append(None, "That information may not be correct, please try again or type Quit to skip.")
                return chat_history, download_btn

            REQUIRED_INFO["current_stage"] = 'eda_generation'
            chat_history.append((None, "✅ Successfully updated the settings according to your need!"))
            yield chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'eda_generation':
            # EDA Generation
            chat_history.append(("🔍 Run exploratory data analysis...", None))
            yield chat_history, download_btn
            my_eda = EDA(global_state)
            my_eda.generate_eda()
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_corr.jpg',)))
            chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/eda_dist.jpg',)))
            yield chat_history, download_btn
            REQUIRED_INFO["current_stage"] = 'algo_selection'

        if REQUIRED_INFO["current_stage"] == 'algo_selection':    
        # Algorithm Selection
            if global_state.algorithm.selected_algorithm is None:
                chat_history.append(("🤖 Select optimal causal discovery algorithm and its hyperparameter...", None))
                yield chat_history, download_btn
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
                yield chat_history, download_btn

                if REQUIRED_INFO["interactive_mode"]:
                    REQUIRED_INFO["current_stage"] = 'user_algo_selection'
                    chat_history.append((None, "Do you want to specify an algorithm instead of the selected one? If so, please choose one from the following: \n"
                                            "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                            "Otherwise please reply NO."))
                    yield chat_history, download_btn
                    return chat_history, download_btn
                else:           
                    REQUIRED_INFO["current_stage"] = 'hyperparameter_selection'     
            else:
                chat_history.append((None, f"✅ Selected algorithm: {global_state.algorithm.selected_algorithm}"))
                chat_history.append(
                    ("🤖 Select optimal hyperparameter for your selected causal discovery algorithm...", None))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'hyperparameter_selection'  

        if REQUIRED_INFO["current_stage"] == 'user_algo_selection':  
            if REQUIRED_INFO["interactive_mode"]:
                chat_history.append((message, None))
                yield chat_history, download_btn
            if message.lower()=='no' or message=='':
                REQUIRED_INFO["current_stage"] = 'hyperparameter_selection'     
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
                yield chat_history, download_btn
            elif message in ['PC', 'FCI', 'CDNOD', 'GES', 'DirectLiNGAM', 'ICALiNGAM', 'NOTEARS']:
                global_state.algorithm.selected_algorithm = message
                #REQUIRED_INFO["current_stage"] = 'algo_selection'
                REQUIRED_INFO["current_stage"] = 'hyperparameter_selection'     
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
                yield chat_history, download_btn
            else: 
                chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                                    "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS"))
                yield chat_history, download_btn
                return chat_history, download_btn

        if REQUIRED_INFO["current_stage"] == 'hyperparameter_selection':  
                filter = Filter(args)
                global_state = filter.forward(global_state)
                reranker = Reranker(args)
                global_state = reranker.forward(global_state)
                hyperparameter_text = ""
                for param, details in global_state.algorithm.algorithm_arguments_json['hyperparameters'].items():
                    value = details['value']
                    explanation = details['explanation']
                    hyperparameter_text += f"  Parameter: {param}\n"
                    hyperparameter_text += f"  Value: {value}\n"
                    hyperparameter_text += f"  Explanation: {explanation}\n\n"
                chat_history.append(
                    (None,
                    f"📖 Hyperparameters for the selected algorithm {global_state.algorithm.selected_algorithm}: \n\n {hyperparameter_text}"))
                yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'algo_running'

        # Causal Discovery
        if REQUIRED_INFO["current_stage"] == 'algo_running':   
            chat_history.append(("🔄 Run causal discovery algorithm...", None))
            yield chat_history, download_btn
            programmer = Programming(args)
            global_state = programmer.forward(global_state)
            REQUIRED_INFO["current_stage"] = 'initial_graph'
        # Visualization for Initial Graph
        if REQUIRED_INFO["current_stage"] == 'initial_graph':  
            chat_history.append(("📊 Generate causal graph visualization...", None))
            yield chat_history, download_btn
            my_visual_initial = Visualization(global_state)
            pos = my_visual_initial.get_pos(global_state.results.converted_graph)
            global_state.results.row_pos = pos
            if global_state.user_data.ground_truth is not None:
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.jpg', global_state.results.row_pos)
                my_visual_initial.plot_pdag(global_state.user_data.ground_truth, 'true_graph.pdf', global_state.results.row_pos)
                chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/true_graph.jpg',)))
                yield chat_history, download_btn
            if global_state.results.converted_graph is not None:
                my_visual_initial.plot_pdag(global_state.results.converted_graph, 'initial_graph.jpg', global_state.results.row_pos)
                my_visual_initial.plot_pdag(global_state.results.converted_graph, 'initial_graph.pdf', global_state.results.row_pos)
                chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/initial_graph.jpg',)))
                yield chat_history, download_btn
                my_report = Report_generation(global_state, args)
                global_state.results.raw_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.converted_graph)
                global_state.logging.graph_conversion['initial_graph_analysis'] = my_report.graph_effect_prompts()
                analysis_clean = global_state.logging.graph_conversion['initial_graph_analysis'].replace('"',"").replace("\\n\\n", "\n\n").replace("\\n", "\n").replace("'", "")
                print(analysis_clean)
                chat_history.append((None, analysis_clean))
                yield chat_history, download_btn

                if REQUIRED_INFO["interactive_mode"]:
                    chat_history.append((None, "Do you want to further prune the initial graph with LLM and analyze the graph reliability?"))
                    REQUIRED_INFO["current_stage"] = 'LLM_prune'
                    yield chat_history, download_btn
                    return chat_history, download_btn
                else:
                    REQUIRED_INFO["current_stage"] = 'revise_graph'

        if REQUIRED_INFO["current_stage"] == 'LLM_prune':
            chat_history.append((message, None))
            yield chat_history, download_btn
            
            class Indicator(BaseModel):
                        indicator: bool
            prompt = """You are a helpful assistant, please identify whether user want to further continue the task and save the boolean result in indicator. """
            parsed_response = LLM_parse_query(Indicator, prompt, message)
            indicator = parsed_response.indicator
            if indicator:
                REQUIRED_INFO["current_stage"] = 'revise_graph'
            else: 
                global_state.results.revised_graph = global_state.results.converted_graph
                if REQUIRED_INFO["interactive_mode"]:
                    REQUIRED_INFO['current_stage'] = 'user_postprocess'
                    chat_history.append((None, "If you are not satisfied with the causal graph, please tell us which edges you want to forbid or add, and we will revise the graph according to your instruction. \n"
                                                "Please follow the templete below, otherwise your input cannot be parsed. \n"
                                                "Add Edges: A1->A2; A3->A4; ... \n"
                                                "Forbid Edges: F1->F2; F3->F4; ... \n"
                                                "Orient Edges: O1->O2; O3->O4; ... \n"))
                    yield chat_history, download_btn
                    return chat_history, download_btn
                else:
                    REQUIRED_INFO['current_stage'] = 'inference_analysis_check'

        # Evaluation for Initial Graph
        if REQUIRED_INFO["current_stage"] == 'revise_graph':  
            chat_history.append(("📝 Evaluate and Revise the initial result...", None))
            yield chat_history, download_btn
            try:
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state, 'markov_blanket', 1)
            except Exception as e:
                print('error during judging:', e)
                judge = Judge(global_state, args)
                global_state = judge.forward(global_state, 'markov_blanket', 1) 
            my_visual_revise = Visualization(global_state)
            global_state.results.revised_edges = convert_to_edges(global_state.algorithm.selected_algorithm, global_state.user_data.processed_data.columns, global_state.results.revised_graph)
            # Plot Bootstrap Heatmap
            paths = my_visual_revise.boot_heatmap_plot()
            chat_history.append(
                (None, f"The following heatmaps show the confidence probability we have on different kinds of edges in the initial graph"))
            yield chat_history, download_btn
            for path in paths:
                chat_history.append((None, (path,)))
                yield chat_history, download_btn
            if args.data_mode=='real':
                # Plot Revised Graph
                if global_state.results.revised_graph is not None:
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.pdf', global_state.results.row_pos)
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.jpg', global_state.results.row_pos)
                    chat_history.append((None, f"This is the revised graph with Bootstrap and LLM techniques"))
                    yield chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/revised_graph.jpg',)))
                    yield chat_history, download_btn
                    # Refutation Graph
                    chat_history.append(("📝 Evaluate the reliability of the revised result...", None))
                    yield chat_history, download_btn
                    global_state.results.refutation_analysis = judge.graph_refutation(global_state)
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/refutation_graph.jpg',)))
                    yield chat_history, download_btn
                    chat_history.append((None, global_state.results.refutation_analysis))
                    yield chat_history, download_btn
            
            chat_history.append((None, "✅ Causal discovery analysis completed"))
            yield chat_history, download_btn
        
            #########
            if REQUIRED_INFO["interactive_mode"]:
                REQUIRED_INFO['current_stage'] = 'user_postprocess'
                chat_history.append((None, "If you are not satisfied with the causal graph, please tell us which edges you want to forbid or add, and we will revise the graph according to your instruction. \n"
                                            "Please follow the templete below, otherwise your input cannot be parsed. \n"
                                            "Add Edges: A1->A2; A3->A4; ... \n"
                                            "Forbid Edges: F1->F2; F3->F4; ... \n"
                                            "Orient Edges: O1->O2; O3->O4; ... \n"))
                yield chat_history, download_btn
                return chat_history, download_btn
            else:
                REQUIRED_INFO['current_stage'] = 'inference_analysis_check'
            #########

        if REQUIRED_INFO['current_stage'] == 'user_postprocess':
            chat_history.append((message, "📝 Start to process your Graph Revision Query..."))
            yield chat_history, download_btn
            user_revise_dict, chat_history, download_btn = parse_user_postprocess(message, chat_history, download_btn)
            print('user_revise_dict', user_revise_dict)
            yield chat_history, download_btn
            if REQUIRED_INFO["current_stage"] == 'postprocess_parse_done':
                judge = Judge(global_state, args)
                global_state = judge.user_postprocess(user_revise_dict)
                my_visual_revise = Visualization(global_state)
                if global_state.results.revised_graph is not None:
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.pdf', global_state.results.row_pos)
                    my_visual_revise.plot_pdag(global_state.results.revised_graph, 'revised_graph.jpg', global_state.results.row_pos)
                    chat_history.append((None, f"This is the revised graph according to your instruction."))
                    yield chat_history, download_btn
                    chat_history.append((None, (f'{global_state.user_data.output_graph_dir}/revised_graph.jpg',)))
                    yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'retry_algo'
            chat_history.append((None, "Do you want to retry other algorithms? If so, please choose one from the following: \n"
                                           "PC, FCI, CDNOD, GES, DirectLiNGAM, ICALiNGAM, NOTEARS\n"
                                           "Otherwise please reply NO."))
            REQUIRED_INFO["current_stage"] = 'retry_algo'
            yield chat_history, download_btn
            return chat_history, download_btn 

        if REQUIRED_INFO["current_stage"] == 'retry_algo': # empty query or postprocess query parsed successfully
            message, chat_history, download_btn = parse_algo_query(message, chat_history, download_btn)
            yield chat_history, download_btn
            if REQUIRED_INFO["current_stage"] == 'algo_selection':
                print(REQUIRED_INFO["current_stage"])
                print(global_state.algorithm.selected_algorithm)
                return process_message(message, chat_history, download_btn)
        
        if REQUIRED_INFO["current_stage"] == 'inference_analysis_check':
            chat_history.append((None, "Do you want to conduct downstream analysis based on the causal discovery result? You can descripbe your needs.\n"
                                        "Otherwise please input 'NO'.\n"
                                           "We support the following tasks: \n"
                                           "1️⃣ Treatment Effect Estimation"
                                           "2️⃣ Anormaly Attribution"
                                           "3️⃣ Feature Importance\n")) 
            REQUIRED_INFO["current_stage"] = 'inference_analysis'
            yield chat_history, download_btn
            return chat_history, download_btn  
        if REQUIRED_INFO["current_stage"] == 'inference_analysis': 
            yield parse_inference_query(message, chat_history, download_btn)
            print('inference_analysis')
        
        if REQUIRED_INFO["current_stage"] == 'analysis_discussion':
            yield parse_inf_discuss_query(message, chat_history, download_btn)
        def parse_inf_discuss_query(message, chat_history, download_btn):
            chat_history.append((message, None))
            global_state.logging.downstream_discuss.append({"role": "user", "content": message})
            #yield chat_history, download_btn
            if message.lower() == 'no' or message == '':
                print('go to report_generation')
                chat_history.append((None, "✅ No need for downstream analysis, continue to the next section..."))
                #yield chat_history, download_btn
                REQUIRED_INFO["current_stage"] = 'report_generation'
            else:
                class DiscussList(BaseModel):
                            indicator: bool
                            answer: str
                prompt = f"""You are a helpful assistant, here is the previous conversation history for your reference:
                        ** Conversation History **
                        {global_state.logging.downstream_discuss}
                        ** Your Task **
                        Firstly identify whether you can answer user's question based on the given history and save the boolean result in indicator. 
                        If the given history is enough to answer the question, set the indicator to True, otherwise set it to False.
                        Secondly, if indicator is True, save your answer to user's question in answer, your answer should be in bullet points; Otherwise set the answer to be None.
                        """
                global_state.logging.downstream_discuss.append({"role": "user", "content": message})
                parsed_response = LLM_parse_query(DiscussList, prompt, message)
                answer_ind, answer_info = parsed_response.indicator, parsed_response.answer 
                print(answer_ind, answer_info)
            
                if answer_ind:
                    chat_history.append((None, answer_info))
                    global_state.logging.downstream_discuss.append({"role": "system", "content": answer_info})
                    #yield chat_history, download_btn
                    chat_history.append((None, "Do you have questions about this analysis? Or do you want to conduct other downstream analysis? \n"
                                                "You can also input 'NO' to end this part. Please describe your needs."))
                    #yield chat_history, download_btn
                    #return chat_history, download_btn
                else:
                    # REQUIRED_INFO["current_stage"] = 'inference_analysis'
                    # chat_history.append((None, "Receive your question! Input 'yes' to analyze it..."))
                    # yield chat_history, download_btn
                    # return process_message(message, chat_history, download_btn)  
                    chat_history, download_btn = parse_inference_query(message, chat_history, download_btn)                   
            return chat_history, download_btn            

        # Report Generation
        if REQUIRED_INFO["current_stage"] == 'report_generation': # empty query or postprocess query parsed successfully
            chat_history.append(("📝 Generate comprehensive report and it may take a few minutes, stay tuned...", None))
            yield chat_history, download_btn
            report_path = call_report_generation(output_dir)
            while not os.path.isfile(report_path):
                chat_history.append((None, "❌ An error occurred during the Report Generation, we are trying again and please wait for a few minutes."))
                yield chat_history, download_btn
                report_path = call_report_generation(output_dir)
            chat_history.append((None, "🎉 Analysis complete!"))
            chat_history.append((None, "📥 You can now download your detailed report using the download button below."))
            download_btn = gr.DownloadButton(
                "📥 Download Exclusive Report",
                size="sm",
                elem_classes=["icon-button"],
                scale=1,
                value=os.path.join(output_dir, 'output_report', 'report.pdf'),
                interactive=True
            )
            yield chat_history, download_btn
            chat_history.append((None, "🧑‍💻 If you still have any questions, just say it and let me help you! If not, just say No"))
            yield chat_history, download_btn
            REQUIRED_INFO["current_stage"] = 'processing_discussion'
            
        # User Discussion Rounds
        if REQUIRED_INFO["current_stage"] == 'processing_discussion':
            report = open(os.path.join(output_dir, 'output_report', 'report.tex')).read()
            discussion = Discussion(args, report)
            global_state.logging.final_discuss = [{"role": "system",
                                     "content": "You are a helpful assistant. Please always refer to the following Causal Analysis information to discuss with the user and answer the user's question\n\n%s" % discussion.report_content}]
            # Answer User Query based on Previous Info
            chat_history.append((message, None))
            yield chat_history, download_btn
            if message.lower() == "no":
                chat_history.append((None, "Thank you for using Causal-Copilot! See you!"))
                yield chat_history, download_btn
                # Re-initialize Status
                REQUIRED_INFO['processing'] = False
                REQUIRED_INFO['data_uploaded'] = False
                REQUIRED_INFO['initial_query'] = False
                REQUIRED_INFO["current_stage"] = 'initial_process'
                return chat_history, download_btn
            else:
                global_state.logging.final_discuss, output = discussion.interaction(global_state.logging.final_discuss, message)
                global_state.logging.final_discuss.append({"role": "system", "content": output})
                chat_history.append((None, output))
                chat_history.append((None, "Do you have any other questions?"))
                yield chat_history, download_btn

        else: # postprocess query cannot be parsed
            yield chat_history, download_btn
            return chat_history, download_btn
    
    except Exception as e:
        chat_history.append((None, f"❌ An error occurred during analysis: {str(e)}, please try again"))
        print(str(e))
        import traceback
        traceback.print_exc()
        yield chat_history, download_btn
        return chat_history, download_btn
    
def call_report_generation(output_dir):
    report_gen = Report_generation(global_state, args)
    report = report_gen.generation(debug=False)
    report_gen.save_report(report)
    report_path = os.path.join(output_dir, 'output_report', 'report.pdf')
    return report_path

def clear_chat():
    global target_path, REQUIRED_INFO, output_dir, chat_history
    # Reset global variables
    target_path = None
    output_dir = None
    chat_history = []

    # Reset required info flags
    REQUIRED_INFO['data_uploaded'] = False
    REQUIRED_INFO['initial_query'] = False
    REQUIRED_INFO['processing_discussion'] = False

    # Return initial welcome message
    return [(None, "👋 Hello! I'm your causal discovery assistant. Want to discover some causal relationships today? \n"
                   "⏫ Some guidances before uploading your dataset: \n"
                   "1️⃣ The dataset should be tabular in .csv format, with each column representing a variable. \n "
                   "2️⃣ Ensure that the features are in numerical format or appropriately encoded if categorical. \n"
                   "3️⃣ For initial query, your dataset has meaningful feature names, please indicate it using 'YES' or 'NO'. \n"
                   "4️⃣ Please mention heterogeneity and its indicator's column name in your initial query if there is any. \n"
                   "💡 Example initial query: 'YES. Use PC algorithm to analyze causal relationships between variables. The dataset has heterogeneity with domain column named 'country'.' \n")],


def load_demo_dataset(dataset_name, chatbot, demo_btn, download_btn):
    global target_path, REQUIRED_INFO, output_dir
    dataset = DEMO_DATASETS[dataset_name]
    source_path = dataset["path"]

    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', '')),
                exist_ok=True)

    target_path = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''),
                               os.path.basename(source_path))
    output_dir = os.path.join(UPLOAD_FOLDER, date_time, os.path.basename(source_path).replace('.csv', ''))
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
                inputs=[gr.Textbox(value=name, visible=False), chatbot, demo_btn, download_btn],
                outputs=[chatbot, demo_btn, download_btn, msg],
                queue=True,
                concurrency_limit=MAX_CONCURRENT_REQUESTS
            ).then(
                fn=process_message,
                inputs=[msg, chatbot, download_btn],
                outputs=[chatbot, download_btn],
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
        inputs=[msg, chatbot, download_btn],
        outputs=[chatbot, download_btn],
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
        outputs=[chatbot],
        queue=False  # No need for queue on reset
    )

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
        inputs=[file_upload, chatbot, file_upload, download_btn],
        outputs=[chatbot, file_upload, download_btn],
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
