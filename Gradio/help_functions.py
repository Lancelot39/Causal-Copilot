import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
from preprocess.stat_info_functions import *
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Union, Any, Optional, Literal
import torch 
from dotenv import load_dotenv
load_dotenv('/Users/wwy/Documents/Project/Causal-Copilot/.env')

def try_numeric(value):
    """Convert string to int first, then float if possible, otherwise return string"""
    try:
        # First try converting to int
        int_val = int(value)
        return int_val
    except ValueError:
        try:
            # If int conversion fails, try float
            float_val = float(value)
            # Check if the float is actually an integer
            if float_val.is_integer():
                return int(float_val)
            return float_val
        except ValueError:
            # Return original string if both conversions fail
            return value.strip().lower()

def generate_hyperparameter_text(global_state):
    hyperparameter_text = ""
    print(global_state.algorithm.algorithm_arguments_json)
    for param, details in global_state.algorithm.algorithm_arguments_json['hyperparameters'].items():
        value = details['value']
        explanation = details['explanation']
        hyperparameter_text += f"  Parameter: {param}\n"
        hyperparameter_text += f"  Value: {value}\n"
        hyperparameter_text += f"  Explanation: {explanation}\n\n"
    return hyperparameter_text, global_state

def LLM_parse_query(format, prompt, message):
    client = OpenAI()
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

def parse_drop_high_miss_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class Bool(BaseModel):
        yes_or_no: bool=None
    prompt = """You are a helpful assistant, please extract the user's query as a boolean value. 
    If user's input is like 'yes', 'Yes', 'YES', 'y', 'Y', the boolean should be True. 
    Otherwise, the boolean should be False.
    If you cannot determine yes or no, save yes_or_no as None.
    """
    parsed_response = LLM_parse_query(Bool, prompt, message)
    yes = parsed_response.yes_or_no
    if yes is None:
        chat_history.append((message, "❌ Your query cannot be parsed, please follow the templete and retry."))
    else:
        if yes:
            global_state.statistics.drop_important_var = True
            chat_history.append((message, "✅ We will drop important variables with missing values greater than 50%."))
        else:
            global_state.statistics.drop_important_var = False
            chat_history.append((message, "✅ We will not drop important variables with missing values greater than 50%, but it may lead to unreliable result."))
        CURRENT_STAGE = 'impute_smaller_miss_30'
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
# process functions
def sample_size_check(n_row, n_col, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE):
    ## Few sample case: must reupload
    if 1<= n_row/n_col < 5:
        info = "⚠️ The dataset provided do not have enough sample size and may result in unreliable analysis. Please upload a larger dataset."
        CURRENT_STAGE = 'reupload_dataset'
    ## Not enough sample case: must reupload
    elif n_row/n_col < 1:
        info = "⚠️ The sample size of dataset provided is less than its feature size. We are not able to conduct further analysis. Please provide more samples. \n"
        CURRENT_STAGE = 'reupload_dataset'
    ## Enough sample case
    else:
        info =  "✅ The sample size is enough for the following analysis. \n"
        CURRENT_STAGE = 'meaningful_feature'
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, info

def meaningful_feature_query(global_state, message, chat_history, download_btn, CURRENT_STAGE):
    prompt = "Does this dataset have meaningful feature names? **Please only respond with 'Yes' or 'No'**."\
    "Column names like 'X1', 'X2', 'X3' are not meaningful."\
    "Column names like 'Age', 'Height', 'Weight' are meaningful."\
    f"Columns: {global_state.user_data.raw_data.columns}"
    f"User provided message: {global_state.user_data.initial_query}"
    response = LLM_parse_query(None, "You are a helpful assistant, help me to answer this question with Yes or No", prompt)
    
    if 'yes' in response.lower():
        global_state.user_data.meaningful_feature = True 
        # chat_history.append((None, "Meaningful Feature Check Summary: \n"\
        #                      "✅ The dataset has meaningful feature names."))
    else:
        global_state.user_data.meaningful_feature = False
        # chat_history.append((None, "Meaningful Feature Check Summary: \n"\
        #                      "✅ The dataset doesn't have meaningful feature names."))
    CURRENT_STAGE = 'heterogeneity'
    return chat_history, download_btn, global_state, CURRENT_STAGE


def heterogeneity_query(global_state, message, chat_history, download_btn, CURRENT_STAGE,args):
    class DomainIndexResponse(BaseModel):
        exist_domain_index: bool 
        candidate_domain_index: Optional[str]  
        
    prompt =  "Please mention heterogeneity if there is any. \n "\
    "**Heterogeneity means that the patterns or trends in the data change over time, instead of staying consistent throughout the entire period.**"\
    "If there is no indicator of heterogeneity, please return an empty list, otherwise please provide the column name of this indicator in the list. \n"\
    "You should only choose 0 or 1 variable as the heterogeneity indicator. The name must be among the features of the dataset!\n"\
    "These are features in the dataset:\n"\
    f"{global_state.user_data.raw_data.columns}"
    "**Definition**: A heterogeneous domain index is a column that indicates which domain, environment, or context each data point belongs to. "\
    "It is often used in causal discovery frameworks like CD-NOD to detect changes in distributions across domains (e.g., time, location, experiment condition).\n\n"\
    "[TASK]\n"
    "1. Identify if any column is a good candidate for the domain index, if yes, set exist_domain_index to be True; If no, set exist_domain_index to be false\n"
    "2. If you set exist_domain_index to be True, choose only one column. The column name must be among the given column name list\n"
    "[Dataset Metadata]\n"
    
    col_types, _ = data_preprocess(global_state.user_data.raw_data)
    for col in global_state.user_data.raw_data.columns:
        col_line = f"- Column name: {col} (type: {col_types[col]}) "
        if col_types[col] == "Category":
            unique_num = global_state.user_data.raw_data[col].nunique()
            col_line += f"Unique values: {unique_num}"
        prompt += col_line + "\n"
    # print(prompt)
    
    parsed_vars = LLM_parse_query(DomainIndexResponse, "You are a helpful assistant, specifically checking for the presence of a heterogeneous domain index in a dataset.", prompt)
    exist_domain_index, candidate_domain_index = parsed_vars.exist_domain_index, parsed_vars.candidate_domain_index
    if exist_domain_index:
        if candidate_domain_index and candidate_domain_index in global_state.user_data.raw_data.columns:
            global_state.statistics.heterogeneous = True
            global_state.statistics.domain_index = candidate_domain_index
            global_state.user_data.important_features = [var for var in global_state.user_data.important_features if var != candidate_domain_index]
            global_state.user_data.selected_features = [var for var in global_state.user_data.selected_features if var != candidate_domain_index]
        else:
            global_state.statistics.heterogeneous = False
            global_state.statistics.domain_index = None
    else:
        global_state.statistics.heterogeneous = False
        global_state.statistics.domain_index = None
        
    CURRENT_STAGE = 'accept_CPDAG'
    return chat_history, download_btn, global_state, CURRENT_STAGE

def parse_preliminary_feedback(chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE, message):
    print(message)
    text = ""
    if not 'no' in message.lower():
        class preliminaryVar(BaseModel):
            selected_variables: list[str]
            meaningful_feature: bool
            heterogeneity: bool
            accept_CPDAG: bool
            domin_index: Optional[str] 
            missing_value: Union[str, int, float]
        prompt = "I'm collecting some informations from the user and I give this template to them, help me to parse information from it and set variables correspondingly.\n"\
        "Firstly, carefully extract which variables are provided by the user, and save them in selected_variables list. \n"\
        "Secondly, save the provided value of variables correspondingly."
        """
        meaningful_feature: True/False
        heterogeneity: True/False
        accept_CPDAG: True/False
        domin_index: The column name of domin_index (Can only be set when heterogeneity is True)
        missing_value: special NA value/False
        """
        "If they provide domin_index, and it's among this column name list, save it in domin_index; Otherwise leave the domin_index as None."
        f"Columns: {global_state.user_data.raw_data.columns}"
        "If user provide some missing value like 0, None, etc, save it in missing_value. Save it as int or string."
        response = LLM_parse_query(preliminaryVar, "You are a helpful assistant, help me to parse the user's message carefully\n"+prompt, message)
        selected_variables, meaningful_feature, heterogeneity, accept_CPDAG, domin_index, missing_value = response.selected_variables, response.meaningful_feature, response.heterogeneity, response.accept_CPDAG, response.domin_index, response.missing_value
        print('1', meaningful_feature, '1', heterogeneity, '1', accept_CPDAG, '1', domin_index, '1', missing_value)
        # text += f"✅ Successfully parsed your provided information. \n"
        if 'meaningful_feature' in selected_variables:
            global_state.user_data.meaningful_feature = meaningful_feature
            text += f"- ✅ Adjusted Meaningful Feature: {meaningful_feature}\n"
        if 'heterogeneity' in selected_variables:
            global_state.statistics.heterogeneous = heterogeneity
            text += f"- ✅ Adjusted Heterogeneity: {heterogeneity}\n"
            if heterogeneity:
                if domin_index and domin_index in global_state.user_data.raw_data.columns:
                    global_state.statistics.domain_index = domin_index
                    # global_state.user_data.important_features.expand(domin_index)
                    text += f"- ✅ Adjusted Domain Index: {domin_index}\n"
                    global_state.user_data.important_features = [var for var in global_state.user_data.important_features if var != domin_index]
                    global_state.user_data.selected_features = [var for var in global_state.user_data.selected_features if var != domin_index]
                elif domin_index and domin_index not in global_state.user_data.raw_data.columns:
                    text += f"- ❌ The provided domain index {domin_index} is not in the dataset, we do not adjust it.\n"
            else:
                global_state.statistics.domain_index = None
        if 'accept_CPDAG' in selected_variables:
            global_state.user_data.accept_CPDAG = accept_CPDAG
            text += f"- ✅ Adjusted Accept CPDAG: {accept_CPDAG}\n"
        if 'missing_value' in selected_variables:
            global_state.user_data.nan_indicator = missing_value
            global_state, nan_detect = numeric_str_nan_detect(global_state)
            if nan_detect:
                info, global_state, CURRENT_STAGE = drop_spare_features(chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE)
                text += f"- ✅ Adjusted Missing Ratio: \n {info}\n"
    return global_state, text

def parse_reupload_query(message, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, next_step):
    class UploadVar(BaseModel):
        upload: bool
    prompt = "I ask the user whether they want to continue the procedure or they want to reupload the dataset."\
    "Help me to analyze user's intention. If they want to continue, set the upload to be False, otherwise set it to be True."
    response = LLM_parse_query(UploadVar, "You are a helpful assistant, help me to answer this question with Yes or No", prompt)
    upload = response.upload
    if not upload:
        chat_history.append((message, "📈 Continue the analysis..."))
        CURRENT_STAGE = next_step
    else:
        REQUIRED_INFO['data_uploaded'] = False
        CURRENT_STAGE = 'initial_process'
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, upload


def process_initial_query(message, chat_history, download_btn, args, REQUIRED_INFO, CURRENT_STAGE):
    print('initial query:', message)
    chat_history.append((message, None))
    REQUIRED_INFO['initial_query'] = True
    if 'yes' in message.lower():
        args.data_mode = 'real'
    elif 'no' in message.lower():
        args.data_mode = 'simulated'     
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE, args    

def parse_mode_query(message, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    message = message.strip()
    if message.lower() == 'yes':
        REQUIRED_INFO["interactive_mode"] = True
        CURRENT_STAGE = 'sparsity_check'
        chat_history.append(("✅ Run with Interactive Mode...", None))
    elif message.lower() == 'no' or message == '':
        REQUIRED_INFO["interactive_mode"] = False
        CURRENT_STAGE = 'sparsity_check_2'
        chat_history.append(("✅ Run with Non-Interactive Mode...", None))
    else: 
        chat_history.append((None, "❌ Invalid input, please try again!"))
    return chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE

def parse_var_selection_query(message, chat_history, download_btn, next_step, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class VarList(BaseModel):
        all: bool
        none: bool
        variables: list[str]
    prompt = "You are a helpful assistant, please help me to understand user's need and extract variable names from their message."
    "I ask them whether in a dataset they have important features"
    "1. If they provide some variable names, please extract variable names as a list. \n"
    "2. If there is only one variable, also save it in list variables"
    "3. If they say 'all of them', 'all' or something like that, set 'all' to be True."
    "4. If they say 'no', 'none', 'nothing' '' or something like that, set 'none' to be True."
    f"Variables must be among this list! {global_state.user_data.raw_data.columns}"
    "If there are 'all of them' or 'all', please return all variables."
    "variables in the returned list MUST be among the list above, and it's CASE SENSITIVE."
    
    parsed_vars = LLM_parse_query(VarList, prompt, message)
    all, none, var_list = parsed_vars.all, parsed_vars.none, parsed_vars.variables
    if all:
        var_list = global_state.user_data.raw_data.columns
        chat_history.append((message, "✅ All variables are selected."))
        CURRENT_STAGE = next_step
    elif none:
        var_list = []
        chat_history.append((message, "✅ No variable is selected."))
        CURRENT_STAGE = next_step
    else:
        if var_list == []:
            chat_history.append((message, "❌ Your variable selection query cannot be parsed, please make sure variables are among your dataset features and retry. \n"
                                ))
        else:
            missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns and var!='']
            if missing_vars != []:
                chat_history.append((message, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry.\n"
                                    "Note that it's CASE SENSITIVE."))
            elif len(var_list) > 20:
                chat_history.append((message, "❌ Number of chosen Variables should be within 20, please check it and retry."))
            else:
                chat_history.append((message, "✅ Successfully parsed your provided variables."))
                CURRENT_STAGE = next_step
    return var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_important_feature_query(message, chat_history, download_btn, CURRENT_STAGE, args, global_state, REQUIRED_INFO):
    print('important_feature_selection')
    var_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_var_selection_query(message, chat_history, download_btn, 'preliminary_check', args, global_state, REQUIRED_INFO, CURRENT_STAGE)
    global_state.user_data.important_features = var_list
    print(global_state.user_data.important_features)
    return args, global_state, REQUIRED_INFO, CURRENT_STAGE, chat_history, download_btn

def LLM_var_selection(message, global_state, chat_history):
    class VarList(BaseModel):
        variables: list[str]
    prompt = f"""
    You are a helpful assistant, these are many variables in the dataset, please help me to select some variables.
    Only select 10 variables, and the variables must be among this list! {global_state.user_data.processed_data.columns}
    variables in the returned list MUST be among the list above, and it's CASE SENSITIVE.
    Save the selected variables in the list variables.
    """
    parsed_vars = LLM_parse_query(VarList, prompt, message)
    var_list = parsed_vars.variables
    missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns]
    var_list = list(set(var_list)-set(missing_vars))
    chat_history.append((None, f"Suggested by LLM, we will visualize the following variables: {', '.join(var_list)}"))
    return var_list, chat_history

        
def parse_ts_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class TimeSeriesInput(BaseModel):
        is_time_series: bool
        time_lag: Union[int, Literal['default'], None]
    prompt = """
            You are helping parse a user's input regarding time-series settings for their dataset.

            The user is asked to specify:

            1. Whether their dataset is time-series.
            2. If it is time-series:
            - They can provide a specific integer `time_lag`, or
            - They can input `'default'` to let the system determine the lag automatically.
            3. If it is NOT a time-series dataset, they must leave `time_lag` as `None`.

            Please extract two fields from the user's input and return them in JSON format:
            - `is_time_series`: a boolean (`true` or `false`)
            - `time_lag`: an integer, `"default"`, or `None`

            ⚠️ Rules:
            - If `is_time_series` is false, `time_lag` must be `None`.
            - If `is_time_series` is true, `time_lag` must be either an integer or `"default"`.
            """
    parsed_response = LLM_parse_query(TimeSeriesInput, prompt, message)
    is_time_series, time_lag = parsed_response.is_time_series, parsed_response.time_lag
    
    class TimeIndex(BaseModel):
        time_index: str
    prompt = f"""
    You are a helpful assistant, I have a time-series data, please help me to extract the time index from these column names.
    These are columns in the dataset: {global_state.user_data.processed_data.columns}
    The time index can be 'Date', 'Time', 'Datetime' or something like that. **It must be among these columns**.
    Just return the column name of time index, do not include any other information.
    """
    
    if is_time_series:
        global_state.statistics.time_series = True
        global_state.statistics.data_type = "Time-series"
        global_state.user_data.heterogeneous = False
        if time_lag is not None and time_lag != 'default':
            global_state.statistics.time_lag = time_lag
        parsed_response = LLM_parse_query(TimeIndex, prompt, message)
        time_index = parsed_response.time_index
        global_state.statistics.time_index = time_index
    else:
        global_state.statistics.time_series = False
    CURRENT_STAGE = 'ts_check_done'
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_sparsity_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    # Select features based on LLM
    if message.upper() == 'LLM' or message == '':
        try:
            global_state = llm_select_dropped_features(global_state)
        except:
            global_state = llm_select_dropped_features(global_state)
        # if message.upper() == 'LLM':
        #     chat_history.append((None, "The following sparse variables suggested by LLM will be dropped: \n"
        #                                     ", ".join(global_state.user_data.llm_drop_features)))
        # elif message == '':
        #     chat_history.append((None, "You do not choose any variables to drop, we will drop the following variables suggested by LLM: \n"
        #                                     ", ".join(global_state.user_data.llm_drop_features)))
        global_state = drop_greater_miss_between_30_50_feature(global_state)
    # Select features based on user query
    else:
        class VarList(BaseModel):
            variables: list[str]
        prompt = "You are a helpful assistant, please extract variable names as a list. \n"
        "If there is only one variable, also save it in list variables"
        f"Variables must be among this list! {global_state.user_data.raw_data.columns}"
        "If there are 'all of them' or 'all', please return all variables."
        "variables in the returned list MUST be among the list above, and it's CASE SENSITIVE."
        "If you cannot find variable names, just return an empty list."
        parsed_vars = LLM_parse_query(VarList, prompt, message)
        var_list = parsed_vars.variables
        if var_list == []:
            chat_history.append((None, "⚠️ Your sparse variable dropping query cannot be parsed, Please follow the templete below and retry. \n"
                                            "Templete: PKA, Jnk, PIP2, PIP3, Mek"))
        else:
            missing_vars = [var for var in var_list if var not in global_state.user_data.raw_data.columns]
            if missing_vars != []:
                chat_history.append((None, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
            else:
                chat_history.append((None, "✅ Successfully parsed your provided variables. These sparse variables you provided will be dropped."))
                global_state.user_data.user_drop_features = var_list
                global_state = drop_greater_miss_between_30_50_feature(global_state)
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def first_stage_sparsity_check(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class NA_Indicator(BaseModel):
                indicator: bool
                na_indicator: str
    prompt = """You are a helpful assistant, please do the following tasks based on the provided context:
    **Context**
    We ask the user: We do not detect NA values in your dataset, do you have the specific value that represents NA? If so, please provide here. Otherwise please input 'NO'.
    Now we need to parse the user's input.
    **Task**
    Firstly, identify whether user answer 'no' or something like that, and save the boolean result in indicator. 
    If user answers 'no' or something like that, the boolean should be **True!**
    If the user provide the na_indicator, the boolean should be **False!**
    Secondly if user provide the na_indicator, identify the indicator user specified in the query, and save the string result in na_indicator. """
    parsed_response = LLM_parse_query(NA_Indicator, prompt, message)
    indicator, na_indicator = parsed_response.indicator, parsed_response.na_indicator
    print(indicator, na_indicator)
    if indicator:
        global_state.user_data.nan_indicator = None
        CURRENT_STAGE = 'sparsity_check_2'
    else:
        global_state.user_data.nan_indicator = na_indicator
        global_state, nan_detect = numeric_str_nan_detect(global_state)
        if nan_detect:
            CURRENT_STAGE = 'sparsity_check_2'
        else:
            chat_history.append((None, "❌ We cannot find the NA value you specified in the dataset, please retry!"))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def drop_spare_features(chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    global_state = missing_ratio_table(global_state) # Update missingness indicator in global state and generate missingness ratio table
    global_state.statistics.sparsity_dict = missing_ratio_check(global_state.user_data.processed_data, global_state.user_data.nan_indicator)
    info = "Missing Ratio Summary: \n"\
            f"1️⃣ High Missing Ratio Variables (>0.5): {', '.join(global_state.statistics.sparsity_dict['high']) if global_state.statistics.sparsity_dict['high']!=[] else 'None'} \n"\
            f"2️⃣ Moderate Missing Ratio Variables: {', '.join(global_state.statistics.sparsity_dict['moderate']) if global_state.statistics.sparsity_dict['moderate']!=[] else 'None'} \n"\
            f"3️⃣ Low Missing Ratio Variables (<0.3): {', '.join(global_state.statistics.sparsity_dict['low']) if global_state.statistics.sparsity_dict['low']!=[] else 'None'} \n"
    
    if global_state.statistics.sparsity_dict['moderate'] != []:
        info += f"📍 The missing ratios of the following variables are greater than 0.3 and smaller than 0.5, we will use LLM to decide which variables to drop. \n"\
                f"{', '.join(global_state.statistics.sparsity_dict['moderate'])} \n"
        chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE = parse_sparsity_query('LLM', chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE)
    if global_state.statistics.sparsity_dict['high'] != []:
        info += f"📍 The missing ratios of the following variables are greater than 0.5, we will drop them: \n"\
                                f"{', '.join(global_state.statistics.sparsity_dict['high'])}  \n"
        global_state = drop_greater_miss_50_feature(global_state)
    if global_state.statistics.sparsity_dict['low'] != []:
        # impute variables with sparsity<0.3 in the following
        info += f"📍 The missing ratios of the following variables are smaller than 0.3, we will impute them: \n" \
                            f"{', '.join(global_state.statistics.sparsity_dict['low'])}  \n"
    CURRENT_STAGE = 'reupload_dataset_done'
    return info, global_state, CURRENT_STAGE
                       

def parse_algo_query(message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    if global_state.statistics.time_series:
        permitted_algo_list = ['PCMCI', 'DYNOTEARS', 'NTSNOTEARS', 'VARLiNGAM'] 
    else:
        if torch.cuda.is_available():
            permitted_algo_list= ['PC', 'PCParallel', 'AcceleratedPC', 'FCI', 'CDNOD', 'AcceleratedCDNOD',
                            'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR',
                            'GES', 'FGES', 'XGES', 'GRaSP',
                            'GOLEM', 'CALM', 'CORL', 'NOTEARSLinear', 'NOTEARSNonlinear',
                            'DirectLiNGAM', 'AcceleratedLiNGAM', 'ICALiNGAM']
        else:
            permitted_algo_list= ['PC', 'PCParallel', 'FCI', 'CDNOD',
                            'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR',
                            'GES', 'FGES', 'XGES', 'GRaSP',
                            'GOLEM', 'CALM', 'CORL', 'NOTEARSLinear', 'NOTEARSNonlinear',
                            'DirectLiNGAM', 'ICALiNGAM']
    class algo_selection(BaseModel):
        indicator: bool
        algo: str
    prompt = f"""You are a helpful assistant, please identify whether user select a causal discovery algorithm. 
    I ask user whether they want to select a causal discovery algorithm from the following:
    Algorithm List: {permitted_algo_list}
    If user do not provide a algorithm, or the algorithm name doesn't belong to the Algorithm List, please save False in indicator.
    If user provide a algorithm name that belongs to the Algorithm List, please save True in indicator and the algorithm name in algo.
    Note: The algorithm name should be exactly the same as the one in the Algorithm List.
    """
    response = LLM_parse_query(algo_selection, prompt, message)
    indicator, algo = response.indicator, response.algo
    if indicator:
        if algo in permitted_algo_list:
            global_state.algorithm.selected_algorithm = algo
            chat_history.append((message, f"✅ We will rerun the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"
                                        "Please press 'enter' in the chatbox to start the running..." ))
            CURRENT_STAGE = 'algo_selection'
        else:
            chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                            f"{', '.join(permitted_algo_list)}\n"
                            "Otherwise please reply NO."))
    else:
        chat_history.append((message, "💬 No algorithm is specified, will go to the next step..."))
        if global_state.user_data.meaningful_feature and not global_state.statistics.time_series:
            CURRENT_STAGE = 'inference_analysis_check'     
        else:
            CURRENT_STAGE = 'report_generation_check'
    return message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_algo_selection(message, global_state, chat_history):
    if global_state.statistics.time_series:
        permitted_algo_list = ['PCMCI', 'DYNOTEARS', 'NTSNOTEARS', 'VARLiNGAM'] 
    else:
        if torch.cuda.is_available():
            permitted_algo_list= ['PC', 'PCParallel', 'AcceleratedPC', 'FCI', 'CDNOD', 'AcceleratedCDNOD',
                            'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR',
                            'GES', 'FGES', 'XGES', 'GRaSP',
                            'GOLEM', 'CALM', 'CORL', 'NOTEARSLinear', 'NOTEARSNonlinear',
                            'DirectLiNGAM', 'AcceleratedLiNGAM', 'ICALiNGAM']
        else:
            permitted_algo_list= ['PC', 'PCParallel', 'FCI', 'CDNOD',
                            'InterIAMB', 'BAMB', 'HITONMB', 'IAMBnPC', 'MBOR',
                            'GES', 'FGES', 'XGES', 'GRaSP',
                            'GOLEM', 'CALM', 'CORL', 'NOTEARSLinear', 'NOTEARSNonlinear',
                            'DirectLiNGAM', 'ICALiNGAM']
    class algo_selection(BaseModel):
        indicator: bool
        algo: str
    prompt = f"""You are a helpful assistant, please identify whether user select a causal discovery algorithm. 
    I ask user whether they want to select a causal discovery algorithm from the following:
    Algorithm List: {permitted_algo_list}
    If user do not provide a algorithm, or the algorithm name doesn't belong to the Algorithm List, please save False in indicator.
    If user provide a algorithm name that belongs to the Algorithm List, please save True in indicator and the algorithm name in algo.
    Note: The algorithm name should be exactly the same as the one in the Algorithm List.
    """
    response = LLM_parse_query(algo_selection, prompt, message)
    indicator, algo = response.indicator, response.algo
    if indicator:
        if algo in permitted_algo_list:
            global_state.algorithm.selected_algorithm = algo
            global_state.algorithm.algorithm_arguments = None 
            CURRENT_STAGE = 'hyperparameter_selection'     
            chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
        else:
            chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                            f"{', '.join(permitted_algo_list)}\n"
                            "Otherwise please reply NO."))
    else:
        CURRENT_STAGE = 'hyperparameter_selection'     
        chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected algorithm: {global_state.algorithm.selected_algorithm}\n"))
    return global_state, chat_history, CURRENT_STAGE
     
def parse_method_selection_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):               
    permitted_algo_list= ['cem', 'propensity_score', 'dml', 'drl', 'iv']
    class algo_selection(BaseModel):
        indicator: bool
        algo: str 
    prompt = f"""You are a helpful assistant, please identify whether user select a causal discovery algorithm. 
    I ask user whether they want to select a causal discovery algorithm from the following:
    Algorithm List: {permitted_algo_list}
    Question for user: "Do you want to change the method? If so, please choose one from the following: "
                                    "1️⃣ PSM (Propensity Score Matching)"
                                    "2️⃣ CEM (Coarsen Exact Matching)"
                                    "3️⃣ DRL (Doubly Robust Learning)"
                                    "4️⃣ DML (Doubly Machine Learning)"
                                    "5️⃣ IV (Instrumental Variable Method)"
                                    "Otherwise please reply NO."
    If user do not provide a algorithm, or the algorithm name doesn't belong to the Algorithm List, please save False in indicator.
    If user provide a algorithm name that belongs to the Algorithm List, please save True in indicator and the algorithm name in algo.
    Note: The algorithm name should be exactly the same as the one in the Algorithm List.
    """               
    response = LLM_parse_query(algo_selection, prompt, message)
    indicator, algo = response.indicator, response.algo
    if indicator:
        if algo in permitted_algo_list:
            if algo == 'iv':
                chat_history.append((None, "⚠️ The instrumental variable doesn't exist in this causal graph, please choose other algorithms. \n"
                                     "💡 An instrumental variable is a variable that affects the treatment but does not directly affect the outcome, except through the treatment."))
            else:
                global_state.inference.task_info[global_state.inference.task_index]['hte_method'] = algo
                CURRENT_STAGE = "analyze_causal_task"    
                chat_history.append((None, f"✅ We will run the Treatment Effect Estimation Procedure with the Selected algorithm: {algo}\n"))
        else:
            chat_history.append((None, "❌ The specified algorithm is not correct, please choose from the following: \n"
                            f"{', '.join(permitted_algo_list)}\n"
                            "Otherwise please reply NO."))
    else:
        CURRENT_STAGE = "analyze_causal_task"
        chat_history.append((None, f"✅ We will run the Treatment Effect Estimation Procedure with the Selected algorithm: {algo}\n"))
    return global_state, chat_history, download_btn, REQUIRED_INFO, CURRENT_STAGE
        
def prepare_hyperparameter_text(global_state, chat_history):
    algorithm = global_state.algorithm.selected_algorithm
    with open(f"algorithm/context/hyperparameters/{algorithm}.json", "r") as f:
        param_hint = json.load(f)
    instruction = "Do you want to specify values for parameters instead of the selected one? If so, please specify your parameter following the template below: \n"
    for key in list(param_hint.keys())[1:]:
        instruction += f"{key}: value\n"
    instruction += "Otherwise please reply NO."
    chat_history.append((None, instruction))
    hint = "Here are instructions for hyper-parameter tuning:\n"
    for key, value in list(param_hint.items())[1:]:
        param_info = f"Parameter Meaning: {value['meaning']}, \n Available Values: {value['available_values']} \n Parameter Selection Suggestion: {value['expert_suggestion']}"
        prompt = "You are an expert in causal discovery, I need to write a hint for the user to choose values for causal discovery algorithm hyper-parameters"\
            "I will give you information about the parameter, please help me to write a short paragraph with bullet points for the user to choose values for this parameter."
        if len(value['available_values']) > 1:
            prompt += f"""
            Example: 
            Brief Explanation for this parameter with 1 sentences
            - **{value['available_values'][0]}**: in which case we recommend to use this value
            - **{value['available_values'][1]}**: in which case we recommend to use this value
            - ......
            """
        param_info = LLM_parse_query(None, prompt, param_info)
        hint += f"- {key}: \n{param_info};\n "
    chat_history.append((None, hint))
    return chat_history
    
def parse_hyperparameter_query(args, message, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE):
    if message.lower()=='no' or message=='':
        CURRENT_STAGE = 'algo_running'    
        hyperparameter_text, global_state = generate_hyperparameter_text(global_state) 
        chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Selected parameters: \n {hyperparameter_text}\n"))
    else:
        try:
            class Param_Selector(BaseModel):
                indicaror: bool
                param_keys: list[str]
                param_values: list[Union[str, int, float]]
                valid_params: bool
                error_messages: str
                
            # Get current algorithm specs
            algorithm = global_state.algorithm.selected_algorithm
            with open(f"algorithm/context/hyperparameters/{algorithm}.json", "r") as f:
                algorithm_specs = json.load(f)
            for param, param_info in algorithm_specs.items():
                # Skip if this is not a parameter dictionary or doesn't have available_values
                if not isinstance(param_info, dict) or 'available_values' not in param_info:
                    continue
                available_values = param_info['available_values']
                # Check if all values in available_values are of the same numeric type
                if available_values and all(isinstance(val, (int, float)) for val in available_values):
                    # Determine if they're all integers or floats
                    if all(isinstance(val, int) for val in available_values):
                        param_info['available_values'] = "integer"
                    else:
                        param_info['available_values'] = "float"
            prompt = f"""You are a parameter validation assistant. Please parse and validate the user's parameter inputs based on the provided context:
            **Context**
            We asked the user: "Do you want to specify values for parameters instead of the selected ones? If so, please specify your parameters."
            The current algorithm selected is: {algorithm}
            Here are the valid parameter specifications for this algorithm:
            {algorithm_specs}

            **Task**
            1. Check if the user has provided any parameters. If not, save indicator as False, and your reason in error message.
            If the user has provided parameters, save indicator as True, and do the following tasks:
            1. Parse the user's input to identify parameter keys and their corresponding values.
            2. Save parameter keys in the list `param_keys` and values in the list `param_values`. The order should match between the two lists.
            3. Check each parameter one by one:
            - Convert numeric values to the appropriate type (int or float) based on the parameter's expected type
            - Validate that string values match one of the allowed options for string parameters
            - Verify numeric values fall within the **acceptable range** for the parameter, int must be int, float must be float, and it's acceptable as long as the value is in the range.
            For example, if the parameter is a float and the min is 0, max is 1.0, check if the value is a float between 0 and 1.0
            if the parameter is an integer and the min is 1, max is 10, check if the value is an integer between 1 and 10
            - If arrays are expected (e.g., hidden_dims), parse them correctly
            4. If the parameter value is valid, set `valid_params` to True and leave `error_messages` as an empty string.
            4. If the parameter value is invalid, set `valid_params` to False and add an appropriate error message to `error_messages` explaining why the value is invalid, and instruct user what values they should input.
            5. If the user does not specify any parameters, return empty lists for both param_keys and param_values.

            Examples of validation: 
            - For alpha parameters with type "float", only check whtether it's a float
            - For indep_test parameters with type "string" and specific allowed values, verify the input matches one of those values
            - For depth parameters with type "integer", only check if it's an integer

            Only validate parameters that exist for the current algorithm. Ignore parameters that aren't defined in the specifications.
            """ 
            parsed_response = LLM_parse_query(Param_Selector, prompt, message)
            indicator, param_keys, param_values, valid_params, error_messages = parsed_response.indicaror, parsed_response.param_keys, parsed_response.param_values, parsed_response.valid_params, parsed_response.error_messages
            if not indicator:
                CURRENT_STAGE = 'algo_running'
                chat_history.append((None, "💬 Continue to the next step..."))
                return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
            specified_params = {param_keys[i].lower(): param_values[i] for i in range(len(param_keys))}
            print('specified_params',specified_params)
            print('valid_params', valid_params)
            if valid_params:
                print('specified_params',specified_params)
                original_params = global_state.algorithm.algorithm_arguments_json['hyperparameters']
                print('original_params',original_params)
                common_keys = original_params.keys() & specified_params.keys()
                if len(common_keys)==0:
                    chat_history.append((None, "❌ The specified parameters are not correct, please follow the template!"))
                    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

                for key in common_keys:
                    global_state.algorithm.algorithm_arguments_json['hyperparameters'][key]['value'] = specified_params[key]
                    global_state.algorithm.algorithm_arguments_json['hyperparameters'][key]['explanation'] = 'User specified'
                    global_state.algorithm.algorithm_arguments[key] = specified_params[key]
                print(global_state.algorithm.algorithm_arguments)
                hyperparameter_text, global_state = generate_hyperparameter_text(global_state) 
                chat_history.append((None, f"✅ We will run the Causal Discovery Procedure with the Specified parameters: \n"
                                        f"{hyperparameter_text}"))
                CURRENT_STAGE = 'algo_running' 
            else:
                chat_history.append((None, "❌ The specified parameters are not correct, the following is the error message: \n"))
                chat_history.append((None, error_messages))
                return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        except Exception as e:
            print(e)
            print(str(e))
            traceback.print_exc()
            chat_history.append((None, "❌ The specified parameters are not correct, please follow the template!"))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_user_postprocess(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    edges_dict = {
        "add_edges": [],
        "forbid_edges": [],
        "orient_edges": []
    }
    print('message:', message)
    try:
        if message == '' or not ('Add Edges' in message or 'Forbid Edges' in message or 'Orient Edges' in message):
             CURRENT_STAGE = 'retry_algo'
             chat_history.append((None, "💬 No valid query is provided, will go to the next step."))
             return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
        else:
            class EditList(BaseModel):
                    add_edges: list[str]
                    forbid_edges: list[str]
                    orient_edges: list[str]
            prompt = f""" You are a helpful assistant, please do the following tasks:
            **Tasks**
            1. Extract node relationships in Add Edges, Forbid Edges, and Orient Edges 
            2. For each relationship, save the node pairs as a list of tuples in add_edges, forbid_edges, and orient_edges respectively.
            For example, Add Edges: Age->Height; Age->Shell Weight; should be save as [(Age, Height), (Age, Shell Weight)] in add_edges.
            3. Add Edges, Forbid Edges, and Orient Edges may not all exist. If there's no relationship, just return an empty list.
            4. All node names must be among this list! {[col.replace('_', '') for col in global_state.user_data.raw_data.columns]}
            **Example**
            Add Edges: Age->Height
            Forbid Edges: Length->Height
            Orient Edges: Age->Diameter
            add_edges = [(Age, Height)]
            forbid_edges = [(Length, Height)]
            orient_edges = [(Age, Diameter)]
            """
            def parse_query(message, edges_dict):
                parsed_response = LLM_parse_query(EditList, prompt, message)
                add_edges, forbid_edges, orient_edges = parsed_response.add_edges, parsed_response.forbid_edges, parsed_response.orient_edges
                edges_dict["add_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in add_edges] if add_edges != [] else []
                edges_dict["forbid_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in forbid_edges] if forbid_edges != [] else []
                edges_dict["orient_edges"] = [(pair.split('->')[0].strip(' '), pair.split('->')[1].strip(' ')) for pair in orient_edges] if orient_edges != [] else []
                return edges_dict
            edges_dict = parse_query(message, edges_dict)
            # Check whether all these variables exist
            variables = [item for sublist in edges_dict.values() for pair in sublist for item in pair]
            missing_vars = [var for var in variables if var not in global_state.user_data.raw_data.columns]
            print(edges_dict)
            print(variables)
            print(global_state.user_data.raw_data.columns)
            if missing_vars != []:
                edges_dict = parse_query(message, edges_dict)
                variables = [item for sublist in edges_dict.values() for pair in sublist for item in pair]
                missing_vars = [var for var in variables if var not in global_state.user_data.raw_data.columns]
                if missing_vars != []:
                    chat_history.append((None, "❌ Variables " + ", ".join(missing_vars) + " are not in the dataset, please check it and retry."))
                    return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
            CURRENT_STAGE = 'postprocess_parse_done'
            return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
    except Exception as e:
        chat_history.append((None, "❌ Your query cannot be parsed, please follow the templete and retry"))
        print(str(e))
        traceback.print_exc()
        return edges_dict, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_report_algo_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    class AlgoList(BaseModel):
        algorithms: list[str]
    algos = global_state.logging.global_state_logging
    prompt = f"""You are a helpful assistant, please do the following tasks:
    **Tasks**
    1. Extract the algorithm name as a list in algorithms.
    2. If there's no algorithm, just return an empty list.
    3. You can only choose from the following algorithms! {', '.join(algos)}
    """
    parsed_response = LLM_parse_query(AlgoList, prompt, message)
    algo_list = parsed_response.algorithms
    if algo_list == []:
        chat_history.append((message, "❌ Your algorithm query cannot be parsed, please choose from the following algorithms!\n"
                             f"{', '.join(algos)}"))
    elif len(algo_list) > 1:
        chat_history.append((message, "⚠️ You can only choose one algorithm at a time, please retry!"))
    else:
        chat_history.append((message, "✅ Successfully parsed your provided algorithm."))
        global_state.results.report_selected_index = algos.index(algo_list[0])
        CURRENT_STAGE = 'report_generation'
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE

def parse_inference_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    message = message.strip()
    if message.lower() == 'no' or message == '':
        chat_history.append((None, "✅ No need for downstream analysis, continue to the next section..."))
        CURRENT_STAGE = 'report_generation_check'
        return None, None, None, None, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE
    else:
        class InfList(BaseModel):
                tasks: list[str]
                reason: str
                descriptions: list[str]
                key_node: list[str]
        columns = global_state.user_data.processed_data.columns
        binary_condition = ""
        with open('causal_analysis/context/query_prompt.txt', 'r') as file:
            query_prompt = file.read()
            query_prompt = query_prompt.replace('[COLUMNS]', f",".join(columns))
            query_prompt = query_prompt.replace('[BINARY]', binary_condition)
        
        global_state.logging.downstream_discuss.append({"role": "user", "content": message})
        parsed_response = LLM_parse_query(InfList, query_prompt, message)
        reason, tasks_list, descs_list, key_node_list = parsed_response.reason, parsed_response.tasks, parsed_response.descriptions, parsed_response.key_node
        print(tasks_list, descs_list, key_node_list)
        chat_history.append((None, "✅ Successfully parsed your query. We will analyze it in the following perspectives:\n"
                                    f"{', '.join(tasks_list)}\n"))
        return reason, tasks_list, descs_list, key_node_list, chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE


def parse_inf_discuss_query(message, chat_history, download_btn, args, global_state, REQUIRED_INFO, CURRENT_STAGE):
    chat_history.append((message, None))
    global_state.logging.downstream_discuss.append({"role": "user", "content": message})
    message = message.strip()
    if message.lower() == 'no' or message == '':
        print('go to report_generation')
        chat_history.append((None, "✅ Discussion is finished, continue to the next section..."))
        CURRENT_STAGE = 'try_other_inference_check'
    else:
        class DiscussList(BaseModel):
                    answer: str
        prompt = f"""You are a helpful assistant, here is the previous conversation history for your reference:
                ** Conversation History **
                {global_state.logging.downstream_discuss}
                ** Your Task **
                Answer user's question based on the given history in bullet points.
                Your answer must be based on the given history, DO NOT include any fake information.
                """
        global_state.logging.downstream_discuss.append({"role": "user", "content": message})
        parsed_response = LLM_parse_query(DiscussList, prompt, message)
        answer_info = parsed_response.answer 
        global_state.inference.task_info[global_state.inference.task_index]['result']['discussion'][message] = answer_info
        print(answer_info)
    
        chat_history.append((None, answer_info))
        global_state.logging.downstream_discuss.append({"role": "system", "content": answer_info})
        chat_history.append((None, "Do you have questions about this analysis?  Please describe your questions.\n"
                                    "You can also input 'NO' to end this discussion."))
    return chat_history, download_btn, global_state, REQUIRED_INFO, CURRENT_STAGE            
                

def parse_treatment(desc, global_state, args):
    prompt = f"""
    I'm doing the Treatment Effect Estimation analysis, please identify the Treatment Variable in this description:
    {desc}
    The variable name must be among these variables: {global_state.user_data.processed_data.columns}
    Only return me with the variable name, do not include anything else.
    """
    treatment = LLM_parse_query(None, 'You are an expert in Causal Discovery.', prompt)
    return treatment

def parse_shift_value(desc, args):
    class ShiftValue(BaseModel):
        shift_value: float
    parsed_response = LLM_parse_query(ShiftValue, "Extract the numerical value from the query and save it in shift_value", desc)
    shift_value = parsed_response.shift_value
    return shift_value


import numpy as np
import networkx as nx
def process_for_json(obj):
    """Convert a complex object to JSON-serializable types recursively"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [process_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): process_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (nx.Graph, nx.DiGraph)):
        return {
            "nodes": list(obj.nodes()),
            "edges": list(obj.edges()),
            "node_attrs": {str(node): process_for_json(attrs) for node, attrs in obj.nodes(data=True)},
            "edge_attrs": {f"{u}->{v}": process_for_json(attrs) for u, v, attrs in obj.edges(data=True)}
        }
    elif hasattr(obj, '__dict__'):
        # For custom objects with a __dict__
        try:
            return {k: process_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        except:
            return str(obj)
    # Add more type conversions as needed
    return obj

# Example application to your GlobalState extraction
def extract_fields_from_global_state(global_state, fields_to_extract):
    result = {}
    
    for class_name, fields in fields_to_extract.items():
        if hasattr(global_state, class_name):
            class_instance = getattr(global_state, class_name)
            result[class_name] = {}
            
            for field in fields:
                if hasattr(class_instance, field):
                    value = getattr(class_instance, field)
                    # Process value to make it JSON serializable
                    result[class_name][field] = process_for_json(value)
    
    return result

def save_documentation(result_folder):
    # Documentation content
    documentation = """# GlobalState Data Structure

## Fields Extracted:

### user_data
- initial_query: User's original query
- knowledge_docs: Background knowledge documents

### statistics
- sample_size: Number of observations
- feature_number: Number of variables
- data_type: Type of data (continuous, discrete, mixed)
- linearity: Assessment of linear relationships
- gaussian_error: Whether data shows Gaussian distribution
- missingness: Information about missing values
- heterogeneous: If dataset contains heterogeneous variables
- domain_index: Domain-specific indexing
- time_series: If data is time series
- time_lag: Lag parameter for time series
- time_index: Time series indexing information

### algorithm
- selected_algorithm: Algorithm used for analysis
- selected_reason: Justification for algorithm choice
- algorithm_arguments: Parameters passed to algorithm

### results
- converted_graph: Original causal adjacency matrix
- revised_graph: Adjacency matrix revised with bootstrap and LLM
- lagged_graph: Causal graph for time-lagged relationships
- bootstrap_probability: Probability values from bootstrap

## Adjacency Matrix Codes:
- 0: none_exist (x) - No evidence for causality
- 1: certain_edges (->) - Strong evidence for causality
- 2: uncertain_edges (-) - Some evidence for causality
- 3: bi_edges (<->) - Bidirectional relationship
- 4: half_certain_edges (o->) - Partial evidence with source uncertainty
- 5: half_uncertain_edges (o-) - Partial evidence with uncertainty at both ends
- 6: none_edges (o-o) - No causal relationship
- 7: highly_correlated_edges (---) - Strong association without clear causality

## Inference Results (Optional):
These files are only present when users conduct specific inference analyses:

- shap_df.csv: Contains SHAP values explaining causal effects
- hte.csv: Heterogeneous treatment effect analysis results
- simulated_shift_intervention.csv: Counterfactual simulation results for interventions"""

    # Save the documentation to a file
    doc_file_path = os.path.join(result_folder, 'README.md')
    with open(doc_file_path, 'w') as f:
        f.write(documentation)
    
    print(f"Documentation saved to: {doc_file_path}")
        
def create_results_folder_and_copy_files(global_state):
    import os
    import shutil
    import pickle
    import glob
    import zipfile
    """
    Create a 'results' folder under global_state.user_data.output_graph_dir and
    copy all JSONs, CSVs, and the report PDF into it.
    
    Args:
        global_state_path: Path to the GlobalState pickle file
    """
    try:
        # Get the output directories
        output_graph_dir = global_state.user_data.output_graph_dir
        output_report_dir = global_state.user_data.output_report_dir
        # Create the results folder
        results_dir = os.path.join(output_graph_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
        
        # Find all JSON files in the current directory
        json_files = glob.glob(f"{output_graph_dir}/*.json")
        print(f"Found {len(json_files)} JSON files")
        # Find all CSV files in the current directory
        csv_files = glob.glob(f"{output_graph_dir}/*.csv")
        print(f"Found {len(csv_files)} CSV files")
        # PDF report path
        pdf_report_path = os.path.join(output_report_dir, 'report.pdf')
        
        # Copy JSON files
        for json_file in json_files:
            dest_path = os.path.join(results_dir, os.path.basename(json_file))
            shutil.copy2(json_file, dest_path)
            print(f"Copied: {json_file} -> {dest_path}")
        
        # Copy CSV files
        for csv_file in csv_files:
            dest_path = os.path.join(results_dir, os.path.basename(csv_file))
            shutil.copy2(csv_file, dest_path)
            print(f"Copied: {csv_file} -> {dest_path}")
        
        # Copy PDF report if it exists
        if os.path.exists(pdf_report_path):
            dest_path = os.path.join(results_dir, 'report.pdf')
            shutil.copy2(pdf_report_path, dest_path)
            print(f"Copied: {pdf_report_path} -> {dest_path}")
        else:
            print(f"Warning: PDF report not found at {pdf_report_path}")
        print(f"All files successfully copied to {results_dir}")
        
        save_documentation(results_dir)
        
        # Create the zip file
        zip_path = f"{results_dir}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(results_dir)))
        print(f"Created zip file: {zip_path}")
        
        return zip_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None
