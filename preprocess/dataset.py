# def load_data(directory):
#     # Xinyue Wang Implemented
#     '''
#     :param path: str for data path or filename
#     :return: pandas dataframe
#     '''
#     import json
#     import numpy as np
#     import pandas as pd
#     import os
#     import numpy as np
#
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f"The directory {directory} does not exist.")
#
#     config_path = f"{directory}/config.json"
#     data_path = f"{directory}/base_data.csv"
#     graph_path = f"{directory}/base_graph.npy"
#
#     if os.path.exists(config_path):
#         with open(config_path, 'r') as f:
#             config = json.load(f)
#     else:
#         config = None
#     if os.path.exists(data_path):
#         data = pd.read_csv(data_path)
#     else:
#         raise FileNotFoundError(f"The data file {data_path} does not exist.")
#     if os.path.exists(graph_path):
#         graph = np.load(graph_path)
#         graph = graph.T
#     else:
#         graph = None
#
#     return config, data, graph
#
#
# def statistics_info(args, data):
#     # Fang Nan Implemented
#     '''
#     :param args: configurations.
#     :param data: Given Tabular Data in Pandas DataFrame format
#     :return: A dict containing all necessary statistics information
#     '''
#     from preprocess.stat_info_functions import stat_info_collection
#
#     statistics_dict, preprocessed_data = stat_info_collection(args=args, data=data)
#
#     return statistics_dict, preprocessed_data
#
#
# def convert_stat_info_to_text(statistics):
#     """
#     Convert the statistical information from Statistics object to natural language.
#
#     :param statistics: Statistics object containing statistical information about the dataset.
#     :return: A string describing the dataset characteristics in natural language.
#     """
#     text = f"The dataset has the following characteristics:\n\n"
#     text += f"The sample size is {statistics.sample_size} with {statistics.feature_number} features. "
#     text += f"This dataset is {'time-series' if statistics.data_type == 'Time-series' else 'not time-series'} data. "
#
#     text += f"Data Type: The overall data type is {statistics.data_type}.\n\n"
#     text += f"Data Quality: {'There are' if statistics.missingness else 'There are no'} missing values in the dataset.\n\n"
#
#     text += "Statistical Properties:\n"
#     text += f"- Linearity: The relationships between variables {'are' if statistics.linearity else 'are not'} predominantly linear.\n"
#     text += f"- Gaussian Errors: The errors in the data {'do' if statistics.gaussian_error else 'do not'} follow a Gaussian distribution.\n"
#     text += f"- Heterogeneity: The dataset {'is' if statistics.heterogeneous else 'is not'} heterogeneous. \n\n"
#
#     if statistics.missingness:
#         text += "3. Imputation techniques should be considered during preprocessing.\n"
#
#     if statistics.domain_index is not None:
#         text += f"If the data is heterogeneous, the column/variable {statistics.domain_index} is the domain index indicating the heterogeneity. "
#         text += f"If the data is not heterogeneous, then the existed domain index is constant.\n\n"
#     else:
#         text += "\n\n"
#
#     return text


def knowledge_info(args, global_state):
    '''
    :param args: configurations
    :param global_state: GlobalState
    :return: GlobalState
    '''
    from openai import OpenAI
    client = OpenAI()
    data = global_state.user_data.processed_data
    table_name = args.data_file
    table_columns = '\t'.join(data.columns._data)
    prompt = ("I will conduct causal discovery on the Dataset %s containing the following Columns: \n\n"
              "%s\n\nPlease provide comprehensive domain knowledge about this data. If variable names are meaningful, analyze in detail. If they're just symbols (like x1, y1), respond with 'No Knowledge'.\n\n"
              "Please cover these aspects with clear structure:\n\n"
              "1. VARIABLE DESCRIPTIONS: Detailed explanation of each variable, its meaning, measurement units, and typical ranges\n\n"
              "2. CAUSAL RELATIONSHIPS: Potential direct and indirect causal connections between variables based on domain expertise\n\n"
              "3. RELATIONSHIP NATURE: Are relationships primarily linear or nonlinear? Explain with examples\n\n"
              "4. DATA DISTRIBUTION: Typical distributions of key variables (e.g., Gaussian, heavy-tailed, multimodal)\n\n"
              "5. CONFOUNDERS: Potential unmeasured variables that might confound relationships\n\n"
              "6. TEMPORAL ASPECTS: Time-dependencies, lags, or sequential relationships if relevant\n\n"
              "7. HETEROGENEITY: Subgroups or contexts where relationships might differ\n\n"
              "8. GRAPH DENSITY: Are causal relationships likely sparse (few connections) or dense (many connections)?\n\n"
              "9. DOMAIN-SPECIFIC CONSTRAINTS: Physical laws, logical impossibilities, or theoretical frameworks that constrain possible causal relationships\n\n"
              "10. RELEVANT LITERATURE: Key studies, papers, or established findings in this domain\n\n"
              "11. DATA QUALITY ISSUES: Typical missing data patterns, measurement errors, or biases in this domain\n\n"
              "12. INTERACTION EFFECTS: Complex variable interactions that might exist (multiplicative, threshold effects)\n\n"
              "13. FEEDBACK LOOPS: Potential cyclic causal relationships that might exist\n\n"
              "14. INSTRUMENTAL VARIABLES: Variables that might serve as valid instruments for causal identification\n\n"
              "15. INTERVENTION HISTORY: Whether any variables reflect experimental interventions or policy changes\n\n"
              "FOR TIME-SERIES DATA (if applicable):\n\n"
              "16. STATIONARITY: Whether variables are expected to be stationary or have trends/seasonality\n\n"
              "17. LAG STRUCTURE: Expected time lags between causes and effects in this domain\n\n"
              "18. REGIME CHANGES: Known historical points where causal mechanisms might have changed\n\n"
              "19. CONTEMPORANEOUS EFFECTS: Which variables might have instantaneous causal effects\n\n"
              "20. PERIODICITY: Cyclical patterns or periodicities in the data generating process\n\n"
              "Please organize your response by these numbered sections, with clear headings and concise, informative content in each section."
              ) % (table_name, table_columns)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a domain expert specializing in causal inference across multiple fields. Analyze dataset variables to extract comprehensive domain knowledge that would help with causal discovery. For each aspect, provide specific insights rather than generic descriptions. If variable names aren't meaningful (e.g., x1, y1), clearly state 'No Knowledge' but suggest what information would be needed from domain experts."},
            {"role": "user", "content": prompt}
        ]
    )
    knowledge_doc = response.choices[0].message.content
    knowledge_docs = [knowledge_doc]
    global_state.user_data.knowledge_docs = knowledge_docs
    
    client2 = OpenAI()
    prompt2 = ("I will conduct causal discovery on the Dataset %s containing the following Columns: \n\n"
              "%s\n\nPlease provide comprehensive domain knowledge about this data. If variable names are meaningful, analyze in detail. If they're just symbols (like x1, y1), respond with 'No Knowledge'.\n\n"
              "Please cover these aspects with clear structure:\n\n"
              "1. VARIABLE DESCRIPTIONS: Detailed explanation of each variable, its meaning, measurement units, and typical ranges\n\n"
              "2. CAUSAL RELATIONSHIPS: Potential direct and indirect causal connections between variables based on domain expertise\n\n"
              "3. RELATIONSHIP NATURE: Are relationships primarily linear or nonlinear? Explain with examples\n\n"
              ) % (table_name, table_columns)
    response2 = client2.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a domain expert specializing in causal inference across multiple fields. Analyze dataset variables to extract comprehensive domain knowledge that would help with causal discovery. "},
            {"role": "user", "content": prompt2}
        ]
    )
    knowledge_doc_for_user = response2.choices[0].message.content
    knowledge_docs_for_user = [knowledge_doc_for_user]
    print(knowledge_docs_for_user)
    global_state.user_data.knowledge_docs_for_user = knowledge_docs_for_user

    

    return global_state

#
# if __name__ == '__main__':
#     data = load_data('../data/20240918_224141_base_nodes20_samples5000/base_data.csv')
#
#     import argparse
#
#
#     def parse_args():
#         parser = argparse.ArgumentParser(description='Causal Learning Tool for Data Analysis')
#
#         # Input data file
#         parser.add_argument(
#             '--data-file',
#             type=str,
#             default="data/20240918_224141_base_nodes20_samples5000/base_data.csv",
#             help='Path to the input dataset file (e.g., CSV format)'
#         )
#
#         # Target variable
#         parser.add_argument(
#             '--target-variable',
#             type=str,
#             help='Name of the target variable in the dataset'
#         )
#
#         # Covariates or features
#         parser.add_argument(
#             '--features',
#             type=str,
#             nargs='+',
#             help='List of feature names to include in the analysis'
#         )
#
#         # Causal model selection
#         parser.add_argument(
#             '--model',
#             type=str,
#             choices=['linear_regression', 'propensity_score_matching', 'causal_forest', 'do_calculus'],
#             help='Causal inference model to use for the analysis'
#         )
#
#         # Hyperparameters for the model
#         parser.add_argument(
#             '--hyperparameters',
#             type=str,
#             help='JSON string or path to JSON file containing hyperparameters for the chosen model'
#         )
#
#         # Output file for results
#         parser.add_argument(
#             '--output-file',
#             type=str,
#             default='results.txt',
#             help='File path to save the analysis results'
#         )
#
#         # Data preprocessing options
#         parser.add_argument(
#             '--normalize',
#             action='store_true',
#             help='Apply normalization to the dataset'
#         )
#         parser.add_argument(
#             '--impute-missing',
#             action='store_true',
#             help='Impute missing values in the dataset'
#         )
#
#         # Data Preprocess Hyper-parameters
#         parser.add_argument(
#             '--ratio',
#             type=float,
#             default=0.5,
#             help=''
#         )
#         parser.add_argument(
#             '--ts',
#             type=bool,
#             default=False,
#             help=''
#         )
#         parser.add_argument(
#             '--num_test',
#             type=int,
#             default=100,
#             help=''
#         )
#         # Verbosity level
#         parser.add_argument(
#             '--alpha',
#             type=float,
#             default=0.1,
#             help='Enable verbose output during analysis'
#         )
#
#         # Max Deliberation Round
#         parser.add_argument(
#             '--max-iterations',
#             type=int,
#             default=10,
#             help='The maximum number of iterations to run the algorithm'
#         )
#
#         # OpenAI Settings
#         parser.add_argument(
#             '--organization',
#             type=str,
#             default="org-5NION61XDUXh0ib0JZpcppqS",
#             help='Organization ID'
#         )
#
#         parser.add_argument(
#             '--project',
#             type=str,
#             default="proj_Ry1rvoznXAMj8R2bujIIkhQN",
#             help='Project ID'
#         )
#
#         parser.add_argument(
#             '--apikey',
#             type=str,
#             default=None,
#             help='API Key'
#         )
#
#         args = parser.parse_args()
#         return args
#     args = parse_args()
#     statistics_dict = statistics_info(args, data)
#     print(statistics_dict)