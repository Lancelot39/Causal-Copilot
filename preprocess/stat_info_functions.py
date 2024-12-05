from distutils.command.clean import clean

import numpy as np
import pandas as pd
import os
import random
import json
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import json
from openai import OpenAI

# new package
from fancyimpute import IterativeImputer


# Correlation checking #################################################################################################
def correlation_check(global_state):
    df = global_state.user_data.raw_data[global_state.user_data.selected_features]
    m = df.shape[1]

    correlation_matrix = df.corr()
    drop_feature = []

    for i in range(m):
        for j in range(i + 1, m):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.9:
                var1 = df.columns[i]
                var2 = df.columns[j]

                if var1 not in global_state.user_data.important_features:
                    drop_feature.append(var1)
                elif var2 not in global_state.user_data.important_features:
                    drop_feature.append(var2)
                else:
                    continue

    # Update global state
    global_state.user_data.high_corr_drop_features = drop_feature
    global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if
                                                element not in drop_feature]

    global_state.user_data.processed_data = global_state.user_data.raw_data[global_state.user_data.selected_features]

    return global_state


# Missingness Checking #################################################################################################
def missing_ratio_table(global_state):

    data = global_state.user_data.raw_data

    if global_state.statistics.heterogeneous and global_state.statistics.domain_index is not None:
        # Drop the domain index column from the data
        domain_index = global_state.statistics.domain_index
        col_domain_index = data[domain_index]
        data = data.drop(columns=[domain_index])

    # Step 0: Initialize selected feature
    global_state.user_data.selected_features = list(data.columns)

    missing_vals = [np.nan]
    missing_mask = data.isin(missing_vals)

    ratio_record = {}
    for column in missing_mask:
        ratio_record[column] = missing_mask[column].mean()

    global_state.statistics.miss_ratio = ratio_record

    ratio_record_df = pd.DataFrame(list(ratio_record.items()), columns=['Feature', 'Missingness Ratio'])

    plt.figure(figsize=(4, 2))
    plt.axis('off')
    table = plt.table(cellText=ratio_record_df.values, colLabels=ratio_record_df.columns, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(ratio_record_df.columns))))

    plt.savefig("missing_ratios_table.png", bbox_inches='tight', dpi=300)

    save_path = global_state.user_data.output_graph_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Saving missingness ratio table to {os.path.join(save_path, 'missing_ratios_table.jpg')}")
    plt.savefig(os.path.join(save_path, 'missing_ratios_table.jpg'))

    if sum(ratio_record.values()) == 0:
        global_state.statistics.missingness = False
    else:
        global_state.statistics.missingness = True

    return global_state


def user_llm_select_feature(global_state, args):
    # Step 1: Drop features whose ratio is greater than 0.5
    ratio_greater_05 = [k for k, v in global_state.statistics.miss_ratio.items() if v >= 0.5]
    ratio_greater_05_drop = [element for element in ratio_greater_05 if element not in global_state.user_data.important_features] # keep important features

    # Update global state
    global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if element not in ratio_greater_05_drop]

    # Step 2: Determine selected features for missingness ratio 0.3~0.5
    user_drop = global_state.user_data.user_drop_features
    if user_drop:
        global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if element not in user_drop]
        # df_dropped = df.drop(columns=user_drop)
        # llm_drop = []
    else:
        # LLM determine dropped features
        ratio_between_05_03 = [k for k, v in global_state.statistics.miss_ratio.items() if 0.5 > v >= 0.3]

        client = OpenAI(organization=args.organization, project=args.project, api_key=args.apikey)
        prompt = (f'Given the list of features of a dataset: {global_state.user_data.selected_features} \n\n,'
                  f'which features listed below do you think may be potential confounders: \n\n {ratio_between_05_03}?'
                  'Your response should be given in a list format, and the name of features should be exactly the same as the feature names I gave.'
                  'You only need to give me the list of features, no other justifications are needed. If there are no features you think should be potential confounder,'
                  'just give me an empty list.')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        llm_select_feature = response.choices[0].message.content
        llm_select_feature = llm_select_feature.replace('```json', '').replace('```', '').strip()

        llm_drop_feature = [element for element in ratio_between_05_03 if element not in llm_select_feature]
        llm_drop_keep_important = [element for element in llm_drop_feature if element not in global_state.user_data.important_features]  # keep important features

        global_state.user_data.llm_drop_features = llm_drop_keep_important
        global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if
                                                     element not in llm_drop_keep_important]

        return global_state



# TIME SERIES PROCESSING ###############################################################################################

def series_lag_est(time_series, nlags = 50):

    autocorr, confint = acf(time_series, nlags=nlags, fft=True, alpha=0.05)
    lag_significant = []

    for lag in range(1, len(autocorr)):
        if autocorr[lag] > confint[lag][1] or autocorr[lag] < confint[lag][0]:
            lag_significant = lag

    if len(lag_significant) != 0:
        est_lag = max(lag_significant)
    else:
        est_lag = np.argmax(autocorr[1:]) + 1

    return est_lag


def time_series_lag_est(df: pd.DataFrame, nlags = 50):
    est_lags = {}

    for i in range(df.shape[1]):
        est_lags[df.columns[i]] = series_lag_est(df.iloc[:, i], nlags=nlags)

    return est_lags

# lags = time_series_lag_est(ts_imputed)
# print(lags)

########################################################################################################################

def data_preprocess (clean_df: pd.DataFrame, ts: bool = False):
    '''
    :param df: Dataset in Panda DataFrame format.
    :param ratio: threshold to remove column.
    :param ts: indicator of time-series data.
    :return: cleaned data, indicator of missingness in cleaned data, overall data type, data type of each feature.
    '''

    # Data Type Classification
    column_type = {}
    overall_type = {}

    for column in clean_df.columns:

        dtype = clean_df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype) and dtype != 'bool':
            column_type[column] = 'Continuous'
        else:
            column_type[column] = 'Category'
        # elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or dtype == 'bool' or dtype == 'int64':
        #     column_type[column] = 'Category'

    all_type = list(column_type.values())
    unique_type = list(set(all_type))

    if not ts:
        if len(unique_type) == 1:
            if unique_type[0] == "Continuous":
                overall_type["Data Type"] = "Continuous"
            elif unique_type[0] == "Category":
                overall_type["Data Type"] = "Category"
        else:
            overall_type["Data Type"] = "Mixture"

    if ts:
        overall_type["Data Type"] = "Time-series"

    # Convert category data to numeric data
    categorical_features = [key for key, value in column_type.items() if value == "Category"]

    for column in categorical_features:
        clean_df[column] = pd.Categorical(clean_df[column])
        clean_df[column] = clean_df[column].cat.codes.replace(-1, np.nan) # Keep NaN while converting    

    return column_type, overall_type

# clean_data, miss_res, each_type, dataset_type = data_preprocess(df = df, ratio = 0.5, ts = False)
# print(clean_data)

def imputation (df: pd.DataFrame, column_type: dict, ts: bool = False):
    '''
    :param df: cleaned and converted data in Pandas DataFrame format.
    :param column_type: data type of each column.
    :param ts: indicator of time-series data.
    :return: imputed data.
    '''

    categorical_features = [key for key, value in column_type.items() if value == "Category"]
    continuous_features = [key for key, value in column_type.items() if value == "Continuous"]

    if not ts:
        # Initialize imputer
        imputer_cat = SimpleImputer(strategy='most_frequent')
        imputer_cont = IterativeImputer(random_state=0)

        # Imputation for continuous data
        df[continuous_features] = imputer_cont.fit_transform(df[continuous_features])

        # Imputation for categorical data
        for column in categorical_features:
            df[column] = imputer_cat.fit_transform(df[[column]]).ravel()

    if ts:
        imputer = IterativeImputer(max_iter=10, random_state=0)
        imputed_data = imputer.fit_transform(df)
        df = pd.DataFrame(imputed_data, columns=df.columns)
        # df = df.ffill()

    # Z-score normalization
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return scaled_df

# imputed_data = imputation(df = clean_data, column_type = each_type, ts = False)


def linearity_check (df: pd.DataFrame, num_test: int = 100, alpha: float = 0.1, path = None):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param num_test: maximum number of tests.
    :param alpha: significance level.
    :return: indicator of linearity, reset testing results for each pair, fitted OLS model.
    '''

    pval = []
    models = []
    m = df.shape[1]

    tot_pairs = m * (m - 1) / 2
    combinations_list = list(combinations(list(range(m)), 2))
    pair_num = min(int(tot_pairs), num_test)
    test_pairs = random.sample(combinations_list, pair_num)

    for i in range(pair_num):
        x = df.iloc[:, test_pairs[i][0]]
        x = sm.add_constant(x).to_numpy()

        y = df.iloc[:, test_pairs[i][1]].to_numpy()

        model = sm.OLS(y, x)
        results = model.fit()
        models.append((results, test_pairs[i]))

        # Ramsey’s RESET - H0: linearity is satisfied
        reset_test = linear_reset(results, power=2)
        pval.append(reset_test.pvalue)

    # Benjamini & Yekutieli procedure: return true for hypothesis that can be rejected for given alpha
    # Return True: reject H0 (p value < alpha) -- linearity is not satisfied
    corrected_result = multipletests(pval, alpha=alpha, method='fdr_by')[0]

    # Once there is one pair of test has been rejected, we conclude non-linearity
    if corrected_result.sum() == 0:
        check_result = {"Linearity": True}
        selected_models = models[:4]
    else:
        check_result = {"Linearity": False}
        # Select one of the non-linear pairs to plot residuals
        non_linear_indices = [i for i, result in enumerate(corrected_result) if result]
        linear_indices = [i for i, result in enumerate(corrected_result) if not result]
        num_nonlinear_pair = len(non_linear_indices)

        if num_nonlinear_pair >= 4:
            selected_models = [models[i] for i in non_linear_indices[:4]]
        else:
            selected_models = [models[i] for i in non_linear_indices[:num_nonlinear_pair]]
            selected_models.extend([models[i] for i in linear_indices[:(4 - num_nonlinear_pair)]])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Residual Plots for Selected Pair of Variables', fontsize=16)

    axs = axs.flatten()
    for idx, (selected_model, selected_pair) in enumerate(selected_models):
        predictions = selected_model.predict()
        residuals = selected_model.resid

        col_x_name = df.columns[selected_pair[0]]
        col_y_name = df.columns[selected_pair[1]]

        axs[idx].scatter(predictions, residuals)
        axs[idx].axhline(y=0, color='r', linestyle='--')
        axs[idx].set_xlabel('Predicted Values')
        axs[idx].set_ylabel('Residuals')
        axs[idx].set_title(f'{col_x_name} vs {col_y_name}')

    # Hide any unused subplots if less than 4 pairs were tested
    for idx in range(len(selected_models), 4):
        fig.delaxes(axs[idx])

    if not os.path.exists(path):
        os.makedirs(path)
    print(f"Saving residuals plot to {os.path.join(path, 'residuals_plot.jpg')}")
    fig.savefig(os.path.join(path, 'residuals_plot.jpg'))

    return check_result

# linearity_res = linearity_check(df = imputed_data, path = '/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causal Copilot/preprocess/stat_figures')
# print(linearity_res)


 # Gaussian error Checking
 #
 # Input: cleaned and transformed data & Linearity testing results
 # Output: testing results
def gaussian_check(df: pd.DataFrame, linearity, num_test: int = 100, alpha: float = 0.1, path=None):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param linearity: indicator of linearity.
    :param num_test: maximum number of tests.
    :param alpha: significance level.
    :return: indicator of gaussian errors.
    '''

    pval = []
    collect_result = []

    m = df.shape[1]
    tot_pairs = m * (m - 1) / 2
    combinations_list = list(combinations(list(range(m)), 2))
    pair_num = min(int(tot_pairs), num_test)
    test_pairs = random.sample(combinations_list, pair_num)

    for i in range(pair_num):
        if linearity:
            x = df.iloc[:, test_pairs[i][0]]
            x = sm.add_constant(x).to_numpy()

            y = df.iloc[:, test_pairs[i][1]].to_numpy()

            model = sm.OLS(y, x)
            results = model.fit()
            residuals = results.resid
            collect_result.append((residuals, test_pairs[i]))

        elif not linearity:
            x = df.iloc[:, test_pairs[i][0]].to_numpy()
            y = df.iloc[:, test_pairs[i][1]].to_numpy()

            # Fit Lowess
            smoothed = lowess(y, x)
            smoothed_x = smoothed[:, 0]
            smoothed_y = smoothed[:, 1]
            smoothed_values = np.interp(x, smoothed_x, smoothed_y)
            residuals = y - smoothed_values

            collect_result.append((residuals, test_pairs[i]))

        # Shapiro-Wilk test - H0: Gaussian errors
        test = stats.shapiro(residuals)
        pval.append(test.pvalue)

        # Benjamini & Yekutieli procedure - True: reject H0 -- Gaussian error assumption is not satisfied
        corrected_result = multipletests(pval, alpha=alpha, method='fdr_by')[0]

        if corrected_result.sum() == 0:
            check_result = {"Gaussian Error": True}
            selected_results = collect_result[:4]
        else:
            check_result = {"Gaussian Error": False}
            non_gaussain_indices = [i for i, result in enumerate(corrected_result) if result]
            gaussian_indices = [i for i, result in enumerate(corrected_result) if not result]
            num_nongaussian_pair = len(non_gaussain_indices)

            if num_nongaussian_pair >= 4:
                selected_results = [collect_result[i] for i in non_gaussain_indices[:4]]
            else:
                selected_results = [collect_result[i] for i in non_gaussain_indices[:num_nongaussian_pair]]
                selected_results.extend([collect_result[i] for i in gaussian_indices[:(4 - num_nongaussian_pair)]])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Q-Q Plots for Selected Pair of Variables', fontsize=16)
    axs = axs.flatten()
    for idx, (selected_results, selected_pair) in enumerate(selected_results):
        res = selected_results

        col_x_name = df.columns[selected_pair[0]]
        col_y_name = df.columns[selected_pair[1]]

        sm.qqplot(res, line='45', ax=axs[idx])
        axs[idx].set_title(f'{col_x_name} vs {col_y_name}')

    # Hide any unused subplots if less than 4 pairs were tested
    for idx in range(len(selected_results), 4):
        fig.delaxes(axs[idx])

    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, 'qq_plot.jpg'))

    return check_result

# gaussian_res = gaussian_check(df = imputed_data, linearity = linearity_res, path='/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causal Copilot/preprocess/stat_figures')
#
# print(gaussian_res)

def heterogeneity_check(df: pd.DataFrame, heterogeneity_indicator: str = "domain_index"):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param test_pairs: maximum number of pairs to be tested.
    :param alpha: significance level.
    :return: indicator of heteroscedasticity.
    '''
    # check if there are multiple domain index
    if heterogeneity_indicator in df.columns:
        if df[heterogeneity_indicator].nunique() > 1:
            return True
    return False


def stationary_check(df: pd.DataFrame, max_test: int = 1000, alpha: float = 0.1):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param max_test: maximum number of test.
    :param alpha: significance level.
    :return: indicator of stationary.
    '''

    ADF_pval = []
    m = df.shape[1]

    if m > max_test:
        num_test = max_test
        index = random.sample(range(m), num_test)
    else:
        num_test = m
        index = range(num_test)

    for i in range(num_test):
        x = df.iloc[:, index[i]].to_numpy()

        # ADF test - H0: this series is non-stationary
        adf_test = adfuller(x)
        ADF_pval.append(adf_test[1])

        # Bonferroni correction
        # True: reject H0 -- this series is stationary
        # False: accept H0 -- this series is non-stationary
        corrected_ADF = multipletests(ADF_pval, alpha=alpha, method='bonferroni')[0]

        if corrected_ADF.sum() == m:
            check_result = {"Stationary": True}
            continue
        else:
            check_result = {"Stationary": False}
            break

    return check_result





def stat_info_collection(global_state):
    '''
    :param data: given tabular data in pandas dataFrame format.
    :param global_state: GlobalState object to update and use.
    :return: updated GlobalState object.
    '''
    if global_state.statistics.heterogeneous and global_state.statistics.domain_index is not None:
        # Drop the domain index column from the data
        domain_index = global_state.statistics.domain_index
        col_domain_index = global_state.user_data.raw_data[domain_index]
    else:
        col_domain_index = None

    data = global_state.user_data.raw_data[global_state.user_data.selected_features]
    n, m = data.shape

    # Update global state
    global_state.statistics.sample_size = n
    global_state.statistics.feature_number = m

    # Data pre-processing
    each_type, dataset_type = data_preprocess(clean_df = data, ts=global_state.statistics.time_series)

    # Update global state
    global_state.statistics.data_type = dataset_type["Data Type"]

    # Imputation
    if global_state.statistics.missingness:
        imputed_data = imputation(df=data, column_type=each_type, ts=global_state.statistics.time_series)
    else:
        imputed_data = data

    # Check assumption for continuous data
    if global_state.statistics.data_type == "Continuous":
        if global_state.statistics.linearity is None:
            # Linearity assumption checking
            linearity_res = linearity_check(df=imputed_data, num_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha,
                                            path=global_state.user_data.output_graph_dir)
            # Update global state
            global_state.statistics.linearity = linearity_res["Linearity"]

        if global_state.statistics.gaussian_error is None:
            # Gaussian error checking
            gaussian_res = gaussian_check(df=imputed_data, linearity=global_state.statistics.linearity, num_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha,
                                          path=global_state.user_data.output_graph_dir)
            # Update global state
            global_state.statistics.gaussian_error = gaussian_res["Gaussian Error"]

    else:
        global_state.statistics.linearity = False
        global_state.statistics.gaussian_error = False

    # Assumption checking for time-series data
    if global_state.statistics.time_series:
        global_state.statistics.linearity = False
        global_state.statistics.gaussian_error = False

        stationary_res = stationary_check(df=imputed_data, max_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha)
        global_state.statistics.stationary = stationary_res["Stationary"]
        if global_state.statistics.time_lag is None:
            global_state.statistics.time_lag =time_series_lag_est(imputed_data, nlags = 50)

    # merge the domain index column back to the data if it exists
    if col_domain_index is not None:
        imputed_data['domain_index'] = col_domain_index

    global_state.user_data.processed_data = imputed_data

    # Convert statistics to JSON for compatibility with existing code
    # stat_info_json = json.dumps(vars(global_state.statistics), indent=4)

    return global_state



def convert_stat_info_to_text(statistics):
    """
    Convert the statistical information from Statistics object to natural language.
    
    :param statistics: Statistics object containing statistical information about the dataset.
    :return: A string describing the dataset characteristics in natural language.
    """
    text = f"The dataset has the following characteristics:\n\n"
    text += f"Data Type: The overall data type is {statistics.data_type}.\n\n"
    text += f"The sample size is {statistics.sample_size} with {statistics.feature_number} features. "
    text += f"This dataset is {'time-series' if statistics.data_type == 'Time-series' else 'not time-series'} data. "
    text += f"Data Quality: {'There are' if statistics.missingness else 'There are no'} missing values in the dataset.\n\n"
    
    text += "Statistical Properties:\n"
    text += f"- Linearity: The relationships between variables {'are' if statistics.linearity else 'are not'} predominantly linear.\n"
    text += f"- Gaussian Errors: The errors in the data {'do' if statistics.gaussian_error else 'do not'} follow a Gaussian distribution.\n"
    text += f"- Heterogeneity: The dataset {'is' if statistics.heterogeneous else 'is not'} heterogeneous. \n\n"

    if statistics.domain_index is not None:
        text += f"If the data is heterogeneous, the column/variable {statistics.domain_index} is the domain index indicating the heterogeneity. "
        text += f"If the data is not heterogeneous, then the existed domain index is constant.\n\n"
    else:
        text += "\n\n"
        
    return text

def sparsity_check(df: pd.DataFrame):
    missing_vals = [np.nan]
    missing_mask = df.isin(missing_vals)

    ratio_record = {}
    for column in missing_mask:
        ratio_record[column] = missing_mask[column].mean()

    # LLM determine dropped features
    sparsity_dict = {'high': [k for k, v in ratio_record.items() if v >= 0.5], # ratio > 0.5
                    'moderate': [k for k, v in ratio_record.items() if 0.5 > v >= 0.3], # 0.5 > ratio >= 0.3
                    'low': [k for k, v in ratio_record.items() if 0 < v < 0.3] # ratio < 0.3
                    }
    return sparsity_dict