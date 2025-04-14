from distutils.command.clean import clean

import numpy as np
import pandas as pd
import os
import random
import json
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import bds
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import mode
from scipy.signal import find_peaks
import json
from openai import OpenAI
# from Gradio.demo import global_state

# new package
from sympy.codegen.ast import Return


# Missingness Detect #################################################################################################
def np_nan_detect(global_state):
    has_nan = global_state.user_data.raw_data.isnull().values.any()
    return has_nan

def numeric_str_nan_detect(global_state):
    nan_value = global_state.user_data.nan_indicator
    data = global_state.user_data.raw_data

    nan_detect = True

    # Check if the nan_value exists in the data
    if isinstance(nan_value, (int, float)):
        if (data == nan_value).any().any():
            data.replace(nan_value, np.nan, inplace=True)
            nan_detect = True
    elif isinstance(nan_value, str):
        # If the value is a string that represents a number, cast it
        try:
            num_value = float(nan_value)
            if (data == num_value).any().any():
                data.replace(num_value, np.nan, inplace=True)
                nan_detect = True
        except ValueError:
            # It's a true string like "missing"
            if data.isin([nan_value]).any().any():
                data.replace(nan_value, np.nan, inplace=True)
                nan_detect = True

    global_state.user_data.raw_data = data
    return global_state, nan_detect


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

def drop_greater_miss_50_feature(global_state):
    # Step 1: Drop features whose ratio is greater than 0.5
    ratio_greater_05 = [k for k, v in global_state.statistics.miss_ratio.items() if v >= 0.5]
    #if global_state.user_data.drop_important_var:
    ratio_greater_05_drop = [element for element in ratio_greater_05 if
                            element not in global_state.user_data.important_features]  # keep important features

    # Update global state
    global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if
                                                element not in ratio_greater_05_drop]

    return global_state



def llm_select_dropped_features(global_state):
    ratio_between_05_03 = [k for k, v in global_state.statistics.miss_ratio.items() if 0.5 > v >= 0.3]

    client = OpenAI()
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
    llm_drop_keep_important = [element for element in llm_drop_feature if
                               element not in global_state.user_data.important_features]  # keep important features
    global_state.user_data.llm_drop_features = llm_drop_keep_important

    return global_state


def drop_greater_miss_between_30_50_feature(global_state):
    # Determine selected features for missingness ratio 0.3~0.5
    user_drop = global_state.user_data.user_drop_features
    if user_drop:
        global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if element not in user_drop]
    else:
        global_state.user_data.selected_features = [element for element in global_state.user_data.selected_features if
                                                     element not in global_state.user_data.llm_drop_features]
    return global_state


def remove_constant(global_state):
    data = global_state.user_data.raw_data
    col_types, _ = data_preprocess(data)
    for col in data.columns:
        if col_types[col] == "Category":
            unique_num = data[col].nunique()
            if unique_num == 1:
                data = data.drop(columns=[col])
                global_state.user_data.selected_features = [feature for feature in global_state.user_data.selected_features if feature != col]
    print("remove_constant", global_state.user_data.selected_features)
    return global_state

# Correlation checking #################################################################################################
def correlation_check(global_state):
    df = global_state.user_data.raw_data[global_state.user_data.selected_features]
    m = df.shape[1]

    for column in df.columns:
        col_data = df[column]
        # Exclude NaN values for type determination
        non_nan_data = col_data.dropna()

        if not pd.api.types.is_numeric_dtype(non_nan_data):
            df[column] = pd.Categorical(df[column])
            df[column] = df[column].cat.codes.replace(-1, np.nan)  # Keep NaN while converting

    correlation_matrix = df.corr()
    print(correlation_matrix)
    correlated_groups = {}
    existed_vars = set()

    m = df.shape[1]
    for i in range(m):
        var1 = df.columns[i]
        if var1 in existed_vars:
            continue
        existed_vars.add(var1)
        # Check the correlation of var1 with all other variables
        for j in range(i + 1, m):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.95:
                var2 = df.columns[j]
                if var2 in existed_vars:
                    continue
                existed_vars.add(var2)
                if var1 not in correlated_groups.keys():
                    correlated_groups[var1] = [var2]
                else:
                    correlated_groups[var1].append(var2)

    highly_correlated_vars = set(list(correlated_groups.keys()) + [item for sublist in correlated_groups.values() for item in sublist])
    remaining_vars = set(df.columns) - highly_correlated_vars
    for var in remaining_vars:
        correlated_groups[var] = []
    
    # Update global state
    # selected_set = set(global_state.user_data.selected_features) - set(drop_feature)
    # selected_set.update(global_state.user_data.important_features)
    # final_drop_feature = list(set(drop_feature) - set(global_state.user_data.important_features))
    global_state.user_data.high_corr_drop_features = highly_correlated_vars

    global_state.user_data.high_corr_feature_groups = correlated_groups
    print('correlated_groups', correlated_groups)

    # print('selected_set', selected_set)
    # print('list_selected_set', list(selected_set))
    # if len(selected_set) > 20:
    #     # Convert back to list
    #     global_state.user_data.selected_features = list(selected_set)
    #     global_state.user_data.processed_data = global_state.user_data.raw_data[global_state.user_data.selected_features]
    #     drop = True
    # else:
    #     drop = False
    drop = False

    return global_state, drop

# TIME SERIES PROCESSING ###############################################################################################
def impute_time_series(df: pd.DataFrame, time_index_feature: str = None) -> pd.DataFrame:
    """
    Impute missing values in a time series DataFrame using time-based interpolation.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the time series data.
        time_index_feature (str): Optional. The column to use as the time index.

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed.
    """
    # Work on a copy to preserve the original DataFrame
    df_copy = df.copy()

    # Determine the time index
    if time_index_feature is None:
        if not np.issubdtype(df_copy.index, np.datetime64):
            try:
                time_index = pd.to_datetime(df_copy.index)
            except Exception as e:
                raise ValueError(f"Cannot convert data index to time index: {e}")
        else:
            time_index = df_copy.index
    else:
        if time_index_feature not in df_copy.columns:
            raise ValueError(f"Column '{time_index_feature}' not found in DataFrame.")
        time_index = pd.to_datetime(df_copy[time_index_feature])

    # Perform interpolation on numeric columns
    for column in df_copy.columns:
        if column != time_index_feature and pd.api.types.is_numeric_dtype(df_copy[column]):
            df_copy[column] = pd.DataFrame(
                df_copy[column].values, index=time_index
            ).interpolate(method='time', limit_direction= 'both').values

    return df_copy



def series_lag_est(time_series, nlags=20, acf_threshold=0.6, pacf_top_pct=0.4):
    '''
    :param time_series: individual time series from df
    :param nlags: maximum lags to check
    :param acf_threshold: threshold for acf test
    :param pacf_top_pct: threshold for pacf test
    :return: estimation of max lag of causal relations
    '''
    # 1. ACF-Based Lag Selection
    acf_values, confint = acf(time_series, nlags=nlags, fft=True, alpha=0.05)
    acf_peaks, _ = find_peaks(acf_values, height=confint[:, 1]*acf_threshold)
    
    # 2. PACF-Based Lag Selection
    pacf_values = pacf(time_series, nlags=nlags)
    pacf_threshold = np.percentile(np.abs(pacf_values), 100 * (1 - pacf_top_pct))
    pacf_significant_lags = np.where(np.abs(pacf_values) > pacf_threshold)[0]

    # 3. AIC/BIC-Based Lag Selection
    best_aic, best_bic = float("inf"), float("inf")
    best_aic_lag, best_bic_lag = 1, 1
    for lag in range(1, nlags + 1):
        try:
            model = sm.tsa.ARIMA(time_series, order=(lag, 0, 0)).fit()
            if model.aic < best_aic:
                best_aic = model.aic
                best_aic_lag = lag
            if model.bic < best_bic:
                best_bic = model.bic
                best_bic_lag = lag
        except:
            continue

    candidate_lags = list(set(acf_peaks) | set(pacf_significant_lags) | {best_aic_lag, best_bic_lag})
    candidate_lags = [lag for lag in candidate_lags if lag > 0] 
    if len(candidate_lags) > 0:
        final_lag = max(set(candidate_lags), key=lambda lag: 
                        (acf_values[lag] if lag in acf_peaks else 0) + 
                        (pacf_values[lag] if lag in pacf_significant_lags else 0) +
                        (0.3 if lag in {best_aic_lag, best_bic_lag} else 0)
                       )
    else:
        final_lag = np.argmax(acf_values[1:]) + 1

    return final_lag


def time_series_lag_est(df: pd.DataFrame, nlags = 30):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param nlags: maximum lags to check
    :return: estimation of max lag of causal relations of each feature
    '''
    est_lags = []

    for i in range(df.shape[1]):
        est_lags.append(series_lag_est(df.iloc[:, i], nlags=nlags))
    
    mode_lag, count = mode(est_lags)
    median_lag = int(np.median(est_lags))
    final_lag = round((mode_lag * 0.6) + (median_lag * 0.4))
        
    return final_lag

def check_instantaneous(df: pd.DataFrame) -> bool:
    '''
    :param df: imputed data in Pandas DataFrame format.
    :return: bool value indicating the presence of instantaneous causal relations
    '''
    n_features = df.shape[1]
    # Cross-correlation based
    corr_matrix = df.corr().abs()
    corr_threshold = 1 / np.sqrt(n_features)  # 
    high_corr_pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)
                       if corr_matrix.iloc[i, j]*n_features > corr_threshold]
    return len(high_corr_pairs) > 0

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

        col_data = clean_df[column]

        # Exclude NaN values for type determination
        non_nan_data = col_data.dropna()

        if pd.api.types.is_numeric_dtype(non_nan_data):
            is_effective_integer = np.all(np.floor(non_nan_data) == non_nan_data)
            # Check if numeric
            if is_effective_integer and non_nan_data.nunique() < 6:
                column_type[column] = "Category"
            else:
                column_type[column] = "Continuous"
        else:
            # Non-numeric data types
            column_type[column] = "Category"

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

# column_type, overall_type = data_preprocess(clean_df = df, ts = False)
# print(column_type)

def imputation(df: pd.DataFrame, column_type: dict, ts: bool = False):
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
        df = impute_time_series(df)

    # Z-score normalization
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return scaled_df

# imputed_data = imputation(df = clean_data, column_type = each_type, ts = False)


def linearity_check (df_raw: pd.DataFrame, global_state):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param num_test: maximum number of tests.
    :param alpha: significance level.
    :return: indicator of linearity, reset testing results for each pair, fitted OLS model.
    '''

    pval = []
    models = []

    # Use information from global state
    num_test = global_state.statistics.num_test
    alpha = global_state.statistics.alpha
    path = global_state.user_data.output_graph_dir

    selected_features = global_state.user_data.selected_features
    visual_selected_features = global_state.user_data.visual_selected_features

    if len(selected_features) >= 10:
        df = df_raw[visual_selected_features]
    else:
        df = df_raw

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
        global_state.statistics.linearity = True
        selected_models = models[:4]
    else:
        global_state.statistics.linearity = False
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

    return global_state


def linearity_check_ts(df_raw: pd.DataFrame, global_state, save_plot=True):
    '''
    :param df: imputed data in Pandas DataFrame format.
    :param global_state
    :return: indicator of linearity, fitting a VAR model and checking the residuals
    '''
    maxlags = global_state.statistics.time_lag
    print("maxlags", maxlags)
    path = global_state.user_data.output_graph_dir
    alpha = global_state.statistics.alpha
    model = VAR(df_raw)
    results = model.fit(maxlags=maxlags)

    residuals = results.resid
    fitted = results.fittedvalues
    columns = residuals.columns
    reset_pvals = {}
    for i in range(residuals.shape[1]):
        y = residuals.iloc[:, i]
        x = fitted.iloc[:, i]
        x_const = sm.add_constant(x)
        ols_model = sm.OLS(y, x_const).fit()
        reset_result = linear_reset(ols_model, power=2, use_f=True)
        reset_pvals[columns[i]] = reset_result.pvalue

    bds_pvals = {}
    for i in range(residuals.shape[1]):
        y = residuals.iloc[:, i]
        try:
            bds_result = bds(y, max_dim=2)
            bds_pvals[columns[i]] = bds_result.pvalue
        except Exception as e:
            bds_pvals[columns[i]] = None
    
    if save_plot:
        num_vars = df_raw.shape[1]
        ncols = 2
        nrows = int(np.ceil(num_vars / ncols))
        
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axs = axs.flatten()

        for i in range(num_vars):
            axs[i].scatter(fitted.iloc[:, i], residuals.iloc[:, i], alpha=0.6)
            axs[i].axhline(0, color='r', linestyle='--')
            axs[i].set_title(f'{df_raw.columns[i]}: Residuals vs Fitted')
            axs[i].set_xlabel('Fitted Values')
            axs[i].set_ylabel('Residuals')

        # Clean up unused subplots
        for i in range(num_vars, len(axs)):
            fig.delaxes(axs[i])

        fig.suptitle('VAR Residual Analysis (Linearity Diagnostic)', fontsize=16)
        
        # Ensure output path exists
        if not os.path.exists(path):
            os.makedirs(path)

        plot_path = os.path.join(path, "var_linearity_residuals.jpg")
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
        fig.savefig(plot_path)

        print(f"Saved VAR residuals plot to: {plot_path}")
        
    reset_significant = any(p < alpha for p in reset_pvals.values() if p is not None)
    bds_significant = any(p < alpha for p in bds_pvals.values() if p is not None)
    linearity = not (reset_significant or bds_significant)
    global_state.statistics.linearity = linearity

    return global_state
# linearity_res = linearity_check(df = imputed_data, path = '/Users/fangnan/Library/CloudStorage/OneDrive-UCSanDiego/UCSD/ML Research/Causal Copilot/preprocess/stat_figures')
# print(linearity_res)


 # Gaussian error Checking
 #
 # Input: cleaned and transformed data & Linearity testing results
 # Output: testing results
def gaussian_check(df_raw, global_state):

    pval = []
    collect_result = []

    # Use information from global state
    linearity = global_state.statistics.linearity
    num_test = global_state.statistics.num_test
    alpha = global_state.statistics.alpha
    path = global_state.user_data.output_graph_dir

    selected_features = global_state.user_data.selected_features
    visual_selected_features = global_state.user_data.visual_selected_features

    if len(selected_features) >= 10:
        df = df_raw[visual_selected_features]
    else:
        df = df_raw

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
            global_state.statistics.gaussian_error = True
            selected_results = collect_result[:4]
        else:
            global_state.statistics.gaussian_error = False
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

    return global_state

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
        index = random.sample(range(m), max_test)
    else:
        index = range(m)

    for i in (index):
        x = df.iloc[:, index[i]].to_numpy()
        adf_test = adfuller(x)
        ADF_pval.append(adf_test[1])

    corrected_ADF = multipletests(ADF_pval, alpha=alpha, method='bonferroni')[0]
    num_stationary = np.sum(corrected_ADF)
    stationary = num_stationary >= len(index) * 0.8  # At least 80% of tested features must be stationary

    return stationary


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
        if domain_index in global_state.user_data.selected_features:
            global_state.user_data.selected_features = global_state.user_data.selected_features.drop(domain_index)
    else:
        col_domain_index = None
    if global_state.statistics.time_series and global_state.statistics.time_index is not None:
        global_state.user_data.selected_features = [feature for feature in global_state.user_data.selected_features if feature != global_state.statistics.time_index]
        data = global_state.user_data.raw_data.set_index(global_state.statistics.time_index)
        data = data.reset_index(drop=True) 
        data = data[global_state.user_data.selected_features]
    else:
        data = global_state.user_data.raw_data[global_state.user_data.selected_features]

    if col_domain_index is not None and domain_index in data.columns:
        data = data.drop(columns=[domain_index])
    
    n, m = data.shape

    # Update global state
    global_state.statistics.sample_size = n
    global_state.statistics.feature_number = m

    # Data pre-processing
    each_type, dataset_type = data_preprocess(clean_df = data, ts=global_state.statistics.time_series)

    # Update global state
    global_state.statistics.data_type = dataset_type["Data Type"]
    global_state.statistics.data_type_column = each_type

    # Imputation
    if global_state.statistics.missingness:
        imputed_data = imputation(df=data, column_type=each_type, ts=global_state.statistics.time_series)
    else:
        imputed_data = data

    # drop domain index from visual selected features
    if global_state.statistics.heterogeneous and global_state.statistics.domain_index in global_state.user_data.visual_selected_features:
        global_state.user_data.visual_selected_features = [feature for feature in global_state.user_data.visual_selected_features if feature != global_state.statistics.domain_index]
    if global_state.statistics.time_series:
        global_state.user_data.visual_selected_features = [feature for feature in global_state.user_data.visual_selected_features if feature != global_state.statistics.time_index]
    # Check assumption for continuous data
    if global_state.statistics.data_type == "Continuous":
        if global_state.statistics.linearity is None:
            # Update global state
            global_state = linearity_check(df_raw=imputed_data, global_state=global_state)

        if global_state.statistics.gaussian_error is None:
            # Update global state
            global_state = gaussian_check(df_raw=imputed_data, global_state=global_state)
    # Assumption checking for time-series data
    elif global_state.statistics.data_type=="Time-series":
        # estimate the time lag
        if global_state.statistics.time_lag is None:
            global_state.statistics.time_lag = time_series_lag_est(df=imputed_data, nlags=20)
        # check linearity
        global_state = linearity_check_ts(df_raw=imputed_data, global_state=global_state)
        # check gaussianity
        global_state = gaussian_check(df_raw=imputed_data, global_state=global_state)
        # check stationarity
        global_state.statistics.stationary = stationary_check(df=imputed_data, max_test=global_state.statistics.num_test, alpha=global_state.statistics.alpha)        
    else:
        global_state.statistics.linearity = False
        global_state.statistics.gaussian_error = False

    # merge the domain index column back to the data if it exists
    if col_domain_index is not None:
        imputed_data['domain_index'] = col_domain_index
        global_state.statistics.data_type_column['domain_index'] = 'Category'

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
    text += f"The sample size is {statistics.sample_size} with {statistics.feature_number} features. \n\n"
    text += f"This dataset is {'time-series' if statistics.data_type == 'Time-series' else 'not time-series'} data. \n\n"
    text += f"Data Quality: {'There are' if statistics.missingness else 'There are no'} missing values in the dataset.\n\n"
    
    text += "Statistical Properties:\n"
    text += f"- Linearity: The relationships between variables {'are' if statistics.linearity else 'are not'} linear.\n"
    text += f"- Gaussian Errors: The errors in the data {'do' if statistics.gaussian_error else 'do not'} follow a Gaussian distribution.\n"
    
    if statistics.time_series:
        text += f"- Time lag: {statistics.time_lag} \n\n"
        text += f"- Stationarity: The dataset {'is' if statistics.stationary else 'is not'} stationary. \n\n"
    else:
        text += f"- Heterogeneity: The dataset {'is' if statistics.heterogeneous and statistics.domain_index is not None else 'is not'} heterogeneous. \n\n"
        
    if statistics.heterogeneous and statistics.domain_index is not None:
        text += f"- Domain Index: {statistics.domain_index}"
    else:
        text += "\n\n"
        
    return text

def missing_ratio_check(df: pd.DataFrame, nan_indicator):
    missing_vals = [np.nan, nan_indicator]
    missing_mask = df.isin(missing_vals)

    ratio_record = {}
    for column in missing_mask:
        ratio_record[column] = missing_mask[column].mean()

    # LLM determine dropped features
    missing_ratio_dict = {'high': [k for k, v in ratio_record.items() if v >= 0.5], # ratio > 0.5
                    'moderate': [k for k, v in ratio_record.items() if 0.5 > v >= 0.3], # 0.5 > ratio >= 0.3
                    'low': [k for k, v in ratio_record.items() if 0 < v < 0.3] # ratio < 0.3
                    }
    return missing_ratio_dict