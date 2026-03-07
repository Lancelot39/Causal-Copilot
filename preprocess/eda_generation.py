import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.api import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import math
from types import SimpleNamespace
from utils.logger import logger

class EDA(object):
    def __init__(self, global_state):
        """
        :param global_state: a dict containing global variables and information
        """
        # Set consistent matplotlib parameters for all plots
        # plt.rcParams['figure.figsize'] = (18, 14)
        # plt.rcParams['figure.dpi'] = 300
        # plt.rcParams["axes.grid"] = False 
        plt.rcParams["font.family"] = "monospace"
        # plt.rcParams['axes.edgecolor'] = 'black'
        # plt.rcParams['figure.frameon'] = False
        # plt.rcParams['axes.spines.left'] = True
        # plt.rcParams['axes.spines.bottom'] = True
        # plt.rcParams['axes.spines.top'] = False
        # plt.rcParams['axes.spines.right'] = False
        # plt.rcParams['axes.linewidth'] = 1.0
        # plt.rcParams['axes.titlesize'] = 14
        # plt.rcParams['axes.labelsize'] = 10
        # plt.rcParams['xtick.labelsize'] = 9
        # plt.rcParams['ytick.labelsize'] = 9
        
        self.global_state = global_state
        
        intersection_features = list(set(global_state.user_data.processed_data.columns).intersection(
            set(global_state.user_data.visual_selected_features)))
        self.data = global_state.user_data.processed_data[intersection_features]
        self.save_dir = global_state.user_data.output_graph_dir
        
        # Store the original feature size for reference
        self.original_feature_count = len(intersection_features)
        
        # Smart feature selection for visualization
        self._select_visualization_features()
        
        # Identify categorical features
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns
        # Check if the data is time series
        self.is_time_series = global_state.statistics.time_series if hasattr(global_state.statistics, 'time_series') else False
    
    def _select_visualization_features(self, max_features=10):
        """
        Intelligently select features for visualization.
        For large feature sets, select a representative subset.
        
        :param max_features: Maximum number of features to include in visualizations
        """
        if self.data.shape[1] <= max_features:
            return  # No need to limit features
            
        df = self.data.copy()
        
        # If important features are defined, prioritize them
        important_features = []
        if hasattr(self.global_state.user_data, 'important_features'):
            if self.global_state.user_data.important_features is not None:
                important_features = [f for f in self.global_state.user_data.important_features 
                                    if f in df.columns]
        
        # Calculate feature importance or variability
        feature_scores = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric: use coefficient of variation
                if df[col].std() > 0:
                    feature_scores[col] = abs(df[col].std() / df[col].mean()) if df[col].mean() != 0 else df[col].std()
                else:
                    feature_scores[col] = 0
            else:
                # For categorical: use normalized entropy
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 1:
                    entropy = -sum(p * np.log(p) for p in value_counts if p > 0)
                    feature_scores[col] = entropy / np.log(len(value_counts))  # Normalized
                else:
                    feature_scores[col] = 0
        
        # Select features: important ones first, then highest scoring ones
        remaining_slots = max_features - len(important_features)
        if remaining_slots > 0:
            # Sort by score (highest first) and select top remaining_slots
            sorted_features = sorted([(k, v) for k, v in feature_scores.items() 
                                     if k not in important_features],
                                    key=lambda x: x[1], reverse=True)
            selected_features = important_features + [f[0] for f in sorted_features[:remaining_slots]]
        else:
            # If we have more important features than max_features, prioritize them
            selected_features = important_features[:max_features]
        
        # Update self.data with selected features
        self.data = df[selected_features]
    
    def get_optimal_layout(self, n_plots, base_width=5, base_height=4, max_cols=5, min_col_width=3):
        """
        Calculate optimal figure layout based on number of plots
        
        :param n_plots: Number of plots to display
        :param base_width: Base width per plot
        :param base_height: Base height per plot
        :param max_cols: Maximum number of columns
        :param min_col_width: Minimum width per column in inches
        :return: tuple (n_rows, n_cols, figsize)
        """
        # Calculate best layout
        n_cols = min(max_cols, n_plots, max(1, int(plt.rcParams['figure.figsize'][0] / min_col_width)))
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # Calculate appropriate figure size
        # Adjust width based on feature name lengths if we have column names
        if hasattr(self, 'data') and self.data is not None:
            # Estimate space needed for column names
            max_name_len = max([len(str(col)) for col in self.data.columns[:n_plots]]) if n_plots > 0 else 10
            width_factor = max(1, min(2, max_name_len / 10))  # Scale between 1-2x based on name length
            fig_width = min(plt.rcParams['figure.figsize'][0], n_cols * base_width * width_factor)
        else:
            fig_width = min(plt.rcParams['figure.figsize'][0], n_cols * base_width)
            
        fig_height = min(plt.rcParams['figure.figsize'][1], n_rows * base_height)
        
        return n_rows, n_cols, (fig_width, fig_height)

    def plot_dist(self):
        df = self.data.copy()
        
        # Number of features and set the number of plots per row
        num_features = len(df.columns)
        logger.debug(f"Number of features for distribution plots: {num_features}", "EDA")
        plots_per_row = 5
        num_rows = (num_features + plots_per_row - 1) // plots_per_row  # Calculate number of rows needed
        logger.debug(f"Distribution plot layout: {num_rows} rows, {plots_per_row} columns", "EDA")
        # Create a grid of subplots
        #plt.rcParams['font.family'] = 'Times New Roman'
        fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(plots_per_row * 5, num_rows * 4))
        # Flatten the axes array for easy indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        # Plot histogram for each feature
        for i, feature in enumerate(df.columns):
            if i < len(axes):  # Safety check
                sns.set(style="whitegrid")
                sns.histplot(df[feature], ax=axes[i], bins=10, color=sns.color_palette("muted")[0], kde=True)
                # Calculate mean and median
                if feature not in self.categorical_features:
                    mean = df[feature].mean()
                    median = df[feature].median()
                    # Add vertical lines for mean and median
                    axes[i].axvline(mean, color='chocolate', linestyle='--', label='Mean')
                    axes[i].axvline(median, color='midnightblue', linestyle='-', label='Median')

                axes[i].set_title(feature, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Value', fontsize=12)
                if (i+1)%plots_per_row != 0:
                    axes[i].set_ylabel('Frequency', fontsize=10)
                
                # Handle tick label rotation if needed
                if isinstance(feature, str) and len(feature) > 10:
                    axes[i].tick_params(axis='x', rotation=45)

        # Hide any unused subplots
        for j in range(num_features, len(axes)):
            axes[j].set_visible(False)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path_dist = os.path.join(self.save_dir, 'eda_dist.jpg')
        plt.savefig(fname=save_path_dist, dpi=1000)
        plt.close(fig)

        return save_path_dist

    def desc_dist(self):
        df = self.data.copy()
        # Generate descriptive statistics for numerical features
        numerical_stats = df.describe()

        # Analyze categorical features
        categorical_analysis = {feature: df[feature].value_counts() for feature in self.categorical_features}
        numerical_analysis = {}

        for feature in df.select_dtypes(include='number').columns:
            numerical_analysis[feature] = {
                'mean': numerical_stats.loc['mean', feature],
                'median': numerical_stats.loc['50%', feature],
                'std_dev': numerical_stats.loc['std', feature],
                'min_val': numerical_stats.loc['min', feature],
                'max_val': numerical_stats.loc['max', feature]
            }

        return numerical_analysis, categorical_analysis

    def plot_corr(self):
        df = self.data.copy()

        # Initialize the LabelEncoder
        label_encoder = LabelEncoder()
        # Convert categorical features using label encoding
        for feature in self.categorical_features:
            df[feature] = label_encoder.fit_transform(df[feature])
        
        # Calculate the correlation matrix
        correlation_matrix = df.corr()
        # Create a heatmap
        plt.figure(figsize=(8, 6))
        #plt.rcParams['font.family'] = 'Times New Roman'
        sns.heatmap(correlation_matrix, annot=True, cmap='PuBu', fmt=".2f", square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        # Save the plot
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path_corr = os.path.join(self.save_dir, 'eda_corr.jpg')
        plt.savefig(fname=save_path_corr, dpi=1000)

        return correlation_matrix, save_path_corr

    def desc_corr(self, correlation_matrix, threshold=0.1):
        """
        :param correlation_matrix: correlation matrix of the original dataset
        :param threshold: correlation below the threshold will not be included in the summary
        :return: string of correlation_summary
        """
        # Prepare a summary of significant correlations
        correlation_summary = {}

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    var_i, var_j = correlation_matrix.columns[i], correlation_matrix.columns[j]
                    correlation_summary[(var_i, var_j)] = correlation_matrix.iloc[i, j]
        return correlation_summary

    def additional_analysis(self):  
        """
        :return: plot path 
        """
        #sns.set_style("dark")

        orange_black = [
            '#fdc029', '#df861d', '#FF6347', '#aa3d01', '#a30e15', '#800000', '#171820'
        ]

        # Setting plot styling.
        #plt.style.use('ggplot') # https://python-charts.com/matplotlib/styles/

        plt.rcParams['figure.figsize'] = (18, 14)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams["axes.grid"] = False 
        # plt.rcParams["grid.color"] = '#b6b6d3' #orange_black[0]
        # plt.rcParams["grid.alpha"] = 0.5
        # plt.rcParams["grid.linestyle"] = '--'
        # plt.rcParams["font.family"] = "monospace"
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['figure.frameon'] = False
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.bottom'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.linewidth'] = 1.0
        
        df = self.data.copy()
        # maybe address Number-Stored-As-String problem later

        # if column only has 0, 1 --> boolean values, classify it as categorical
        bool_features = df.columns[df.apply(lambda col: col.isin([0, 1]).all())] # find columns with only boolean values 
        num_cols = df.select_dtypes(include=['number']).columns.difference(bool_features)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.union(bool_features)
   
        # Top 6 focused variable's correlational graph
        label_encoders = {}
        # Convert categorical features using label encoding
        for feature in cat_cols:
            label_encoders[feature] = LabelEncoder()
            df[feature] = label_encoders[feature].fit_transform(df[feature])
        
        corr_matrix = df.corr()
        corr_values = corr_matrix.unstack().reset_index()
        corr_values.columns = ['Var1', 'Var2', 'Correlation']
        corr_values = corr_values[corr_values['Var1'] != corr_values['Var2']]
        corr_values = corr_values.drop_duplicates(subset = 'Correlation', keep = 'first').reset_index()

        top_corr = corr_values.iloc[corr_values['Correlation'].abs().sort_values(ascending = False).index, :].head(6)

        # decode labeled data after correlation analysis
        for feature in cat_cols:
            df[feature] = label_encoders[feature].inverse_transform(df[feature])

        # plot subplots based on types of data:
        # num-num --> scatter, num-cat --> violin, cat-cat --> count with legend
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 4 * 2))
        #fig.suptitle("Plots of Top 6 Correlated Focus Variables", fontweight='bold', fontsize=12)
        idx = 0
        axes = axes.reshape(-1)
        for i, row in top_corr.iterrows():
            var1, var2 = row['Var1'], row['Var2']
            corr_value = row['Correlation']
            if var1 in num_cols and var2 in num_cols:
                sns.scatterplot(x=var1, y=var2, data=df, ax = axes[idx], s = 15, color=sns.color_palette("muted")[0])
                axes[idx].set_title(f'Scatterplot (Corr: {corr_value:.2f})', fontsize = 12, fontweight='bold')
                idx += 1
            elif var1 in num_cols and var2 in cat_cols or var1 in cat_cols and var2 in num_cols:
                cat_var = var1 if var1 in cat_cols else var2
                num_var = var1 if var1 in num_cols else var2
                sns.violinplot(x=cat_var, y=num_var, data=df, ax=axes[idx], color=sns.color_palette("muted")[0])
                axes[idx].set_title(f'Violinplot (Corr: {corr_value:.2f})', fontsize = 12, fontweight='bold')
                idx += 1 
            elif var1 in cat_cols and var2 in cat_cols:
                sns.countplot(x=var1, hue=var2, data=df, ax=axes[idx])
                axes[idx].set_title(f'Count Plot (Corr: {corr_value:.2f})', fontsize = 12, fontweight='bold')
                axes[idx].legend(
                    title= " ".join(var2.split("_")[:2]), 
                    loc='best', 
                    frameon=True, 
                    borderpad=0.5, 
                    labelspacing=0.5, 
                    borderaxespad=0.5, 
                    title_fontsize=8,
                    fontsize=6
                )
                idx += 1 

        for ax in axes.reshape(-1):
            ax.xaxis.label.set_size(8)
            ax.tick_params(axis = 'x', labelsize = 6, rotation = 45)
        plt.tight_layout()
        save_path_additional = os.path.join(self.save_dir, 'eda_additional.jpg')
        plt.savefig(save_path_additional, dpi=1000)

        return save_path_additional
    
    def multivariate_time_series_plot(self):
        """
        Create a visualization of multivariate time series data with each variable in its own subplot.
        
        :return: Path to the saved plot
        """
        if not self.is_time_series:
            return None
            
        df = self.data.copy()
        
        # Get number of series
        n_series = df.shape[1]
        
        # Get optimal layout
        n_rows, n_cols, figsize = self.get_optimal_layout(n_series, base_height=3, base_width=4)
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
        
        # Ensure axes is always a 2D array for consistent indexing
        if n_series == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif not hasattr(axes, 'shape'):  # For single subplot case
            axes = np.array([[axes]])
        
        # Create color palette with enough colors
        palette = sns.color_palette("viridis", n_series)
        
        # Create pseudo time index if not available
        time_index = np.arange(df.shape[0])
        
        # Plot each series in its own subplot
        # Sort the columns alphabetically for consistent visualization
        sorted_columns = sorted(df.columns)
        df = df[sorted_columns]
        for i, col in enumerate(df.columns):
            row_idx = i // n_cols
            col_idx = i % n_cols
            ax = axes[row_idx, col_idx]
            
            # Extract series
            series = df[col]
            
            # Plot the series
            ax.plot(time_index, series, color=palette[i], linewidth=1.5, alpha=0.8)
            
            # Add rolling mean if series length is sufficient
            if len(series) > 10:
                window_size = max(5, len(series) // 20)  # Adaptive window size
                rolling_mean = series.rolling(window=window_size).mean()
                ax.plot(time_index, rolling_mean, color='red', linewidth=1, alpha=0.7, 
                       linestyle='--', label='Rolling Mean')
            
            # Style enhancements
            ax.set_title(col, fontsize=11, pad=5)
            ax.grid(color='#E0E0E0', linestyle=':', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Only add x-label for bottom row
            if row_idx == n_rows - 1:
                ax.set_xlabel('Time', labelpad=5)
            
            # Add y-label for leftmost column
            if col_idx == 0:
                ax.set_ylabel('Value', labelpad=5)
        
        # Hide unused subplots
        for i in range(n_series, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            fig.delaxes(axes[row_idx, col_idx])
        
        # Add legend for rolling mean at the bottom of the figure
        handles, labels = [], []
        handles.append(plt.Line2D([0], [0], color='red', linewidth=1, linestyle='--'))
        labels.append('Rolling Mean')
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.0), 
                  ncol=1, frameon=True, fontsize=10)
    
        # Add main title
        fig.suptitle("Multivariate Time Series Analysis", fontweight='bold', fontsize=16, y=1.0)
        
        # Add subtle background color
        fig.patch.set_facecolor('#F8F9FA')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, 'eda_multivariate_ts.jpg')
        plt.savefig(save_path, dpi=1000)
        plt.close(fig)
        
        return save_path
    
    def lag_correlation_heatmap(self, max_lag=5, interval=1):
        """
        Create a heatmap visualization of cross-correlations at different lags.
        
        :param max_lag: Maximum lag to compute cross-correlations for
        :param interval: Interval between lags to compute (e.g., interval=2 will compute lags 0, 2, 4, ...)
        :return: Path to the saved plot
        """
        if not self.is_time_series:
            return None
            
        df = self.data.copy()
        
        # Get variable names and count
        variables = df.columns
        n_vars = len(variables)
        
        # Calculate which lags to use based on interval
        lags_to_use = list(range(0, max_lag + 1, interval))
        if not lags_to_use:  # Ensure at least lag 0 is included
            lags_to_use = [0]
        
        # Prepare result container
        lag_corr_matrix = np.zeros((n_vars, n_vars * len(lags_to_use)))
        
        # Calculate cross-correlation at each lag
        for lag_idx, lag in enumerate(lags_to_use):
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    # Calculate position in the combined matrix
                    col_idx = j + (lag_idx * n_vars)
                    
                    if i == j and lag == 0:
                        lag_corr_matrix[i, col_idx] = 1.0
                    else:
                        # Shift series for lag
                        s1 = df[var1].iloc[:-lag].values if lag > 0 else df[var1].values
                        s2 = df[var2].iloc[lag:].values if lag > 0 else df[var2].values
                        
                        # Compute correlation
                        if len(s1) > 1 and len(s2) > 1:  # Ensure enough data points
                            try:
                                corr, _ = pearsonr(s1, s2)
                                lag_corr_matrix[i, col_idx] = corr
                            except Exception as e:
                                logger.warning(f"Error in correlation calculation for {var1} and {var2} at lag {lag}: {e}")
                                lag_corr_matrix[i, col_idx] = np.nan

        summary = self.generate_lag_correlation_summary(lag_corr_matrix, variables, max_lag)
        # Create figure with appropriate size
        # Adjust figure size based on number of variables and lags
        width = max(10, min(20, 2 * n_vars * len(lags_to_use)))
        height = max(8, min(16, 1.5 * n_vars))
        fig, ax = plt.subplots(figsize=(width, height))
        cmap = sns.color_palette("PuBu", as_cmap=True)
        
        # Create column labels with lag information
        col_labels = []
        for lag in lags_to_use:
            for var in variables:
                col_labels.append(f"{var} (Lag {lag})")
        
        # Adjust tick label size based on number of variables
        tick_fontsize = max(6, min(10, 10 * (10 / max(10, n_vars))))
        
        # Make cells square by adjusting aspect ratio
        aspect = width / height  # Current aspect ratio
        cell_aspect = (n_vars * len(lags_to_use)) / n_vars  # Ideal cell aspect for square cells
        aspect_ratio = aspect / cell_aspect
        
        # Create heatmap with square cells
        sns.heatmap(lag_corr_matrix, ax=ax, cmap=cmap, vmin=-1, vmax=1, center=0,
                  linewidths=.5, cbar_kws={"shrink": .8}, square=True,
                  xticklabels=col_labels, yticklabels=variables)
        
        # Adjust tick labels - increase right margin to prevent truncation
        plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)
        
        # Add vertical lines to separate lag groups
        for lag_idx in range(1, len(lags_to_use)):
            ax.axvline(x=lag_idx * n_vars, color='black', linestyle='-', linewidth=1)
        
        # Add main title with extra padding to prevent truncation
        if interval > 1:
            plt.title(f"Cross-Correlation at Lags (0 to {max_lag}, interval={interval})", fontweight='bold', fontsize=16, pad=20)
        else:
            plt.title("Cross-Correlation at Different Lags", fontweight='bold', fontsize=16, pad=20)
        
        # Add subtle background color to distinguish lag groups
        for lag_idx in range(len(lags_to_use)):
            if lag_idx % 2 == 0:  # Alternate background colors
                ax.axvspan(lag_idx * n_vars, (lag_idx + 1) * n_vars, alpha=0.1, color='lightblue')
        
        # Adjust figure size to ensure square cells
        fig.set_size_inches(width, width / cell_aspect)
        
        # Adjust layout for readability with square cells and prevent truncation
        plt.tight_layout(rect=[0, 0, 0.98, 0.95])  # Leave more space for title and labels
        
        # Add extra space at the bottom for x-labels
        plt.subplots_adjust(bottom=0.15)
        
        # Save the plot
        save_path = os.path.join(self.save_dir, 'eda_lag_correlation.jpg')
        plt.savefig(save_path, dpi=1000, bbox_inches='tight')  # Use bbox_inches to prevent truncation
        plt.close(fig)
        
        return save_path, summary
    
    def generate_lag_correlation_summary(self, lag_corr_matrix, variables, max_lag):
        """
        Generate summary statistics from the lag correlation matrix.
        :return: Dictionary with summary statistics
        """
        n_vars = len(variables)
        summary = {
            'strongest_autocorrelations': [],
            'potential_granger_causality': [],
            'correlation_by_lag': {lag: [] for lag in range(1, max_lag+1)}
        }
        
        # Create a more structured representation of the correlation matrix
        structured_corr = []
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                for lag in range(max_lag + 1):
                    if i == j and lag == 0:
                        continue  # Skip self-correlation at lag 0 (always 1.0)
                    
                    col_idx = j + (lag * n_vars)
                    corr_value = lag_corr_matrix[i, col_idx]
                    
                    structured_corr.append({
                        'var1': var1,
                        'var2': var2,
                        'lag': lag,
                        'correlation': corr_value
                    })
        
        # Analyze autocorrelations (same variable, different lags)
        autocorrelations = [x for x in structured_corr if x['var1'] == x['var2'] and x['lag'] > 0]
        strongest_autocorr = sorted(autocorrelations, key=lambda x: abs(x['correlation']), reverse=True)[:10]
        summary['strongest_autocorrelations'] = strongest_autocorr
        
        # Group correlations by lag for overall lag analysis
        for item in structured_corr:
            if item['lag'] > 0:  # Only interested in actual lags
                summary['correlation_by_lag'][item['lag']].append(abs(item['correlation']))
        
        # Calculate average absolute correlation by lag
        summary['avg_abs_correlation_by_lag'] = {
            lag: np.mean(corrs) if corrs else 0 
            for lag, corrs in summary['correlation_by_lag'].items()
        }
        
        # Find potential Granger causality relationships
        # (where X at lag > 0 has strong correlation with Y)
        cross_correlations = [x for x in structured_corr if x['var1'] != x['var2'] and x['lag'] > 0]
        
        # Group by variable pairs to find maximum correlation at any lag
        var_pairs = {}
        for item in cross_correlations:
            pair_key = (item['var1'], item['var2'])
            if pair_key not in var_pairs or abs(item['correlation']) > abs(var_pairs[pair_key]['correlation']):
                var_pairs[pair_key] = item
        # Sort by absolute correlation and take top relationships
        potential_granger = sorted(var_pairs.values(), key=lambda x: abs(x['correlation']), reverse=True)[:10]
        # Reformat for easier reading
        summary['potential_granger_causality'] = [
            {
                'cause': item['var1'], 
                'effect': item['var2'], 
                'lag': item['lag'], 
                'correlation': item['correlation']
            }
            for item in potential_granger
        ]
        logger.debug(f"Potential Granger causality: {summary['potential_granger_causality']}", "EDA")
        return summary
    
    
    def time_series_diagnostics(self, max_vars=3, max_lags=5, interval=2):
        """
        Create detailed time series diagnostic plots including ACF, PACF, and stationarity information.
        
        :param max_vars: Maximum number of variables to analyze
        :param max_lags: Maximum lags for ACF/PACF
        :param interval: Interval between lags to compute (e.g., interval=2 will compute lags 0, 2, 4, ...)
        :return: Path to the saved plot
        """
        if not self.is_time_series:
            return None
            
        # Import stationarity test
        from statsmodels.tsa.stattools import adfuller
        
        df = self.data.copy()
        
        # Select series to analyze - limit if too many
        if df.shape[1] > max_vars:
            # Try to include important features first
            if hasattr(self.global_state.user_data, 'important_features') and self.global_state.user_data.important_features is not None:
                important_features = [f for f in self.global_state.user_data.important_features 
                                    if f in df.columns][:max_vars]
                remaining = max_vars - len(important_features)
                if remaining > 0:
                    other_features = [f for f in df.columns if f not in important_features][:remaining]
                    columns = important_features + other_features
                else:
                    columns = important_features
            else:
                columns = df.columns[:max_vars]
        else:
            columns = df.columns
            max_vars = len(columns)
        
        summary = {
        "stationary_variables": [],
        "non_stationary_variables": []
    }
        # Reduce number of displayed lags to reduce clutter
        display_lags = min(15, max_lags)
        
        # Create pseudo time index if not available
        time_index = np.arange(df.shape[0])
        
        # Create color palette for the time series
        palette = sns.color_palette("viridis", max_vars)
        
        # Set up the figure - one page per variable for better spacing
        for idx, col in enumerate(columns):
            series = df[col].dropna()
            
            # Create a new figure for each variable - horizontal layout
            plt.figure(figsize=(12, 8))
            
            # Create a 2x3 grid layout
            gs = plt.GridSpec(2, 3, height_ratios=[1, 1.2], width_ratios=[3, 3, 1],
                            wspace=0.25, hspace=0.4)
            
            # === TOP ROW ===
            # Plot 1: Time Series
            ax_series = plt.subplot(gs[0, :2])
            ax_series.plot(time_index, series, color=palette[idx], linewidth=1.8, label=col)
            
            # Add moving average for trend visualization
            if len(series) > 10:
                window_size = max(5, len(series) // 20)
                rolling_mean = series.rolling(window=window_size).mean()
                ax_series.plot(time_index, rolling_mean, color='red', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label='Moving Avg')
            # Calculate and add first difference transform
            first_diff = series.diff().fillna(0)  # First value will be 0
            ax_series.plot(time_index, first_diff, color='darkblue', linestyle='-.', 
                        linewidth=1.5, alpha=0.7, label='First Difference')
            
            # ax_series.set_title(f"Time Series: {col}", fontweight='bold', fontsize=14)
            ax_series.grid(True, color='#E0E0E0', linestyle=':', linewidth=0.5, alpha=0.5)
            ax_series.spines['top'].set_visible(False)
            ax_series.spines['right'].set_visible(False)
            ax_series.set_xlabel("Time", fontsize=10)
            ax_series.set_ylabel("Value", fontsize=10)
            ax_series.legend(loc='upper right', fontsize=9)
            
            # Reduce number of x-ticks
            ax_series.xaxis.set_major_locator(plt.MaxNLocator(6))
            
            # Plot 2: Statistics
            ax_stats = plt.subplot(gs[0, 2])
            ax_stats.axis('off')
            
            # Calculate basic statistics
            stats_text = f"Statistics\n\n"
            stats_text += f"Mean: {series.mean():.2f}\n"
            stats_text += f"Std: {series.std():.2f}\n"
            stats_text += f"Min: {series.min():.2f}\n"
            stats_text += f"Max: {series.max():.2f}\n"
            
            # === BOTTOM ROW ===
            # Plot 3: Stationarity test
            #ax_stationarity = plt.subplot(gs[1, 0])
            
            # try:
            # Perform ADF test
            adf_result = adfuller(series, autolag='AIC')
            adf_stat, p_value, _, _, critical_values, _ = adf_result
            
            # Determine if the series is stationary
            is_stationary = p_value < 0.05
            status_color = 'green' if is_stationary else 'red'
            status_text = "STATIONARY" if is_stationary else "NON-STATIONARY"
                
                # # Plot differenced series if non-stationary to show improvement
                # if not is_stationary:
                #     # Calculate first difference
                #     diff_series = series.diff().dropna()
                #     # Plot original vs differenced
                #     ax_stationarity.plot(time_index[1:], diff_series, color='green', alpha=0.7, 
                #                      label='First Difference')
                #     ax_stationarity.plot(time_index, series, color='gray', alpha=0.3, label='Original')
                #     ax_stationarity.legend(loc='best', fontsize=8)
                #     ax_stationarity.set_title("First Difference Transform", fontsize=12)
                # else:
                #     # Just plot the stationary series
                #     ax_stationarity.plot(time_index, series, color='green', alpha=0.7)
                #     ax_stationarity.set_title("Original Series (Stationary)", fontsize=12)
                
                # # Style the plot
                # ax_stationarity.grid(True, color='#E0E0E0', linestyle=':', linewidth=0.5, alpha=0.5)
                # ax_stationarity.spines['top'].set_visible(False)
                # ax_stationarity.spines['right'].set_visible(False)
                # ax_stationarity.set_xlabel("Time", fontsize=10)
                # ax_stationarity.set_ylabel("Value", fontsize=10)
                # ax_stationarity.xaxis.set_major_locator(plt.MaxNLocator(6))
                
                # Add stationarity text in the plot
            stats_text += f"\n ADF Test:\n\n{status_text}\np-value: {p_value:.4f}"
                # text_box = dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, 
                #               edgecolor=status_color, linewidth=2)
                # ax_stationarity.text(0.05, 0.95, stat_text, transform=ax_stationarity.transAxes,
                #                   fontsize=9, verticalalignment='top', bbox=text_box)
                
            # except:
            #     print(f"Error in stationarity test for {col}")
                # ax_stationarity.text(0.5, 0.5, "Cannot compute stationarity test", 
                #                  ha='center', va='center', fontsize=10)
                # ax_stationarity.axis('off')
            # Add text with a subtle background
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                        verticalalignment='top', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F8F8', 
                                alpha=0.6, edgecolor='#CCCCCC'))
            # Plot 4: ACF
            ax_acf = plt.subplot(gs[1, 0])
            try:
                # Calculate lags to use based on interval
                lags_to_use = list(range(0, display_lags + 1, interval))
                if not lags_to_use:  # Ensure at least lag 0 is included
                    lags_to_use = [0]
                
                # Compute ACF for specified lags
                acf_values = acf(series, nlags=display_lags)
                
                # Plot only the lags we want
                lags_array = np.array(lags_to_use)
                acf_values_filtered = acf_values[lags_array]
                
                # Manual plotting to show only selected lags
                ax_acf.vlines(lags_array, [0], acf_values_filtered, color='#1f77b4', lw=1.5)
                ax_acf.scatter(lags_array, acf_values_filtered, marker='o', color='#1f77b4', s=25)
                
                # Add confidence intervals (approximately 95%)
                conf_level = 1.96 / np.sqrt(len(series))
                ax_acf.axhspan(-conf_level, conf_level, alpha=0.2, color='gray')
                
                # Style the plot
                ax_acf.set_title("ACF" + (f" (interval={interval})" if interval > 1 else ""), fontsize=12)
                ax_acf.grid(True, alpha=0.3)
                ax_acf.spines['top'].set_visible(False)
                ax_acf.spines['right'].set_visible(False)
                ax_acf.set_xlim(-0.5, max(lags_to_use) + 0.5)
                ax_acf.set_ylim(-1.1, 1.1)
                ax_acf.set_xlabel("Lag")
                ax_acf.set_ylabel("Correlation")
            except Exception:
                ax_acf.text(0.5, 0.5, "Cannot compute ACF", ha='center', va='center')
                ax_acf.axis('off')
            
            # Plot 5: PACF
            ax_pacf = plt.subplot(gs[1, 1])
            try:
                # Calculate lags to use based on interval
                lags_to_use = list(range(0, display_lags + 1, interval))
                if not lags_to_use:  # Ensure at least lag 0 is included
                    lags_to_use = [0]
                
                # Compute PACF for specified lags
                pacf_values = pacf(series, nlags=display_lags)
                
                # Plot only the lags we want
                lags_array = np.array(lags_to_use)
                pacf_values_filtered = pacf_values[lags_array]
                
                # Manual plotting to show only selected lags
                ax_pacf.vlines(lags_array, [0], pacf_values_filtered, color='#1f77b4', lw=1.5)
                ax_pacf.scatter(lags_array, pacf_values_filtered, marker='o', color='#1f77b4', s=25)
                
                # Add confidence intervals (approximately 95%)
                conf_level = 1.96 / np.sqrt(len(series))
                ax_pacf.axhspan(-conf_level, conf_level, alpha=0.2, color='gray')
                
                # Style the plot
                ax_pacf.set_title("PACF" + (f" (interval={interval})" if interval > 1 else ""), fontsize=12)
                ax_pacf.grid(True, alpha=0.3)
                ax_pacf.spines['top'].set_visible(False)
                ax_pacf.spines['right'].set_visible(False)
                ax_pacf.set_xlim(-0.5, max(lags_to_use) + 0.5)
                ax_pacf.set_ylim(-1.1, 1.1)
                ax_pacf.set_xlabel("Lag")
                ax_pacf.set_ylabel("Partial Correlation")
            except Exception:
                ax_pacf.text(0.5, 0.5, "Cannot compute PACF", ha='center', va='center')
                ax_pacf.axis('off')
            
            # Add main title for this variable's page
            plt.suptitle(f"Time Series Diagnostics: {col}", fontweight='bold', fontsize=16, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save the plot for this variable
            var_save_path = os.path.join(self.save_dir, f'eda_ts_diagnostics_{idx+1}_{col}.jpg')
            plt.savefig(var_save_path, dpi=1000, bbox_inches='tight')
            plt.close()
        
        ################
        # Create an index page with thumbnails of all the variables
        n_cols = min(2, max_vars)
        n_rows = (max_vars + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        fig.suptitle("Time Series Diagnostics Overview", fontweight='bold', fontsize=16, y=0.98)
        
        # Create thumbnails for the index page
        for i, col in enumerate(columns):
            if i < len(axes):
                # Plot a simplified series for the thumbnail
                axes[i].plot(time_index, df[col], color=palette[i], linewidth=1.5)
                axes[i].set_title(col, fontsize=12)
                axes[i].grid(False)
                axes[i].set_xticklabels([])
                axes[i].set_yticklabels([])
                
                # Perform ADF test for stationarity badge
                try:
                    adf_result = adfuller(df[col].dropna(), autolag='AIC')
                    p_value = adf_result[1]
                    is_stationary = p_value < 0.05
                    status_color = 'green' if is_stationary else 'red'
                    status_text = "STATIONARY" if is_stationary else "NON-STATIONARY"
                    
                    # Add a small badge with stationarity status
                    axes[i].text(0.5, 0.1, status_text, transform=axes[i].transAxes,
                              ha='center', fontsize=8, weight='bold', color='white',
                              bbox=dict(boxstyle="round,pad=0.2", facecolor=status_color, alpha=0.8))
                    # Store result in appropriate list
                    if is_stationary:
                        summary["stationary_variables"].append(col)
                    else:
                        summary["non_stationary_variables"].append(col)
                except Exception:
                    pass
        summary["total_variables_analyzed"] = len(columns)
        summary["total_stationary"] = len(summary["stationary_variables"])
        summary["total_non_stationary"] = len(summary["non_stationary_variables"])
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the index plot
        index_save_path = os.path.join(self.save_dir, 'eda_ts_diagnostics.jpg')
        plt.savefig(index_save_path, dpi=1000, bbox_inches='tight')
        plt.close()
        
        return index_save_path, summary
        
    def analyze_var_model(self, max_lag=3):
        """
        Analyze the time series using a VAR model.
        
        :param max_lag: Maximum lag to use in the VAR model (if None, will use the one from global state)
        :return: Path to the saved plot and VAR model summary
        """
        if not self.is_time_series:
            return None, None
            
        df = self.data.copy()
        
        # Use max_lag from global state if not provided
        if max_lag is None and hasattr(self.global_state.statistics, 'time_lag'):
            max_lag = self.global_state.statistics.time_lag
        elif max_lag is None:
            max_lag = min(5, df.shape[0] // 5)  # Default: 5 or 20% of data points
        
        try:
            # Check if we already have a VAR analysis from stat_info_functions.py
            var_plot_path = os.path.join(self.global_state.user_data.output_graph_dir, 'var_analysis.jpg')
            
            # If the VAR plot already exists from stat_info_functions.py, use it
            # if os.path.exists(var_plot_path):
            #     # Copy the existing plot to our EDA directory
            #     eda_var_path = os.path.join(self.save_dir, 'eda_var_analysis.jpg')
            #     import shutil
            #     shutil.copy(var_plot_path, eda_var_path)
                
            #     # Create a basic summary since we don't have the model results
            #     var_summary = {
            #         'lag_order': max_lag,
            #         'note': 'Using pre-computed VAR analysis from statistical information collection'
            #     }
                
            #     return eda_var_path, var_summary
            
            # If no existing plot, perform the VAR analysis
            # Fit VAR model
            model = VAR(df)
            results = model.fit(maxlags=max_lag)
            
            # Get fitted values and residuals
            fitted = results.fittedvalues
            residuals = results.resid
            
            # Get number of variables
            n_vars = df.shape[1]
            
            # Calculate optimal layout with more space between plots
            # Use fewer columns to allow more horizontal space
            n_cols = min(2, n_vars) if n_vars <= 4 else min(3, n_vars)
            n_rows = math.ceil(n_vars / n_cols)
            
            # Calculate figure size with more space per subplot
            fig_width = n_cols * 4 + 2  # More width per subplot + extra padding
            fig_height = n_rows * 3 + 1  # More height per subplot + extra padding
            
            # Create figure
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            # Create GridSpec with increased spacing between subplots
            gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.8)
            
            for i, col in enumerate(df.columns):
                if i < n_rows * n_cols:
                    row_idx = i // n_cols
                    col_idx = i % n_cols
                    
                    # Create subplot
                    ax = fig.add_subplot(gs[row_idx, col_idx])
                    
                    # Plot with improved visibility
                    ax.scatter(fitted.iloc[:, i], residuals.iloc[:, i], 
                              alpha=0.6, s=45, color='#1f77b4', edgecolor='none')
                    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
                    
                    # Use lighter grid for better readability
                    ax.grid(True, linestyle=':', alpha=0.3, color='#cccccc')
                    
                    # Set clear titles and labels with adequate padding
                    ax.set_title(f"{col}", fontsize=12, fontweight='bold', pad=10)
                    ax.set_xlabel("Fitted Values", fontsize=10, labelpad=8)
                    ax.set_ylabel("Residuals", fontsize=10, labelpad=8)
                    
                    # Add "Residuals vs Fitted" as subtitle with smaller font
                    ax.text(0.5, 0.02, "Residuals vs Fitted", 
                           transform=ax.transAxes, ha='center', fontsize=8, 
                           style='italic', color='#666666')
                    
                    # Clean up appearance
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Ensure adequate padding around plot area
                    x_range = fitted.iloc[:, i].max() - fitted.iloc[:, i].min()
                    y_range = residuals.iloc[:, i].max() - residuals.iloc[:, i].min()
                    ax.set_xlim(fitted.iloc[:, i].min() - 0.1 * x_range, 
                              fitted.iloc[:, i].max() + 0.1 * x_range)
                    ax.set_ylim(residuals.iloc[:, i].min() - 0.1 * y_range, 
                              residuals.iloc[:, i].max() + 0.1 * y_range)
                    
                    # Use fewer tick marks for cleaner look
                    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                    
                    # Adjust tick parameters for better spacing
                    ax.tick_params(axis='both', which='major', labelsize=9, pad=5)
            
            # Add main title with better positioning
            fig.suptitle(f"VAR Model Analysis (Lag={max_lag})", 
                        fontweight='bold', fontsize=16, y=0.98)
            
            # Ensure proper layout with more space
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
            
            # Save the plot
            save_path = os.path.join(self.save_dir, 'eda_var_analysis.jpg')
            plt.savefig(save_path, dpi=1000, bbox_inches='tight')
            plt.close(fig)
            
            # Get model summary
            var_summary = {
                'lag_order': results.k_ar,
                'num_observations': results.nobs,
                'aic': results.aic,
                'bic': results.bic,
                'fpe': results.fpe,
                'hqic': results.hqic
            }
            
            return save_path, var_summary
            
        except Exception as e:
            logger.error(f"Error in VAR analysis: {str(e)}")
            return None, None
        
    def generate_eda(self):
        import joblib
        from joblib import Parallel, delayed
        
        eda_result = {}
        
        if self.is_time_series:
            # Time series specific EDA
            # Run time series analyses in parallel
            results = Parallel(n_jobs=-1)(
                delayed(func)() for func in [
                    # self.plot_dist,
                    # self.plot_corr,
                    # self.desc_dist,
                    # self.multivariate_time_series_plot,
                    lambda: self.lag_correlation_heatmap(
                        max_lag=min(getattr(self.global_state.statistics, 'time_lag', 5), 5)
                    ),
                    lambda: self.time_series_diagnostics(
                        max_vars=4,
                        max_lags=min(getattr(self.global_state.statistics, 'time_lag', 5), 10),
                        interval=getattr(self.global_state.statistics, 'time_lag_interval', 2)
                    ),
                    # self.analyze_var_model
                ]
            )
            
            # Unpack results
            # plot_path_dist = results[0]
            # corr_mat, plot_path_corr = results[1]
            # numerical_analysis, categorical_analysis = results[2]
            # plot_path_multivariate = results[3]
            plot_path_lag_corr = results[0][0]
            lag_corr_summary = results[0][1]
            plot_path_diagnostics = results[1][0]
            diagnostics_summary = results[1][1]
            # plot_path_var, var_summary = results[6]
            
            # Process correlation analysis separately as it depends on corr_mat
            # corr_analysis = self.desc_corr(corr_mat)
            # plot_path_additional = self.additional_analysis()
            
            # Build time series EDA result dictionary
            eda_result = {
                # 'plot_path_dist': plot_path_dist,
                # 'plot_path_corr': plot_path_corr,
                # 'dist_analysis_num': numerical_analysis,
                # 'dist_analysis_cat': categorical_analysis,
                # 'corr_analysis': corr_analysis,
                # 'plot_path_additional': plot_path_additional
            }
            
            # Add time series specific results
            # if plot_path_multivariate:
            #     eda_result['plot_path_multivariate'] = plot_path_multivariate
            if plot_path_lag_corr:
                eda_result['plot_path_lag_corr'] = plot_path_lag_corr
            if lag_corr_summary:
                eda_result['lag_corr_summary'] = lag_corr_summary
            if plot_path_diagnostics:
                eda_result['plot_path_diagnostics'] = plot_path_diagnostics
            if diagnostics_summary:
                eda_result['diagnostics_summary'] = diagnostics_summary
            # if plot_path_var:
            #     eda_result['plot_path_var'] = plot_path_var
            #     eda_result['var_analysis'] = var_summary
                
        else:
            # Regular tabular data EDA
            # Run standard analyses in parallel
            results = Parallel(n_jobs=-1)(
                delayed(func)() for func in [
                    self.plot_dist,
                    self.plot_corr,
                    self.desc_dist,
                    self.additional_analysis
                ]
            )
            
            # Unpack results
            plot_path_dist = results[0]
            corr_mat, plot_path_corr = results[1]
            numerical_analysis, categorical_analysis = results[2]
            plot_path_additional = results[3]
            
            # Process correlation analysis separately as it depends on corr_mat
            corr_analysis = self.desc_corr(corr_mat)
            
            # Build tabular EDA result dictionary
            eda_result = {
                'plot_path_dist': plot_path_dist,
                'plot_path_corr': plot_path_corr,
                'dist_analysis_num': numerical_analysis,
                'dist_analysis_cat': categorical_analysis,
                'corr_analysis': corr_analysis,
                'plot_path_additional': plot_path_additional
            }
        
        # Add feature information
        eda_result['feature_info'] = {
            'total_features': self.original_feature_count,
            'displayed_features': self.data.shape[1]
        }
        
        self.global_state.results.eda = eda_result

def test_timeseries_eda():
    """
    Test function to demonstrate time series EDA functionality.
    Creates dummy time series data and runs the EDA analysis.
    """
    import numpy as np
    import pandas as pd
    import os
    from types import SimpleNamespace
    
    logger.section("Running Time Series EDA Test")
    
    # Create a temporary directory for output
    output_dir = "ts_eda_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dummy time series data (3 variables, 100 time points)
    np.random.seed(42)  # For reproducibility
    n_samples = 100
    
    # Create base signals with different patterns
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Variable 1: Sine wave
    x1 = np.sin(t) + np.random.normal(0, 0.2, n_samples)
    
    # Variable 2: Cosine wave (related to x1 with lag)
    x2 = np.cos(t) + 0.5 * np.roll(x1, 5) + np.random.normal(0, 0.2, n_samples)
    
    # Variable 3: Square wave
    x3 = np.where(np.sin(t/2) > 0, 1, -1) + np.random.normal(0, 0.15, n_samples)
    
    # Variable 4: Exponential growth
    x4 = np.exp(t/10) / 10 + np.random.normal(0, 0.3, n_samples)
    
    # Variable 5: Logarithmic pattern
    x5 = np.log(t + 1) + 0.2 * np.roll(x2, 7) + np.random.normal(0, 0.25, n_samples)
    
    # Variable 6: Sawtooth wave
    x6 = (t % (np.pi/2)) / (np.pi/2) + np.random.normal(0, 0.2, n_samples)
    
    # Variable 7: Damped oscillation
    x7 = np.exp(-t/10) * np.sin(t) + 0.3 * np.roll(x1, 10) + np.random.normal(0, 0.15, n_samples)
    
    # Variable 8: Combined sine waves (different frequencies)
    x8 = 0.5 * np.sin(t) + 0.3 * np.sin(2*t) + 0.2 * np.sin(3*t) + np.random.normal(0, 0.2, n_samples)
    
    # Variable 9: Step function with noise
    x9 = np.where(t > np.median(t), 2, 1) + 0.4 * np.roll(x3, 6) + np.random.normal(0, 0.2, n_samples)
    
    # Variable 10: Polynomial trend
    x10 = 0.01 * t**2 - 0.1 * t + 0.5 * np.roll(x5, 3) + np.random.normal(0, 0.25, n_samples)
    
    # Variable 3: Trend + seasonality
    trend = 0.05 * t
    seasonality = 0.5 * np.sin(t/2)
    x3 = trend + seasonality + 0.3 * np.roll(x2, 3) + np.random.normal(0, 0.15, n_samples)
    
    # Combine into DataFrame
    df = pd.DataFrame({
        f"Signal_{i+1}": x for i, x in enumerate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    })
    
    # Create minimal global state with required attributes
    global_state = SimpleNamespace()
    global_state.user_data = SimpleNamespace()
    global_state.user_data.processed_data = df
    global_state.user_data.visual_selected_features = df.columns.tolist()
    global_state.user_data.output_graph_dir = output_dir
    
    global_state.statistics = SimpleNamespace()
    global_state.statistics.time_series = True
    global_state.statistics.time_lag = 5
    
    global_state.results = SimpleNamespace()
    
    # Create EDA instance
    eda = EDA(global_state)
    
    # Test individual functions
    logger.process("Testing multivariate_time_series_plot")
    mv_path = eda.multivariate_time_series_plot()
    logger.save("Multivariate plot saved", mv_path)
    
    logger.process("Testing lag_correlation_heatmap")
    lag_path, summary = eda.lag_correlation_heatmap(max_lag=5)
    logger.save("Lag correlation heatmap saved", lag_path)
    
    logger.process("Testing time_series_diagnostics")
    diag_path, summary = eda.time_series_diagnostics(max_vars=3, max_lags=15)
    logger.save("Time series diagnostics saved", diag_path)
    
    logger.process("Testing analyze_var_model")
    var_path, var_summary = eda.analyze_var_model(max_lag=3)
    logger.save("VAR model plot saved", var_path)
    logger.debug(f"VAR summary: {var_summary}", "EDA")
    
    logger.process("Testing complete generate_eda method")
    eda.generate_eda()
    logger.success("All time series EDA results saved to global state")
    
    # Log locations of all generated plots
    eda_result = global_state.results.eda
    logger.section("Generated plots")
    for key, value in eda_result.items():
        if key.startswith('plot_path'):
            logger.detail(f"{key}: {value}")
    
    logger.success(f"All test outputs saved to: {output_dir}")
    logger.checkpoint("Time Series EDA Test Completed")
    
    return global_state.results.eda

if __name__ == "__main__":
    test_timeseries_eda()






