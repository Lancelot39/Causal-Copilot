import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

class EDA(object):
    def __init__(self, global_state):
        """
        :param global_state: a dict containing global variables and information
        """
        self.global_state = global_state
        self.data = global_state.user_data.processed_data[global_state.user_data.visual_selected_features]
        self.save_dir = global_state.user_data.output_graph_dir
        # limit the number of features contained
        if self.data.shape[1] > 10:
            df = self.data.copy()
            # choose 10 columns randomly
            random_columns = np.random.choice(df.columns, size=10, replace=False)
            self.data = df[random_columns]
        # Identify categorical features
        self.categorical_features = self.data.select_dtypes(include=['object', 'category']).columns

    def plot_dist(self):
        df = self.data.copy()
        # limit the number of features contained
        if df.shape[1] > 10:
            # choose 10 columns randomly
            random_columns = np.random.choice(df.columns, size=10, replace=False)
            df = df[random_columns]

        # Number of features and set the number of plots per row
        num_features = len(df.columns)
        plots_per_row = 5
        num_rows = (num_features + plots_per_row - 1) // plots_per_row  # Calculate number of rows needed

        # Create a grid of subplots
        #plt.rcParams['font.family'] = 'Times New Roman'
        fig, axes = plt.subplots(nrows=num_rows, ncols=plots_per_row, figsize=(plots_per_row * 5, num_rows * 4))

        # Flatten the axes array for easy indexing
        axes = axes.flatten()

        # Plot histogram for each feature
        for i, feature in enumerate(df.columns):
            #plt.rcParams['font.family'] = 'Times New Roman'
            sns.set(style="whitegrid")
            sns.histplot(df[feature], ax=axes[i], bins=10, color=sns.color_palette("muted")[0], kde=True)
            # Calculate mean and median
            if feature not in self.categorical_features:
                mean = df[feature].mean()
                median = df[feature].median()
                # Add vertical lines for mean and median
                axes[i].axvline(mean, color='chocolate', linestyle='--', label='Mean')
                axes[i].axvline(median, color='midnightblue', linestyle='-', label='Median')

            axes[i].set_title(feature, fontsize=16, fontweight='bold')
            axes[i].set_xlabel('Value', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Frequency', fontsize=14, fontweight='bold')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path_dist = os.path.join(self.save_dir, 'eda_dist.jpg')
        plt.savefig(fname=save_path_dist, dpi=1000)

        return save_path_dist

    def desc_dist(self):
        df = self.data.copy()
        # Generate descriptive statistics for numerical features
        numerical_stats = df.describe()

        # Analyze categorical features
        categorical_analysis = {feature: df[feature].value_counts() for feature in self.categorical_features}
        numerical_analysis = {}
        #analysis_input = "Numerical Features:\n"

        for feature in df.select_dtypes(include='number').columns:
            numerical_analysis[feature] = {
                'mean': numerical_stats.loc['mean', feature],
                'median': numerical_stats.loc['50%', feature],
                'std_dev': numerical_stats.loc['std', feature],
                'min_val': numerical_stats.loc['min', feature],
                'max_val': numerical_stats.loc['max', feature]
            }
        #     mean = numerical_stats.loc['mean', feature]
        #     median = numerical_stats.loc['50%', feature]
        #     std_dev = numerical_stats.loc['std', feature]
        #     min_val = numerical_stats.loc['min', feature]
        #     max_val = numerical_stats.loc['max', feature]

        #     analysis_input += (
        #         f"Feature: {feature}\n"
        #         f"Mean: {mean:.2f}, Median: {median:.2f}, Standard Deviation: {std_dev:.2f}\n"
        #         f"Min: {min_val}, Max: {max_val}\n\n"
        #     )

        # analysis_input += "\nCategorical Features:\n"

        # for feature in self.categorical_features:
        #     counts = df[feature].value_counts()
        #     analysis_input += f"Feature: {feature}\n{counts}\n\n"

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
                    # correlation_summary += (
                    #     f"Correlation between {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: "
                    #     f"{correlation_matrix.iloc[i, j]:.2f}\n"
                    # )
        return correlation_summary

    def generate_eda(self):
        plot_path_dist = self.plot_dist()
        corr_mat, plot_path_corr = self.plot_corr()

        numerical_analysis, categorical_analysis = self.desc_dist()
        corr_analysis = self.desc_corr(corr_mat)

        eda_result = {'plot_path_dist': plot_path_dist,
                      'plot_path_corr': plot_path_corr,
                      'dist_analysis_num': numerical_analysis,
                      'dist_analysis_cat': categorical_analysis,
                      'corr_analysis': corr_analysis}

        self.global_state.results.eda = eda_result






