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
        
        intersection_features = list(set(global_state.user_data.processed_data.columns).intersection(
            set(global_state.user_data.visual_selected_features)))
        self.data = global_state.user_data.processed_data[intersection_features]
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
        print(num_features)
        plots_per_row = 5
        num_rows = (num_features + plots_per_row - 1) // plots_per_row  # Calculate number of rows needed
        print(num_rows)
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
        plt.rcParams["font.family"] = "monospace"
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
    
    def generate_eda(self):
        plot_path_dist = self.plot_dist()
        corr_mat, plot_path_corr = self.plot_corr()

        numerical_analysis, categorical_analysis = self.desc_dist()
        corr_analysis = self.desc_corr(corr_mat)
        plot_path_additional = self.additional_analysis()

        eda_result = {'plot_path_dist': plot_path_dist,
                      'plot_path_corr': plot_path_corr,
                      'dist_analysis_num': numerical_analysis,
                      'dist_analysis_cat': categorical_analysis,
                      'corr_analysis': corr_analysis,
                      'plot_path_additional': plot_path_additional}

        self.global_state.results.eda = eda_result






