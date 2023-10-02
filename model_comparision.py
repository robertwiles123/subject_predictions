import numpy as np
from scipy import stats
import pandas as pd

# Load the data from the CSV files into DataFrames
ridge_df = pd.read_csv('/workspaces/subject_predictions/ridge_scores/scores.csv')
#linear_df = pd.read_csv('/workspaces/subject_predictions/linear_regression_scores/scores.csv')
random_forest_df = pd.read_csv('/workspaces/subject_predictions/random_forest_scores/scores.csv')
# teacher_df = pd.read_csv('teacher_scores/scores.csv')

df1 = ridge_df
df2 = random_forest_df

try:
    if df1.equals(teacher_df) or df2.equals(teacher_df):
        metric_names = ['MSE', 'RMSE', 'R2']
except NameError:
    metric_names = ['MSE', 'RMSE', 'R2', 'Mean cross validation', 'Mean Absolute Error']

# Extract the performance metrics as NumPy arrays
df1_metrics = df1[metric_names].values
df2_metrics = df2[metric_names].values

# Calculate the differences in performance metrics
differences = df1_metrics - df2_metrics

zeros_array = np.zeros(differences.shape)

# Paired t-test
t_statistic, p_value_t = stats.ttest_rel(differences, zeros_array)

# Wilcoxon signed-rank test
w_statistic, p_value_w = stats.wilcoxon(differences, zero_method='wilcox')

results_df = pd.DataFrame({'metric': metric_names, 'T-test': p_value_t, 'Wilcox': p_value_w})

print(results_df.to_string(index=False))

# Define a significance level
alpha = 0.05

# Select metrics where either Wilcoxon signed-rank test or paired t-test p-value is below alpha
selected_metrics = [metric for i, metric in enumerate(metric_names) if p_value_w[i] < alpha or p_value_t[i] < alpha]

# Calculate the average performance scores for selected metrics
average_scores = np.mean(df1_metrics[:, [i for i, metric in enumerate(metric_names) if metric in selected_metrics]], axis=1)

# Report individual metrics and their averages
for metric in selected_metrics:
    print(f"{metric}: df1={df1_metrics[:, metric_names.index(metric)].mean():.4f}, df2={df2_metrics[:, metric_names.index(metric)].mean():.4f}")
