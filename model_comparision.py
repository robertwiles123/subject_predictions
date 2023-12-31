import numpy as np
from scipy import stats
import pandas as pd

# Load the data from the CSV files into DataFrames
ridge_df = pd.read_csv('/workspaces/subject_predictions/ridge_scores/scores.csv')
# linear_df = pd.read_csv('/workspaces/subject_predictions/linear_regression_scores/scores.csv')
# random_forest_df = pd.read_csv('/workspaces/subject_predictions/random_forest_scores/scores.csv')
teacher_df = pd.read_csv('teacher_scores/scores.csv')
# xbg_df = pd.read_csv('xgb_scores/scores.csv')
# svr_df = pd.read_csv('svr_scores/scores.csv')
# bridge_df = pd.read_csv('bridge_scores/scores.csv')
# feature_df = pd.read_csv('feature_scores/scores.csv')

df1 = ridge_df
df2 = teacher_df

try:
    if df1.equals(teacher_df) or df2.equals(teacher_df):
        metric_names = ['MSE', 'RMSE', 'R2', 'Mean Absolute Error']
except NameError:
    try:
        if df1.equals(feature_df) or df2.equals(feature_df):
            metric_names = ['MSE', 'RMSE', 'R2', 'Mean Absolute Error']
    except NameError:
            metric_names = ['MSE', 'RMSE', 'R2', 'Mean cross validation', 'Mean Absolute Error']

if len(metric_names) == 0:
    metric_names = ['MSE', 'RMSE', 'R2', 'Mean Absolute Error']

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
for metric in metric_names:
    print(f"{metric}: df1={df1_metrics[:, metric_names.index(metric)].mean():.4f}, df2={df2_metrics[:, metric_names.index(metric)].mean():.4f}")
