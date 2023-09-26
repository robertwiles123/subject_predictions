import numpy as np
from scipy import stats
import pandas as pd

# Load the data from the CSV files into DataFrames
ridge_df = pd.read_csv('/workspace/subject_predictions/ridge_scores/scores.csv')
# linear_df = pd.read_csv('/workspace/subject_predictions/linear_regression_scores/scores.csv')
random_forest_df = pd.read_csv('/workspace/subject_predictions/random_forest_scores/scores.csv')

# Extract the performance metrics as NumPy arrays
#linear_regression_metrics = linear_df[['MSE', 'RMSE', 'R2', 'Mean cross validation', 'Mean Absolute Error']].values
ridge_regression_metrics = ridge_df[['MSE', 'RMSE', 'R2', 'Mean cross validation', 'Mean Absolute Error']].values
random_forest__metrics = random_forest_df[['MSE', 'RMSE', 'R2', 'Mean cross validation', 'Mean Absolute Error']].values

# Calculate the differences in performance metrics
differences = ridge_regression_metrics - random_forest__metrics

zeros_array = np.zeros(differences.shape)


# Paired t-test
t_statistic, p_value_t = stats.ttest_rel(differences, zeros_array)

# Wilcoxon signed-rank test
w_statistic, p_value_w = stats.wilcoxon(differences, zero_method='wilcox')

print(f"Paired t-test p-value: {p_value_t}")
print(f"Wilcoxon signed-rank test p-value: {p_value_w}")
