import pandas as pd
from sklearn.metrics import mean_absolute_error

# Assuming you have a DataFrame named 'df' with columns: linear, random_forest, decision_tree, ridge, Average prediction, GCSE
prediction_columns = ['linear', 'random_forest', 'decision_tree', 'ridge', 'Average prediction']

real = pd.read_csv('read.csv')

test = pd.read_csv('prediction_test_full_actual.csv')

# Iterate over the prediction columns
for column in prediction_columns:
    # Calculate the mean absolute error between the current prediction column and the 'GCSE' column
    mae = mean_absolute_error(real['GCSE'], test[column])
    print(f"Mean Absolute Error for {column}: {mae}")
