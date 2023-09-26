import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import GridSearchCV
import subject_list
import re

# Define subject and year here
subjects = subject_list.prediction_subjects()

results_df = pd.DataFrame(columns=['subject', 'max_depth', 'n_estimators', 'min_samples_split',
           'min_samples_leaf', 'max_features', 'max_samples', 'bootstrap',
           'random_state'])

for subject in subjects:
    # regex for none standard grade endings
    regex_pattern = re.compile(rf'{subject}.*_real')

    # Load the data for the model
    subject_model = subject + '.csv'
    predictor = pd.read_csv('model_csvs/' + subject_model)

    predictor.drop(columns=['upn'], inplace=True)

    # Create an empty DataFrame to store the encoded columns
    encoded_columns = pd.DataFrame()

    for col in predictor.columns:
        if 'btec' in subject or 'tech_' in subject:
            grade_mapping = subject_list.grades_mapped()
            if re.match(rf'{subject}.*_(ap1|ap2|real)', col):
                encoded_columns[col] = predictor[col].map(grade_mapping)
        if col == 'gender_ap2':
            # Use pd.get_dummies for the gender column
            gender_encoded = pd.get_dummies(predictor[col], prefix=col)
            encoded_columns = pd.concat([encoded_columns, gender_encoded], axis=1)
        else:
            # Keep non-matching columns as they are
            if col not in encoded_columns.columns:
                encoded_columns = pd.concat([encoded_columns, predictor[[col]]], axis=1)

    X = encoded_columns.drop(columns=[col for col in encoded_columns.columns if isinstance(col, str) and regex_pattern.match(col) and col != fr'{subject}.*_real'])

    X.columns = X.columns.astype(str)

    if 'btec' in subject or 'tech_' in subject:
        y_column_pattern = subject + '.*_real.*$'
    else:
        # assign patter that is needed for this subject
        y_column_pattern = subject + '.*_real$'
    
    # assin y based on pattern
    y = encoded_columns.filter(regex=y_column_pattern)

    y = y.values.ravel()


    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'max_samples': [None, 0.5, 0.8],
        'bootstrap': [True, False],
        'random_state': [42]
    }


    # Starting model
    rf_model = RandomForestRegressor()

   # Perform grid search with cross-validation
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # Print the best hyperparameters and corresponding score
    print(f"Best hyperparameters for {subject}:")
    print(grid_search.best_params_)
    print("Best Score (Negative MSE):", -grid_search.best_score_)

    # Append the results to the DataFrame
    best_params = grid_search.best_params_
    best_params['subject'] = subject


    df_to_add = pd.DataFrame([best_params])
    results_df = pd.concat([results_df, df_to_add], ignore_index=True)

# Save the DataFrame to a CSV file
results_df.to_csv('/workspace/subject_predictions/models/params.csv', index=False)
print('Saved')





