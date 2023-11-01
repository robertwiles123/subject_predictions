import pandas as pd
import re
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import subject_list
import numpy as np
from sklearn.metrics import make_scorer, mean_squared_error

# Define a list of subjects
subjects = subject_list.prediction_subjects()

# Create a DataFrame to store the alpha scores
alpha_scores = []

def rmse_scorer(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

# Create a scorer from the custom RMSE scorer
rmse_scorer = make_scorer(rmse_scorer, greater_is_better=False)

# Loop through the subjects
for subject in subjects:
    print(subject)
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
            if re.match(rf'{subject}.*', col):
                encoded_columns[col] = predictor[col].map(grade_mapping)
        if col in subject_list.columns_encode():
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

    # Define a range of alpha values to search
    alphas = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100]
 # Create the Ridge regression model
    ridge = Ridge()

    # Perform grid search to find the best alpha
    param_grid = {'alpha': alphas}
    grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring=rmse_scorer)  # You can change the scoring metric
    y_rounded = np.round(y)
    grid_search.fit(X, y_rounded)

    # Get the best alpha from the grid search
    best_alpha = grid_search.best_params_['alpha']
    ridge = Ridge(alpha=best_alpha)
    # Append the results to the list
    alpha_scores.append({'subject': subject, 'alpha': best_alpha})
    ridge.fit(X, y)
    feature_coefficients = ridge.coef_
    feature_names = X.columns.tolist()

    for feature, coefficient in zip(feature_names, ridge.coef_):
        print(f"{feature}: {coefficient:.4f}")

# Create a DataFrame from the list
alpha_scores_df = pd.DataFrame(alpha_scores)

# Save the alpha scores to a CSV file
alpha_scores_df.to_csv('/workspaces/subject_predictions/models/ridge_params.csv', index=False)
print('alpha saved')
