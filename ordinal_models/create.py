import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import re
import joblib
import sys
sys.path.append('/workspaces/subject_predictions') 
import subject_list

# Code to assign model and name of model based on inported model
if 'RandomForestRegresssor' in globals():
    model_name, model = subject_list.get_models(x=globals(), name=subject)
else:
    model_name, model = subject_list.get_models(globals())


subjects = subject_list.prediction_subjects()

results = []
# Iterate over each subject
for subject in subjects:
    print(f'{subject} scores:')

    # Regular expression pattern for matching column names related to the subject
    regex_pattern = re.compile(rf'{subject}.*_real')

    # Load the data for the model (CSV files)
    subject_model = subject + '.csv'
    predictor = pd.read_csv('/workspaces/subject_predictions/model_csvs/' + subject_model)

    # Remove the 'upn' column from the dataset
    predictor.drop(columns=['upn'], inplace=True)

    # Create an empty DataFrame to store encoded columns
    encoded_columns = pd.DataFrame()

    # Iterate over columns in the dataset
    for col in predictor.columns:
        if 'btec' in subject or 'tech_' in subject:
            # Map grades for certain subjects
            grade_mapping = subject_list.grades_mapped()
            if re.match(rf'{subject}.*_(ap1|ap2|real)', col):
                encoded_columns[col] = predictor[col].map(grade_mapping)
        if col == 'gender_ap2':
            # One-hot encode the 'gender_ap2' column
            gender_encoded = pd.get_dummies(predictor[col], prefix=col)
            encoded_columns = pd.concat([encoded_columns, gender_encoded], axis=1)
        else:
            # Keep non-matching columns as they are
            if col not in encoded_columns.columns:
                encoded_columns = pd.concat([encoded_columns, predictor[[col]]], axis=1)

    # Prepare input features (X) and target variable (y)
    X = encoded_columns.drop(columns=[col for col in encoded_columns.columns if isinstance(col, str) and regex_pattern.match(col) and col != fr'{subject}.*_real'])
    X.columns = X.columns.astype(str)

    if 'btec' in subject or 'tech_' in subject:
        y_column_pattern = subject + '.*_real.*$'
    else:
        y_column_pattern = subject + '.*_real$'

    # Assign the target variable (y) based on the pattern
    y = encoded_columns.filter(regex=y_column_pattern)

    if model_name == 'random_forest':
        y = y.values.ravel()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=27)

    ordinal_model = sm.OLS(y_train, sm.add_constant(X_train)).fit()

    # Print the summary of the model
    print(ordinal_model.summary())

    # Now, let's use the model to make predictions on the test data (X_test)
    # Add a constant to X_test for the intercept term
    X_test = sm.add_constant(X_test)

    # Predict the ordinal values for X_test
    predicted_values = ordinal_model.predict(X_test)

    # Convert the predicted values to ordinal categories (e.g., rounding to the nearest integer)
    predicted_categories = np.round(predicted_values).astype(int)

    # Print the predicted ordinal categories
    print("Predicted Categories:")
    print(predicted_categories)