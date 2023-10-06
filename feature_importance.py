from sklearn.linear_model import Ridge
import pandas as pd
import subject_list
import numpy as np
import re

subjects = subject_list.prediction_subjects()

for subject in subjects:

    predictor = pd.read_csv('/workspaces/subject_predictions/testing_full_clean/testing.csv')

    predictor.drop(columns='upn', inplace=True)
    # Regular expression pattern for matching column names related to the subject
    regex_pattern = re.compile(rf'{subject}.*_real')

    # Create an empty DataFrame to store encoded columns
    encoded_columns = pd.DataFrame()
    # Iterate over columns in the dataset
    for col in predictor.columns:
        if 'btec' in subject or 'tech_' in col:
            # Map grades for certain subjects
            grade_mapping = subject_list.grades_mapped()
            if re.match(rf'{subject}.*_(ap1|ap2|real)', col):
                encoded_columns[col] = predictor[col].map(grade_mapping)
        if col == 'gender':
            # One-hot encode the 'gender_ap2' column
            gender_encoded = pd.get_dummies(predictor[col], prefix=col)
            encoded_columns = pd.concat([encoded_columns, gender_encoded], axis=1)
        else:
            # Keep non-matching columns as they are
            if col not in encoded_columns.columns:
                encoded_columns[col] = predictor[col]
    encoded_columns.drop(columns='Unnamed: 0', inplace=True)
        
    for col in encoded_columns:
        encoded_columns[col] = encoded_columns[col].fillna(0)
        try:
            encoded_columns[col] = encoded_columns[col].str.replace(['U', 'X'], '0').fillna(0).astype(int)
        except AttributeError:
            continue


    # Prepare input features (X) and target variable (y)
    # Create a list of column names to keep (those that don't match the pattern)
    columns_to_keep = [col for col in encoded_columns.columns if not regex_pattern.search(col)]

    # Create the DataFrame with the desired columns
    X = encoded_columns[columns_to_keep]

    if 'btec' in subject or 'tech_' in subject:
        y_column_pattern = subject + '.*_real.*$'
    else:
        y_column_pattern = subject + '.*_real$'
        

    # Assign the target variable (y) based on the pattern
    y_columns = [col for col in encoded_columns.columns if regex_pattern.search(col)]
    y = encoded_columns[y_columns]
    # Create a Ridge model with a specific alpha (regularization strength)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)

        # Get the coefficients of the Ridge model
    coefficients = ridge_model.coef_[0]

    # Create a DataFrame to store helping variables and their coefficients
    variable_coeff_df = pd.DataFrame({'Variable': columns_to_keep, 'Coefficient': coefficients})

    # Sort the DataFrame by the absolute values of coefficients in descending order
    variable_coeff_df = variable_coeff_df.reindex(variable_coeff_df['Coefficient'].abs().sort_values(ascending=False).index)

    # Separate helping and not helping variables based on the threshold (e.g., 1e-10)
    helping_variables = variable_coeff_df[variable_coeff_df['Coefficient'].abs() >= 1e-10]
    not_helping_variables = variable_coeff_df[variable_coeff_df['Coefficient'].abs() < 1e-10]

    # Print the sorted DataFrame of helping variables
    print(f"Helping Variables for {subject} (sorted by coefficient magnitude):")
    print(helping_variables)