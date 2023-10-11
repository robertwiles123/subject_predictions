# from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import subject_list
import numpy as np
import re
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

subjects = subject_list.prediction_subjects()

for subject in subjects:

    predictor = pd.read_csv('/workspaces/subject_predictions/testing_full_clean/testing.csv')

    predictor.drop(columns='upn', inplace=True)
    # Regular expression pattern for matching column names related to the subject
    regex_pattern = re.compile(rf'{subject}.*_real')

    matching_columns = [col for col in predictor.columns if regex_pattern.match(col)]

    # Drop columns that don't match the pattern

    columns_to_drop = [col for col in predictor.columns if '_real' in col and not regex_pattern.match(col)]

    predictor = predictor.drop(columns=columns_to_drop)

 
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
    try:
        encoded_columns.drop(columns='Unnamed: 0', inplace=True)
    except KeyError:
        continue
        
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
    ridge_model = RandomForestRegressor(
    n_estimators=100,          # Number of trees in the forest
    criterion='squared_error',           # Mean squared error (you can also try 'mae' for mean absolute error)
    max_depth=None,            # Maximum depth of each tree (None means nodes are expanded until all leaves are pure or contain less than min_samples_split samples)
    min_samples_split=2,       # Minimum number of samples required to split an internal node
    min_samples_leaf=1,        # Minimum number of samples required to be at a leaf node
    max_features='sqrt',       # Number of features to consider when looking for the best split ('auto' is sqrt(n_features))
    random_state=42,           # Seed for random number generation (for reproducibility)
    n_jobs=-1                  # Number of CPU cores to use for parallelism (-1 means use all available cores)
)
    ridge_model.fit(X, y)

        # Get the coefficients of the Ridge model
    coefficients = ridge_model.feature_importances_

    # Create a DataFrame to store helping variables and their coefficients
    variable_coeff_df = pd.DataFrame({'Variable': columns_to_keep, 'Coefficient': coefficients})

    # Sort the DataFrame by the absolute values of coefficients in descending order
    variable_coeff_df = variable_coeff_df.reindex(variable_coeff_df['Coefficient'].abs().sort_values(ascending=False).index)

    # Separate helping and not helping variables based on the threshold (e.g., 1e-10)
    helping_variables = variable_coeff_df[variable_coeff_df['Coefficient'].abs() >= 1e-10]

    # Print the sorted DataFrame of helping variables
    print(f"Helping Variables for {subject} (sorted by coefficient magnitude):")
    print(helping_variables)
    print()
    print()
    helping_variable_names = helping_variables['Variable'].tolist()

    try:
        # Subset the original feature matrix X to include only the helping variables
        X_helping = X[helping_variable_names]
        X_train, X_test, y_train, y_test = train_test_split(X_helping, y, test_size=0.3, random_state=27)

        # Fit the model
        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)

        # Round predictions to whole numbers
        y_pred_rounded = np.round(y_pred)
        y_pred_true = y_pred_rounded.astype(int)

        # Calculate various evaluation metrics
        mse = mean_squared_error(y_test, y_pred_true)
        rmse = mean_squared_error(y_test, y_pred_true, squared=False)
        r2 = r2_score(y_test, y_pred_true)

        # Perform cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=68)
        scores = cross_val_score(ridge_model, X, y, cv=kf, scoring='r2')
        scores.sort()
        cross_val_mean = np.mean(scores)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print("Mean Absolute Error:", mae)
        print(f"R-squared (R2) Score: {r2:.2f}")
        print('Cross-validation scores:', scores)
        print(f'Mean cross-validation scores: {cross_val_mean}')

    # Dictionary to save scores
        scores_dict = {
            'subject': subject,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'Mean Absolute Error': mae,
        }

        # Load an existing scores CSV file
        scores_df = pd.read_csv(f'feature_scores/scores.csv')

        # Create a mask to filter out rows with the same subject
        mask = scores_df['subject'] != subject

        # Remove rows with the same subject from scores_df
        scores_df = scores_df[mask]

        new_scores_df = pd.DataFrame([scores_dict])

        # Update the scores DataFrame with new scores
        scores_df = pd.concat([scores_df, new_scores_df], ignore_index=True)
        scores_df.reset_index(drop=True, inplace=True)

        # Sort the DataFrame by the "subject" column
        scores_df = scores_df.sort_values(by='subject')

        # Save the updated scores DataFrame to a CSV file
        scores_df.to_csv('feature_scores/scores.csv', index=False, float_format='%.3f')

        print('Scores updated in scores.csv')
    except ValueError:
        continue