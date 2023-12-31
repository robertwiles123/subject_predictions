import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
#from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import subject_list



subjects = subject_list.prediction_subjects()

results = []
# Iterate over each subject
for subject in subjects:

    model_name, model = subject_list.get_models(globals(), name=subject)

    print(f'{subject} scores:')

    # Regular expression pattern for matching column names related to the subject
    regex_pattern = re.compile(rf'{subject}.*_real')

    # Load the data for the model (CSV files)
    subject_model = subject + '.csv'
    predictor = pd.read_csv('model_csvs/' + subject_model)

    # Remove the 'upn' column from the dataset
    predictor.drop(columns=['upn'], inplace=True)

    # Create an empty DataFrame to store encoded columns
    encoded_columns = pd.DataFrame()

    # Iterate over columns in the dataset
    for col in predictor.columns:
        if 'btec' in subject or 'tech_' in subject:
            if re.match(rf'{subject}.*', col) or col == 'fft':
                grade_mapping = subject_list.grades_mapped()
                encoded_columns[col] = predictor[col].map(grade_mapping)
        if col in subject_list.columns_encode():
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Round predictions to whole numbers
    y_pred_rounded = np.round(y_pred)
    y_pred_true = y_pred_rounded.astype(int)

    # Calculate various evaluation metrics
    mse = mean_squared_error(y_test, y_pred_true)
    rmse = mean_squared_error(y_test, y_pred_true, squared=False)
    r2 = r2_score(y_test, y_pred_true)

    # Perform cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=68)
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    scores.sort()
    cross_val_mean = np.mean(scores)
    mae = mean_absolute_error(y_test, y_pred_true)

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
        'Cross-validation': scores,  # Convert to list
        'Mean cross-validation': cross_val_mean  # Store the mean value directly
    }

    # Load an existing scores CSV file
    scores_df = pd.read_csv(f'{model_name}_scores/scores.csv')

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
    scores_df.to_csv(f'{model_name}_scores/scores.csv', index=False, float_format='%.3f')

    print('Scores updated in scores.csv')

    # to help with analysis using ordinal testing
    true_grades = y_test
    true_grades = np.array(true_grades).astype(int)
    model_predictions = y_pred_true
    model_predictions = np.array(model_predictions).astype(int)

    teacher_predictions_full = X_test.filter(regex=f'{subject}.*_ap2').values

    # Calculate accuracy for model predictions
    model_accuracy = accuracy_score(true_grades, model_predictions)
    # Calculate accuracy for teacher predictions
    teacher_df = pd.read_csv(f'model_csvs/{subject}_2223.csv')
    teacher_predictions = teacher_df[f'{subject}_ap2']
    teacher_true = teacher_df[f'{subject}_real']
    teacher_accuracy = accuracy_score(teacher_true, teacher_predictions)

    # Count predictions off by one or two or more
    off_by_one_model = sum(abs(true - pred) == 1 for true, pred in zip(true_grades, model_predictions))
    off_by_more_than_two_model = sum(abs(true - pred) > 2 for true, pred in zip(true_grades, model_predictions))

    off_by_one_teacher = sum(abs(true - pred) == 1 for true, pred in zip(true_grades, teacher_predictions_full))
    off_by_more_than_two_teacher = sum(abs(true - pred) > 2 for true, pred in zip(true_grades, teacher_predictions_full))

    results.append({
        "subject": subject,
        "Model_Accuracy": model_accuracy,
        "Teacher_Accuracy": teacher_accuracy,
        "Model_Off_One": off_by_one_model,
        "Model_Off_2+": off_by_more_than_two_model,
        "Teacher_Off_One": off_by_one_teacher,
        "Teacher_Off_2+": off_by_more_than_two_teacher
    })

    # Create a learning curve graph
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the learning curve
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training score')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test score')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xlabel('Number of training samples')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.ylim([0, 1])

    # Show the learning curve graph

    # Save the learning curve graph as an image
    plt.savefig(f"{model_name}_scores/{subject}_{model_name}.png")
    plt.clf()
    print('Learning curve graph saved')

    # Refit the model on the entire dataset
    model.fit(X, y)

    # Save the trained model as a joblib file
    joblib.dump(model, f'models/{subject}_{model_name}.pkl')

    print(f'{subject} Model saved')
    conf_matrix = confusion_matrix(y_test, y_pred_true)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {subject}")
    plt.savefig(f'{model_name}_scores/{subject}_confusion.png')
    plt.clf()
    
    print()
    print()
    """
    # Get the coefficients of the Ridge model
    coefficients = model.coef_[0]

    # Create a DataFrame to store helping variables and their coefficients
    variable_coeff_df = pd.DataFrame({'Variable': X, 'Coefficient': coefficients})

    # Sort the DataFrame by the absolute values of coefficients in descending order
    variable_coeff_df = variable_coeff_df.reindex(variable_coeff_df['Coefficient'].abs().sort_values(ascending=False).index)

    # Separate helping and not helping variables based on the threshold (e.g., 1e-10)
    helping_variables = variable_coeff_df[variable_coeff_df['Coefficient'].abs() >= 1e-10]
    not_helping_variables = variable_coeff_df[variable_coeff_df['Coefficient'].abs() < 1e-10]

    # Print the sorted DataFrame of helping variables
    print(f"Helping Variables for {subject} (sorted by coefficient magnitude):")
    print(helping_variables)
    print("\nNot Helping Variables (coefficients close to zero):")
    print(not_helping_variables)
    """

# Save the DataFrame to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(f"accuracy_scores/{model_name}_accuracy.csv", index=False)
print('Accuracy csv saved')