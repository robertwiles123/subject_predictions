# Import necessary libraries
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
from math import sqrt
import re
import subject_list  # Assuming this is a custom module containing subject-related functions
import matplotlib.pyplot as plt
import seaborn as sns

# Get a list of subjects from the subject_list module
subjects = subject_list.prediction_subjects()

# Iterate through each subject
for subject in subjects:
    # Read a CSV file for the current subject
    df = pd.read_csv(f'model_csvs/{subject}.csv')
    print(f'subject: {subject}')
    escaped_subject = re.escape(subject)
    regex_pattern = fr'^{escaped_subject}.*?_ap2$'

    # Find columns in the DataFrame that match a specific pattern
    matching_columns = [col for col in df.columns if re.match(regex_pattern, col)]

    # Check if the matching columns were found in the DataFrame
    if matching_columns:
        # Select the first matching column (you can modify this logic as needed)
        try:
            ap2_column = matching_columns[0]
            # Construct the corresponding 'real' column name
            real_column = ap2_column.replace('_ap2', '_real')

            # If the subject contains 'btec' or 'tech_', map grades using a custom mapping
            if 'btec' in subject or 'tech_' in subject:
                for col in df.columns:
                    grade_mapping = subject_list.grades_mapped()
                    if re.match(rf'{subject}.*_(ap2|real)', col):
                        df[col] = df[col].map(grade_mapping)

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(df[real_column], df[ap2_column])

            # Calculate Root Mean Squared Error (RMSE)
            rmse = sqrt(mse)

            # Calculate R-squared (R2)
            r2 = r2_score(df[real_column], df[ap2_column])

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(df[real_column], df[ap2_column])

            conf_matrix = confusion_matrix(df[real_column], df[ap2_column])

            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"Confusion Matrix for {subject}")
            plt.savefig(f'teacher_scores/{subject} teacher.png')
            plt.clf()

        except ValueError:
            continue

        # Print the results for the current subject
        print(f"Mean Absolute Error (MAE) for {subject}: {mae}")
        print(f"Mean Squared Error (MSE) for {subject}: {mse}")
        print(f"Root Mean Squared Error (RMSE) for {subject}: {rmse}")
        print(f"R-squared (R2) for {subject}: {r2}")
        print()
        print()
        print()

        # Load "scores.csv" into a DataFrame
        scores_df = pd.read_csv('teacher_scores/scores.csv')

        # Check if the subject already exists in "scores.csv"
        if subject in scores_df['subject'].values:
            # Update the existing row with the new values
            scores_df.loc[scores_df['subject'] == subject, ['MSE', 'RMSE', 'R2', 'Mean Absolute Error']] = [mse, rmse, r2, mae]
        else:
            # Add a new row for the subject
            new_row = {'subject': subject, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Mean Absolute Error': mae}
            scores_df = pd.concat([scores_df, pd.DataFrame([new_row])], ignore_index=True)

        # Save the updated DataFrame back to "scores.csv"
        scores_df.to_csv('teacher_scores/scores.csv', index=False)
