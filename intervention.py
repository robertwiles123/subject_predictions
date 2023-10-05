import pandas as pd
import subject_list
import joblib
import re
import numpy as np

subjects = subject_list.prediction_subjects()

for subject in subjects:
    # import csv file
    full_df = pd.read_csv(f'model_csvs/{subject}.csv')

    print(subject)

    # change non numerical to numerical
    if 'btec' in subject or 'tech_' in subject:
        grade_mapping = subject_list.grades_mapped()
        for col in full_df.columns:
            if re.match(rf'{subject}.*_(ap1|ap2|real)', col):
                full_df[col] = full_df[col].map(grade_mapping)

    # change to target after predicted included as this will allow for comparing to targe grade               
    regex_target = re.compile(rf'{subject}.*_real')
    regex_ap2 = re.compile(rf'{subject}.*_ap2')


    # Find columns matching the regex patterns
    target_columns = [col for col in full_df.columns if regex_target.match(col)]
    ap2_columns = [col for col in full_df.columns if regex_ap2.match(col)]

    #load model
    model = joblib.load(f'models/{subject}_ridge.pkl')

    encoded_columns = pd.DataFrame()

    for col in full_df.columns:
        if col == 'gender_ap2':
            # Use pd.get_dummies for the gender column
            gender_encoded = pd.get_dummies(full_df[col], prefix=col)
            encoded_columns = pd.concat([encoded_columns, gender_encoded], axis=1)
        else:
            # Keep non-matching columns as they are
            if col not in encoded_columns.columns:
                encoded_columns = pd.concat([encoded_columns, full_df[[col]]], axis=1)


    regex_real = re.compile(rf'{subject}.*_real')

    # Columns to exclude from dropping
    columns_to_exclude = ['upn']

    # Create a list of columns to drop based on the regex pattern and exclusion list
    columns_to_drop = [col for col in encoded_columns.columns if regex_real.match(col) or col in columns_to_exclude]

    # Drop the selected columns from the DataFrame
    X_prediction = encoded_columns.drop(columns=columns_to_drop)
    y_prediction = model.predict(X_prediction)

    y_prediction_rounded = np.round(y_prediction)
    y_prediction_rounded = y_prediction_rounded.astype(int)


    y_prediction_rounded_flat = y_prediction_rounded.ravel()

    # Create a new DataFrame with the 'prediction' column
    prediction_df = pd.DataFrame({'prediction': y_prediction_rounded_flat})
    # Concatenate the 'prediction_df' with 'X_prediction' horizontally
    result_df = pd.concat([X_prediction, prediction_df], axis=1)
    # Calculate the difference between "real" and "ap2" columns

    result_df[target_columns] = full_df[target_columns]
    result_df['upn'] = full_df['upn']

    for target_col, ap2_col in zip(target_columns, ap2_columns):
        result_df['teacher_difference'] = result_df[target_col] - result_df[ap2_col]
        result_df['model_difference'] = result_df[target_col] - result_df['prediction']

    # Calculate the 25th percentile for 'teacher_difference' and 'model_difference'
    teacher_difference_q1 = result_df['teacher_difference'].quantile(0.25)
    model_difference_q1 = result_df['model_difference'].quantile(0.25)

    # Create boolean masks for the lowest quartile for each column
    teacher_mask = result_df['teacher_difference'] <= teacher_difference_q1
    model_mask = result_df['model_difference'] <= model_difference_q1

    # Use boolean indexing to filter rows in the lowest quartile for both columns
    upn_model = result_df[model_mask].copy()  # Create a copy to avoid modifying the original DataFrame
    upn_teacher = result_df[teacher_mask].copy()  # Create a copy to avoid modifying the original DataFrame

    merged_df = upn_model[['upn', 'model_difference']].merge(
    upn_teacher[['upn', 'teacher_difference']],
    on='upn',
    how='outer',
    suffixes=('_model', '_teacher')
    )

    # Print the merged DataFrame
    merged_df.to_csv(f'intervention/{subject}.csv')
    print('csv saved')