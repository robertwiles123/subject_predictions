# Import necessary libraries
import os
import pandas as pd
import numpy as np
import sys

# Add a custom folder to the Python path for additional imports
sys.path.append('/workspaces/subject_predictions') 
import subject_list

csv_files = [file for file in os.listdir() if file.endswith(".csv")]

years = ['1819', '2122','2223']

# Define a list of subjects using a function from the subject_list module
subjects = subject_list.full_subjects()

# Specify the folder path where the CSV files are located
folder_path = "/workspaces/subject_predictions/grades_split_dirty"

combined_dataframes = {}

# Loop through subjects and years
for subject in subjects:
    for year in years:
        key = f"{subject}_{year}.csv"
        # Construct the full file path
        file_path = os.path.join(folder_path, key)
        # Check if the file exists before reading it
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if subject == 'ict_btec' and year == '2223':
                df['ict_btec_ap1'] = df['ict_btec']
                df['ict_btec_ap2'] = df['ict_btec']
                df.drop(columns=['ict_btec'], inplace=True)
            if subject not in combined_dataframes:
                combined_dataframes[subject] = df
            else:
            # If it's in the dictionary, check for missing columns
                missing_columns = df.columns.difference(combined_dataframes[subject].columns)
                
                # Add missing columns with 0 values to the existing DataFrame
                for col in missing_columns:
                    combined_dataframes[subject][col] = '0'

                # Concatenate the DataFrames
                combined_dataframes[subject] = pd.concat([combined_dataframes[subject], df], ignore_index=True)
                

for subject, df in combined_dataframes.items():
    regex_pattern = f'^{subject}.*_ap[12]$'
    # Filter the DataFrame columns based on the regex pattern
    filtered_columns = df.filter(regex=regex_pattern)

    # Loop through the filtered columns
    for col in filtered_columns.columns:
        target_col_name = col.replace('ap1', 'ap2').replace('ap2', 'ap1')

        try:
            # Attempt to cast the target column to the same data type as the current column
            df[target_col_name] = df[target_col_name].astype(df[col].dtype)
        except ValueError:
            # Handle the ValueError by replacing 'U' with 0
            df[target_col_name] = df[target_col_name].replace('U', 0).astype(df[col].dtype)

        # Use np.where to fill NaN values based on a condition
        df[col] = np.where(df[col].notna(), df[col], df[target_col_name])

    # Remove rows with any remaining NaN values
    df.dropna(inplace=True)

    # Replace 'U' with '0' in all columns except 'gender_ap2'
    columns_to_replace = df.columns.difference(['gender_ap2', 'upn'])
    try:
        df[columns_to_replace] = df[columns_to_replace].replace(['U','u', 'X', 'x'], '0')
        df[columns_to_replace] = df[columns_to_replace].astype(int)
    except ValueError:
        pass

    # Convert non-'gender_ap2' columns to integers
    for column in df.columns:
        if column != 'upn':
            try:
                df[column] = df[column].astype(int)
            except ValueError:
                # If the conversion to int raises a ValueError, skip to the next column
                continue

    # Define a custom function to apply the condition
    def map_to_boolean(value):
        if value == 0:
            value = '0'
        if pd.isna(value) or value.lower() in ['n', '0']:
            return 0
        elif value.lower() in ['f', 'k', 'e']:
            return 1
        elif value.lower() in ['t']:
            return 2
        
        else:
            print(f'{subject} {value} not value')

    # Apply the custom function to the 'SEN' column and create a new 'SEN_bool' column
    try:
        """df['sen_bool'] = df['sen_real'].apply(map_to_boolean)
        df.drop('sen_real', axis=1, inplace=True)"""

        df['pp_bool'] = df['pp_real'].apply(map_to_boolean)
        df.drop('pp_real', axis=1, inplace=True)
        
        df['fsm_bool'] = df['fsm_real'].apply(map_to_boolean)
        df.drop('fsm_real', axis=1, inplace=True)

        df['eal_bool'] = df['eal_real'].apply(map_to_boolean)
        df.drop('eal_real', axis=1, inplace=True)

        columns_to_convert = ['pp_bool', 'fsm_bool', 'eal_bool']

        # Convert the specified columns to bool
        df[columns_to_convert] = df[columns_to_convert].astype(int)
    except KeyError:
        print('error')

    # If the subject is 'science_double', drop columns with '2nd' in their names
    if subject == 'science_double':
        df.drop(columns=df.columns[df.columns.str.contains('2nd')], inplace=True)

    file_name = f'{subject}.csv'
    output_directory = '/workspaces/subject_predictions/model_csvs'
    file_path = os.path.join(output_directory, file_name)
    df.to_csv(file_path, index=False)
    print(f'{subject} saved')




"""
# Iterate through CSV files in the specified folder
for root, dirs, files in os.walk(folder_path):
for file in files:
if file.endswith(".csv") and year in file:
# Extract the subject name from the file name
subject = file.split(year)[0]

# Check if the subject is in the list of subjects
if subject in subjects:
    # Read the CSV file
    csv_path = os.path.join(root, file)
    df = pd.read_csv(csv_path)
    
    # Define a regex pattern based on the subject name
    regex_pattern = f'^{subject}.*_ap[12]$'

    # Filter the DataFrame columns based on the regex pattern
    filtered_columns = df.filter(regex=regex_pattern)

    # Loop through the filtered columns
    for col in filtered_columns.columns:
        target_col_name = col.replace('ap1', 'ap2').replace('ap2', 'ap1')

        try:
            # Attempt to cast the target column to the same data type as the current column
            df[target_col_name] = df[target_col_name].astype(df[col].dtype)
        except ValueError:
            # Handle the ValueError by replacing 'U' with 0
            df[target_col_name] = df[target_col_name].replace('U', 0).astype(df[col].dtype)

        # Use np.where to fill NaN values based on a condition
        df[col] = np.where(df[col].notna(), df[col], df[target_col_name])

    # Remove rows with any remaining NaN values
    df.dropna(inplace=True)

    # Replace 'U' with '0' in all columns except 'gender_ap2'
    columns_to_replace = df.columns.difference(['gender_ap2', 'upn'])
    try:
        df[columns_to_replace] = df[columns_to_replace].replace(['U','u', 'X', 'x'], '0')
        df[columns_to_replace] = df[columns_to_replace].astype(int)
    except ValueError:
        pass

    # Convert non-'gender_ap2' columns to integers
    for column in df.columns:
        if column != 'upn':
            try:
                df[column] = df[column].astype(int)
            except ValueError:
                # If the conversion to int raises a ValueError, skip to the next column
                continue

    # If the subject is 'science_double', drop columns with '2nd' in their names
    if subject == 'science_double':
        df.drop(columns=df.columns[df.columns.str.contains('2nd')], inplace=True)

    # Save the processed DataFrame to CSV files in different directories based on the year
    if year[1].lower() == 'm':
        year = '_model'
        file_name = f'{subject}.csv'
        output_directory = '/workspaces/subject_predictions/model_csvs'
        file_path = os.path.join(output_directory, file_name)
        df.to_csv(file_path, index=False)
        print(f'{subject} saved')
    elif year == '_2223':
        file_name = f'{subject}{year}.csv'
        output_directory = '/workspaces/subject_predictions/to_be_predicted'
        file_path = os.path.join(output_directory, file_name)
        df.to_csv(file_path, index=False)
        print(f'{subject} saved')
    else:
        print(f'{subject} not saved')
"""