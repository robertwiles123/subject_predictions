import pandas as pd
import re
import os
import sys
# import folder needed for following imports
sys.path.append('/workspace/subject_predictions') 
import subject_list

type = input('Create models or predictions? Input p or m: ')

type = type[0].lower()

if type == 'm':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    ap1 = pd.read_excel('Y11 2122 Ap1 MLG P8.xlsx', header=1)
    ap2 = pd.read_excel('Y11 2122 Ap2 MLG P8.xlsx', header=1)
    p8 = pd.read_excel('Y11 2122 Actual Results P8.xlsx', header=1)

    # to ensure there is no issues with the columns names
    def clean_column_names(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    dataframes = (ap1, ap2, p8)
    for df in dataframes:
        clean_column_names(df)

    #ensure upn for merge is clean
    for df in dataframes:
        df['upn'] = df['upn'].str.strip().str.lower()
        df.dropna(subset=['upn'], inplace=True)

    full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))

    upn_index = p8.columns.get_loc('upn')

    # Move the 'upn' column to position 0
    p8 = p8[[p8.columns[upn_index]] + [col for col in p8.columns if col != 'upn']]

    # Give the remaining columns a suffix "_real"
    p8.columns = ['upn'] + [f'{col}_real' for col in p8.columns if col != 'upn']


    #merge real grades with the outcomes
    full = pd.merge(full, p8, on='upn', how='outer')

    #list of duplicated columns to drop after merge
    columns_to_drop = ['gender_ap1', 'rm_scaled_score_ap1', 'rm_scaled_score_ap2', 'actual_ap1', 'actual_ap2', 'difference_ap1', 'difference_ap2', 'difference_real', 'entries_ap1', 'score_ap1', 'scoreadjusted_ap1', 'entries_ap2', 'score_ap2', 'scoreadjusted_ap2', 'entries_real', 'score_real']

    # Drop the specified columns from the DataFrame
    full.drop(columns=columns_to_drop, inplace=True)

    #subjects in dataframe
    subjects = subject_list.full_subjects()

    # Create a dictionary to store DataFrames
    subject_dataframes = {}

    # Loop through subjects and create DataFrames
    for subject in subjects:
        # Create regex pattern for the subject
        subject_pattern = f'{subject}.*'
        
        # Select columns using regex pattern and add common columns
        columns_to_select = ['upn', 'estimate_real', 'actual_real', 'gender_ap2'] + [col for col in full.columns if re.search(subject_pattern, col)]
        
        # Create the DataFrame for the subject
        subject_df = full[columns_to_select]
        
        # Add the subject DataFrame to the dictionary with a key based on the subject
        subject_dataframes[subject] = subject_df

    # Access individual DataFrames dynamically by subject name
    for subject in subjects:
        globals()[f'{subject}_df'] = subject_dataframes[subject]


    # Define the output directory where CSV files will be saved
    output_directory = '/workspace/subject_predictions/grades_split_dirty'
    # Loop through the DataFrames and save each as a CSV file

    for subject in subjects:
        final = globals()[f'{subject}_df']
        file_name = f'{subject}_model.csv'
        file_path = os.path.join(output_directory, file_name)
        final.to_csv(file_path, index=False)
    print('Files saved')
elif type == 'p':
    ap1 = pd.read_excel('Y11 2223 FFT P8.xlsx')
    ap2 = pd.read_excel('Y11 2223 FFT P8.xlsx')

    # to ensure there is no issues with the columns names
    def clean_column_names(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    dataframes = (ap1, ap2)
    for df in dataframes:
        clean_column_names(df)

    #ensure upn for merge is clean
    for df in dataframes:
        df['upn'] = df['upn'].str.strip().str.lower()
        df.dropna(subset=['upn'], inplace=True)

    full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))

    #list of duplicated columns to drop after merge
    columns_to_drop = ['em_fine_points_ap2', 'em_fine_points_ap1', 'gender_ap1', 'difference_ap1', 'difference_ap2', 'entries_ap1', 'score_ap1', 'entries_ap2', 'score_ap2']


    # Drop the specified columns from the DataFrame
    full.drop(columns=columns_to_drop, inplace=True)

    #subjects in dataframe
    subjects = subject_list.full_subjects()

    # Create a dictionary to store DataFrames
    subject_dataframes = {}
    print(full.columns)
    # Loop through subjects and create DataFrames
    for subject in subjects:
        # Create regex pattern for the subject
        subject_pattern = f'{subject}.*'
        
        # Select columns using regex pattern and add common columns
        columns_to_select = ['upn', 'estimate_ap2', 'actual_ap2', 'gender_ap2'] + [col for col in full.columns if re.search(subject_pattern, col)]
        
        # Create the DataFrame for the subject
        subject_df = full[columns_to_select]
        
        # Add the subject DataFrame to the dictionary with a key based on the subject
        subject_dataframes[subject] = subject_df

    # Access individual DataFrames dynamically by subject name
    for subject in subjects:
        globals()[f'{subject}_df'] = subject_dataframes[subject]


    # Define the output directory where CSV files will be saved
    output_directory = '/workspace/subject_predictions/grades_split_dirty'

    # Loop through the DataFrames and save each as a CSV file
    for subject in subjects:
        final = globals()[f'{subject}_df']
        file_name = f'{subject}_2223.csv'
        file_path = os.path.join(output_directory, file_name)
        final.to_csv(file_path, index=False)
    print('Files saved')
else:
    "Error neither chosen."