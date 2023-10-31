import pandas as pd
import re
import os
import sys
# import folder needed for following imports
sys.path.append('/workspaces/subject_predictions') 
import subject_list

year = '2223'
# to ensure there is no issues with the columns names
def clean_column_names(df):
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

if year == '2122' or '2223':
    ap1 = pd.read_excel(f'Y11 {year} Ap1 MLG P8.xlsx', header=1)
    ap2 = pd.read_excel(f'Y11 {year} Ap2 MLG P8.xlsx', header=1)
    p8 = pd.read_excel(f'Y11 {year} Actual Results P8.xlsx', header=1)
     
    dataframes = (ap1, ap2, p8)
    
    for df in dataframes:
        clean_column_names(df)


    full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))
    print(p8.columns)
    upn_index = p8.columns.get_loc('upn')

    # Move the 'upn' column to position 0
    p8 = p8[[p8.columns[upn_index]] + [col for col in p8.columns if col != 'upn']]

    # Give the remaining columns a suffix "_real"
    p8.columns = ['upn'] + [f'{col}_real' for col in p8.columns if col != 'upn']


    #merge real grades with the outcomes
    full = pd.merge(full, p8, on='upn', how='outer')

    #list of duplicated columns to drop after merge
    columns_to_drop = ['gender_ap1', 'rm_scaled_score_ap1', 'rm_scaled_score_ap2', 'actual_ap1', 'actual_ap2', 'difference_ap1', 'difference_ap2', 'difference_real', 'entries_ap1', 'score_ap1', 'scoreadjusted_ap1', 'entries_ap2', 'score_ap2', 'scoreadjusted_ap2', 'entries_real', 'score_real', 'sen_ap1', 'sen_ap2', 'pp_ap1', 'pp_ap2', 'fsm_ap1', 'fsm_ap2', 'eal_ap1', 'eal_ap2']

    for col in columns_to_drop:
        if col in full.columns:
            full.drop(columns=col, inplace=True)

    #subjects in dataframe
    subjects = subject_list.full_subjects()

    # Create a dictionary to store DataFrames
    subject_dataframes = {}

    # Loop through subjects and create DataFrames
    for subject in subjects:
        # Create regex pattern for the subject
        subject_pattern = f'{subject}.*'
        
        # Select columns using regex pattern and add common columns
        selected_columns  = ['upn', 'gender_ap2', 'fsm_real', 'sen_real', 'pp_real', 'eal_real'] + [col for col in full.columns if re.search(subject_pattern, col)]
        
        columns_to_select = [col for col in selected_columns if col in full.columns]

        # Create the DataFrame for the subject
        subject_df = full[columns_to_select]
        
        # Add the subject DataFrame to the dictionary with a key based on the subject
        subject_dataframes[subject] = subject_df
    # Access individual DataFrames dynamically by subject name
    for subject in subjects:
        globals()[f'{subject}_df'] = subject_dataframes[subject]

    # Define the output directory where CSV files will be saved
    output_directory = '/workspaces/subject_predictions/grades_split_dirty'
    # Loop through the DataFrames and save each as a CSV file

    for subject in subjects:
        final = globals()[f'{subject}_df']
        file_name = f'{subject}_{year}.csv'
        file_path = os.path.join(output_directory, file_name)
        final.to_csv(file_path, index=False)
    print('Files saved')
elif type == 'p':
    ap1 = pd.read_excel('Y11 2223 FFT P8.xlsx')
    ap2 = pd.read_excel('Y11 2223 FFT P8.xlsx')

    dataframes = (ap1, ap2)
    for df in dataframes:
        clean_column_names(df)

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