import pandas as pd
import re
import os
import sys
# import folder needed for following imports
sys.path.append('/workspace/subject_predictions') 
import subject_list

# Define a function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# Define a function to create subject-specific dataframes
def create_subject_dataframes(full, subjects, estimate_column, actual_column, gender_column):
    subject_dataframes = {}
    
    for subject in subjects:
        subject_pattern = f'{subject}.*'
        columns_to_select = ['upn', estimate_column, actual_column, gender_column] + [col for col in full.columns if re.search(subject_pattern, col)]
        subject_df = full[columns_to_select]
        subject_dataframes[subject] = subject_df
    
    return subject_dataframes

# Define a function to save dataframes as CSV
def save_dataframes_as_csv(dataframes, output_directory, file_suffix):
    for subject, final in dataframes.items():
        file_name = f'{subject}_{file_suffix}.csv'
        file_path = os.path.join(output_directory, file_name)
        final.to_csv(file_path, index=False)

# Main program
type = input('Create models or predictions? Input p or m: ')
type = type[0].lower()

if type == 'm':
    ap1 = pd.read_excel('Y11 2122 Ap1 MLG P8.xlsx', header=1)
    ap2 = pd.read_excel('Y11 2122 Ap2 MLG P8.xlsx', header=1)
    p8 = pd.read_excel('Y11 2122 Actual Results P8.xlsx', header=1)
    
    # Convert all column names to lowercase
    ap1.columns = ap1.columns.str.lower()
    ap2.columns = ap2.columns.str.lower()
    p8.columns = p8.columns.str.lower()

    clean_column_names(ap1)
    clean_column_names(ap2)

    ap1['upn'] = ap1['upn'].str.strip().str.lower()
    ap2['upn'] = ap2['upn'].str.strip().str.lower()
    p8['upn'] = p8['upn'].str.strip().str.lower()

    ap1.dropna(subset=['upn'], inplace=True)
    ap2.dropna(subset=['upn'], inplace=True)
    p8.dropna(subset=['upn'], inplace=True)

    full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))

    upn_index = p8.columns.get_loc('upn')
    p8 = p8[[p8.columns[upn_index]] + [col for col in p8.columns if col != 'upn']]
    p8.columns = ['upn'] + [f'{col}_real' for col in p8.columns if col != 'upn']
    
    full = pd.merge(full, p8, on='upn', how='outer')
    
    columns_to_drop = ['gender_ap1', 'rm_scaled_score_ap1', 'rm_scaled_score_ap2', 'actual_ap1', 'actual_ap2', 'difference_ap1', 'difference_ap2', 'difference_real', 'entries_ap1', 'score_ap1', 'scoreadjusted_ap1', 'entries_ap2', 'score_ap2', 'scoreadjusted_ap2', 'entries_real', 'score_real']
    full.drop(columns=columns_to_drop, inplace=True)
    
    subjects = subject_list.full_subjects()
    subject_dataframes = create_subject_dataframes(full, subjects, 'estimate_real', 'actual_real', 'gender_ap2')
    save_dataframes_as_csv(subject_dataframes, '/workspace/subject_predictions/grades_split_dirty', 'model')
    print('Files saved')
elif type == 'p':
    ap1 = pd.read_excel('Y11 2223 FFT P8.xlsx')
    ap2 = pd.read_excel('Y11 2223 FFT P8.xlsx')
    
    # Convert all column names to lowercase
    ap1.columns = ap1.columns.str.lower()
    ap2.columns = ap2.columns.str.lower()

    clean_column_names(ap1)
    clean_column_names(ap2)

    ap1['upn'] = ap1['upn'].str.strip().str.lower()
    ap2['upn'] = ap2['upn'].str.strip().str.lower()

    ap1.dropna(subset=['upn'], inplace=True)
    ap2.dropna(subset=['upn'], inplace=True)

    full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))
    
    columns_to_drop = ['em_fine_points_ap2', 'em_fine_points_ap1', 'gender_ap1', 'difference_ap1', 'difference_ap2', 'entries_ap1', 'score_ap1', 'entries_ap2', 'score_ap2']
    full.drop(columns=columns_to_drop, inplace=True)
    
    subjects = subject_list.full_subjects()
    subject_dataframes = create_subject_dataframes(full, subjects, 'estimate_ap2', 'actual_ap2', 'gender_ap2')
    save_dataframes_as_csv(subject_dataframes, '/workspace/subject_predictions/grades_split_dirty', '2223')
    print('Files saved')
else:
    print("Error: Neither option chosen.")
