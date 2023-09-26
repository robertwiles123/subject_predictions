# All but science double are giving the right amount out. Science double is missing any end 
import os
import pandas as pd
import numpy as np
import sys
# import folder needed for following imports
sys.path.append('/workspace/subject_predictions') 
import subject_list

# will need to update so loads in multiple models if model selected and join them together
year = input('Input year of prediction or model')
year = '_'+year

# Define a list of subjects
subjects = subject_list.full_subjects()

# Specify the folder path where the CSV files are located
folder_path = "/workspace/subject_predictions/grades_split_dirty"


# Iterate through CSV files in the folder
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
                regex_pattern = f'^{subject}.*_ap[12]$'

                # Filter the DataFrame based on the regex pattern
                filtered_columns = df.filter(regex=regex_pattern)

                for col in filtered_columns.columns:
                    target_col_name = col.replace('ap1', 'ap2').replace('ap2', 'ap1')

                    try:
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
                    # Convert non-'gender_ap2' columns to integers
                except ValueError:
                    pass

                for column in df.columns:
                    if column != 'upn':
                        try:
                            df[column] = df[column].astype(int)
                        except ValueError:
                            # If the conversion to int raises a ValueError, skip to the next column
                            continue
                if subject == 'science_double':
                    df.drop(columns='science_double_2nd_mark_real', inplace=True)
                #Save files
                if year == '_model':
                    file_name = f'{subject}.csv'
                    output_directory = '/workspace/subject_predictions/model_csvs'
                    file_path = os.path.join(output_directory, file_name)
                    df.to_csv(file_path, index=False)
                    print(f'{subject} saved')
                elif year == '_2223':
                    file_name = f'{subject}{year}.csv'
                    output_directory = '/workspace/subject_predictions/to_be_predicted'
                    file_path = os.path.join(output_directory, file_name)
                    df.to_csv(file_path, index=False)
                    print(f'{subject} saved')
                else:
                    print(f'{subject} not saved')

                

