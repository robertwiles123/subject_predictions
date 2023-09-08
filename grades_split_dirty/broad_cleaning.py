# All but science double are giving the right amount out. Science double is missing any end 
import os
import pandas as pd
year = input('What year to be cleaned? ')
year = '_'+year

# Define a list of subjects
subjects = ['english_language', 'english_literature', 'maths', 'biology', 'chemistry', 'computer_science', 'french_language', 'geography', 'german', 'history', 'physics', 'science_double', 'spanish', 'art_&_design', 'business_studies', 'd_&_t_product_design', 'd_&_t_textiles_technology', 'drama', 'food_technology', 'ict_btec', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'product_design']

# Specify the folder path where the CSV files are located
folder_path = "/workspaces/subject_predictions/grades_split_dirty"

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

                # Fill missing values based on the regex pattern
                for col in filtered_columns.columns:
                    df[col].fillna(df[col.replace('ap1', 'ap2').replace('ap2', 'ap1')], inplace=True)
                #remove NA 
                df.dropna(inplace=True)
                # Replace 'U' with '0' in all columns except 'gender_ap2'
                columns_to_replace = df.columns.difference(['gender_ap2', 'upn'])
                try:
                    df[columns_to_replace] = df[columns_to_replace].replace(['U','u'], '0')
                    df[columns_to_replace] = df[columns_to_replace].astype(int)
                    # Convert non-'gender_ap2' columns to integers
                except ValueError:
                    continue

                #Save files
                if year == '_2122':
                    file_name = f'{subject}{year}.csv'
                    output_directory = '/workspaces/subject_predictions/grades_full_clean'
                    file_path = os.path.join(output_directory, file_name)
                    df.to_csv(file_path, index=False)
                    print('File saved')
                elif year == '_2223':
                    file_name = f'{subject}{year}.csv'
                    output_directory = '/workspaces/subject_predictions/to_be_predicted'
                    file_path = os.path.join(output_directory, file_name)
                    df.to_csv(file_path, index=False)
                    print('File saved')

                

