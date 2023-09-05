# All but science double are giving the right amount out. Science double is missing any end 
import os
import pandas as pd

# Define a list of subjects
subjects = ['english_language', 'english_literature', 'maths', 'biology', 'chemistry', 'computer_science', 'french_language', 'geography', 'german', 'history', 'physics', 'science_double', 'spanish', 'art_&_design', 'business_studies', 'd_&_t_product_design', 'd_&_t_textiles_technology', 'drama', 'food_technology', 'ict_btec', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'product_design']

# Specify the folder path where the CSV files are located
folder_path = "/workspaces/subject_predictions/grades_split_dirty"

# Iterate through CSV files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv") and "_2122" in file:
            # Extract the subject name from the file name
            subject = file.split("_2122")[0]           
            # Check if the subject is in the list of subjects
            if subject in subjects:
                # Read the CSV file
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                #remove NA 
                df.dropna(inplace=True)
                # Annoce those removed
                print(subject)
                print(len(df))
                #Save files
                file_name = f'{subject}_2122.csv'
                output_directory = '/workspaces/subject_predictions/grades_full_clean'
                file_path = os.path.join(output_directory, file_name)
                df.to_csv(file_path, index=False)
                print('File saved')
                

