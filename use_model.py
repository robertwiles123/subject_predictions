import os
import joblib
import pandas as pd
# need to populate a folder for it to extract

subjects = ['english_language', 'english_literature', 'maths', 'biology', 'chemistry', 'computer_science', 'french_language', 'geography', 'german', 'history', 'physics', 'spanish', 'art_&_design', 'business_studies', 'd_&_t_product_design', 'd_&_t_textiles_technology', 'drama', 'food_technology', 'ict_btec', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'product_design']

# change to year of year 11 to create predictions.
year = '2223'
year = '_' + year

folder_path = "/workspaces/subject_predictions/to_be_predicted"

# Iterate through CSV files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv") and year in file:
            # Extract the subject name from the file name
            subject = file.split(year)[0]           
            # Check if the subject is in the list of subjects
            if subject in subjects:
                model = joblib.load('/workspaces/subject_predictions/models' + subject + '.joblib')
                print('No error')