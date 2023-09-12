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
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)

                # Due to slight differencing in names
                df.rename(columns={'actual_ap2': 'actual_real'}, inplace=True)

                model = joblib.load('/workspaces/subject_predictions/models/' + subject + '_ridge.joblib')
                
                df_encoded = pd.get_dummies(df['gender_ap2'], prefix='Gender')

                # Concatenate the encoded gender columns with the original DataFrame
                df_final = pd.concat([df, df_encoded], axis=1)

                # Drop the original "Gender" column if needed
                df_final.drop(columns=['gender_ap2'], inplace=True)

                pattern = rf'^{subject}.*real$'
                print(pattern)
                print(df_final.columns)
                # Use the filter method to drop columns matching the pattern
                columns_to_drop = df_final.filter(regex=pattern)

                print(columns_to_drop.columns)
"""
                X = df_final.drop(columns=['upn', columns_to_drop.columns])

                y = df_final[columns_to_drop.columns]



"""