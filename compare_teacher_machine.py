import pandas as pd

subjects = ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science', 'drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'maths', 'music_studies', 'physics', 'spanish']
model = 'ridge'


for subject in subjects:

    prediction_df = pd.read_csv(f'{model}_predicted/{subject}_2223_prediction.csv')
    prediction_df_basics = prediction_df[['upn', f'{subject}_prediction']]
    
    teacher_prediction = pd.read_csv()