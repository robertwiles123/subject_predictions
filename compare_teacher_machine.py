import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import re

# List of stopics that have data
subjects = ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science', 'drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'maths', 'music_studies', 'physics', 'spanish']

# model that predictions will be made from
model = 'ridge'

# loop to go over topics
for subject in subjects:
    # load in predictions from model
    prediction_df = pd.read_csv(f'{model}_predicted/{subject}_2223_prediction.csv')
    prediction_df_basics = prediction_df[['upn', f'{subject}_prediction']]
    
    # load predictions made by teachers
    teacher_prediction = pd.read_csv(f'to_be_predicted/{subject}_2223.csv')
    teacher_filtered = teacher_prediction.filter(regex=f'{subject}.*_ap2$')

    # load real grades
    real_grades = pd.read_csv()

    # complete and print MSE and MAE for machine predictions
    mae_prediction = abs(prediction_df_basics[f'{subject}_prediction'] - real_grades['real_grades']).mean()
    print("Mean Absolute Error (MAE) prediction:", mae_prediction)

    mse_prediction = ((prediction_df_basics[f'{subject}_prediction']  - real_grades['real_grades']) ** 2).mean()
    print("Mean Squared Error (MSE) prediction:", mse_prediction)

    # complete and print MSE and MAE for teacher predictions
    mae_teacher = abs(teacher_prediction[teacher_filtered] - real_grades['real_grades']).mean()
    print("Mean Absolute Error (MAE) prediction:", mse_teacher)

    mse_teacher = ((teacher_prediction[teacher_filtered] - real_grades['real_grades']) ** 2).mean()
    print("Mean Squared Error (MSE) prediction:", mse_teacher)
   