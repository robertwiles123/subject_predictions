import pandas as pd

subjects = ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science', 'drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'maths', 'music_studies', 'physics', 'spanish']
model = 'ridge'


for subject in subjects:

    prediction_df = pd.read_csv(f'{model}_predicted/{subject}_2223_prediction.csv')
    prediction_df_basics = prediction_df[['upn', f'{subject}_prediction']]
    
    teacher_prediction = pd.read_csv(f'to_be_predicted/{subject}_2223.csv')
    teacher_filtered = teacher_prediction.filter(regex=f'{subject}.*_ap2$')

    real_grades = pd.read_csv()

    mae_prediction = abs(prediction_df_basics[f'{subject}_prediction'] - real_grades['real_grades']).mean()
    print("Mean Absolute Error (MAE) prediction:", mae_prediction)

    mse_prediction = ((prediction_df_basics[f'{subject}_prediction']  - real_grades['real_grades']) ** 2).mean()
    print("Mean Squared Error (MSE) prediction:", mse_prediction)

    mae_teacher = abs(teacher_prediction[teacher_filtered] - real_grades['real_grades']).mean()
    print("Mean Absolute Error (MAE) prediction:", mse_teacher)

    mse_teacher = ((teacher_prediction[teacher_filtered] - real_grades['real_grades']) ** 2).mean()
    print("Mean Squared Error (MSE) prediction:", mse_teacher)