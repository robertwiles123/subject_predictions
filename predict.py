import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import re
import joblib
import subject_list

model_name = subject_list.get_model_name(globals())

year_prediction = '2223'

# Define subject and year here
subjects = subject_list.prediction_subjects()

for topic in subjects:
    subject = topic
    model = joblib.load(f'models/{subject}_{model_name}.pkl')
     
    # load in dataframe to predict
    subject_prediction = subject + '_' + year_prediction + '.csv'
    prediction = pd.read_csv('to_be_predicted/' + subject_prediction)

    prediction.rename(columns={'actual_ap2': 'actual_real',
                            'estimate_ap2': 'estimate_real'}, inplace=True)

    encoded_columns = pd.DataFrame()

    for col in prediction.columns:
        if 'btec' in subject or 'tech_' in subject:
            grade_mapping = subject_list.grades_mapped()
            if re.match(rf'{subject}.*_(ap1|ap2)', col):
                encoded_columns[col] = prediction[col].map(grade_mapping)
        if col == 'gender_ap2':
            # Use pd.get_dummies for the gender column
            gender_encoded = pd.get_dummies(prediction[col], prefix=col)
            encoded_columns = pd.concat([encoded_columns, gender_encoded], axis=1)
        else:
            # Keep non-matching columns as they are
            if col not in encoded_columns.columns:
                encoded_columns = pd.concat([encoded_columns, prediction[[col]]], axis=1)


    X_prediction = encoded_columns.drop(columns=['upn'])

    y_prediction = model.predict(X_prediction)

    y_prediction_rounded = np.round(y_prediction)

    # save prediction
    subject_prediction_column_name = subject + '_prediction'
    if 'btec' in subject or 'tech_' in subject:
        original_mapping = subject_list.grades_mapped()
        reverse_mapping = {v: k for k, v in original_mapping.items()}
        y_prediction_rounded = y_prediction_rounded.astype(int)
        y_prediction_rounded = y_prediction_rounded.flatten()
        y_pred_btec = [reverse_mapping[value] for value in y_prediction_rounded]
        prediction[subject_prediction_column_name] = y_pred_btec
    else:
        try:
            prediction[subject_prediction_column_name] = y_prediction_rounded.astype(int)
        except ValueError:
            prediction[subject_prediction_column_name] = y_prediction_rounded[:, 0].astype(int)

    


    prediction.to_csv(f'{model_name}_predicted/' + subject + '_' + year_prediction + '_prediction.csv')
    print(f'{subject} Prediction saved')
