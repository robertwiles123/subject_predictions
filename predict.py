import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import re
import joblib
import subject_list

model_name = 'linear_regression'
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

    if 'btec' in subject or 'tech_' in subject:
        btec_pattern = pattern = rf'{subject}.*_(ap1|ap2|real)'
        columns_to_encode = [col for col in prediction.columns if re.match(btec_pattern, col)]

        for col in columns_to_encode:
            encoded_column = pd.get_dummies(prediction[col], prefix=col)
            prediction = pd.concat([prediction, encoded_column], axis=1)
            prediction.drop(columns=[col], inplace=True)

    gender_encoded_prediction = pd.get_dummies(prediction['gender_ap2'], prefix='Gender')

    # Concatenate the encoded gender columns with the original DataFrame
    prediction_final = pd.concat([prediction, gender_encoded_prediction], axis=1)

    

    # Drop the original "Gender" column if needed
    prediction_final.drop(columns=['gender_ap2'], inplace=True)

    X_prediction = prediction_final.drop(columns=['upn'])

    y_prediction =model.predict(X_prediction)

    y_prediction_rounded = np.round(y_prediction)

    # save prediction
    subject_prediction_column_name = subject + '_prediction'

    try:
        prediction[subject_prediction_column_name] = y_prediction_rounded.astype(int)
    except ValueError:
        prediction[subject_prediction_column_name] = y_prediction_rounded[:, 0].astype(int)


    prediction.to_csv(f'{model_name}_predicted/' + subject + '_' + year_prediction + '_prediction.csv')
    print(f'{subject} Prediction saved')
