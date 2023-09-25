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


# change model that is being used. Also update line 66
model_name = 'linear_regression'
year_prediction = '2223'

# Define subject and year here
subjects = subject_list.prediction_subjects()
"""
Removed subjects
d_&_t_product_design
d_&_t_textiles_technology
ict_btec
music_tech_grade
pearson_btec_sport
product_design
"""

for topic in subjects:
    subject = topic
    model = joblib.load(f'models/{subject}_{model_name}.pkl')
     
    # load in dataframe to predict
    subject_prediction = subject + '_' + year_prediction + '.csv'
    prediction = pd.read_csv('to_be_predicted/' + subject_prediction)

    prediction.rename(columns={'actual_ap2': 'actual_real',
                            'estimate_ap2': 'estimate_real'}, inplace=True)

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
    prediction[subject_prediction_column_name] = y_prediction_rounded.astype(int)
    prediction.to_csv(f'{model_name}_predicted/' + subject + '_' + year_prediction + '_prediction.csv')
    print(f'{subject} Prediction saved')