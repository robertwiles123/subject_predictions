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
    
    # Graphs that shows all predicted errors

    # Convert y_test to a NumPy array
    y_test_array = y_test.to_numpy()

    # Calculate the residuals (the differences between true and predicted values)
    residuals = y_test_array - y_pred_true

    # Calculate the absolute errors (useful for visualization)
    absolute_errors = np.abs(residuals)

    # Find the indices of the data points where the model made incorrect predictions (absolute error > 0)
    incorrect_indices = np.where(absolute_errors > 0)

    # Extract the corresponding values from y_test_array and y_pred_true for incorrect predictions
    incorrect_y_test = y_test_array[incorrect_indices]
    incorrect_y_pred = y_pred_true[incorrect_indices]

    # Now, you have the true and predicted values for all data points where the model made incorrect predictions.
    # You can use these values for visualization or further analysis.
    try:
        # Adjust the axis increments (both axis increase by 1)
        plt.xticks(np.arange(min(incorrect_y_test), max(incorrect_y_test)+1, 1))
        plt.yticks(np.arange(min(incorrect_y_pred), max(incorrect_y_pred)+1, 1))
    except ValueError:
        continue

    # Make grid lines visible
    plt.grid(True)

    # Visualize the errors, for example, in a scatter plot
    plt.scatter(incorrect_y_test, incorrect_y_pred, label='Incorrect Predictions', color='red')
    plt.title(f'{model_name.capitalize()} {subject} wrong predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f'error_graphs/{subject}_{model_name}.png')
    print('Errors saved')