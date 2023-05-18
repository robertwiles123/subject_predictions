#currently only works on LinearRegression, will add others
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# dict of avalible models, this is to allow me to test the models in the real world. Currently the linear regressions seems to be the most appropriate, but with more real world testing that may not be the case
combined_models_to_predict = ['combined_linear']
combined_models_to_predict_dict ={}

for s in combined_models_to_predict:
    model = load(s+'.joblib')
    combined_models_to_predict_dict[s] = model

print(combined_models_to_predict_dict)


# ask for data input
data = input('What data do you want to use? If a file include .csv: ')

# check if csv then import csv file
if data.endswith('.csv'):
    data = pd.read_csv(data)

# to run the data through the model
def run_test(data, models=combined_models_to_predict_dict):
    # to check if data is as a string of 2 grades or a dataframe
    outcomes = {}
    for k, v in combined_models_to_predict_dict.items():
        if isinstance(data, str):
        # need to take the string of grades, split in to two then use to predict
            print('oops')
        # to take a dataframe and extract the grades within to use those to predict
        elif isinstance(data, pd.DataFrame):
            le = LabelEncoder()
            encoded_data = data.copy()
            col_to_encode = ['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']
            for col in col_to_encode:
                encoded_data[col] = le.fit_transform(data[col])
            load=v
            prediction = v.predict(encoded_data)
            print(prediction)
            outcomes[k] = prediction
        else:
            print('Neither')
    return outcomes


outcome = run_test(data)

# will print the dict that contains the key of the name wand the value, as an array, of the now grades
print(outcome)

# to join the prediction to the data frame if a csv
if isinstance(data, pd.DataFrame):
    outcome_df = pd.DataFrame(outcome)
    predicted_data = data.join(outcome_df)

print(predicted_data)