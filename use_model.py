#currently only works on LinearRegression, will add others
import pandas as pd
from joblib import load
from sklearn.linear_model import LinearRegression

# dict of avalible models, this is to allow me to test the models in the real world. Currently the linear regressions seems to be the most appropriate, but with more real world testing that may not be the case
models_to_predict = {'Linear Regression': 'combined_inear.joblib'}

# ask for data input
data = input('What data do you want to use? If a file include .csv: ')

#import csv file
if data.endswith('.csv'):
    file = pd.read_csv(data)

# to run the data through the model
def run_test(data, models=models_to_predict):
    # to check if data is as a string of 2 grades or a dataframe
    if isinstance(data, str):
           print('oops')
    elif isinstance(data, pd.DataFrame):
        input_grades = [[data['grade_1'], data['grade_2']]]
        for k, v in models.items():
            outcomes = {}
            load=v
            prediction = v.predict(input_grades)
            outcomes[k] = prediction

    
        

outcome = run_test(models_to_predict)

print(outcome)