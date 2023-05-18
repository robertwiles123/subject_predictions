import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

combined_models_to_predict = ['combined_linear']
combined_models_to_predict_dict = {}

for s in combined_models_to_predict:
    model = load(s + '.joblib')
    combined_models_to_predict_dict[s] = model

print(combined_models_to_predict_dict)

data = input('What data do you want to use? If a file, include .csv: ')

if data.endswith('.csv'):
    data = pd.read_csv(data)

def run_test(data, models=combined_models_to_predict_dict):
    outcomes = {}
    for k, v in combined_models_to_predict_dict.items():
        if isinstance(data, str):
            print('oops')
        elif isinstance(data, pd.DataFrame):
            le = LabelEncoder()
            encoded_data = data.copy()
            col_to_encode = ['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']
            for col in col_to_encode:
                encoded_data[col] = le.fit_transform(data[col])
            prediction = v.predict(encoded_data)
            print(prediction)
            outcomes[k] = prediction
        else:
            print('Neither')
    return outcomes

outcome = run_test(data)
print(outcome)

# Convert the outcome dictionary to a DataFrame
outcome_df = pd.DataFrame(outcome)

# Join the predicted values with the original DataFrame
predicted_data = data.join(outcome_df)

print(predicted_data)