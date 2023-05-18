import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

combined = pd.read_csv('combined_clean.csv')


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
        #currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
        elif isinstance(data, pd.DataFrame):
            le = LabelEncoder()
            encoded_data = data.copy()
            col_to_encode = ['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2', 'Combined MOCK GRADE Term 4']
            combined_data = pd.concat([combined, encoded_data])
            for col in col_to_encode:
                combined_data[col] = le.fit_transform(combined_data[col])
            encoded_prediction_data = combined_data[len(combined):]
            encoded_prediction_data = encoded_prediction_data[['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']]
            prediction = v.predict(encoded_prediction_data)

            # to be used when i have more info and can reverse the data
            # outcomes[k] = le.inverse_transform(prediction)
            # being used to check that something is being outputted
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