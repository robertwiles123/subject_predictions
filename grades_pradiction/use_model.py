import pandas as pd
from joblib import load
import encoding
import numpy as np

data = input('What do you want to predict? ')
# clean combined dataframe

# current models I have, though this will be added to and taken away with more data, with accuracy
combined_models_to_predict = ['combined_linear']
combined_models_to_predict_dict = {}

# populate a dict of model with the file so can apply them later as neeeded
for s in combined_models_to_predict:
    model = load(s + '.joblib')
    combined_models_to_predict_dict[s] = model

# check that the dict of files and models are correct
print(combined_models_to_predict_dict)

# for later, can take input of grades if prefered
# data = input('What data do you want to use? If a file, include .csv: ')

if data.endswith('.csv'):
    data = pd.read_csv(data)

outcomes = {}
for k, v in combined_models_to_predict_dict.items():
    # currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
    if isinstance(data, str):
        print('oops')
    elif isinstance(data, pd.DataFrame):
        # combined and trimple have different encodings
        encoder = load('testing.joblib')
        encoded_data = encoding.new_data_one_hot(data, encoder)
        prediction = v.predict(encoded_data)
        print(type(prediction))
        pred_df = pd.DataFrame({k+'Predicted grades': prediction})
        df_with_predictions = pd.concat([data, pred_df], axis=1)

    else:
        print('Neither')

columns_to_skip = encoding.combined_columns()
        for column in df_with_predictions.columns:
            if column not in columns_to_skip:
                df_with_predictions[column] = df_with_predictions[column].apply(lambda x: round(x * 2) / 2)
        print(df_with_predictions)