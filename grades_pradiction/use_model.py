import pandas as pd
from joblib import load
import encoding
import numpy as np

data = input('What do you want to predict? ')
# clean combined dataframe
combined = pd.read_csv('combined_clean.csv')

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
        type = input('Is the dataframe combined or triple? ')
        """combined_data = pd.concat([combined, data])
        encoded_prediction_data, encoder = encoding.le_science(combined_data, type, train=False, file=True)
        encoded_prediction_data = combined_data.loc[data.index]
        encoded_prediction_data = encoded_prediction_data[['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']]
        prediction = v.predict(encoded_prediction_data)
        # not able to return encoded data, test file includes data it would have seen 
        closest_list = []
        print(f"lenght prediction {len(prediction)} \n length encoded {len(data)} \n length combined {len(combined_data)}")
        print(prediction

        for i in range(len(encoded_prediction_data)):
            try:
                outcomes[k] = encoding.le_return(prediction[i], encoder)
            except (ValueError, KeyError):
                distances = np.abs(encoder.classes_ - encoded_prediction_data.iloc[i].values.reshape(-1, 1))
                closest_index = np.argmin(distances)
                closest_value = encoder.classes_[closest_index]
                print(f"closest value - {closest_value} \n predict i = {prediction[i]}")
                closest_list.append(closest_value)
                print(closest_list)
                if i == (len(encoded_prediction_data) - 2):
                    print(f"in the loop: {closest_list}")
                    closest_value_unencoded = encoding.le_return(np.array([closest_list]), encoder)
                    # Replace the prediction with the closest neighbor
                    encoded_prediction_data.iloc[i] = closest_value
                    # Update outcomes with the unencoded closest neighbor
                    outcomes[k] = closest_value_unencoded
                    """
        encoder = load('testing.joblib')
        encoded_data = encoding.new_data_one_hot(data, encoder)
        prediction = v.predict(encoded_data)
        expanded_prediction = np.expand_dims(prediction, axis=1)
        joined_encoded = np.concatenate((encoded_data, expanded_prediction), axis=1)
        unencoded = encoder.inverse_transform(joined_encoded[:, 1:])
        # this seems to be creating a np array of the correct data, however, it is probably missing the predictin data
        print(unencoded)

    else:
        print('Neither')
