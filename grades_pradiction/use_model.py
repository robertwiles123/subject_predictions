import pandas as pd
from joblib import load
import encoding
import grades_pradiction.df_columns as df_columns

data = input('What do you want to predict? ')
# clean combined dataframe
type = input('Is the data for combined or triple? ')

# current models I have, though this will be added to and taken away with more data, with accuracy
if type.lower()[0] == 'c':
    combined_models_to_predict = ['combined_linear', 'random_forest', 'descition_tree']
    combined_models_to_predict_dict = {}
elif type.lower()[0] == 't':
    tripe_models_to_predict = ['combined_linear_triple', 'random_forest_triple', 'descition_tree_tripe']
    triple_models_to_predict_dict = {}
else:
    print('Failed to detarmine, no model loaded')
    

# populate a dict of model with the file so can apply them later as neeeded
for s in combined_models_to_predict:
    model = load(s + '.joblib')
    combined_models_to_predict_dict[s] = model

# check that the dict of files and models are correct
print(combined_models_to_predict_dict)

if data.endswith('.csv'):
    data = pd.read_csv(data)


outcomes = {}
if type.lower()[0] == 'c':
    for k, v in combined_models_to_predict_dict.items():
        # currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
        if isinstance(data, str):
            print('oops')
        elif isinstance(data, pd.DataFrame):
                encoder = load(k + '_encoding.joblib')
                encoded_data = encoding.new_data_one_hot(data, encoder)
                prediction = v.predict(encoded_data)
                pred_df = pd.DataFrame({k+'_predicted grades': prediction})
                outcomes[k] = prediction
        else:
                print('Error')
elif type.lower()[0] == 't':
    for k, v in combined_models_to_predict_dict.items():
        # currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
        if isinstance(data, str):
            print('oops')
        elif isinstance(data, pd.DataFrame):
                encoder = load(k + '_encoding_triple.joblib')
                encoded_data = encoding.new_data_one_hot(data, encoder)
                prediction = v.predict(encoded_data)
                pred_df = pd.DataFrame({k+'Predicted grades': prediction})
                data = pd.concat([data, pred_df], axis=1)
        else:
             print('error')

             
df_with_predictions = data.assign(**outcomes)

columns_to_skip = df_columns.combined_columns()
for column in df_with_predictions.columns:
    if column not in columns_to_skip:
        df_with_predictions[column] = df_with_predictions[column].apply(lambda x: round(x * 2) / 2)

print(df_with_predictions)
