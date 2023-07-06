import pandas as pd
from joblib import load
from grade_packages import df_columns, encoding

data = input('What do you want to predict? ')
# clean combined dataframe
type = input('Is the data for combined or triple? ')


models_to_predict = ['linear', 'random_forest', 'descition_tree', 'ridge']
# current models I have, though this will be added to and taken away with more data, with accuracy
if type.lower()[0] == 'c':
    combined_models_to_predict_dict = {}
    for s in models_to_predict:
        model = load('combined_models/combined_' + s + '.joblib')
        combined_models_to_predict_dict[s] = model
    # check that the dict of files and models are correct
    print(combined_models_to_predict_dict)
elif type.lower()[0] == 't':
    triple_models_to_predict_dict = {}
    # populate a dict of model with the file so can apply them later as neeeded
    for s in models_to_predict:
        model = load('triple_models/triple_' + s + '.joblib')
        triple_models_to_predict_dict[s] = model
        # check that the dict of files and models are correct
    print(triple_models_to_predict_dict)
else:
    print('Failed to detarmine, no model loaded')
    

if data.endswith('.csv'):
    data = pd.read_csv(data)


outcomes = {}
if type.lower()[0] == 'c':
    for k, v in combined_models_to_predict_dict.items():
        # currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
        if isinstance(data, str):
            print('oops')
        elif isinstance(data, pd.DataFrame):
                encoder = load('combined_models/combined_' + k + '_encoding.joblib')
                encoded_data = encoding.new_data_one_hot(data, encoder)
                prediction = v.predict(encoded_data)
                prediction = prediction.ravel()
                pred_df = pd.DataFrame({k+'_predicted grades': prediction})
                outcomes[k] = prediction
        else:
                print('Error')
    df_with_predictions = data.assign(**outcomes)
    columns_to_skip = data.columns
    for column in df_with_predictions.columns:
        if column not in columns_to_skip:
            print(column)
            df_with_predictions[column] = df_with_predictions[column].apply(lambda x: round(x * 2) / 2)
            # to return predicted grades back to how they should be, however, not doing original files
            grade_mapping = {
                                    0: 'U',
                                    1.0: '1-1',
                                    1.5: '2-1',
                                    2.0: '2-2',
                                    2.5: '3-2',
                                    3.0: '3-3',
                                    3.5: '4-3',
                                    4.0: '4-4',
                                    4.5: '5-4',
                                    5.0: '5-5',
                                    5.5: '6-5',
                                    6.0: '6-6',
                                    6.5: '7-6',
                                    7.0: '7-7',
                                    7.5: '8-7',
                                    8.0: '8-8',
                                    8.5: '9-8',
                                    9.0: '9-9'
                                }
            df_with_predictions.loc[:, column] = df_with_predictions[column].map(grade_mapping)
    # Forces display to print the whole df
    pd.set_option('display.max_columns', None)
    print(df_with_predictions)
elif type.lower()[0] == 't':
    df_with_predictions = pd.DataFrame()
    for k, v in triple_models_to_predict_dict.items():
        print(k, v)
        # currently does not work, as I do not have enough training data and have missing encoudings. This happens with one shot encouding as well
        if isinstance(data, str):
            print('oops')
        elif isinstance(data, pd.DataFrame):
        # Doesn't work, probably because 3 numpy array at a guess
                encoder = load('triple_models/triple_' + k + '_encoding.joblib')
                encoded_data = encoding.new_data_one_hot(data, encoder)
                prediction = v.predict(encoded_data)
                outcomes[k] = prediction
        #need to take the dict and change in to a dataframe
    for key, value in outcomes.items():
        names = ['bio', 'chem', 'phys']
    # Iterate over each predicted output column
        for i in range(value.shape[1]):
            col_name = f"{key}_{names[i]}"  # Create column name using the key and index
            df_with_predictions[col_name] = value[:, i]  # Add the predicted output column to the dataframe
    # To round columns that where predicted    
    for column in df_with_predictions.columns:
        df_with_predictions[column] = df_with_predictions[column].apply(lambda x: round(x))
    print(df_with_predictions)
        
    



         

