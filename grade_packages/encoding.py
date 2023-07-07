import pandas as pd
import df_columns
from sklearn.preprocessing import OneHotEncoder

def one_hot_fit(df, type):
    type = type.lower()[0]
    if type == 'c':
        encoder = OneHotEncoder()
        X_columns = df_columns.combined_dependent()
        y_columns = df_columns.combined_independent()
        bool_columns = df_columns.combined_bool()
        X = df[X_columns]
        y = df[y_columns]
        X = X.drop(bool_columns, axis=1)
        bool_data = df[bool_columns]
        X_encoded = encoder.fit_transform(X)
        encoded_names = encoder.get_feature_names_out(X.columns)
        # Convert the encoded X matrix to a DataFrame
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_names)
        # Rejoin the bool_data DataFrame with X_encoded_df
        X_with_bool = pd.concat([X_encoded_df, bool_data], axis=1)
        return encoder, X_with_bool, y
    elif type.lower()[0] == 't':
        encoder = OneHotEncoder()
        X_columns = df_columns.triple_dependent()
        y_columns = df_columns.triple_independent()
        bool_columns = df_columns.triple_bool()
        X = df[X_columns]
        y = df[y_columns]
        X = X.drop(bool_columns, axis=1)
        bool_data = df[bool_columns]
        X_encoded = encoder.fit_transform(X)
        encoded_names = encoder.get_feature_names_out(X.columns)
        # Convert the encoded X matrix to a DataFrame
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_names)
        # Rejoin the bool_data DataFrame with X_encoded_df
        X_with_bool = pd.concat([X_encoded_df, bool_data], axis=1)#
        return encoder, X_with_bool, y
    else:
        print('No correct type chosen')
    
def new_data_one_hot(df, encoder, type):
    if type.lower()[0] == 'c':
        X_columns = df_columns.combined_dependent()
        bool_columns = df_columns.combined_bool()
        X = df[X_columns]
        X = X.drop(bool_columns, axis=1)
        bool_data = df[bool_columns]
        X_encoded = encoder.transform(X)
        encoded_names = encoder.get_feature_names_out(X.columns)
        # Convert the encoded X matrix to a DataFrame
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_names)
        # Rejoin the bool_data DataFrame with X_encoded_df
        X_with_bool = pd.concat([X_encoded_df, bool_data], axis=1)# Convert the encoded X matrix to a DataFrame
        return X_with_bool
    elif type.lower()[0] == 't':
        X_columns = df_columns.triple_dependent()
        bool_columns = df_columns.triple_bool()
        X = df[X_columns]
        X = X.drop(bool_columns, axis=1)
        bool_data = df[bool_columns]
        X_encoded = encoder.transform(X)
        encoded_names = encoder.get_feature_names_out(X.columns)
        # Convert the encoded X matrix to a DataFrame
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoded_names)
        # Rejoin the bool_data DataFrame with X_encoded_df
        X_with_bool = pd.concat([X_encoded_df, bool_data], axis=1)# Convert the encoded X matrix to a DataFrame
        return X_with_bool
    else:
        print('Incorrect type')
