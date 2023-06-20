# at some point this needs to be returned to grades_packages, but was having issues with the installing and for now would rather work on the models
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.preprocessing import OneHotEncoder

# takes in a data frame and a type and encoudes the data
#  dataframe is the imported dataframe to use
# type is to check if combined or triple
# train is to return X and y
# file is to return a label encoder file
def le_science(dataframe, type, train=True, file=False):
    learning_grades = dataframe.copy()
    if type.lower()[0] == 'c':
        columns_to_encode = ['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2', 'Combined MOCK GRADE Term 4']
        le = LabelEncoder()
        for col in columns_to_encode:
            learning_grades[col] = le.fit_transform(dataframe[col])
        if train:
            X = learning_grades[['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']]
            y = learning_grades['Combined MOCK GRADE Term 4']
    elif type.lower()[0] == 't':
        columns_to_encode = ['FFT20', 'year 10 bio grade', 'year 10 chem grade', 'year 10 phys grade', 
                            'year 11 paper 1 bio grade', 'year 11 paper 1 chem grade', 'year 11 paper 1 phys grade',
                            'year 11 paper 2 bio grade', 'year 11 paper 2 chem grade', 'year 11 paper 2 phys grade']
        for col in columns_to_encode:
            learning_grades[col] = le.fit_transform(dataframe[col])
        if train:
            X = learning_grades[['year 10 bio grade', 'year 10 chem grade', 'year 10 phys grade', 
                            'year 11 paper 1 bio grade', 'year 11 paper 1 chem grade', 'year 11 paper 1 phys grade']]
            y = learning_grades[['year 11 paper 2 bio grade', 'year 11 paper 2 chem grade', 'year 11 paper 2 phys grade']]
    else:
        print('Neither science selected')
    if train:
        if not file:
            return learning_grades, X, y
        else:
            return learning_grades, X, y, le

    else:
        if not file:
            return learning_grades
        else:
            return learning_grades, le

# returns data from an encoding
def le_return(data, le):
        return le.inverse_transform(data)

# to save an encoder
def le_save(le, filename):
    joblib.dump(le, filename+'.joblib')

# to load an encoder
def le_load(filename):
    return joblib.load

def one_hot_fit(dataframe, type, train=True):
    if type.lower()[0] == 'c':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X = dataframe.drop('Combined MOCK GRADE Term 4', axis=1)
        y = dataframe['Combined MOCK GRADE Term 4']
        X_encoded = encoder.fit_transform(X)
        joblib.dump(encoder, 'testing.joblib')
        return encoder, X_encoded, y
    elif type.lower()[0] == 't':
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        columns_to_encode = ['FFT20', 'year 10 bio grade', 'year 10 chem grade', 'year 10 phys grade', 
                            'year 11 paper 1 bio grade', 'year 11 paper 1 chem grade', 'year 11 paper 1 phys grade',
                            'year 11 paper 2 bio grade', 'year 11 paper 2 chem grade', 'year 11 paper 2 phys grade']
        train_encoded = dataframe.copy()
        for col in columns_to_encode:
            train_encoded[col] = encoder.fit_transform(dataframe[col].values.reshape(-1, 1))
        X = train_encoded.loc[:, ['FFT20', 'year 10 bio grade', 'year 10 chem grade', 'year 10 phys grade', 
                            'year 11 paper 1 bio grade', 'year 11 paper 1 chem grade', 'year 11 paper 1 phys grade',
                            'year 11 paper 2 bio grade', 'year 11 paper 2 chem grade', 'year 11 paper 2 phys grade']]
        y = dataframe[['year 11 paper 2 bio grade', 'year 11 paper 2 chem grade', 'year 11 paper 2 phys grade']]
        return train_encoded, X, y
     
    else:
        print('Neither science selected')

def one_hot_inverse(data, encoder):
    decoded = encoder.inverse_transform(data)
    return decoded

def new_data_one_hot(data, encoder):
    return encoder.transform(data)
