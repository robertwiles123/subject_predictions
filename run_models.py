import pandas as pd
from grade_packages import encoding
import models as md
from sklearn.model_selection import train_test_split
import importlib

type = input("Combined or triple? ")

type_science = type.lower()[0]

while True:
    if type_science == 'c':
        learning_grades = pd.read_csv('csv_clean/clean_combined.csv')
        break
    elif type_science == 't':
        learning_grades = pd.read_csv('csv_clean/clean_triple.csv')
        break
    else:
        print('Incorrect type of science')
        type = input("Combined or triple? ")
        type_science = type.lower()[0]

encoder, X, y = encoding.one_hot_fit(learning_grades, type_science)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

training_models = ['linear_regression', 'descition_tree', 'random_forest', 'ridge']

if type_science == 'c':
    file_name = 'combined'
else:
    file_name = 'triple'

for model in training_models:
    print(f"Scored for {model}:")
    print()
    module_name = 'models.' + model
    module = importlib.import_module(module_name)
    function_to_call = getattr(module, model)
    function_to_call(X, y, X_train, X_test, y_train, y_test, type_science, encoder, file_name)
    print()
    print()

print('All models complete')
    