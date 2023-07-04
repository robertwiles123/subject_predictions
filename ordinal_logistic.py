import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from grades_packages import encoding
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

combined = pd.read_csv('csv_clean/ordinal_clean_combined.csv')

X_ordinal, y = encoding.ordinal_encoder_combined(combined)

# convert bool to np array
PP_array = combined['PP'].values.reshape(-1, 1)
SEN_bool_array = combined['SEN bool'].values.reshape(-1, 1)
' combined features'
X = np.concatenate((X_ordinal, PP_array, SEN_bool_array), axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=64)

model = LogisticRegression(multi_class='multinomial', solver='newton-cg')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

                                                    
save = input('should it be saved? ')
if save[0].strip().lower() == 'y':
    dump(model, 'ordinal_combined_models/logistic_combined_.joblib')
    print('Model save')
