import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt

learning_grades = pd.read_csv('clean_prediction.csv')

le = LabelEncoder()
learning_grades['Year 10 Combined MOCK GRADE'] = le.fit_transform(learning_grades['Year 10 Combined MOCK GRADE'])
learning_grades['Combined MOCK GRADE term 2'] = le.fit_transform(learning_grades['Combined MOCK GRADE term 2'])
learning_grades['Combined MOCK GRADE Term 4'] = le.fit_transform(learning_grades['Combined MOCK GRADE Term 4'])

X = learning_grades[['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2']]
y = learning_grades['Combined MOCK GRADE Term 4']

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

rc = RandomForestClassifier(random_state=1)

bc = BaggingClassifier(estimator=rc, n_estimators=50, n_jobs=-1)

bc.fit(X_train, y_train)

y_pred = bc.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("R2 score:", r2)

#The above R2 scoreis so hight I will be checking that the data is not overfit

kfold = KFold(n_splits=10, shuffle=True, random_state=15)
scores = cross_val_score(bc, X, y, cv=kfold, scoring='r2')
#the cross-validated scores are very similar, reducing the chance that the model is overfitted
print('Cross-validation scores:', scores)

train_sizes, train_scores, test_scores = learning_curve(bc, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                                                        
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
                                                        
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training score')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test score')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()

#this is definelty overfitting and should not be used