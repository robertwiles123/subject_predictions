# As the model is only accurate after about 20 samples, and due to the high R2 score there is a possiblity of over fitting I will also run a linear regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import encoding
# using joblib as it is more secure and will be predicting data based on GDPR and as all within sklearn the compatability isn't an issue
from joblib import dump

# import whatever file I want to test
file_name = input('What file do you want to test? ')

learning_grades = pd.read_csv(file_name)

# determine the type of science, triple and combined have different headings
type_science = input('Is it triple or combined? ')

# uses code to return an encoded dataframe as well as an X and y catagory
# does not predict new grades well, probable as there are grades this could not see
# learning_grades, X, y = encoding.le_science(learning_grades, type_science)

# the label encoder was not effective at creating predicted grades. When chaging to a one hot encoder the model became less effective
# May be worth adding in new variables and seeing how this affects the data
encoder, X, y = encoding.one_hot_fit(learning_grades, type_science)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create a linear regression object
lr = LinearRegression()

# fit the model to the training data
lr.fit(X_train, y_train)

# make predictions on the test data
y_pred = lr.predict(X_test)

# evaluate the model performance using mean squred error, root and r squared
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


y_train_size = y_pred.shape[0]
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print('Train MSE {:.2f}'.format(mean_squared_error(y_train[:y_train_size], y_pred)))
print('testMSE {:.2f}'.format(mean_squared_error(y_test, y_pred)))
print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("R2 score:", r2)

# this again has an extremely high R2 score, so may be over fit

# create a kfold to check scores are similar
kfold = KFold(n_splits=10, shuffle=True, random_state=15)
scores = cross_val_score(lr, X, y, cv=kfold, scoring='r2')
# the cross-validated scores are very similar, reducing the chance that the model is overfitted
print('Cross-validation scores:', scores)

# create a learning curve to see if data is overfitting
train_sizes, train_scores, test_scores = learning_curve(lr, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                                                        
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
plt.ylim([0.99, 1.01])
plt.show()


print(encoder)
# if data is good give option to save and name to save
save = input('should it be saved? ')
if save[0].strip().lower() == 'y':
    name = input('what is the name of the program? ')
    dump(lr, name+'.joblib')
    encoder = input('name encoder ')
    dump(encoder, encoder+'.joblib')

