# could try importing more features, such as FFT20, intervention, PPE etc. and looking at importance to see what has the greatest effect on predicting grades
# need to add in more features and compare which is most useful, things like FFT20, PPE, inteventions using something like bellow
"""
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()
"""
import encoding
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt

file_name = input('What file do you want to test? ')
learning_grades = pd.read_csv(file_name)

type_science = input('Is it triple or combined? ')

learning_grades, X, y = encoding.le_science(learning_grades, type_science)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=1, n_estimators=50, max_depth=3, min_samples_leaf=3)


rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE): {:.2f}".format(mse))
# thows an error due to difference in size, needs to be looked at
y_train_size = y_pred.shape[0]
print('Train MSE {:.2f}'.format(mean_squared_error(y_train[:y_train_size], y_pred)))
print('testMSE {:.2f}'.format(mean_squared_error(y_test, y_pred)))

print("Root Mean Squared Error (RMSE): {:.2f}".format(rmse))
print("R2 score:", r2)


# The above R2 scoreis so hight I will be checking that the data is not overfit

kfold = KFold(n_splits=10, shuffle=True, random_state=15)
scores = cross_val_score(rf, X, y, cv=kfold, scoring='r2')
# the cross-validated scores are very similar, reducing the chance that the model is overfitted
print('Cross-validation scores:', scores)

train_sizes, train_scores, test_scores = learning_curve(rf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

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

# It seems the model is good for making predictions if there are enough data points. I will continue to work on this if there is more data avalible
