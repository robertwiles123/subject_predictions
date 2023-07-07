import encoding
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt
from joblib import dump
# from sklearn.model_selection import GridSearchCV

file_name = input('What file do you want to test? ')
learning_grades = pd.read_csv('../csv_clean/' + file_name + '.csv')

type_science = input('Is it triple or combined? ')

encoder, X, y = encoding.one_hot_fit(learning_grades, type_science)

if type_science.lower()[0] == 't':
    y = y.values.ravel()

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
"""
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [50, 100, 200],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'max_samples': [None, 0.5, 0.8],
    'bootstrap': [True, False],
    'random_state': [42]
}

# Starting model
rf_model = RandomForestRegressor()

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding score
print("Best hyperparameters:")
print(grid_search.best_params_)
print("Best Score (Negative MSE):", -grid_search.best_score_)
"""
if type_science.lower()[0] == 'c':
    rf = RandomForestRegressor(random_state=42, bootstrap=False, max_depth=10, max_features='sqrt', max_samples=None, min_samples_leaf=1, min_samples_split=2, n_estimators=200)
elif type_science.lower()[0] == 't':
    rf = RandomForestRegressor(bootstrap=False, max_depth=10, max_features='log2', max_samples=None, min_samples_leaf=1, min_samples_split=2, n_estimators=50, random_state=42)
else:
    print('No model selected')
    

rf.fit(X_train, y_train)

y_pred_unrounded = rf.predict(X_test)

y_pred = np.vectorize(lambda x: round(x * 2) / 2)(y_pred_unrounded)

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
plt.show()
plt.savefig("../model_graphs/" + file_name + "_random.png", )

# It seems the model is good for making predictions if there are enough data points. I will continue to work on this if there is more data avalible

save = input('should it be saved? ')
if save[0].strip().lower() == 'y':
    if type_science.lower()[0] == 'c':
        dump(rf, '../combined_models/combined_random_forest.joblib')
        dump(encoder, '../combined_models/combined_random_forest_encoding.joblib')
        print('Model save')
    else:
        dump(rf, '../triple_models/triple_random_forest.joblib')
        dump(encoder, '../triple_models/triple_random_forest_encoding.joblib')
        print('Model save')
