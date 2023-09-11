# This is to test importing, encoding and predicting of just Enlgish. 
# Change subject and X y on liness 11, 24, 26
# Change prediction subject and X on lines 48, 55
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

subject = 'english_language'

subject_full = subject+'_2122.csv'

predictor = pd.read_csv('grades_full_clean/' + subject_full)

gender_encoded = pd.get_dummies(predictor['gender_ap2'], prefix='Gender')

# Concatenate the encoded gender columns with the original DataFrame
predictor_final = pd.concat([predictor, gender_encoded], axis=1)

# Drop the original "Gender" column if needed
predictor_final.drop(columns=['gender_ap2'], inplace=True)

X = predictor_final.drop(columns=['upn', 'english_language_gcse_level_1/2_real'])

y = predictor_final['english_language_gcse_level_1/2_real']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

model = Ridge(alpha=1)      

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_rounded = np.round(y_pred)

y_pred_true = y_pred_rounded.astype(int)

mse = mean_squared_error(y_test, y_pred_true)

# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred_true, squared=False)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred_true)

kf = KFold(n_splits=5, shuffle=True, random_state=68)

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
# the cross-validated scores are very similar, reducing the chance that the model is overfitted

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
print('Cross-validation scores:', scores)

train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
                                                            
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
plt.ylim([0, 1])
plt.show()
plt.savefig("overfit_graph/" + subject + "_ridge.png", )
plt.clf()
print('Graph saved')

subject_prediction = subject + '_2223.csv'

prediction = pd.read_csv('to_be_predicted/' + subject_prediction)

prediction.rename(columns={'actual_ap2': 'actual_real',
                           'estimate_ap2': 'estimate_real'}, inplace=True)

gender_encoded_prediction = pd.get_dummies(prediction['gender_ap2'], prefix='Gender')

# Concatenate the encoded gender columns with the original DataFrame
prediction_final = pd.concat([prediction, gender_encoded_prediction], axis=1)

# Drop the original "Gender" column if needed
prediction_final.drop(columns=['gender_ap2'], inplace=True)

X_prediction = prediction_final.drop(columns=['upn'])

y_prediction =model.predict(X_prediction)

y_prediction_rounded = np.round(y_prediction)

subject_prediction_column_name = subject + '_prediction'
prediction[subject_prediction_column_name] = y_prediction_rounded.astype(int)

prediction.to_csv('predicted/'+subject+'_2223_prediction.csv')
print('Prediction saved')