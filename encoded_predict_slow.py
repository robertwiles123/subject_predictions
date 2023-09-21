# Work with predicting and encoding any subject that has numerical grades automated
# change the subject on line 26 for now
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import re

model_name = 'linear_regression'

# Define subject and year here
subjects = ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science', 'drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'maths', 'music_studies', 'physics', 'spanish']
"""
Removed subjects
d_&_t_product_design
d_&_t_textiles_technology
ict_btec
music_tech_grade
pearson_btec_sport
product_design
"""

for topic in subjects:

    subject = topic
    year_model = '2122'
    year_prediction = '2223'

    # regex for none standard grade endings

    regex_pattern = re.compile(rf'{subject}.*_real$')

    # Load the data for the model
    subject_model = subject + '_' + year_model + '.csv'
    predictor = pd.read_csv('grades_full_clean/' + subject_model)

    # Load the data for prediction
    subject_prediction = subject + '_' + year_prediction + '.csv'
    prediction = pd.read_csv('to_be_predicted/' + subject_prediction)

    gender_encoded = pd.get_dummies(predictor['gender_ap2'], prefix='Gender')

    # Concatenate the encoded gender columns with the original DataFrame
    predictor_final = pd.concat([predictor, gender_encoded], axis=1)

    # Drop the original "Gender" column if needed
    predictor_final.drop(columns=['gender_ap2'], inplace=True)

    X = predictor_final.drop(columns=['upn'] + [col for col in predictor_final.columns if regex_pattern.match(col)])

    y_column_pattern = subject + '_.*_real$'

    y = predictor_final.filter(regex=y_column_pattern)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    model = LinearRegression()

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

    scores.sort()

    cross_val_mean = np.mean(scores)

    mae = mean_absolute_error(y_test, y_pred)

    print(f'{subject} scores:')
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("Mean Absolute Error:", mae)
    print(f"R-squared (R2) Score: {r2:.2f}")
    print('Cross-validation scores:', scores)
    print(f'Mean cross validation scores: {cross_val_mean}')

    # dict to save scores
    scores_dict = {
        'subject': subject,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'Mean Absolute Error': mae,
        'Cross-validation': scores,
        'Mean cross validation': cross_val_mean
    }
    # load excel sheet
    scores_df = pd.read_csv(f'{model_name}_scores/scores.csv')

    # Create a mask to filter out rows with the same subject
    mask = scores_df['subject'] != subject

    # Remove rows with the same subject from scores_df
    scores_df = scores_df[mask]

    new_scores_df = pd.DataFrame([scores_dict])

    # update excel sheet will make copy of same suject if there
    scores_df = pd.concat([scores_df, new_scores_df], ignore_index=True)

    scores_df.reset_index(drop=True, inplace=True)

    # Sort the DataFrame by the "subject" column
    scores_df = scores_df.sort_values(by='subject')

    scores_df.to_csv(f'{model_name}_scores/scores.csv', index=False, float_format='%.3f')

    print('Scores updated in scores.xlsx')

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
    plt.savefig(f"{model_name}_scores/{subject}_{model}.png", )
    plt.clf()
    print('Graph saved')

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
    prediction.to_csv(f'{model_name}_predicted/' + subject + '_' + year_prediction + '_prediction.csv')
    print(f'{subject} Prediction saved')


    # Graphs that shows all predicted errors

    # Convert y_test to a NumPy array
    y_test_array = y_test.to_numpy()

    # Calculate the residuals (the differences between true and predicted values)
    residuals = y_test_array - y_pred_true

    # Calculate the absolute errors (useful for visualization)
    absolute_errors = np.abs(residuals)

    # Find the indices of the data points where the model made incorrect predictions (absolute error > 0)
    incorrect_indices = np.where(absolute_errors > 0)

    # Extract the corresponding values from y_test_array and y_pred_true for incorrect predictions
    incorrect_y_test = y_test_array[incorrect_indices]
    incorrect_y_pred = y_pred_true[incorrect_indices]

    # Now, you have the true and predicted values for all data points where the model made incorrect predictions.
    # You can use these values for visualization or further analysis.
    try:
        # Adjust the axis increments (both axis increase by 1)
        plt.xticks(np.arange(min(incorrect_y_test), max(incorrect_y_test)+1, 1))
        plt.yticks(np.arange(min(incorrect_y_pred), max(incorrect_y_pred)+1, 1))
    except ValueError:
        continue

    # Make grid lines visible
    plt.grid(True)

    # Visualize the errors, for example, in a scatter plot
    plt.scatter(incorrect_y_test, incorrect_y_pred, label='Incorrect Predictions', color='red')
    plt.title(f'{model_name.capitalize()} {subject} wrong predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(f'error_graphs/{subject}_{model_name}.png')
    print('Errors saved')