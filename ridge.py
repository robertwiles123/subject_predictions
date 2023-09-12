"""
Subjects that seems to be doing well are:
"music_tech_grade," "biology," "history," "german," "d_&_t_textiles_technology," "geography," "business_studies," and "computer_science" 
Though none are doing amazing
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import re
import numpy as np
from sklearn.model_selection import KFold
import joblib


# does not include double science due to issue with data collection
# Define a list of subjects
subjects = ['english_language', 'english_literature', 'maths', 'biology', 'chemistry', 'computer_science', 'french_language', 'geography', 'german', 'history', 'physics', 'spanish', 'art_&_design', 'business_studies', 'd_&_t_product_design', 'd_&_t_textiles_technology', 'drama', 'food_technology', 'ict_btec', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'product_design']

# change to year of year 11 to create predictions.
year = '2122'
year = '_' + year

# Specify the folder path where the CSV files are located
folder_path = "/workspaces/subject_predictions/grades_full_clean"

# Iterate through CSV files in the folder
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".csv") and year in file:
            # Extract the subject name from the file name
            subject = file.split(year)[0]           
            # Check if the subject is in the list of subjects
            if subject in subjects:
               # Read the CSV file
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)

                columns_to_encode = []

                for column in df.columns:
                    if column != 'upn' and not pd.api.types.is_integer_dtype(df[column]) and not pd.api.types.is_float_dtype(df[column]):
                        columns_to_encode.append(column)
                print(f'columns to encode {columns_to_encode}')

                df_encoded = pd.DataFrame()

                for column in columns_to_encode:
                    if column == 'gender_ap2':
                        prefix = 'Gender'
                    else:
                        prefix = f'{subject}_{column.split("_")[2]}'  # Extract the number from the column name
                    
                    encoded_column = pd.get_dummies(df[column], prefix=prefix)
                    df_encoded = pd.concat([df_encoded, encoded_column], axis=1)

                # Concatenate the encoded gender columns with the original DataFrame
                df_final = pd.concat([df, df_encoded], axis=1)

                # Drop the original "Gender" column if needed
                df_final.drop(columns=columns_to_encode, inplace=True)

                pattern = rf'^{subject}.*real$'

                # Use the filter method to drop columns matching the pattern
                columns_to_drop = df_final.filter(regex=pattern)

                columns_to_drop_list = columns_to_drop.columns.tolist()

                columns_to_drop_list.append('upn')

                print(f'columns to drop {columns_to_drop_list}')

                print(f'Final columns{df_final.columns}')

                X = df_final.drop(columns=columns_to_drop_list)

                y = df_final[columns_to_drop.columns]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

                # Print the shapes of X_train and y_train for debugging
                ridge_model = Ridge(alpha=0.5)      
                ridge_model.fit(X_train, y_train)
                y_pred = ridge_model.predict(X_test)
                y_pred = np.round(y_pred)

                

                # Calculate Mean Squared Error (MSE)
                mse = mean_squared_error(y_test, y_pred)

                # Calculate Root Mean Squared Error (RMSE)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                # Calculate R-squared (R2) score
                r2 = r2_score(y_test, y_pred)

                # Initialize a KFold cross-validator with 5 folds
                kf = KFold(n_splits=5)

                mse_scores = []

                # Perform cross-validation
                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    
                    # Fit the model on the training data
                    ridge_model.fit(X_train, y_train)
                    
                    # Make predictions on the test data
                    y_pred = ridge_model.predict(X_test)
                    
                    # Calculate the Mean Squared Error for this fold
                    mse = mean_squared_error(y_test, y_pred)
                    mse_scores.append(mse)

                # Calculate the average MSE across all folds
                average_mse = sum(mse_scores) / len(mse_scores)
                
                print(subject)
                print(f"Mean Squared Error (MSE): {mse:.2f}")
                print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                print(f"R-squared (R2) Score: {r2:.2f}")
                print(f"Cross value score: {average_mse:.2f}")
                print()


                model_filename = '/workspaces/subject_predictions/models/' + subject + '_ridge.joblib'

                joblib.dump(ridge_model, model_filename)
                
                print('Model saved')
                print()
                print()
                print()