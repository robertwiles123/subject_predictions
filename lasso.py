"""
Subjects that seems to be doing well are:
"music_tech_grade," "biology," "history," "german," "d_&_t_textiles_technology," "geography," "business_studies," and "computer_science" 
Though none are doing amazing
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import re
import numpy as np
from sklearn.model_selection import KFold

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
                print(subject)
               # Read the CSV file
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                df_unchanged = df.copy()
                # List of categorical columns to be one-hot encoded (excluding 'UPN')
                categorical_columns = [col for col in df.columns if col != 'upn']
                
                # Use pandas get_dummies to perform one-hot encoding
                data_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True) 
                # Construct the regex pattern
                regex_pattern = f'^.*{subject}.*_real.*$'

                matching_columns = data_encoded.columns

                y_columns = [col for col in matching_columns if re.match(regex_pattern, col)]
                # Filter the DataFrame based on the selected columns
                y = data_encoded[y_columns]

                # Define the features (X)
                X = data_encoded.drop(columns=['upn'] + y.columns.tolist())
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

                # Print the shapes of X_train and y_train for debugging
                lasso_model = Lasso(alpha=1.0, random_state=62)    
                lasso_model.fit(X_train, y_train)
                y_pred = lasso_model.predict(X_test)
                y_pred = np.round(y_pred)

                # Calculate Mean Squared Error (MSE)
                mse = mean_squared_error(y_test, y_pred)

                # Calculate Root Mean Squared Error (RMSE)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                # Calculate R-squared (R2) score
                r2 = r2_score(y_test, y_pred)

                print(subject)
                print(f"Mean Squared Error (MSE): {mse:.2f}")
                print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
                print(f"R-squared (R2) Score: {r2:.2f}")
                # Initialize a KFold cross-validator with 5 folds
                kf = KFold(n_splits=5)

                mse_scores = []

                # Perform crdoss-validation
                for i, (train_index, test_index) in enumerate(kf.split(X)):
                    
                    # Fit the model on the training data
                    lasso_model.fit(X_train, y_train)
                    
                    # Make predictions on the test data
                    y_pred = lasso_model.predict(X_test)
                    
                    # Calculate the Mean Squared Error for this fold
                    mse = mean_squared_error(y_test, y_pred)
                    mse_scores.append(mse)

                # Calculate the average MSE across all folds
                average_mse = sum(mse_scores) / len(mse_scores)
                print(f"Cross value score: {average_mse:.2f}")
                print()
                print()