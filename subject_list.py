import pandas as pd

def full_subjects():
    return ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science', 'd_&_t_product_design', 'd_&_t_textiles_technology', 'drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'ict_btec', 'maths', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'physics', 'product_design', 'science_double', 'spanish']

def prediction_subjects():
    return ['art_&_design', 'biology', 'business_studies', 'chemistry', 'computer_science','drama', 'english_language', 'english_literature', 'food_technology', 'french_language', 'geography', 'german', 'history', 'ict_btec', 'maths', 'music_studies', 'music_tech_grade', 'pearson_btec_sport', 'physics', 'product_design', 'science_double', 'spanish']

"""
Removed subjects
d_&_t_product_design
d_&_t_textiles_technology
"""

def grades_mapped():
    return {'0': 0, 'P1': 1, 'P2': 2, 'M1': 3, 'M2': 4, 'D1': 5, 'D2': 6, 'D*1': 7, 'D*2': 8}


def get_model_name(x):
    # Check if RandomForestRegressor is imported, if yes, return 'random_forest', otherwise return 'linear_regression'
    if 'RandomForestRegressor' in x:
        return 'random_forest'
    elif 'LinearRegression' in x:
        return 'linear_regression'
    elif 'Ridge' in x:
        return 'ridge'    

def get_models(x, name = None):
    # Check if RandomForestRegressor is imported, if yes, return 'random_forest', otherwise return 'linear_regression'
    if 'RandomForestRegressor' in x:
        from sklearn.ensemble import RandomForestRegressor
        df = pd.read_csv('/workspace/subject_predictions/models/params.csv')
        hyperparameters = {}
        subject_row = df[df['subject'] == name]
        if subject_row.empty or subject_row.isnull().values.any():
            print(f"Hyperparameters for subject '{name}' not found in the CSV.")
            model = RandomForestRegressor(max_depth= 5, n_estimators=100, min_samples_split=5, min_samples_leaf=2, max_features='sqrt',max_samples=0.5, bootstrap=False, random_state=42)
        else:
            # Extract hyperparameters as a dictionary
            for column in df.columns:
                value = subject_row[column].values[0]
                if pd.notna(value) and column != 'subject':
                    hyperparameters[column] = value
        return RandomForestRegressor(**hyperparameters)
    elif 'LinearRegression' in x:
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif 'Ridge' in x:
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1)   
