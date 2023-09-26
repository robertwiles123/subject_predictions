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

