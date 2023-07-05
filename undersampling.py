import pandas as pd
from sklearn.utils import resample
from grades_packages import df_columns

grades_normal = pd.read_csv('csv_clean/clean_combined.csv')

majority_class = pd.DataFrame()
minority_class = pd.DataFrame()

for col in grades_normal.columns:
    if col in df_columns.combined_grades():
        majority_class[col] = grades_normal.loc[grades_normal[col] != 0, col]
        minority_class[col] = grades_normal.loc[grades_normal[col] == 0, col]
    else:
        majority_class[col] = grades_normal[col]
        minority_class[col] = grades_normal[col]

# Undersample the majority class to match the number of samples in the minority class
undersampled_majority = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)

# Combine the undersampled majority class with the original minority class
undersampled_df = pd.concat([undersampled_majority, minority_class])

# Drop rows with NaN values
undersampled_df = undersampled_df.dropna()

# Shuffle the DataFrame to randomize the order of samples
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

undersampled_df.to_csv('csv_clean/combined_undersample.csv')