import pandas as pd
from sklearn.utils import resample
import df_columns

grades_normal = pd.read_csv('csv_clean/clean_combined.csv')

# Create empty DataFrames for majority and minority classes
majority_class = pd.DataFrame(columns=grades_normal.columns)
minority_class = pd.DataFrame(columns=grades_normal.columns)

for col in grades_normal.columns:
    if col in df_columns.combined_full_clean():
        majority_class[col] = grades_normal.loc[grades_normal[col] != 0, col]
        minority_class[col] = grades_normal.loc[grades_normal[col] == 0, col]
    else:
        majority_class[col] = grades_normal[col]
        minority_class[col] = grades_normal[col]


# Calculate the desired number of samples from the majority class
undersample_size = 130

# Undersample the majority class to the desired size
undersampled_majority = resample(majority_class, replace=False, n_samples=undersample_size, random_state=86)

# Combine the undersampled majority class with the original minority class
undersampled_df = pd.concat([undersampled_majority, minority_class])

# Drop rows with NaN values
undersampled_df = undersampled_df.dropna()

# Shuffle the DataFrame to randomize the order of samples
undersampled_df = undersampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

undersampled_df = undersampled_df.drop("Unnamed: 0", axis=1)

# Save the undersampled DataFrame to CSV without the index column
undersampled_df.to_csv("csv_clean/combined_undersample.csv")

print('CSV saved')