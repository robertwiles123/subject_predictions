# improting and cleaning data
import numpy as np
import pandas as pd
from grade_packages import df_columns

# allows user to import either a file full of combined grades with format of 1-1, 2-1, 2-2 or triple with just 1 2 3
file_name = input('What file do you want cleaned, include file type? ')
full = pd.read_csv('csv_dirty/' + file_name)
full.columns = full.columns.str.strip()

# to clean based on triple or combined
type_science = input('triple or combined? ')
type_science = type_science.lower()[0]

# just checking first letter as currenlt just me using so to help with typo management
if type_science == 'c':
    just_grades = full[df_columns.combined_full_original()]
elif type_science == 't':
    just_grades = full[df_columns.triple_full_original()]
else:
    print('Neither selected')
# to see what needs to be cleared
print(just_grades.describe())

just_grades['PP'] = just_grades['PP'].str.replace('Yes', 'True')
just_grades['PP'] = just_grades['PP'].fillna(False)

just_grades['PP'] = just_grades ['PP'].astype(bool)

just_grades['SEN need(s)'] = just_grades ['SEN need(s)'].fillna("n")

# To see if I can just clean the missing data
print(just_grades.isna().sum().sort_values(), len(just_grades) * 0.05)
# less then 5% of the data is nan so can be removed


# 17 FFT20 there should only be 8 for values 1-9 inclusive
# display(pd.unique(just_grades['FFT20']))

just_grades_no_nan = just_grades.dropna(axis=0)
just_grades_clean_FFT20 = just_grades_no_nan.copy()
just_grades_clean_FFT20['FFT20'] = just_grades_no_nan['FFT20'].str.replace('[^\d]', '')


# data has been cleaned and simplified to allow greater comparison
# display(pd.unique(just_grades_clean_FFT20['FFT20']))

# there are now too many grades in all three scores as well as U. This is due to scores being 1-1, 1-2 etc. it would be better to show it as 1, 1.5 respectably. And to change the U to a 0
# display(just_grades_clean_FFT20.describe())

# collect column names to clean one after another
columns_in_df = []
for col in just_grades_clean_FFT20.columns:
    columns_in_df.append(col)

clean_grades = just_grades_clean_FFT20.copy()


# When looking through the data it seems that year 10 has 'Ab' to show absance and 'Combined OMCK GRADE term 4' as 'ABS'
# display(clean_grades.info())

for col in columns_in_df:
    clean_grades[col] = clean_grades[col].replace({'Ab': np.nan, 'Abs': np.nan})
    clean_grades[col] = clean_grades[col].astype('object')

full_clean_grades = clean_grades.dropna(axis=0)


# All the same legth, all objects
print(full_clean_grades.info())

# all the correct possible unique values
for col in columns_in_df:
    print(full_clean_grades[col].unique())

if type_science == 'c':
    # after working with this data I think converting the grades in to '0' '1' '1.5' '2' would work better
    grade_mapping = {
        'U': 0,
        '1-1': 1.0,
        '2-1': 1.5,
        '2-2': 2.0,
        '3-2': 2.5,
        '3-3': 3.0,
        '4-3': 3.5,
        '4-4': 4.0,
        '5-4': 4.5,
        '5-5': 5.0,
        '6-5': 5.5,
        '6-6': 6.0,
        '7-6': 6.5,
        '7-7': 7.0,
        '8-7': 7.5,
        '8-8': 8.0,
        '9-8': 8.5,
        '9-9': 9.0
    }

    for col in ['Year 10 Combined MOCK GRADE', 'Combined MOCK GRADE term 2', 'Combined MOCK GRADE Term 4']:
        full_clean_grades.loc[:, col] = full_clean_grades[col].map(grade_mapping)
        
else:
    def convert_to_int(value):
        if isinstance(value, float):
            return int(value)
        elif value == 'U':
            return 0
        elif '+' in value or '-' in value:
            return int(value[:-1])
        else:
            return int(value)

    for column in full_clean_grades.columns:
        if column not in df_columns.triple_non_grades():
            full_clean_grades[column] = full_clean_grades[column].map(convert_to_int)

full_clean_grades['SEN bool'] = full_clean_grades['SEN need(s)'].apply(lambda x: False if x == 'n' else True)

drop = ['SEN need(s)']

full_clean_grades = full_clean_grades.drop(drop, axis=1)

pd.set_option('display.max_columns', None)

full_clean_grades.to_csv('csv_clean/clean_' + file_name)
print('CSV saved')
