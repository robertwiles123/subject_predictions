import pandas as pd

ap1 = pd.read_excel('Y11 2122 Ap1 MLG P8.xlsx', header=1)
ap2 = pd.read_excel('Y11 2122 Ap2 MLG P8.xlsx', header=1)
p8 = pd.read_excel('Y11 2122 FFT P8.xlsx', header=1)

# to ensure there is no issues with the columns names
def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

dataframes = (ap1, ap2, p8)

for df in dataframes:
    clean_column_names(df)

full = pd.merge(ap1, ap2, on='upn', how='outer', suffixes=('_ap1', '_ap2'))

# give the p8 columns a suffix else won't have it on the join
p8.columns = ['upn'] + [f'{col}_real' for col in p8.columns if col != 'upn']

#merge real grades with the outcomes
full = pd.merge(full, p8, on='upn', how='outer')

#list of duplicated columns to drop after merge
columns_to_drop = ['gender_ap1', 'gender_ap2', 'rm_scaled_score_ap1', 'rm_scaled_score_ap2', 'actual_ap1', 'actual_ap2', 'difference_ap1', 'difference_ap2', 'difference_real', 'entries_ap1', 'score_ap1', 'scoreadjusted_ap1', 'entries_ap2', 'score_ap2', 'scoreadjusted_ap2', 'entries_real', 'score_real']

# Drop the specified columns from the DataFrame
full.drop(columns=columns_to_drop, inplace=True)

