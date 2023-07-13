import pandas as pd

df = pd.read_csv('csv_clean/clean_combined.csv')

df = df.drop('Mock 1', axis=1)

print(df.columns)

map = {
    'Mock 2': 'Mock 1',
    'Mock 3': 'Mock 2'
}

df = df.rename(columns=map)
          
df.to_csv('test_full_actual.csv')