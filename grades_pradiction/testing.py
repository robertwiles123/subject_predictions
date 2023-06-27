import pandas as pd
import df_columns


df = pd.read_csv('triple.csv')

print(df.columns)

df = df[df_columns.triple_columns()]

for col in df.columns:
    print(df[col].unique())
