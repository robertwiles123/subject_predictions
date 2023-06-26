import pandas as pd
import df_columns


df = pd.read_csv('combined.csv')

df = df[df_columns.combined_columns()]

for col in df.columns:
    print(df[col].unique())
