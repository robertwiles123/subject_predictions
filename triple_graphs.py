import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from grade_packages import df_columns

# import csv file
grades = pd.read_csv('csv_clean/clean_triple.csv')

print(grades.columns)

# create subplots for each
"""
# Create a figure and subplots
fig, ax = plt.subplots(3,3, figsize=(12, 18))

# Set spacing between subplots
fig.subplots_adjust(hspace=0.5)

# Graphs bellow are producing a bar and hist plot for SEN and PP against grades at different points of the year

# 3 bar graphs to show balues for scores and how SEN affects
sns.histplot(data=grades, x='year 10 bio grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[0, 0])
sns.histplot(data=grades, x='year 11 paper 1 bio grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[1, 0])
sns.histplot(data=grades, x='year 11 paper 2 bio grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[2, 0])

sns.histplot(data=grades, x='year 10 chem grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[0, 1])
sns.histplot(data=grades, x='year 11 paper 1 chem grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[1, 1])
sns.histplot(data=grades, x='year 11 paper 2 chem grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[2, 1])

sns.histplot(data=grades, x='year 10 phys grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[0, 2])
sns.histplot(data=grades, x='year 11 paper 1 phys grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[1, 2])
sns.histplot(data=grades, x='year 11 paper 2 phys grade', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[2, 2])

plt.savefig("triple_graphs/SEN_and_grades.png", )


# break between graphs

"""
# scatter plot for examining how the grades affect the grade looking for

# X vaiable stored as list so this allows to grab in in code
X = df_columns.combined_independent()

# to sort df so that graph is ordered properly

grades_sorted = grades.sort_values(X[0], ascending=True)

sns.lineplot(data=grades, x=X[0], y='Year 10 Combined MOCK GRADE', label='Year 10 mocks')

sns.lineplot(data=grades, x=X[0], y='Combined MOCK GRADE term 2', label='Year 11 mocks')

ax = plt.gca()
ax.set_xticks(range(len(grade_order)))
ax.set_xticklabels(grade_order)
ax.set_yticks(range(len(grade_order)))
ax.set_yticklabels(grade_order)
# Reverse the order of the y-axis ticks
ax.set_ylim(ax.get_ylim()[::-1])

plt.legend()

plt.savefig("triple_graphs/grades_line.png")