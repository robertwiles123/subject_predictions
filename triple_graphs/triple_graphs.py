import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from grade_packages import df_columns
import itertools

# import csv file
grades = pd.read_csv('../csv_clean/clean_triple.csv')

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

#Grades vsv grades
y_variations = df_columns.triple_grades()  # Replace y1, y2, y3 with your actual Y column variations
x_variations = df_columns.triple_independent()  # Replace x1, x2 with your actual X column variations

num_rows = len(y_variations)
num_cols = len(x_variations)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

# Generate all combinations
combinations = list(itertools.product(y_variations, x_variations))

# Create line plots
for i, (y, x) in enumerate(combinations):
    row = i // num_cols
    col = i % num_cols
    ax = axes[row, col]

    # Plot the line on the respective subplot
    sns.lineplot(data=grades, x=x, y=y, ax=ax)

    # Set title for the subplot
    ax.set_title(f'Combination {i+1}')

# Remove empty subplots if the number of combinations is less than the grid size
if len(combinations) < (num_rows * num_cols):
    for i in range(len(combinations), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.4)

plt.savefig("triple_graphs/ind_grades_vs_dep_grades.png", )