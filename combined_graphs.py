import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from grades_packages import df_columns

# import csv file
grades = pd.read_csv('csv_clean/clean_combined.csv')

# Converts grades back from numaric
for column in grades.columns:
        if column in df_columns.combined_grades():
            # to return predicted grades back to how they should be, however, not doing original files
            grade_mapping = {
                                                0: 'U',
                                                1.0: '1-1',
                                                1.5: '2-1',
                                                2.0: '2-2',
                                                2.5: '3-2',
                                                3.0: '3-3',
                                                3.5: '4-3',
                                                4.0: '4-4',
                                                4.5: '5-4',
                                                5.0: '5-5',
                                                5.5: '6-5',
                                                6.0: '6-6',
                                                6.5: '7-6',
                                                7.0: '7-7',
                                                7.5: '8-7',
                                                8.0: '8-8',
                                                8.5: '9-8',
                                                9.0: '9-9'
                                            }

            grades.loc[:, column] = grades[column].map(grade_mapping)

print(grades.columns)

# create a bool for SEN to examine if SEN affects grades
grades['SEN bool'] = grades['SEN need(s)'].apply(lambda x: False if x == 'n' else True)

# So count plot is ordered from lowest to heighst grade
grade_order = ['U', '1-1', '2-1', '2-2', '3-2', '3-3', '4-3', '4-4', '5-4', '5-5', '6-5', '6-6', '7-6', '7-7', '8-7', '8-8', '9-8', '9-9']

# create subplots for each 

# Create a figure and subplots
fig, ax = plt.subplots(3, 2, figsize=(12, 18))

# Set spacing between subplots
fig.subplots_adjust(hspace=0.5)

# Graphs bellow are producing a bar and hist plot for SEN and PP against grades at different points of the year
"""
# 3 bar graphs to show balues for scores and how SEN affects
sns.countplot(data=grades, x='Year 10 Combined MOCK GRADE', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[0, 0], order=grade_order)
sns.countplot(data=grades, x='Combined MOCK GRADE term 2', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[1, 0], order=grade_order)
sns.countplot(data=grades, x='Combined MOCK GRADE Term 4', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[2, 0], order=grade_order)

ax[0, 0].set_xticklabels(grade_order, rotation=45)
ax[1, 0].set_xticklabels(grade_order, rotation=45)
ax[2, 0].set_xticklabels(grade_order, rotation=45)

sns.histplot(data=grades, x='Year 10 Combined MOCK GRADE', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[0, 1], element='step', bins=len(grade_order))
ax[0, 1].set_xticks(range(len(grade_order)))
ax[0, 1].set_xticklabels(grade_order, rotation=45)


sns.histplot(data=grades, x='Combined MOCK GRADE term 2', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[1, 1], element='step', bins=len(grade_order))
ax[1, 1].set_xticks(range(len(grade_order)))
ax[1, 1].set_xticklabels(grade_order, rotation=45)

sns.histplot(data=grades, x='Combined MOCK GRADE Term 4', hue='SEN bool', palette=['blue', 'yellow'], ax=ax[2, 1], element='step', bins=len(grade_order))
ax[2, 1].set_xticks(range(len(grade_order)))
ax[2, 1].set_xticklabels(grade_order, rotation=45)

plt.savefig("combined_graphs/SEN_and_grades.png", )
"""

# break between graphs

"""

#These grades are for PP students

sns.countplot(data=grades, x='Year 10 Combined MOCK GRADE', hue='PP', palette=['blue', 'yellow'], ax=ax[0, 0], order=grade_order)
sns.countplot(data=grades, x='Combined MOCK GRADE term 2', hue='PP', palette=['blue', 'yellow'], ax=ax[1, 0], order=grade_order)
sns.countplot(data=grades, x='Combined MOCK GRADE Term 4', hue='PP', palette=['blue', 'yellow'], ax=ax[2, 0], order=grade_order)

ax[0, 0].set_xticklabels(grade_order, rotation=45)
ax[1, 0].set_xticklabels(grade_order, rotation=45)
ax[2, 0].set_xticklabels(grade_order, rotation=45)

sns.histplot(data=grades, x='Year 10 Combined MOCK GRADE', hue='PP', palette=['blue', 'yellow'], ax=ax[0, 1], element='step', bins=5)
ax[0, 1].set_xticks(range(len(grade_order)))
ax[0, 1].set_xticklabels(grade_order, rotation=45)

sns.histplot(data=grades, x='Combined MOCK GRADE term 2', hue='PP', palette=['blue', 'yellow'], ax=ax[1, 1], element='step', bins=5)
ax[1, 1].set_xticks(range(len(grade_order)))
ax[1, 1].set_xticklabels(grade_order, rotation=45)

sns.histplot(data=grades, x='Combined MOCK GRADE Term 4', hue='PP', palette=['blue', 'yellow'], ax=ax[2, 1], element='step', bins=5)
ax[2, 1].set_xticks(range(len(grade_order)))
ax[2, 1].set_xticklabels(grade_order, rotation=45)

plt.savefig("combined_graphs/PP_and_grades.png")
"""

