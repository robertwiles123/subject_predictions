import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('ridge_accuracy.csv')

# Calculate the t-test
t_statistic, t_p_value = stats.ttest_rel(data['Model_Accuracy'], data['Teacher_Accuracy'])

# Calculate the Wilcoxon signed-rank test
wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(data['Model_Accuracy'], data['Teacher_Accuracy'])

# Calculate and print the mean and median
model_mean = data['Model_Accuracy'].mean()
model_median = data['Model_Accuracy'].median()
teacher_mean = data['Teacher_Accuracy'].mean()
teacher_median = data['Teacher_Accuracy'].median()

print(f"Model Mean Accuracy: {model_mean:.2%}")
print(f"Model Median Accuracy: {model_median:.2%}")
print(f"Teacher Mean Accuracy: {teacher_mean:.2%}")
print(f"Teacher Median Accuracy: {teacher_median:.2%}")

# Print the results of the tests
print(f"T-Test - t-statistic: {t_statistic}, p-value: {t_p_value}")
print(f"Wilcoxon Test - statistic: {wilcoxon_statistic}, p-value: {wilcoxon_p_value}")

# Determine which is better based on the t-test
if t_p_value < 0.05:
    if t_statistic > 0:
        print("T-Test: Model is significantly better than Teacher.")
    else:
        print("T-Test: Teacher is significantly better than Model.")
else:
    print("T-Test: No significant difference between Model and Teacher.")

# Determine which is better based on the Wilcoxon test
if wilcoxon_p_value < 0.05:
    if wilcoxon_statistic > 0:
        print("Wilcoxon Test: Model is significantly better than Teacher.")
    else:
        print("Wilcoxon Test: Teacher is significantly better than Model.")
else:
    print("Wilcoxon Test: No significant difference between Model and Teacher.")

data['Teacher_Off_One'] = data['Teacher_Off_One'].apply(lambda x: int(x.strip('[]')))
data['Model_Off_One'] = data['Model_Off_One'].apply(lambda x: int(x.strip('[]')))


plt.figure(figsize=(20, 10))
width = 0.35  # Width of the bars

# Calculate the positions for the bars for 'full' and 'teacher' data
x = range(len(data['subject']))
x_full_positions = [pos - width/2 for pos in x]
x_teacher_positions = [pos + width/2 for pos in x]

# Create the bars for 'full' data
plt.bar(x_full_positions, data['Model_Off_One'], label='Model off by one', width=width)

# Create the bars for 'teacher' data
plt.bar(x_teacher_positions, data['Teacher_Off_One'], label='Teacher off by one', width=width, alpha=0.7)
plt.xticks(x, data['subject'])
plt.xticks(rotation=45)
plt.ylabel('Amount off by one')
plt.title('Difference between Teacher and Model Off by one for Each Subject')

# Add a legend
plt.legend()

plt.savefig('off_by_one.png')
plt.clf()

plt.figure(figsize=(20, 10))
width = 0.35  # Width of the bars

data['Teacher_Off_2+'] = data['Teacher_Off_2+'].apply(lambda x: int(x.strip('[]')))
data['Model_Off_2+'] = data['Model_Off_2+'].apply(lambda x: int(x.strip('[]')))

# Calculate the positions for the bars for 'full' and 'teacher' data
x = range(len(data['subject']))
x_full_positions = [pos - width/2 for pos in x]
x_teacher_positions = [pos + width/2 for pos in x]

# Create the bars for 'full' data
plt.bar(x_full_positions, data['Model_Off_2+'], label='Model off by 2+', width=width)

# Create the bars for 'teacher' data
plt.bar(x_teacher_positions, data['Teacher_Off_2+'], label='Teacher off by 2+', width=width, alpha=0.7)
plt.xticks(x, data['subject'])
plt.xticks(rotation=45)

plt.ylabel('Amount off by one')
plt.yticks(range(0, max(data['Teacher_Off_2+']) + 1, 1))

plt.title('Difference between Teacher and Model Off by one for Each Subject')

# Add a legend
plt.legend()

plt.savefig('off_by_two.png')
plt.clf()


# Create a KDE plot for Model_Accuracy with a smaller bandwidth
sns.kdeplot(data['Model_Accuracy'], label='Model Accuracy')

# Create a KDE plot for Teacher_Accuracy with a smaller bandwidth
# 
sns.kdeplot(data['Teacher_Accuracy'], label='Teacher Accuracy')

# Add labels and a legend
plt.xlabel('Accuracy Scores')
plt.ylabel('Density')
plt.title('Distribution of Ridge and Teacher accuracy')
plt.legend()

plt.savefig('hist_accuracy.png')
plt.clf

scores = pd.read_csv('/workspaces/subject_predictions/ridge_scores/scores.csv')

full = pd.merge(data, scores, on ='subject')

full['Model_Accuracy_percent'] = (full['Model_Accuracy'] * 100)

x_full = full['subject']
accuracy_full = full['Model_Accuracy_percent']
rmse_full = full['RMSE']

teacher = pd.read_csv('/workspaces/subject_predictions/teacher_scores/scores.csv')

teacher_full = pd.merge(data, teacher, on='subject')

teacher_full['Teacher_Accuracy_percent'] = (teacher_full['Teacher_Accuracy'] * 100)

accuracy_teacher = teacher_full['Teacher_Accuracy_percent']
rmse_teacher = teacher_full['RMSE']

# Create the bar chart with error bars for 'full' data
plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
width = 0.35  # Width of the bars

# Calculate the positions for the bars for 'full' and 'teacher' data
x = range(len(x_full))
x_full_positions = [pos - width/2 for pos in x]
x_teacher_positions = [pos + width/2 for pos in x]

# Create the bars for 'full' data
plt.bar(x_full_positions, accuracy_full, yerr=rmse_full, capsize=5, label='Full Accuracy with RMSE Error Bars', width=width)

# Create the bars for 'teacher' data
plt.bar(x_teacher_positions, accuracy_teacher, yerr=rmse_teacher, capsize=5, label='Teacher Accuracy with RMSE Error Bars', width=width, alpha=0.7)

# Customize the chart
plt.xlabel('Subject')
plt.ylabel('Accuracy')
plt.title('Accuracy with RMSE Error Bars for Each Subject')
plt.xticks(x, x_full)  # Use the 'x_full' as labels for x-axis
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_error.png')