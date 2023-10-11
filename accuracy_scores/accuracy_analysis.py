import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

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

# Calculate the difference between Teacher_Off_One and Model_Off_One
data['Difference_Off_One'] = (pd.to_numeric(data['Teacher_Off_One']) - pd.to_numeric(data['Model_Off_One']))

# Create a bar chart
plt.figure(figsize=(15, 10))
plt.bar(data['Subject'], data['Difference_Off_One'], color='blue')
plt.xlabel('Subject')
plt.ylabel('Difference (Teacher - Model) Off One')
plt.title('Difference between Teacher and Model Off by One for Each Subject')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid()


plt.savefig('off_by_one.png')
plt.clf()

# Extract elements from the lists in 'Teacher_Off_2+' and 'Model_Off_2+' columns
data['Teacher_Off_2+'] = data['Teacher_Off_2+'].apply(lambda x: int(x.strip('[]')))
data['Model_Off_2+'] = data['Model_Off_2+'].apply(lambda x: int(x.strip('[]')))

# Calculate the difference between Teacher_Off_2+ and Model_Off_2+
data['Difference_Off_2+'] = data['Teacher_Off_2+'] - data['Model_Off_2+']

# Create a bar chart
plt.figure(figsize=(15, 10))
plt.bar(data['Subject'], data['Difference_Off_2+'], color='green')
plt.xlabel('Subject')
plt.ylabel('Difference (Teacher - Model) Off Two')
plt.title('Difference between Teacher and Model Off by Two for Each Subject')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('off_by_two.png')
plt.clf()