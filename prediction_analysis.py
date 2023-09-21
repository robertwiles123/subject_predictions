import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

model = 'linear_regression'

df = pd.read_csv(f'{model}_scores/scores.csv')

# Extract subject names, MSE, and RMSE values from the DataFrame
subjects = df['subject']
mse_values = df['MSE']
rmse_values = df['RMSE']

# Create a figure with a larger size
plt.figure(figsize=(10, 6))

# Create an axis
ax = plt.subplot(111)

# Plot MSE bars in blue
mse_bars = ax.bar(np.arange(len(subjects)), mse_values, label='MSE', color='blue')

# Plot RMSE bars in red
rmse_bars = ax.bar(np.arange(len(subjects)) + 0.4, rmse_values, label='RMSE', color='red')

# Set the x-axis labels to be the subject names
ax.set_xticks(np.arange(len(subjects)) + 0.2)
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and legend
ax.set_xlabel('Subjects')
ax.set_ylabel('Error Values')
ax.set_title(f'{model.capitalize()} MSE and RMSE by Subject')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Save the plot with the title and file extension
plt.savefig(f'analysis_graphs/{model}_MSE_RMSE_by_subject.png')

print('Figure saved')
plt.clf

# Extract subject names and R2 scores from the DataFrame
subjects = df['subject']
r2_scores = df['R2']

# Create an array of indices for the x-axis
x = range(len(subjects))

# Create the bar chart for R2 scores
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed

# Plot R2 scores in a single color (e.g., blue)
r2_bars = ax.bar(x, r2_scores, color='blue')

# Set the x-axis labels to be the subject names
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and ylabel
ax.set_xlabel('Subjects')
ax.set_ylabel('R2 Score')
ax.set_title(f'{model.capitalize()} R2 Score by Subject')

# Adjust the layout
plt.tight_layout()
plt.savefig(f'analysis_graphs/{model} R2 by subject.png')
print('Fig2 saved')

plt.clf

# Extract subject names and MCVS scores from the DataFrame
subjects = df['subject']
r2_scores = df['Mean cross validation']

# Create an array of indices for the x-axis
x = range(len(subjects))

# Create the bar chart for MCVS scores
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figsize as needed

# Plot MCSV scores in a single color (e.g., blue)
r2_bars = ax.bar(x, r2_scores, color='blue')

# Set the x-axis labels to be the subject names
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and ylabel
ax.set_xlabel('Subjects')
ax.set_ylabel('Mean cross validation Score')
ax.set_title(f'{model.capitalize()} mean cross validation Score by Subject')

# Adjust the layout
plt.tight_layout()
plt.savefig(f'analysis_graphs/{model} MCVS by subject.png')
print('Fig3 saved')

plt.clf

def extract_and_convert(s):
    s = s.strip('[]')  # Remove square brackets
    values = [float(x) for x in s.split()]  # Convert the space-separated values to floats
    return values

# Apply the function to the "Cross-validation" column
df['Cross-validation'] = df['Cross-validation'].apply(extract_and_convert)

# Explode the "Cross-validation" column to create separate rows for each value
df = df.explode('Cross-validation')

plt.figure(figsize=(10, 6))
sns.barplot(x='subject', y='Cross-validation', data=df, estimator=np.mean, errorbar="sd")
plt.title(f'{model.capitalize()} Cross-validation Scores by Subject with Error Bars')
plt.xlabel('Subject')
plt.ylabel('Cross-validation Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(f'analysis_graphs/{model} cross-val range by subject.png')
print('Fig4 saved')
plt.clf