import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Update to change model imported and same saved
model = 'ridge'

# load in dataframe from the model name
df = pd.read_csv(f'{model}_scores/scores.csv')

# Extract subject names, MSE, and RMSE values from the DataFrame
subjects = df['subject']
mae_values = df['Mean Absolute Error']
rmse_values = df['RMSE']

# Create a figure with a larger size
plt.figure(figsize=(15, 10))

# Create an axis
ax = plt.subplot(111)

# Width of each bar
bar_width = 0.4

# Plot MSE bars in blue
mae_bars = ax.bar(np.arange(len(subjects)), mae_values, width=bar_width, label='MAE', color='blue')

# Plot RMSE bars in red, adjust the x-positions
rmse_bars = ax.bar(np.arange(len(subjects)) + bar_width, rmse_values, width=bar_width, label='RMSE', color='red')

# Set the x-axis labels to be the subject names
ax.set_xticks(np.arange(len(subjects)) + bar_width / 2)
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and legend
ax.set_xlabel('Subjects')
ax.set_ylabel('Error Values')
ax.set_title(f'{model.capitalize()} MAE and RMSE by Subject')
ax.legend()
# line to show average MSE
plt.axhline(y=0.4216, color='blue', linestyle='--', label='Teacher MAE')
plt.axhline(y=0.8203, color='red', linestyle='--', label='Teacher RMSE')

# Show the plot
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
# Save the plot with the title and file extension
plt.savefig(f'analysis_graphs/{model}_MSE_RMSE_by_subject.png')

print('Figure saved')
# clear plt for next graph
plt.clf

# Extract subject names and R2 scores from the DataFrame
subjects = df['subject']
r2_scores = df['R2']

# Create an array of indices for the x-axis
x = range(len(subjects))

# Create the bar chart for R2 scores
fig, ax = plt.subplots(figsize=(16, 10))  # Adjust the figsize as needed

# Plot R2 scores in a single color (e.g., blue)
r2_bars = ax.bar(x, r2_scores, color='blue')

# Set the x-axis labels to be the subject names
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and ylabel
ax.set_xlabel('Subjects')
ax.set_ylabel('R2 Score')
ax.set_title(f'{model.capitalize()} R2 Score by Subject')

plt.axhline(y=0.8126, color='blue', linestyle='--', label='Teacher R2')

# Adjust the layout
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f'analysis_graphs/{model} R2 by subject.png')
print('Fig2 saved')

plt.clf

# extract a string of numbers in to floats
def extract_and_convert(s):
    s = s.strip('[]')  # Remove square brackets
    values = [float(x) for x in s.split()]  # Convert the space-separated values to floats
    return values

# Apply the function to the "Cross-validation" column
df['Cross-validation'] = df['Cross-validation'].apply(extract_and_convert)

# Explode the "Cross-validation" column to create separate rows for each value
df = df.explode('Cross-validation')

plt.figure(figsize=(15, 10))
sns.barplot(x='subject', y='Cross-validation', data=df, estimator=np.mean, errorbar="sd")
plt.title(f'{model.capitalize()} Cross-validation Scores by Subject with Error Bars')
plt.xlabel('Subject')
plt.ylabel('Cross-validation Score')
plt.xticks(rotation=45)
plt.ylim(0, 1)

plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.savefig(f'analysis_graphs/{model} cross-val range by subject.png')
print('Fig3 saved')
plt.clf