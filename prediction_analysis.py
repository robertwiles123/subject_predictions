import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('model_scores/scores.csv')

# Extract subject names, MSE, and RMSE values from the DataFrame
subjects = df['subject']
mse_values = df['MSE']
rmse_values = df['RMSE']

# Create an array of indices for the x-axis
x = range(len(subjects))

# Set bar width
bar_width = 0.35

# Create the bar chart
fig, ax = plt.subplots()

# Plot MSE bars in blue
mse_bars = ax.bar(x, mse_values, bar_width, label='MSE', color='blue')

# Plot RMSE bars in red
rmse_bars = ax.bar([i + bar_width for i in x], rmse_values, bar_width, label='RMSE', color='red')

# Set the x-axis labels to be the subject names
ax.set_xticks([i + bar_width/2 for i in x])
ax.set_xticklabels(subjects, rotation=45, ha='right')

# Add labels, title, and legend
ax.set_xlabel('Subjects')
ax.set_ylabel('Error Values')
ax.set_title('MSE and RMSE by Subject')
ax.legend()


plt.subplots_adjust(bottom=0.5, top=1)
plt.savefig('analysis_graphs/MSE&RMSE by subject')

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
ax.set_title('R2 Score by Subject')

# Adjust the layout
plt.tight_layout()
plt.savefig('analysis_graphs/ R2 by subject')
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
ax.set_title('Mean cross validation Score by Subject')

# Adjust the layout
plt.tight_layout()
plt.savefig('analysis_graphs/ MCVS by subject')
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
plt.title('Cross-validation Scores by Subject with Error Bars')
plt.xlabel('Subject')
plt.ylabel('Cross-validation Score')
plt.xticks(rotation=45)
plt.ylim(-1, 1)

plt.tight_layout()
plt.savefig('analysis_graphs/ cross-val range by subject')
print('Fig4 saved')
plt.clf