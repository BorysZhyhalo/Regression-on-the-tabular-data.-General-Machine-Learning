import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('train.csv')

# Display basic information about the df set
print("Basic Information:")
print(df .info())

# Display the first few rows of the dataset
print("\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df.describe())

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

#Calculating skewness and kurtosis for'target'
target_skewnes = df['target'].skew()
target_kurtosis = df['target'].kurtosis()

print(f'Skewness: {target_skewnes}')
print(f'Kurtosis: {target_kurtosis}')

# Plot the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df['target'], kde=True, bins=30)
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.show()

# Plot pairwise correlations using a heatmap
plt.figure(figsize=(40, 30))
correlation_matrix = df.corr()
#creating mask for better view
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5, annot=True, mask = mask)
plt.title('Feature Correlation Matrix')
plt.show()

# Plot the correlations of the target variable with other features
plt.figure(figsize=(15, 8))
correlation_with_target = correlation_matrix['target'].drop('target').sort_values(ascending=False)
sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values)
plt.xticks(rotation=90)
plt.title('Correlation of Features with Target')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.show()

# Plot scatter plots of the target variable with other features
plt.figure(figsize=(20, 15))
correlation_with_target = correlation_matrix['target'].drop('target').sort_values(ascending=False)
for i, feature in enumerate(correlation_with_target.index, 1):
    plt.subplot(6, 9, i)
    sns.scatterplot(x=df[feature], y=df['target'])
    plt.title(f'Scatter Plot: Target vs {feature}')
    plt.xlabel(feature)
    plt.ylabel('Target')
plt.tight_layout()
plt.show()

