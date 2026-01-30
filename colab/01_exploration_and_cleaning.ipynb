# 01_exploration_and_cleaning.ipynb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv('https://noldo.fr/dev/ml/vl-model-installation-time/installation_data_clean_v2.csv')  # ou upload avec files.upload()

# Convert 'installation_time_in_ms' to numeric, coercing errors, and then to seconds
df['installation_time_in_ms'] = pd.to_numeric(df['installation_time_in_ms'], errors='coerce')
# Drop rows where 'installation_time_in_ms' became NaN after conversion, as these are problematic
df.dropna(subset=['installation_time_in_ms'], inplace=True)
df['installation_time_seconds'] = df['installation_time_in_ms'] / 1000

# Replace NaN values in 'software_addon' and 'software_addon_version' with 'not_defined' for df
df['software_addon'] = df['software_addon'].fillna('not_defined')
df['software_addon_version'] = df['software_addon_version'].fillna('not_defined')

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

print("Dataset size :", df.shape)
print("\nOverview :")
display(df.head(100))

print("\nStatistics:")
display(df.describe())

# Scatter plot colored by software
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='software', y='installation_time_seconds',
                hue='version', s=100, palette='deep', alpha=0.8)
plt.title("Installation time by software and by version", fontsize=16)
plt.ylabel("Time (seconds)")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='software', y='installation_time_seconds')
plt.title("Boxplot per software – visible outliers")
plt.ylabel("Time (seconde)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Global Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['installation_time_seconds'], bins=30, kde=True)
plt.title("Distribution of installation time")
plt.xlabel("Seconds")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Outliers w/ IQR
Q1 = df['installation_time_seconds'].quantile(0.25)
Q3 = df['installation_time_seconds'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Define conditions for removal
condition_iqr_outlier = (df['installation_time_seconds'] < lower) | (df['installation_time_seconds'] > upper)
condition_software_exclusion = (df['software'] == "verylys-prerender") | (df['software'] == "nginx-server")

# Combine conditions: remove if either is true
to_remove_condition = condition_iqr_outlier | condition_software_exclusion

outliers = df[to_remove_condition]
print(f"\nFound outliers (IQR or Software excluded) : {len(outliers)} lines")
display(outliers)

df_clean = df[~to_remove_condition].copy()

# Replace NaN values in 'software_addon' and 'software_addon_version' with 'not_defined' for df_clean
df_clean['software_addon'] = df_clean['software_addon'].fillna('not_defined')
df_clean['software_addon_version'] = df_clean['software_addon_version'].fillna('not_defined')

# Save cleaned dataset
df_clean.to_csv('installation_data_clean_v2.csv', index=False)
print(f"\nCleaned Dataset saved : {df_clean.shape[0]} lines")