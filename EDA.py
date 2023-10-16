import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
data = pd.read_csv('crime_rate_Spain.csv')
df = pd.DataFrame(data)

# Overview of the data
print(df.head())

# Basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())



##### BASIC GRAPHS #####

# Total cases by location and year Barplot 
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Location', y='Total cases', hue='Year')
plt.title('Total Cases by Location and Year')
plt.show()

# Total Cases by Location Swarmplot
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['Location'], y=df['Total cases'], inner=None, color="0.8")
sns.swarmplot(x=df['Location'], y=df['Total cases'], edgecolor="black", alpha=0.7)
plt.title('Total Cases by Location')
plt.show()

# Distribution of total cases
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Total cases', kde=True)
plt.title('Distribution of Total Cases')
plt.show()

# Boxplot to understand spread of data across locations
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Location', y='Total cases')
plt.title('Spread of Total Cases by Location')
plt.show()

# Pivot and fill missing values with zero
pivot_table = df.pivot_table(index='Location', columns='Year', values='Total cases', aggfunc=np.sum, fill_value=0)

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
plt.title('Heatmap of Total Cases by Location and Year')
plt.show()



