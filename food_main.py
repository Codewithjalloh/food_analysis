#%%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

#%%
# Load and Initial Data Exploration
# --------------------------------
# Load the dataset
df = pd.read_csv('food_prices.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

#%%
# Basic Statistical Analysis
# -------------------------
print("\nBasic Statistical Summary:")
print(df.describe())

#%%
# Missing Values Analysis
# ----------------------
print("\nMissing Values Analysis:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

#%%
# Data Visualization
# -----------------
# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Food Prices Analysis', fontsize=16)

# Plot 1: Price Distribution
sns.histplot(data=df, x='Average Price', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Average Price Distribution')

# Plot 2: Box Plot of Prices
sns.boxplot(data=df, y='Average Price', ax=axes[0, 1])
axes[0, 1].set_title('Average Price Box Plot')

# Plot 3: Correlation Heatmap
numeric_columns = ['Year', 'Average Price', 'Price in USD']
numeric_df = df[numeric_columns]
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Correlation Heatmap')

# Plot 4: Time Series Analysis
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
df.set_index('Date')['Average Price'].plot(ax=axes[1, 1])
axes[1, 1].set_title('Price Trend Over Time')

plt.tight_layout()
plt.show()

#%%
# Advanced Analysis
# ----------------
# Add any specific analysis based on your dataset's characteristics
# For example:
# - Price trends by category
# - Seasonal patterns
# - Geographic analysis
# - Price comparisons between different items

#%%
# Data Quality Checks
# ------------------
print("\nData Quality Checks:")
print("\nNumber of unique values in each column:")
print(df.nunique())

print("\nDuplicate rows:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

#%%
# Export Analysis Results
# ----------------------
# Save the processed data if needed
# df.to_csv('processed_food_prices.csv', index=False) 