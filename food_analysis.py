#%%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="viridis")

#%%
# Load and prepare data
# --------------------
"""
Data Loading and Preparation Notes:
- Loading the food prices dataset from CSV
- Cleaning column names by removing extra whitespace
- Converting Year and Month to datetime for time series analysis
- This step ensures data is properly formatted for all subsequent analyses
"""
df = pd.read_csv('food_prices.csv')
df.columns = df.columns.str.strip()  # Clean column names
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))

#%%
# 1. Price Trend Analysis by Country
# --------------------------------
"""
Analysis Notes:
- Shows how food prices have changed over time for different countries
- First 5 countries are plotted for clarity (can be adjusted)
- Line plot with markers makes it easy to track price changes
- Useful for identifying:
  * Long-term price trends
  * Country-specific price patterns
  * Potential price convergence/divergence
"""
plt.figure(figsize=(12, 6))
for country in df['Country'].unique()[:5]:  # Plot first 5 countries for clarity
    country_data = df[df['Country'] == country]
    plt.plot(country_data['Date'], country_data['Average Price'], label=country, marker='o')
plt.title('Price Trends by Country')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 2. Seasonal Price Patterns
# ------------------------
"""
Analysis Notes:
- Examines how food prices vary by month across all years
- Bar plot shows average price for each month
- Helps identify:
  * Seasonal price patterns
  * Months with typically higher/lower prices
  * Potential seasonal buying opportunities
"""
plt.figure(figsize=(10, 6))
monthly_avg = df.groupby('Month')['Average Price'].mean().reset_index()
sns.barplot(x='Month', y='Average Price', data=monthly_avg, palette='viridis')
plt.title('Average Food Prices by Month')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.show()

#%%
# 3. Food Item Price Comparison
# ---------------------------
"""
Analysis Notes:
- Compares average prices across different food items
- Sorted in descending order for easy comparison
- Useful for:
  * Identifying most/least expensive food items
  * Understanding price ranges
  * Budget planning and cost comparison
"""
plt.figure(figsize=(12, 6))
food_prices = df.groupby('Food Item')['Average Price'].mean().sort_values(ascending=False)
sns.barplot(x=food_prices.index, y=food_prices.values, palette='viridis')
plt.title('Average Prices by Food Item')
plt.xlabel('Food Item')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 4. Currency Impact Analysis
# -------------------------
"""
Analysis Notes:
- Examines relationship between local currency and USD prices
- Scatter plot shows price distribution
- Color-coded by currency type
- Helps understand:
  * Currency conversion impacts
  * Price parity across currencies
  * Exchange rate effects
"""
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Average Price', y='Price in USD', hue='Currency', palette='viridis')
plt.title('Local Currency vs USD Price Comparison')
plt.xlabel('Local Currency Price')
plt.ylabel('Price in USD')
plt.show()

#%%
# 5. Quality vs Price Analysis
# --------------------------
"""
Analysis Notes:
- Box plot showing price distribution by quality level
- Helps understand:
  * Price range for different quality levels
  * Price premium for higher quality
  * Value for money at different quality levels
"""
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Quality', y='Average Price', palette='viridis')
plt.title('Price Distribution by Quality Level')
plt.xlabel('Quality')
plt.ylabel('Average Price')
plt.show()

#%%
# 6. Availability Impact Study
# --------------------------
"""
Analysis Notes:
- Examines how availability affects food prices
- Box plot shows price distribution by availability level
- Useful for understanding:
  * Price impact of supply changes
  * Relationship between availability and price
  * Market dynamics
"""
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Availability', y='Average Price', palette='viridis')
plt.title('Price Distribution by Availability')
plt.xlabel('Availability')
plt.ylabel('Average Price')
plt.show()

#%%
# 7. Regional Price Comparisons
# ---------------------------
"""
Analysis Notes:
- Compares average food prices across different countries
- Sorted in descending order
- Helps identify:
  * Most/least expensive countries
  * Regional price differences
  * Potential market opportunities
"""
plt.figure(figsize=(12, 6))
country_avg = df.groupby('Country')['Average Price'].mean().sort_values(ascending=False)
sns.barplot(x=country_avg.index, y=country_avg.values, palette='viridis')
plt.title('Average Food Prices by Country')
plt.xlabel('Country')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 8. Price Volatility Analysis
# --------------------------
"""
Analysis Notes:
- Measures price stability across different food items
- Standard deviation used as volatility measure
- Helps identify:
  * Most/least stable priced items
  * Risk factors in food pricing
  * Items requiring careful price monitoring
"""
plt.figure(figsize=(12, 6))
volatility = df.groupby('Food Item')['Average Price'].std().sort_values(ascending=False)
sns.barplot(x=volatility.index, y=volatility.values, palette='viridis')
plt.title('Price Volatility by Food Item')
plt.xlabel('Food Item')
plt.ylabel('Price Standard Deviation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%%
# 9. Inflation Rate Analysis
# ------------------------
"""
Analysis Notes:
- Calculates year-over-year price changes
- Shows percentage change in average prices
- Important for:
  * Tracking food price inflation
  * Identifying years with significant price changes
  * Understanding long-term price trends
"""
plt.figure(figsize=(10, 6))
yearly_avg = df.groupby('Year')['Average Price'].mean().reset_index()
yearly_avg['Price Change %'] = yearly_avg['Average Price'].pct_change() * 100
sns.barplot(x='Year', y='Price Change %', data=yearly_avg, palette='viridis')
plt.title('Year-over-Year Price Changes')
plt.xlabel('Year')
plt.ylabel('Price Change (%)')
plt.show()

#%%
# 10. Price Correlation Analysis
# ----------------------------
"""
Analysis Notes:
- Examines relationships between different numeric variables
- Heatmap shows correlation coefficients
- Red indicates positive correlation
- Blue indicates negative correlation
- Helps understand:
  * Relationships between different factors
  * Potential causal relationships
  * Key drivers of price changes
"""
plt.figure(figsize=(10, 8))
numeric_cols = ['Year', 'Month', 'Average Price', 'Price in USD', 'Availability']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Variables')
plt.tight_layout()
plt.show()

#%%
# Additional Insights
# -----------------
"""
Summary Statistics Notes:
- Provides quick reference for key findings
- Includes top performers in different categories
- Useful for quick decision-making and reporting
"""
print("\nTop 5 Most Expensive Food Items:")
print(df.groupby('Food Item')['Average Price'].mean().sort_values(ascending=False).head())

print("\nTop 5 Countries with Highest Average Prices:")
print(df.groupby('Country')['Average Price'].mean().sort_values(ascending=False).head())

print("\nFood Items with Highest Price Volatility:")
print(volatility.head())

print("\nYear with Highest Average Price Increase:")
max_increase_year = yearly_avg.loc[yearly_avg['Price Change %'].idxmax()]
print(f"Year: {max_increase_year['Year']}, Price Increase: {max_increase_year['Price Change %']:.2f}%") 