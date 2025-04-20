#%%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="viridis")

#%%
# Load and prepare data
# Read CSV with proper column handling
df = pd.read_csv('food_prices.csv', skipinitialspace=True)

# Clean column names by removing extra spaces
df.columns = df.columns.str.strip()

# Convert numeric columns to appropriate types
numeric_columns = ['Year', 'Month', 'Average Price', 'Price in USD', 'Availability']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Create Date column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))

# Verify data types
print("\nData Types:")
print(df.dtypes)

# Verify data structure
print("\nData Overview:")
print(df.head())

#%%
# 1. Global Price Trends Analysis
# ---------------------------
"""
Global Price Trends:
- Overall price trends across all countries
- Price changes over time
- Major price movements
"""
# Calculate global average prices
global_avg = df.groupby('Date')['Price in USD'].mean()

# Plot global price trends
plt.figure(figsize=(12, 6))
global_avg.plot()
plt.title('Global Food Price Trends (2018-2023)')
plt.xlabel('Date')
plt.ylabel('Average Price in USD')
plt.grid(True)
plt.show()

# Calculate price changes
price_changes = global_avg.pct_change() * 100
print("\nGlobal Price Changes:")
print(f"Average annual increase: {price_changes.mean():.2f}%")
print(f"Maximum monthly increase: {price_changes.max():.2f}%")
print(f"Maximum monthly decrease: {price_changes.min():.2f}%")

#%%
# 2. Country-Specific Analysis
# ------------------------
"""
Country Analysis:
- Price levels by country
- Price stability
- Currency impact
"""
# Calculate average prices by country
country_prices = df.groupby('Country')['Price in USD'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

# Plot country price levels
plt.figure(figsize=(12, 6))
sns.barplot(x=country_prices.index, y='mean', data=country_prices)
plt.title('Average Food Prices by Country (USD)')
plt.xlabel('Country')
plt.ylabel('Average Price in USD')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate price stability (lower std = more stable)
print("\nPrice Stability by Country:")
print(country_prices.sort_values('std').head(5))  # Most stable
print("\nMost Volatile Prices:")
print(country_prices.sort_values('std', ascending=False).head(5))  # Most volatile

#%%
# 3. Food Item Analysis
# -----------------
"""
Food Item Analysis:
- Price differences between items
- Most expensive/cheapest items
- Price stability by item
"""
# Calculate statistics by food item
food_stats = df.groupby('Food Item')['Price in USD'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

# Plot food item prices
plt.figure(figsize=(12, 6))
sns.barplot(x=food_stats.index, y='mean', data=food_stats)
plt.title('Average Prices by Food Item (USD)')
plt.xlabel('Food Item')
plt.ylabel('Average Price in USD')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nMost Expensive Food Items:")
print(food_stats.head(5))
print("\nLeast Expensive Food Items:")
print(food_stats.tail(5))

#%%
# 4. Seasonal Analysis
# -----------------
"""
Seasonal Patterns:
- Monthly price patterns
- Seasonal variations
- Peak and low seasons
"""
# Calculate monthly averages
monthly_avg = df.groupby('Month')['Price in USD'].mean()

# Plot seasonal patterns
plt.figure(figsize=(12, 6))
monthly_avg.plot(kind='bar')
plt.title('Average Monthly Food Prices')
plt.xlabel('Month')
plt.ylabel('Average Price in USD')
plt.xticks(rotation=0)
plt.show()

# Identify peak and low seasons
print("\nSeasonal Analysis:")
print(f"Highest average prices in month: {monthly_avg.idxmax()}")
print(f"Lowest average prices in month: {monthly_avg.idxmin()}")
print(f"Seasonal variation: {(monthly_avg.max() - monthly_avg.min()) / monthly_avg.mean() * 100:.2f}%")

#%%
# 5. Quality Impact Analysis
# ----------------------
"""
Quality Impact:
- Price differences by quality
- Quality premium
- Quality distribution
"""
# Calculate price differences by quality
quality_prices = df.groupby('Quality')['Price in USD'].agg(['mean', 'std', 'count'])

# Plot quality impact
plt.figure(figsize=(10, 6))
sns.boxplot(x='Quality', y='Price in USD', data=df)
plt.title('Price Distribution by Quality Level')
plt.xlabel('Quality')
plt.ylabel('Price in USD')
plt.show()

print("\nQuality Impact Analysis:")
print(quality_prices)
print(f"\nQuality Premium (High vs Low): {(quality_prices.loc['High', 'mean'] - quality_prices.loc['Low', 'mean']) / quality_prices.loc['Low', 'mean'] * 100:.2f}%")

#%%
# 6. Availability Impact
# ------------------
"""
Availability Impact:
- Price changes with availability
- Supply-demand relationship
- Availability patterns
"""
# Calculate price differences by availability
availability_prices = df.groupby('Availability')['Price in USD'].agg(['mean', 'std', 'count'])

# Plot availability impact
plt.figure(figsize=(10, 6))
sns.boxplot(x='Availability', y='Price in USD', data=df)
plt.title('Price Distribution by Availability')
plt.xlabel('Availability')
plt.ylabel('Price in USD')
plt.show()

print("\nAvailability Impact Analysis:")
print(availability_prices)
print(f"\nPrice Impact of Low Availability: {(availability_prices.loc[0, 'mean'] - availability_prices.loc[1, 'mean']) / availability_prices.loc[1, 'mean'] * 100:.2f}%")

#%%
# 7. Currency Analysis
# -----------------
"""
Currency Analysis:
- Currency impact on prices
- Exchange rate effects
- Currency stability
"""
# Calculate average prices by currency
currency_prices = df.groupby('Currency')['Price in USD'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

# Plot currency impact
plt.figure(figsize=(12, 6))
sns.barplot(x=currency_prices.index, y='mean', data=currency_prices)
plt.title('Average Prices by Currency')
plt.xlabel('Currency')
plt.ylabel('Average Price in USD')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nCurrency Analysis:")
print(currency_prices)
print(f"\nCurrency with highest average prices: {currency_prices.index[0]}")
print(f"Currency with lowest average prices: {currency_prices.index[-1]}")

#%%
# 8. Comprehensive Insights
# ---------------------
"""
Comprehensive Insights:
- Key findings
- Important patterns
- Significant relationships
"""
print("\nComprehensive Analysis Summary:")
print("\n1. Global Trends:")
print(f"- Average global food price: ${global_avg.mean():.2f}")
print(f"- Total price change (2018-2023): {(global_avg.iloc[-1] - global_avg.iloc[0]) / global_avg.iloc[0] * 100:.2f}%")

print("\n2. Country Analysis:")
print(f"- Country with highest average prices: {country_prices.index[0]} (${country_prices['mean'].iloc[0]:.2f})")
print(f"- Country with lowest average prices: {country_prices.index[-1]} (${country_prices['mean'].iloc[-1]:.2f})")

print("\n3. Food Items:")
print(f"- Most expensive item: {food_stats.index[0]} (${food_stats['mean'].iloc[0]:.2f})")
print(f"- Least expensive item: {food_stats.index[-1]} (${food_stats['mean'].iloc[-1]:.2f})")

print("\n4. Seasonal Patterns:")
print(f"- Highest prices typically in month: {monthly_avg.idxmax()}")
print(f"- Lowest prices typically in month: {monthly_avg.idxmin()}")

print("\n5. Quality Impact:")
print(f"- High quality premium: {(quality_prices.loc['High', 'mean'] - quality_prices.loc['Low', 'mean']) / quality_prices.loc['Low', 'mean'] * 100:.2f}%")

print("\n6. Availability Impact:")
print(f"- Price increase during low availability: {(availability_prices.loc[0, 'mean'] - availability_prices.loc[1, 'mean']) / availability_prices.loc[1, 'mean'] * 100:.2f}%")

print("\n7. Currency Analysis:")
print(f"- Currency with highest average prices: {currency_prices.index[0]}")
print(f"- Currency with lowest average prices: {currency_prices.index[-1]}") 