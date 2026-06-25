"""Script to backtest a MA(5,20) crossover on AAPL and plot results.

This file downloads AAPL price data, computes 5-day and 20-day SMAs,
generates a simple long-only MA crossover signal, computes returns, and
plots cumulative performance for comparison.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests import Session

# Create a custom network session to make the Yahoo Finance request look more like a normal browser request
session = Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})

# Download AAPL stock data for the past 1 year using the custom session above
data = yf.download("AAPL", period = "1y", session=session)

# Calculate Simple Moving Averages (SMA)
# SMA_5: 5-day moving average - captures short-term trends
# SMA_20: 20-day moving average - captures long-term trends
data['SMA_5'] = data['Close'].rolling(window = 5).mean()
data['SMA_20'] = data['Close'].rolling(window = 20).mean()

# Remove rows with NaN values (from moving average calculation)
data = data.dropna()

# Generate trading signal: 1 if SMA_5 > SMA_20 (bullish), 0 otherwise (bearish)
data['Signal'] = np.where(data['SMA_5'] > data['SMA_20'], 1, 0)

# Calculate daily returns
data['Market_Return'] = data['Close'].pct_change()

# Calculate strategy returns: market return multiplied by previous day's signal - shift(1) to avoid lookahead bias
# This simulates buying/holding (signal=1) or not holding (signal=0)
data['Strategy_Return'] = data['Market_Return'] * data['Signal'].shift(1)

# Calculate cumulative returns to track wealth growth over time
# Cumulative_Market: returns if holding the stock throughout the period
# Cumulative_Strategy: returns using the moving average trading signal
data['Cumulative_Market'] = (1 + data['Market_Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

# Display backtesting results: final cumulative returns for comparison
print("\n===== Backtesting Results =====")
print(data[['Cumulative_Market', 'Cumulative_Strategy']].tail(1))


# Create a comparison chart for the buy-and-hold benchmark and the moving average strategy
plt.figure(figsize = (10, 6))

# Plot the market return curve as a dashed blue line
plt.plot(data['Cumulative_Market'], label = 'Market (Buy & Hold)', linestyle = '--', color = 'blue')

# Plot the strategy return curve as a solid red line
plt.plot(data['Cumulative_Strategy'], label = 'MA Strategy', color = 'red')

# Add chart labels and styling so the comparison is easy to read
plt.title('Backtest: MA(5, 20) Strategy vs Buy & Hold (AAPL)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (1.0 = Initial Capital)')
plt.legend()
plt.grid(True)

# Show the finished plot window
# Blocking call on some environments; keep it for interactive use
plt.show()

