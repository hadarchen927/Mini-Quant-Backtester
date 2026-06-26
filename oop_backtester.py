"""Simple moving-average backtester (object-oriented).

This module provides `MABacktester`, a minimal OOP backtesting helper that
downloads historical price data using `yfinance`, computes short/long SMAs,
generates a simple long-only signal (short SMA > long SMA), and compares the
strategy performance against buy-and-hold.
"""

import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from requests import Session

# Use a requests Session with common browser headers to reduce chance of
# request blocking when yfinance fetches data.
session = Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
})


class MABacktester:
    """A minimal moving-average crossover backtester.

    Attributes:
        ticker (str): ticker symbol passed to yfinance (e.g., "AAPL").
        short_window (int): window size for the short SMA.
        long_window (int): window size for the long SMA.
        period (str): history length string accepted by yfinance (e.g., "1y").
        data (pd.DataFrame): downloaded price data and computed columns after
            calling `fetch_data()` and `run_backtest()`.
    """

    def __init__(self, ticker: str, short_window: int, long_window: int, period: str = "1y") -> None:
        """
        Initialize the backtester with ticker, SMA windows, and period.
        :param ticker: str, ticker symbol (e.g., "AAPL")
        :param short_window: int, window size for the short SMA
        :param long_window: int, window size for the long SMA
        :param period: str, history length string accepted by yfinance (default: "1y")
        """

        self.ticker = ticker
        self.short_window = short_window
        self.long_window = long_window
        self.period = period

    def fetch_data(self):
        """Download OHLCV data for `self.ticker` into `self.data`.

        Uses the shared `session` so requests include browser-like headers.
        """
        print(f"Fetching data for {self.ticker} for the past {self.period}...")
        self.data = yf.download(self.ticker, period=self.period, session=session)

    def run_backtest(self):
        """Compute indicators, signals, returns, and cumulative performance.

        Steps:
        - Compute short and long SMAs on close prices
        - Create a binary `Signal` column (1 when short > long, else 0)
        - Compute daily market returns and strategy returns (signal applied
          with a one-day shift to avoid lookahead)
        - Compute cumulative returns for market and strategy
        """
        # Simple moving averages on the close price
        self.data['SMA_short'] = self.data['Close'].rolling(window = self.short_window).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window = self.long_window).mean()
        # Drop initial NaN rows introduced by rolling windows
        self.data = self.data.dropna()

        # 1 when short SMA is above long SMA, otherwise 0
        self.data['Signal'] = np.where(self.data['SMA_short'] > self.data['SMA_long'], 1, 0)
        # Daily percentage change of the close price
        self.data['Market_Return'] = self.data['Close'].pct_change()
        # Strategy return uses prior day's signal to avoid lookahead bias
        self.data['Strategy_Return'] = self.data['Market_Return'] * self.data['Signal'].shift(1)

        # Apply trading costs to the strategy returns
        # cost_rate: transaction cost per trade (0.1% in this case)
        cost_rate = 0.001
        # Trades: detect when signal changes (0->1 or 1->0), indicating a buy/sell trade
        self.data['Trades'] = self.data['Signal'].diff().abs()
        # Strategy_Return_Net: gross strategy return minus trading costs applied on each trade
        self.data["Strategy_Return_Net"] = self.data['Strategy_Return'] - (self.data['Trades'] * cost_rate)

        self.data = self.data.dropna() 
        
        # Cumulative product of returns to simulate growth of $1 starting capital
        self.data['Cumulative_Market'] = (1 + self.data['Market_Return']).cumprod()
        # Strategy cumulative returns including trading costs (gross returns minus transaction fees)
        self.data['Cumulative_Strategy'] = (1 + self.data['Strategy_Return_Net']).cumprod()

    def print_performance(self):
        """Print final percentage returns for market and strategy."""
        market_return = (self.data['Cumulative_Market'].iloc[-1] - 1) * 100
        strategy_return = (self.data['Cumulative_Strategy'].iloc[-1] - 1) * 100
        print(f"\n===== Backtesting Results for {self.ticker} (Average Line {self.short_window} & {self.long_window}) =====")
        print(f"Market Return (Buy & Hold): {market_return:.2f}%")
        print(f"Strategy Return (MA Strategy): {strategy_return:.2f}% \n")

    def plot_performance(self):
        """Plot cumulative performance of market vs. strategy (including trading costs)."""
        # Visualize the cumulative returns over time to compare strategy vs. buy-and-hold benchmark
        plt.figure(figsize = (10, 6))
        # Buy-and-hold benchmark (dashed blue line)
        plt.plot(self.data['Cumulative_Market'], label = 'Market (Buy & Hold)', linestyle = '--', color = 'blue')
        # MA strategy with trading costs accounted for (solid orange line)
        plt.plot(self.data['Cumulative_Strategy'], label = 'MA Strategy', color = 'orange')
        plt.title(f"Cumulative Performance: {self.ticker} ({self.short_window}/{self.long_window} SMA)")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns ($1 Starting Capital)")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":

    # 1. Create an argument parser to allow command-line input for ticker, short SMA, long SMA, and period
    parser = argparse.ArgumentParser(description="Simple Moving Average Backtester")

    # 2. Define command-line arguments with default values and help descriptions
    parser.add_argument("--ticker", type = str, default = "APPL", help = "Ticker symbol (default: AAPL)")
    parser.add_argument("--short", type = int, default = 10, help = "Short SMA window (default: 10)")
    parser.add_argument("--long", type = int, default = 50, help = "Long SMA window (default: 50)")

    # 3. Parse the command-line arguments into the `args` namespace
    args = parser.parse_args()

    # 4. Print the backtest configuration and run the backtester
    print(f"Running backtest for {args.ticker} with short SMA = {args.short} and long SMA = {args.long}... \n")
    tester = MABacktester(ticker = args.ticker, short_window = args.short, long_window = args.long)
    tester.fetch_data()
    tester.run_backtest()
    tester.print_performance()