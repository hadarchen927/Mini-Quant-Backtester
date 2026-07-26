# Mini Quant Backtester

A Python-based object-oriented backtesting framework for quantitative trading strategies.
It downloads historical price data from Yahoo Finance, runs a moving-average crossover test,
plots cumulative performance, and lets you analyze multiple tickers in one interactive session.

## Features
- **OOP Design**: Easy to instantiate and reuse for different tickers.
- **Slippage & Fees**: Includes transaction costs calculation for realistic results.
- **Grid Search**: Built-in parameter optimization for moving averages.

## How to use
Run the `oop_backtester.py` script, then enter a ticker when prompted.
After each backtest, the program asks which stock you want to analyze next.

Example:

```bash
python oop_backtester.py
```

You can also pass an initial ticker and moving-average windows:

```bash
python oop_backtester.py --ticker AAPL --short 10 --long 50 --period 1y
```