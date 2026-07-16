from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from mini_quant_backtester.strategies.ma_crossover import generate_ma_crossover_signals

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    final_equity: float
    total_return: float

def run_ma_crossover_backtest(
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        fast_window: int = 20,
        slow_window: int = 50,
        commisssion_per_trade: float = 1.0,
        slippage_bps: float = 2.0,
        price_col: str = "close",
) -> BacktestResult:
    if price_col not in data.columns:
        raise ValueError(f"Missing required column: '{price_col}")
    
    df = generate_ma_crossover_signals(
        data = data,
        fast_window = fast_window,
        slow_window = slow_window,
        price_col = price_col,
    ).copy()

    # Strategy return: use previous bar signal to avoid look-ahead bias
    asset_ret = df[price_col].pct_change().fillna(0.0)
    strategy_ret = asset_ret * df["signal"].shift(1).fillna(0)

    # Cost model when position changes
    trades = df["position_change"].abs() # 1 for buy/sell switch, can be 2 on flip
    slippage_rate = slippage_bps / 10_000.0
    total_cost_rate = trades * slippage_rate
    total_cost_flat = trades * (commisssion_per_trade / initial_cash)

    net_ret = strategy_ret - total_cost_rate - total_cost_flat
    equity_curve = (1.0 + net_ret).cumprod() * initial_cash

    final_equity = float(equity_curve.iloc[-1])
    total_return = (final_equity - initial_cash) - 1.0

    return BacktestResult(
        equity_curve = equity_curve,
        returns = net_ret,
        final_equity = final_equity,
        total_return = float(total_return),
    )