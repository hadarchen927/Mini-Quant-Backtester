from __future__ import annotations

import pandas as pd

def generate_ma_crossover_signals(
        data: pd.DataFrame,
        fast_window: int = 20,
        slow_window: int = 50,
        price_col: str = "close",
) -> pd.DataFrame:
    if price_col not in data.columns:
        raise ValueError(f"Missing required column: '{price_col}")
    
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("fast_window and slow_window must be positive integers")
    if fast_window >= slow_window:
        raise ValueError("fast_window should be smaller than slow_window")
    

    df = data.copy()
    df["fast_ma"] = df[price_col].rolling(fast_window).mean()
    df["slow_ma"] = df[price_col].rolling(slow_window).mean()

    # 1 when fast > slow else 0
    df["signal"] = (df["fast_ma"] > df["slow_ma"]).astype(int)

    # Position change (entry/exit makers)
    df["position_change"] = df["signal"].diff().fillna(0)

    return df
