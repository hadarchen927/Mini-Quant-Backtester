from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding = "utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML object at the root.")
    return data

def run_backtest(config: dict[str, Any]) -> int:
    import pandas as pd
    from mini_quant_backtester.core.engine import run_ma_crossover_backtest

    # Temporary synthetic data until data loader is added
    n = 400
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series((100 + pd.Series(range(n)).values * 0.03), index=rng) + \
        pd.Series(pd.np.random.normal(0, 1, n), index=rng).cumsum() * 0.2  # type: ignore[attr-defined]
    data = pd.DataFrame({"close": close}, index=rng)

    strategy_cfg = config.get("strategy", {})
    costs_cfg = config.get("costs", {})

    result = run_ma_crossover_backtest(
        data=data,
        initial_cash=float(config.get("initial_cash", 100_000)),
        fast_window=int(strategy_cfg.get("fast_window", 20)),
        slow_window=int(strategy_cfg.get("slow_window", 50)),
        commission_per_trade=float(costs_cfg.get("commission_per_trade", 1.0)),
        slippage_bps=float(costs_cfg.get("slippage_bps", 2.0)),
    )

    print("=== Backtest Complete ===")
    print(f"Final equity: {result.final_equity:,.2f}")
    print(f"Total return: {result.total_return * 100:.2f}%")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog = "mqb",
        description = "Mini Quant Backtester CLI",
    )
    subparsers = parser.add_subparsers(dest = "command", required = True)

    run_parser = subparsers.add_parser("run", help = "Run a backtest from YAML config")
    run_parser.add_argument(
        "--config",
        "-c",
        required = True,
        help = "Path to YAML config file",
    )

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "run":
            config = load_config(args.config)
            code = run_backtest(config)
            raise SystemExit(code)
    
        raise SystemExit(2)
    except Exception as exc: # keep broad for CLI surface
        print(f"Error: {exc}", file = sys.stderr)
        raise SystemExit(1)
    
if __name__ == "__main__":
    main()

