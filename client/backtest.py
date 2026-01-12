"""client/backtest.py

Backtest runner that matches your UI contract:

  run_backtest_from_yahoo(bot: BotConfig, start: datetime, end: datetime) -> BacktestResult

It uses:
- client/yahoo_data.py to download data
- client/strategies.py to execute strategy logic

This file is local-only (desktop).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import pandas as pd

from shared.models import BotConfig, Trade


from client.yahoo_data import YahooLoadOptions, fetch_ohlcv
from client import strategies


@dataclass
class BacktestResult:
    strategy_id: str
    symbols: List[str]
    start: datetime
    end: datetime
    initial_capital: float

    trades: List[Trade]
    equity_curve: pd.Series

    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int


def _compute_kpis(equity_curve: pd.Series, initial_capital: float) -> Dict[str, float]:
    if equity_curve is None or equity_curve.empty:
        final_equity = float(initial_capital)
        return {
            "final_equity": final_equity,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    final_equity = float(equity_curve.iloc[-1])
    total_return_pct = ((final_equity / float(initial_capital)) - 1.0) * 100.0

    # max drawdown
    running_max = equity_curve.cummax()
    dd = (equity_curve / running_max) - 1.0
    max_dd = float(dd.min()) * 100.0  # negative

    return {
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_dd),
    }


def run_backtest_from_yahoo(bot: BotConfig, start: datetime, end: datetime) -> BacktestResult:
    """Main entry called by UI.

    Requirements (UI must provide):
      - bot.bot_id, bot.user_id
      - bot.symbols
      - bot.capital
      - bot.strategy.strategy_id
      - bot.strategy.params
      - start/end datetimes

    This function:
      1) loads OHLCV from Yahoo
      2) calls the strategy backtest function
      3) computes KPIs
      4) returns BacktestResult
    """

    sid = bot.strategy.strategy_id
    params = bot.strategy.params or {}

    
    timeframe = str(params.get("timeframe", "1D"))

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    # Load data per symbol
    data_by_symbol: Dict[str, pd.DataFrame] = {}
    opts = YahooLoadOptions()

    for sym in bot.symbols:
        df = fetch_ohlcv(sym, start=start_s, end=end_s, timeframe=timeframe, options=opts)
        # Ensure required columns exist
        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise ValueError(f"Missing {col} for {sym}")
        data_by_symbol[sym] = df

    # Dispatch
    if sid == "mean_reversion_losers":
        trades, equity = strategies.backtest_mean_reversion_losers(bot, data_by_symbol)
    elif sid == "moving_average":
        trades, equity = strategies.backtest_moving_average(bot, data_by_symbol)
    elif sid == "rsi_reversion":
        trades, equity = strategies.backtest_rsi_reversion(bot, data_by_symbol)
    elif sid == "macd_trend":
        trades, equity = strategies.backtest_macd_trend(bot, data_by_symbol)
    else:
        raise KeyError(f"Unknown strategy_id: {sid}")

    k = _compute_kpis(equity, bot.capital)

    return BacktestResult(
        strategy_id=sid,
        symbols=list(bot.symbols),
        start=start,
        end=end,
        initial_capital=float(bot.capital),
        trades=list(trades),
        equity_curve=equity,
        final_equity=float(k["final_equity"]),
        total_return_pct=float(k["total_return_pct"]),
        max_drawdown_pct=float(k["max_drawdown_pct"]),
        total_trades=int(len(trades)),
    )
