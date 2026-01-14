"""client/strategies.py

FILE OVERVIEW:
This file contains the core logic for the **Backtesting Engine**.
Unlike the live bot which runs on the server, this code runs locally on your computer.
It simulates how simulated trades would have performed in the past using historical data.

It contains:
1. Helper functions to manage data (sorting, validating).
2. The logic for 4 specific trading strategies:
   - Mean Reversion (Top Losers)
   - Moving Average Crossover
   - RSI Mean Reversion
   - MACD Trend
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from uuid import uuid4

import pandas as pd

from shared.models import BotConfig, Trade, OrderSide


@dataclass
class Position:
    """
    A simple data structure to track what we own during the simulation.
    qty: How many shares we hold.
    avg_price: The average price we paid for them.
    """
    qty: int = 0
    avg_price: float = 0.0

def _has_series_data(s) -> bool:
    """Checks if a pandas Series exists and has data."""
    return isinstance(s, pd.Series) and (not s.empty)

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures that the historical data is sorted by date (Oldest -> Newest).
    If dates are out of order, the strategy simulation would fail.
    """
    if not df.index.is_monotonic_increasing:
        return df.sort_index()
    return df


def _bar_close(df: pd.DataFrame) -> pd.Series:
    """Extracts the 'Close' price column from the dataframe."""
    if "Close" not in df.columns:
        raise ValueError("Missing Close column")
    return df["Close"].astype(float)


def _mk_trade(bot: BotConfig, symbol: str, side: OrderSide, qty: int, price: float, ts: datetime) -> Trade:
    """
    Creates a 'Trade' object to record a transaction in our history.
    Since this is a simulation, we generate a fake random UUID for the trade ID.
    """
    return Trade(
        trade_id=str(uuid4()),
        bot_id=bot.bot_id,
        user_id=bot.user_id,
        symbol=symbol,
        side=side,
        quantity=int(qty),
        price=float(price),
        executed_at=ts,
    )


def _apply_stop_tp(entry_price: float, current_price: float, stop_loss_pct: float, take_profit_pct: float) -> bool:
    """
    Checks if a position should be closed based on Risk Management rules.
    
    Returns True (Sell Signal) if:
    1. The price dropped below the Stop Loss threshold (limit losses).
    2. The price rose above the Take Profit threshold (secure gains).
    """
    if entry_price <= 0:
        return False
    
    # Check Stop Loss
    if stop_loss_pct and stop_loss_pct > 0:
        if current_price <= entry_price * (1.0 - float(stop_loss_pct)):
            return True
            
    # Check Take Profit
    if take_profit_pct and take_profit_pct > 0:
        if current_price >= entry_price * (1.0 + float(take_profit_pct)):
            return True
            
    return False


# =============================================================================
# Strategy 1: Daily Mean Reversion — Top Losers
# =============================================================================

def backtest_mean_reversion_losers(
    bot: BotConfig,
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[List[Trade], pd.Series]:
    """
    LOGIC:
    1. Look at yesterday's performance for all symbols.
    2. Identify the top N 'Losers' (stocks that dropped the most).
    3. Buy them at today's Open price, betting they will bounce back.
    4. Sell them at today's Close price (Intraday trade).
    """
    params = bot.strategy.params or {}
    losers_to_buy = int(params.get("losers_to_buy", 5))
    losers_to_buy = max(1, losers_to_buy)

    # Clean and sort data for all symbols
    dfs = {s: _ensure_sorted(df) for s, df in data_by_symbol.items()}
    
    # Find the common dates where we have data for ALL symbols
    base_index = None
    for df in dfs.values():
        base_index = df.index if base_index is None else base_index.intersection(df.index)
    if base_index is None or len(base_index) < 3:
        raise ValueError("Not enough data to backtest mean reversion strategy.")

    trades: List[Trade] = []
    equity_points = []
    cash = float(bot.capital)

    # Loop through each day (starting from index 1 because we need previous day's data)
    for i in range(1, len(base_index)):
        ts = base_index[i]       # Today
        prev_ts = base_index[i - 1] # Yesterday

        # --- STEP 1: Calculate Returns ---
        returns = []
        for sym, df in dfs.items():
            if ts not in df.index or prev_ts not in df.index:
                continue
            
            # Calculate % change. 
            # Logic: If we have 2 days of history, we check PrevClose vs PrevPrevClose.
            # Otherwise we use PrevOpen vs PrevClose.
            if i >= 2:
                prevprev_ts = base_index[i - 2]
                if prevprev_ts in df.index:
                    r = (float(df.loc[prev_ts, "Close"]) / float(df.loc[prevprev_ts, "Close"])) - 1.0
                else:
                    r = (float(df.loc[prev_ts, "Close"]) / float(df.loc[prev_ts, "Open"])) - 1.0
            else:
                r = (float(df.loc[prev_ts, "Close"]) / float(df.loc[prev_ts, "Open"])) - 1.0
            returns.append((sym, r))

        if not returns:
            continue

        # --- STEP 2: Rank Losers ---
        # Sort by return (lowest first)
        returns.sort(key=lambda x: x[1])
        # Pick top N symbols
        picked = [sym for sym, _ in returns[: min(losers_to_buy, len(returns))]]

        # --- STEP 3: Execute Trades ---
        # Split cash equally among the picked stocks
        alloc = cash / float(len(picked)) if picked else 0.0

        day_open_cost = 0.0
        day_close_value = 0.0

        for sym in picked:
            df = dfs[sym]
            open_px = float(df.loc[ts, "Open"])   # Buy Price
            close_px = float(df.loc[ts, "Close"]) # Sell Price
            
            if open_px <= 0: continue
            
            # Calculate quantity we can afford
            qty = int(alloc // open_px)
            if qty <= 0: continue
            
            # BUY Transaction
            cost = qty * open_px
            cash -= cost
            day_open_cost += cost
            trades.append(_mk_trade(bot, sym, OrderSide.BUY, qty, open_px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
            
            # SELL Transaction (End of day)
            cash += qty * close_px
            day_close_value += qty * close_px
            trades.append(_mk_trade(bot, sym, OrderSide.SELL, qty, close_px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))

        # Record daily equity (cash balance)
        equity = cash
        equity_points.append((ts, equity))

    # Format the Equity Curve for the chart
    if not equity_points:
        equity_curve = pd.Series([float(bot.capital)], index=[base_index[-1]])
    else:
        equity_curve = pd.Series([v for _, v in equity_points], index=[t for t, _ in equity_points])

    return trades, equity_curve


# =============================================================================
# Strategy 2: Moving Average Trend (SMA/EMA cross)
# =============================================================================

def backtest_moving_average(
    bot: BotConfig,
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[List[Trade], pd.Series]:
    """
    LOGIC:
    1. Calculate two averages: Short (Fast) and Long (Slow).
    2. Buy when Short crosses ABOVE Long (Golden Cross).
    3. Sell when Short crosses BELOW Long (Death Cross).
    """
    params = bot.strategy.params or {}
    ma_type = str(params.get("ma_type", "SMA")).upper()
    short_p = int(params.get("short_period", 20))
    long_p = int(params.get("long_period", 50))
    stop_loss = float(params.get("stop_loss_pct", 0.0) or 0.0)
    take_profit = float(params.get("take_profit_pct", 0.0) or 0.0)

    # Ensure Long period is at least 1 greater than Short
    if long_p <= short_p:
        long_p = short_p + 1

    dfs = {s: _ensure_sorted(df) for s, df in data_by_symbol.items()}
    # Create a unified timeline (union of all dates)
    index = None
    for df in dfs.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    # Track positions per symbol
    positions: Dict[str, Position] = {s: Position() for s in dfs.keys()}
    entry_price: Dict[str, float] = {s: 0.0 for s in dfs.keys()}

    cash = float(bot.capital)
    trades: List[Trade] = []
    equity_vals = []

    # --- Precompute Indicators ---
    ind = {}
    for sym, df in dfs.items():
        close = _bar_close(df)
        if ma_type == "EMA":
            s_ma = close.ewm(span=short_p, adjust=False).mean()
            l_ma = close.ewm(span=long_p, adjust=False).mean()
        else:
            s_ma = close.rolling(window=short_p, min_periods=short_p).mean()
            l_ma = close.rolling(window=long_p, min_periods=long_p).mean()
        ind[sym] = (s_ma, l_ma, close)

    symbols = list(dfs.keys())
    per_symbol_budget = cash / max(1, len(symbols)) # Equal allocation

    # --- Main Time Loop ---
    for ts in index:
        # Update Portfolio Value (Mark to Market)
        equity = cash
        for sym, pos in positions.items():
            if pos.qty > 0:
                df = dfs[sym]
                if ts in df.index:
                    px = float(df.loc[ts, "Close"])
                else:
                    px = float(ind[sym][2].loc[:ts].iloc[-1]) # Last known price
                equity += pos.qty * px

        equity_vals.append((ts, equity))

        # Check signals for each symbol
        for sym in symbols:
            df = dfs[sym]
            if ts not in df.index: continue

            s_ma, l_ma, close = ind[sym]
            if ts not in s_ma.index or ts not in l_ma.index: continue
            if pd.isna(s_ma.loc[ts]) or pd.isna(l_ma.loc[ts]): continue

            px = float(df.loc[ts, "Close"])
            pos = positions[sym]

            # 1. Check Stop Loss / Take Profit
            if pos.qty > 0 and _apply_stop_tp(entry_price[sym], px, stop_loss, take_profit):
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0
                continue

            # 2. Check Crossover Logic
            # BUY if Short > Long (and we don't have position)
            if pos.qty == 0 and float(s_ma.loc[ts]) > float(l_ma.loc[ts]):
                qty = int(per_symbol_budget // px)
                if qty > 0:
                    cash -= qty * px
                    pos.qty = qty
                    entry_price[sym] = px
                    trades.append(_mk_trade(bot, sym, OrderSide.BUY, qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
            
            # SELL if Short < Long (and we have position)
            elif pos.qty > 0 and float(s_ma.loc[ts]) < float(l_ma.loc[ts]):
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0

    equity_curve = pd.Series([v for _, v in equity_vals], index=[t for t, _ in equity_vals])
    return trades, equity_curve


# =============================================================================
# Strategy 3: RSI Mean Reversion
# =============================================================================

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Calculates Relative Strength Index (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def backtest_rsi_reversion(
    bot: BotConfig,
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[List[Trade], pd.Series]:
    """
    LOGIC:
    1. Calculate RSI.
    2. Buy when RSI is Low (Oversold) -> Expecting price to go UP.
    3. Sell when RSI is High (Overbought) -> Expecting price to go DOWN.
    """
    params = bot.strategy.params or {}
    period = int(params.get("rsi_period", 14))
    oversold = int(params.get("oversold", 30))
    overbought = int(params.get("overbought", 70))
    stop_loss = float(params.get("stop_loss_pct", 0.0) or 0.0)
    take_profit = float(params.get("take_profit_pct", 0.0) or 0.0)

    dfs = {s: _ensure_sorted(df) for s, df in data_by_symbol.items()}
    index = None
    for df in dfs.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    positions: Dict[str, Position] = {s: Position() for s in dfs.keys()}
    entry_price: Dict[str, float] = {s: 0.0 for s in dfs.keys()}
    cash = float(bot.capital)
    trades: List[Trade] = []
    equity_vals = []

    symbols = list(dfs.keys())
    per_symbol_budget = cash / max(1, len(symbols))

    indicators = {}
    for sym, df in dfs.items():
        close = _bar_close(df)
        indicators[sym] = (_rsi(close, period), close)

    for ts in index:
        equity = cash
        for sym, pos in positions.items():
            if pos.qty > 0:
                df = dfs[sym]
                px = float(df.loc[ts, "Close"]) if ts in df.index else float(indicators[sym][1].loc[:ts].iloc[-1])
                equity += pos.qty * px
        equity_vals.append((ts, equity))

        for sym in symbols:
            df = dfs[sym]
            if ts not in df.index: continue
            rsi_s, close = indicators[sym]
            if ts not in rsi_s.index or pd.isna(rsi_s.loc[ts]): continue

            px = float(df.loc[ts, "Close"])
            pos = positions[sym]

            # Risk Checks
            if pos.qty > 0 and _apply_stop_tp(entry_price[sym], px, stop_loss, take_profit):
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0
                continue

            r = float(rsi_s.loc[ts])
            
            # Buy Signal: RSI <= Oversold
            if pos.qty == 0 and r <= oversold:
                qty = int(per_symbol_budget // px)
                if qty > 0:
                    cash -= qty * px
                    pos.qty = qty
                    entry_price[sym] = px
                    trades.append(_mk_trade(bot, sym, OrderSide.BUY, qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
            
            # Sell Signal: RSI >= Overbought
            elif pos.qty > 0 and r >= overbought:
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0

    equity_curve = pd.Series([v for _, v in equity_vals], index=[t for t, _ in equity_vals])
    return trades, equity_curve


# =============================================================================
# Strategy 4: MACD Trend
# =============================================================================

def backtest_macd_trend(
    bot: BotConfig,
    data_by_symbol: Dict[str, pd.DataFrame],
) -> Tuple[List[Trade], pd.Series]:
    """
    LOGIC:
    1. Compute MACD Line (Fast EMA - Slow EMA).
    2. Compute Signal Line (EMA of MACD).
    3. Buy when MACD crosses ABOVE Signal.
    4. Sell when MACD crosses BELOW Signal.
    """
    params = bot.strategy.params or {}
    fast_p = int(params.get("fast_period", 12))
    slow_p = int(params.get("slow_period", 26))
    sig_p = int(params.get("signal_period", 9))
    stop_loss = float(params.get("stop_loss_pct", 0.0) or 0.0)
    take_profit = float(params.get("take_profit_pct", 0.0) or 0.0)

    if slow_p <= fast_p:
        slow_p = fast_p + 1

    dfs = {s: _ensure_sorted(df) for s, df in data_by_symbol.items()}
    index = None
    for df in dfs.values():
        index = df.index if index is None else index.union(df.index)
    index = index.sort_values()

    positions: Dict[str, Position] = {s: Position() for s in dfs.keys()}
    entry_price: Dict[str, float] = {s: 0.0 for s in dfs.keys()}
    cash = float(bot.capital)
    trades: List[Trade] = []
    equity_vals = []

    symbols = list(dfs.keys())
    per_symbol_budget = cash / max(1, len(symbols))

    macd_data = {}
    for sym, df in dfs.items():
        close = _bar_close(df)
        ema_fast = close.ewm(span=fast_p, adjust=False).mean()
        ema_slow = close.ewm(span=slow_p, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=sig_p, adjust=False).mean()
        macd_data[sym] = (macd, signal, close)

    prev_rel = {sym: None for sym in symbols}

    for ts in index:
        equity = cash
        for sym, pos in positions.items():
            if pos.qty > 0:
                df = dfs[sym]
                px = float(df.loc[ts, "Close"]) if ts in df.index else float(macd_data[sym][2].loc[:ts].iloc[-1])
                equity += pos.qty * px
        equity_vals.append((ts, equity))

        for sym in symbols:
            df = dfs[sym]
            if ts not in df.index: continue

            macd, signal, close = macd_data[sym]
            if ts not in macd.index or ts not in signal.index: continue
            if pd.isna(macd.loc[ts]) or pd.isna(signal.loc[ts]): continue

            px = float(df.loc[ts, "Close"])
            pos = positions[sym]

            # Risk Checks
            if pos.qty > 0 and _apply_stop_tp(entry_price[sym], px, stop_loss, take_profit):
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0
                prev_rel[sym] = None
                continue

            # Calculate logic for crossover
            rel = float(macd.loc[ts]) - float(signal.loc[ts])
            prev = prev_rel[sym]
            prev_rel[sym] = rel

            if prev is None: continue

            crossed_up = (prev <= 0) and (rel > 0)
            crossed_down = (prev >= 0) and (rel < 0)

            # Buy Signal
            if pos.qty == 0 and crossed_up:
                qty = int(per_symbol_budget // px)
                if qty > 0:
                    cash -= qty * px
                    pos.qty = qty
                    entry_price[sym] = px
                    trades.append(_mk_trade(bot, sym, OrderSide.BUY, qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
            
            # Sell Signal
            elif pos.qty > 0 and crossed_down:
                cash += pos.qty * px
                trades.append(_mk_trade(bot, sym, OrderSide.SELL, pos.qty, px, ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts))
                pos.qty = 0
                entry_price[sym] = 0.0

    equity_curve = pd.Series([v for _, v in equity_vals], index=[t for t, _ in equity_vals])
    return trades, equity_curve