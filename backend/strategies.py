"""backend/strategies.py

FILE OVERVIEW:
This file contains the **Live Trading Logic**.
While the client side had "simulations", this code runs on the server and makes REAL trades (or paper trades) using the Alpaca API.

Key Components:
1. `LiveStrategy`: The base class that all strategies must inherit from.
2. `StrategyTickContext`: A snapshot of the current market time and status.
3. Specific Strategies:
   - DailyMeanReversionTopLosers (Buys losers at open)
   - MovingAverageTrend (Golden/Death Cross)
   - RSIMeanReversion (Buy low, sell high)
   - MACDTrend (Momentum)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from backend import db
from backend.discord import send_discord

import pandas as pd

from shared.models import BotConfig


# =============================================================================
# Timeframe normalization
# =============================================================================

# Valid Alpaca timeframes
ALPACA_TIMEFRAMES = {"1Min", "5Min", "15Min", "30Min", "1Hour", "1Day"}

# Helper map to convert "1m" -> "1Min"
_TIMEFRAME_ALIASES = {
    "1m": "1Min",
    "1min": "1Min",
    "5m": "5Min",
    "5min": "5Min",
    "15m": "15Min",
    "15min": "15Min",
    "30m": "30Min",
    "30min": "30Min",
    "1h": "1Hour",
    "1hour": "1Hour",
    "60m": "1Hour",
    "1d": "1Day",
    "1day": "1Day",
    "day": "1Day",
    # canonical shared short forms
    "1h_short": "1Hour",
    "1d_short": "1Day",
}

def normalize_timeframe(tf: str) -> str:
    """
    Converts user input (e.g. '5m') into the strict format Alpaca requires (e.g. '5Min').
    Defaults to '5Min' if the input is unknown.
    """
    raw = (tf or "").strip()
    if raw.upper() == "1H":
        return "1Hour"
    if raw.upper() == "1D":
        return "1Day"
    low = raw.lower()
    mapped = _TIMEFRAME_ALIASES.get(low, raw)
    if mapped not in ALPACA_TIMEFRAMES:
        # default safe
        return "5Min"
    return mapped


def _to_utc(ts: Optional[datetime]) -> datetime:
    """Helper to ensure a datetime object is set to UTC timezone."""
    if ts is None:
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _approx_bar_seconds(tf: str) -> int:
    """
    Returns how many seconds are in a specific timeframe.
    Used to calculate how long the bot should sleep between checks.
    """
    tf = normalize_timeframe(tf)
    return {
        "1Min": 60,
        "5Min": 300,
        "15Min": 900,
        "30Min": 1800,
        "1Hour": 3600,
        "1Day": 86400,
    }.get(tf, 300)


# =============================================================================
# Alpaca data helpers
# =============================================================================

def fetch_recent_bars(alpaca: Any, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetches historical candle data (Open, High, Low, Close) from Alpaca.
    Returns a Pandas DataFrame formatted for easy calculation.
    """
    tf = normalize_timeframe(timeframe)
    bars = alpaca.get_bars(symbol, tf, limit=limit)
    # alpaca-trade-api returns a Bars object with .df (multi-index)
    df = getattr(bars, "df", None)
    if df is None:
        # some versions return DataFrame directly
        df = pd.DataFrame(bars)

    if df is None or df.empty:
        return pd.DataFrame()

    # If multi-index (symbol, time) -> select symbol specifically
    if isinstance(df.index, pd.MultiIndex):
        if symbol in df.index.get_level_values(0):
            df = df.xs(symbol)
        else:
            # pick first symbol level if exact match fails
            df = df.xs(df.index.get_level_values(0)[0])

    df = df.copy()
    # normalize columns to lower-case names used by Alpaca DF
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"t": "time"})
    
    # Ensure datetime index is standard UTC
    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df["timestamp"], utc=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)

    df = df.sort_index()
    return df


def last_trade_price(alpaca: Any, symbol: str, fallback_timeframe: str = "1Min") -> Optional[float]:
    """
    Gets the absolute latest price for a symbol.
    1. Tries to get the last real-time trade tick.
    2. If that fails, gets the closing price of the last 1-minute bar.
    """
    try:
        lt = alpaca.get_last_trade(symbol)
        price = getattr(lt, "price", None) or getattr(lt, "p", None)
        if price is not None:
            return float(price)
    except Exception:
        pass

    # Fallback: fetch 1 bar of history
    try:
        df = fetch_recent_bars(alpaca, symbol, fallback_timeframe, 1)
        if not df.empty:
            return float(df["close"].iloc[-1])
    except Exception:
        pass

    return None


def _floor_qty(cash_to_spend: float, price: float) -> int:
    """Calculates max shares we can buy with given cash (rounding down)."""
    if cash_to_spend <= 0 or price <= 0:
        return 0
    return int(math.floor(cash_to_spend / price))


# =============================================================================
# Tick context + base strategy
# =============================================================================

@dataclass
class StrategyTickContext:
    """
    A snapshot of the market passed to the strategy every time it wakes up.
    Contains: current time, is market open?, when does it close?
    """
    now_utc: datetime
    clock: Any
    is_market_open: bool
    next_open_utc: Optional[datetime]
    next_close_utc: Optional[datetime]


def make_tick_context(alpaca: Any) -> StrategyTickContext:
    """Queries Alpaca API to build the current time/market context."""
    clock = alpaca.get_clock()
    is_open = bool(getattr(clock, "is_open", False))

    ts = getattr(clock, "timestamp", None) or getattr(clock, "ts", None)
    now_utc = _to_utc(ts if isinstance(ts, datetime) else None)

    next_open = getattr(clock, "next_open", None)
    next_close = getattr(clock, "next_close", None)

    next_open_utc = _to_utc(next_open) if isinstance(next_open, datetime) else None
    next_close_utc = _to_utc(next_close) if isinstance(next_close, datetime) else None

    return StrategyTickContext(
        now_utc=now_utc,
        clock=clock,
        is_market_open=is_open,
        next_open_utc=next_open_utc,
        next_close_utc=next_close_utc,
    )


class LiveStrategy:
    """
    THE PARENT CLASS.
    All trading strategies must inherit from this.
    It sets up the common links to the Bot Config and Alpaca Connection.
    """
    strategy_id: str

    def __init__(self, bot: BotConfig, alpaca: Any):
        self.bot = bot
        self.alpaca = alpaca
        self.params = bot.strategy.params or {}
        self.symbols = list(bot.symbols or [])
        
        # Initialize Webhook URL from settings DB
        self.webhook_url = None
        try:
            settings = db.get_user_settings(bot.user_id)
            self.webhook_url = settings.get("discord_webhook")
        except Exception:
            pass

    def tick(self, ctx: StrategyTickContext) -> None:
        """
        The heartbeat of the strategy.
        The BotRunner calls this method repeatedly (e.g., every 5 minutes).
        Child classes MUST overwrite this with their own logic.
        """
        raise NotImplementedError


# =============================================================================
# Strategy 1: Daily Mean Reversion — Top Losers
# =============================================================================

class DailyMeanReversionTopLosers(LiveStrategy):
    """
    STRATEGY: Buy stocks that crashed yesterday.
    Hypothesis: They were oversold and will bounce back today.
    Action: Buy at Open, Sell at Close.
    """
    strategy_id = "mean_reversion_losers"

    def __init__(self, bot: BotConfig, alpaca: Any):
        super().__init__(bot, alpaca)
        self.losers_to_buy = int(self.params.get("losers_to_buy", 5))
        self.losers_to_buy = max(1, min(self.losers_to_buy, 15))
        self._last_trade_day: Optional[date] = None

    def _already_traded_today(self, now_utc: datetime) -> bool:
        """Prevents buying the same things twice in one day."""
        d = now_utc.date()
        return self._last_trade_day == d

    def _mark_traded_today(self, now_utc: datetime) -> None:
        self._last_trade_day = now_utc.date()

    def _buy_losers_at_open(self) -> None:
        """Logic to calculate losers and submit BUY orders."""
        rets: List[Tuple[str, float]] = []
        for sym in self.symbols:
            # Need 3 days of data to safely calculate yesterday's return
            df = fetch_recent_bars(self.alpaca, sym, "1Day", limit=3)
            if df is None or df.empty or len(df) < 2:
                continue
            
            # Calculate Return: (Yesterday Close / DayBeforeYesterday Close) - 1
            r = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-2])) - 1.0
            rets.append((sym, r))

        if not rets:
            return

        # Sort: Smallest return (biggest loss) first
        rets.sort(key=lambda x: x[1])  
        picked = [s for s, _ in rets[: min(self.losers_to_buy, len(rets))]]
        if not picked:
            return

        # Calculate Position Size (Total Capital / Number of Stocks)
        acct = self.alpaca.get_account()
        broker_cash = float(getattr(acct, "cash", 0.0))
        bot_budget = float(self.bot.capital)
        usable_cash = min(bot_budget, broker_cash)
        alloc = usable_cash / float(len(picked))

        if alloc < 1.0:
            print("DEBUG: Alloc too small to buy losers.")
            return

        for sym in picked:
            try:
                # Don't buy if we already have an open order
                existing = self.alpaca.list_orders(status="open", symbols=[sym])
                if existing: continue
            except: pass

            px = last_trade_price(self.alpaca, sym, fallback_timeframe="1Min")
            if px:
                print(f"BUYING LOSER: {sym} for ${alloc:.2f}")
                msg = f"🚀 BUY SIGNAL: Bot {self.bot.name} buying ${alloc:.2f} of {sym}"
                print(msg)
                if self.webhook_url: send_discord(self.webhook_url, msg)
                
                # Submit Market Order using 'notional' (dollar amount)
                self.alpaca.submit_order(
                    symbol=sym, 
                    notional=alloc, 
                    side="buy", 
                    type="market", 
                    time_in_force="day"
                )

    def close_all_at_day_end(self) -> None:
        """Panic Button: Sell EVERYTHING associated with this bot."""
        try:
            positions = self.alpaca.list_positions()
        except Exception:
            positions = []
        for p in positions:
            sym = getattr(p, "symbol", None)
            if sym and (not self.symbols or sym in self.symbols):
                qty = int(float(getattr(p, "qty", 0)))
                if qty > 0:
                    msg = f"📉 SELL SIGNAL: Bot {self.bot.name} selling {qty} {sym} (End of Day)"
                    print(msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")

    def tick(self, ctx: StrategyTickContext) -> None:
        """Main Loop: Runs periodically."""
        if not ctx.is_market_open:
            return

        # 1. Entry Logic (Run once per day)
        if not self._already_traded_today(ctx.now_utc):
            self._buy_losers_at_open()
            self._mark_traded_today(ctx.now_utc)

        # 2. Stop Loss Monitoring (Runs continuously)
        try:
            positions = self.alpaca.list_positions()
            for p in positions:
                sym = getattr(p, "symbol", "")
                if sym not in self.symbols: continue
                
                qty = float(getattr(p, "qty", 0))
                avg_entry = float(getattr(p, "avg_entry_price", 0.0))
                current_price = float(getattr(p, "current_price", 0.0))

                if qty > 0 and avg_entry > 0:
                    pct_change = (current_price - avg_entry) / avg_entry
                    stop_pct = float(self.params.get("stop_loss_pct", 0.03)) 
                    
                    # If lost more than X%, sell immediately
                    if stop_pct > 0 and pct_change <= -stop_pct:
                        print(f"STOP LOSS (Losers): {sym} down {pct_change:.2%}")
                        msg = f"🛑 STOP LOSS: Bot {self.bot.name} sold {sym} (Down {pct_change:.2%})"
                        print(msg)
                        if self.webhook_url: send_discord(self.webhook_url, msg)
                        self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
        except Exception as e:
            print(f"Error in TopLosers tick: {e}")


# =============================================================================
# Strategy 2: Moving Average Trend (SMA/EMA)
# =============================================================================

class MovingAverageTrend(LiveStrategy):
    """
    STRATEGY: Golden Cross / Death Cross.
    Buy when Short MA crosses above Long MA.
    Sell when Short MA crosses below Long MA.
    """
    strategy_id = "moving_average"

    def __init__(self, bot: BotConfig, alpaca: Any):
        super().__init__(bot, alpaca)
        self.ma_type = str(self.params.get("ma_type", "SMA")).upper()
        self.short_p = int(self.params.get("short_period", 20))
        self.long_p = int(self.params.get("long_period", 50))
        if self.long_p <= self.short_p:
            self.long_p = self.short_p + 1
        self.timeframe = normalize_timeframe(str(self.params.get("timeframe", "5Min")))
        self.stop_loss = float(self.params.get("stop_loss_pct", 0.0) or 0.0)
        self.take_profit = float(self.params.get("take_profit_pct", 0.0) or 0.0)
        self._last_bar_ts: Dict[str, datetime] = {}

    def _ma(self, s: pd.Series, period: int) -> pd.Series:
        """Calculates SMA or EMA."""
        if self.ma_type == "EMA":
            return s.ewm(span=period, adjust=False).mean()
        return s.rolling(window=period, min_periods=period).mean()

    def tick(self, ctx: StrategyTickContext) -> None:
        if not ctx.is_market_open:
           return

        for sym in self.symbols:
            # 1. Fetch Data (Enough to cover the long period)
            df = fetch_recent_bars(self.alpaca, sym, self.timeframe, limit=max(self.long_p + 5, 60))
            if df is None or df.empty or len(df) < self.long_p + 1:
                continue

            bar_ts = df.index[-1].to_pydatetime()
            self._last_bar_ts[sym] = bar_ts

            # 2. Compute Indicators
            close = df["close"].astype(float)
            s_ma = self._ma(close, self.short_p).iloc[-1]
            l_ma = self._ma(close, self.long_p).iloc[-1]
            if pd.isna(s_ma) or pd.isna(l_ma):
                continue
            price = float(close.iloc[-1])

            # 3. Check Account Status
            qty = 0
            avg_entry = 0.0
            try:
                existing = self.alpaca.list_orders(status="open", symbols=[sym])
                if existing: continue

                pos = self.alpaca.get_position(sym)
                qty = float(getattr(pos, "qty", 0))
                avg_entry = float(getattr(pos, "avg_entry_price", 0.0))
            except Exception:
                pass

            # 4. Risk Management (Stop Loss / Take Profit)
            if qty > 0 and avg_entry > 0:
                pct_change = (price - avg_entry) / avg_entry
                
                if self.stop_loss > 0 and pct_change <= -self.stop_loss:
                    print(f"STOP LOSS: {sym} {pct_change:.2%}")
                    msg = f"🛑 STOP LOSS: Bot {self.bot.name} sold {sym} (Down {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue
                
                if self.take_profit > 0 and pct_change >= self.take_profit:
                    print(f"TAKE PROFIT: {sym} {pct_change:.2%}")
                    msg = f"💰 TAKE PROFIT: Bot {self.bot.name} sold {sym} (Up {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue

            # 5. Trading Logic
            # BUY: Short MA > Long MA
            if qty == 0 and float(s_ma) > float(l_ma):
                acct = self.alpaca.get_account()
                broker_cash = float(getattr(acct, "cash", 0.0))
                bot_budget = float(self.bot.capital)
                usable_cash = min(bot_budget, broker_cash)
                alloc = usable_cash / max(1, len(self.symbols))
                
                if alloc >= 1.0:
                    msg = f"🚀 BUY SIGNAL: Bot {self.bot.name} buying ${alloc:.2f} of {sym}"
                    print(msg)
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, notional=alloc, side="buy", type="market", time_in_force="day")

            # SELL: Short MA < Long MA
            elif qty > 0 and float(s_ma) < float(l_ma):
                msg = f"📉 SELL SIGNAL: Bot {self.bot.name} selling {qty} {sym}"
                print(msg)
                if self.webhook_url: send_discord(self.webhook_url, msg)
                self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")


# =============================================================================
# Strategy 3: RSI Mean Reversion
# =============================================================================

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """Standard RSI calculation."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


class RSIMeanReversion(LiveStrategy):
    """
    STRATEGY: Buy Low (Oversold), Sell High (Overbought).
    """
    strategy_id = "rsi_reversion"

    def __init__(self, bot: BotConfig, alpaca: Any):
        super().__init__(bot, alpaca)
        self.period = int(self.params.get("rsi_period", 14))
        self.oversold = int(self.params.get("oversold", 30))
        self.overbought = int(self.params.get("overbought", 70))
        self.timeframe = normalize_timeframe(str(self.params.get("timeframe", "5Min")))
        self.stop_loss = float(self.params.get("stop_loss_pct", 0.0) or 0.0)
        self.take_profit = float(self.params.get("take_profit_pct", 0.0) or 0.0)
        self._last_bar_ts: Dict[str, datetime] = {}

    def tick(self, ctx: StrategyTickContext) -> None:
        
        if not ctx.is_market_open:
            return

        for sym in self.symbols:
            # 1. Fetch Data
            df = fetch_recent_bars(self.alpaca, sym, self.timeframe, limit=max(self.period + 10, 60))
            if df is None or df.empty or len(df) < self.period + 1:
                continue

            self._last_bar_ts[sym] = df.index[-1].to_pydatetime()

            # 2. Calc RSI
            close = df["close"].astype(float)
            price = float(close.iloc[-1])
            rsi = _rsi(close, self.period).iloc[-1]
            if pd.isna(rsi):
                continue

            # 3. Check Holdings
            qty = 0
            avg_entry = 0.0
            try:
                existing_orders = self.alpaca.list_orders(status="open", symbols=[sym])
                if existing_orders: continue
                
                pos = self.alpaca.get_position(sym)
                qty = float(getattr(pos, "qty", 0))
                avg_entry = float(getattr(pos, "avg_entry_price", 0.0))
            except Exception:
                pass

            # 4. Risk Checks
            if qty > 0 and avg_entry > 0:
                pct_change = (price - avg_entry) / avg_entry
                if self.stop_loss > 0 and pct_change <= -self.stop_loss:
                    msg = f"🛑 STOP LOSS: Bot {self.bot.name} sold {sym} (Down {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue

                if self.take_profit > 0 and pct_change >= self.take_profit:
                    msg = f"💰 TAKE PROFIT: Bot {self.bot.name} sold {sym} (Up {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue

            # 5. Logic
            # Buy if RSI <= Oversold (e.g. 30)
            if qty == 0 and float(rsi) <= self.oversold:
                acct = self.alpaca.get_account()
                broker_cash = float(getattr(acct, "cash", 0.0))
                bot_budget = float(self.bot.capital)
                usable_cash = min(bot_budget, broker_cash)
                alloc = usable_cash / max(1, len(self.symbols))
                
                if alloc >= 1.0:
                    msg = f"🚀 BUY SIGNAL: Bot {self.bot.name} buying ${alloc:.2f} of {sym} (RSI {float(rsi):.1f})"
                    print(msg)
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, notional=alloc, side="buy", type="market", time_in_force="day")

            # Sell if RSI >= Overbought (e.g. 70)
            elif qty > 0 and float(rsi) >= self.overbought:
                msg = f"📉 SELL SIGNAL: Bot {self.bot.name} selling {qty} {sym} (RSI {float(rsi):.1f})"
                print(msg)
                if self.webhook_url: send_discord(self.webhook_url, msg)
                self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")


# =============================================================================
# Strategy 4: MACD Trend
# =============================================================================

class MACDTrend(LiveStrategy):
    """
    STRATEGY: MACD Crossover.
    Buy when MACD Line > Signal Line (Upward momentum).
    Sell when MACD Line < Signal Line (Downward momentum).
    """
    strategy_id = "macd_trend"

    def __init__(self, bot: BotConfig, alpaca: Any):
        super().__init__(bot, alpaca)
        self.fast = int(self.params.get("fast_period", 12))
        self.slow = int(self.params.get("slow_period", 26))
        if self.slow <= self.fast:
            self.slow = self.fast + 1
        self.signal = int(self.params.get("signal_period", 9))
        self.timeframe = normalize_timeframe(str(self.params.get("timeframe", "5Min")))
        self.stop_loss = float(self.params.get("stop_loss_pct", 0.0) or 0.0)
        self.take_profit = float(self.params.get("take_profit_pct", 0.0) or 0.0)
        self._last_bar_ts: Dict[str, datetime] = {}
        self._prev_rel: Dict[str, Optional[float]] = {}

    def tick(self, ctx: StrategyTickContext) -> None:
        if not ctx.is_market_open:
            return

        for sym in self.symbols:
            # 1. Fetch Data
            df = fetch_recent_bars(self.alpaca, sym, self.timeframe, limit=max(self.slow + self.signal + 10, 100))
            if df is None or df.empty or len(df) < self.slow + self.signal + 5:
                continue

            self._last_bar_ts[sym] = df.index[-1].to_pydatetime()

            # 2. Calc MACD
            close = df["close"].astype(float)
            ema_fast = close.ewm(span=self.fast, adjust=False).mean()
            ema_slow = close.ewm(span=self.slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            sig = macd.ewm(span=self.signal, adjust=False).mean()
            
            # Rel = MACD - Signal
            # Rel > 0 means Bullish. Rel < 0 means Bearish.
            rel = float(macd.iloc[-1] - sig.iloc[-1])
            prev = self._prev_rel.get(sym)
            self._prev_rel[sym] = rel
            
            if prev is None: continue
            
            price = float(close.iloc[-1])

            # 3. Check Holdings
            qty = 0
            avg_entry = 0.0
            try:
                existing = self.alpaca.list_orders(status="open", symbols=[sym])
                if existing: continue
                pos = self.alpaca.get_position(sym)
                qty = float(getattr(pos, "qty", 0))
                avg_entry = float(getattr(pos, "avg_entry_price", 0.0))
            except Exception:
                pass

            # 4. Risk Checks
            if qty > 0 and avg_entry > 0:
                pct_change = (price - avg_entry) / avg_entry
                if self.stop_loss > 0 and pct_change <= -self.stop_loss:
                    msg = f"🛑 STOP LOSS: Bot {self.bot.name} sold {sym} (Down {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue
                if self.take_profit > 0 and pct_change >= self.take_profit:
                    msg = f"💰 TAKE PROFIT: Bot {self.bot.name} sold {sym} (Up {pct_change:.2%})"
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")
                    continue

            # 5. Logic
            # Crossover Up (Yesterday negative, today positive)
            crossed_up = (prev <= 0) and (rel > 0)
            # Crossover Down (Yesterday positive, today negative)
            crossed_down = (prev >= 0) and (rel < 0)

            if qty == 0 and crossed_up:
                acct = self.alpaca.get_account()
                broker_cash = float(getattr(acct, "cash", 0.0))
                bot_budget = float(self.bot.capital)
                usable_cash = min(bot_budget, broker_cash)
                alloc = usable_cash / max(1, len(self.symbols))
                
                if alloc >= 1.0:
                    msg = f"🚀 BUY SIGNAL: Bot {self.bot.name} buying ${alloc:.2f} of {sym}"
                    print(msg)
                    if self.webhook_url: send_discord(self.webhook_url, msg)
                    self.alpaca.submit_order(symbol=sym, notional=alloc, side="buy", type="market", time_in_force="day")

            elif qty > 0 and crossed_down:
                msg = f"📉 SELL SIGNAL: Bot {self.bot.name} selling {qty} {sym}"
                print(msg)
                if self.webhook_url: send_discord(self.webhook_url, msg)
                self.alpaca.submit_order(symbol=sym, qty=qty, side="sell", type="market", time_in_force="day")


# =============================================================================
# Factory
# =============================================================================

def build_live_strategy(bot: BotConfig, alpaca: Any) -> LiveStrategy:
    """
    Factory Function:
    Takes a Bot Configuration and returns the correct Strategy Class instance.
    """
    sid = bot.strategy.strategy_id
    if sid == "mean_reversion_losers":
        return DailyMeanReversionTopLosers(bot, alpaca)
    if sid == "moving_average":
        return MovingAverageTrend(bot, alpaca)
    if sid == "rsi_reversion":
        return RSIMeanReversion(bot, alpaca)
    if sid == "macd_trend":
        return MACDTrend(bot, alpaca)
    raise KeyError(f"Unknown strategy_id: {sid}")