"""
shared/strategy_specs.py

Minimal shared strategy metadata.

Purpose:
- Single source of truth for:
  - Strategy IDs and names
  - Strategy descriptions (UI text)
  - Allowed parameters, ranges, defaults, examples
  - Symbol list rules (e.g., Mean Reversion universe 20–50)

Design:
- plain dictionaries
- no Alpaca/Yahoo code
- no database code
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


# Canonical timeframes exposed to UI and stored in BotConfig.strategy.params["timeframe"].
# Each runtime maps this value:
# - Live (Alpaca): maps to "1Min/5Min/.../1Hour/1Day"
# - Backtest (Yahoo): maps to "1m/5m/.../1h/1d"
TIMEFRAMES: List[str] = [
    "1Min",
    "5Min",
    "15Min",
    "30Min",
    "1H",
    "1D",
]


ParamType = Literal["int", "float", "str", "bool"]


class ParamSpec(TypedDict, total=False):
    type: ParamType
    description: str
    required: bool

    default: object
    example: object

    min: float
    max: float

    allowed: List[object]


class StrategySpec(TypedDict):
    id: str
    name: str
    description: str

    # How to interpret BotConfig.symbols for this strategy
    # - "universe": symbols is the universe from which the strategy selects
    # - "symbols": symbols are the traded symbols
    symbols_role: Literal["universe", "symbols"]

    # symbol list constraints
    min_symbols: int
    max_symbols: Optional[int]

    params: Dict[str, ParamSpec]


STRATEGY_SPECS: Dict[str, StrategySpec] = {
    "mean_reversion_losers": {
        "id": "mean_reversion_losers",
        "name": "Daily Mean Reversion — Top Losers",
        "description": (
            "At market open, rank the user-selected universe by previous-day return, "
            "buy the worst performers (market orders), then sell all positions at market close."
        ),
        "symbols_role": "universe",
        "min_symbols": 20,
        "max_symbols": 50,
        "params": {
            "losers_to_buy": {
                "type": "int",
                "description": "How many of the worst-performing stocks to buy each day.",
                "required": True,
                "min": 1,
                "max": 15,
                "default": 5,
                "example": 5,
            },
            "timeframe": {
                "type": "str",
                "description": "Decision timeframe (daily is typical for this strategy).",
                "required": True,
                "allowed": ["1D"],
                "default": "1D",
                "example": "1D",
            },
            "enable_stop_loss": {
                "type": "bool",
                "description": "Optional intraday stop-loss protection.",
                "required": False,
                "default": False,
                "example": False,
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Stop-loss percentage (e.g., 0.03 = 3%). Used only if enable_stop_loss is true.",
                "required": False,
                "min": 0.001,
                "max": 0.20,
                "default": 0.03,
                "example": 0.03,
            },
        },
    },

    "moving_average": {
        "id": "moving_average",
        "name": "Moving Average Trend (SMA / EMA)",
        "description": (
            "Trend-following strategy based on moving average crossovers. "
            "Enter when short MA crosses above long MA; exit on opposite signal (plus optional risk rules)."
        ),
        "symbols_role": "symbols",
        "min_symbols": 1,
        "max_symbols": None,
        "params": {
            "ma_type": {
                "type": "str",
                "description": "Moving average type.",
                "required": True,
                "allowed": ["SMA", "EMA"],
                "default": "SMA",
                "example": "EMA",
            },
            "short_period": {
                "type": "int",
                "description": "Short moving average period.",
                "required": True,
                "min": 2,
                "max": 300,
                "default": 20,
                "example": 20,
            },
            "long_period": {
                "type": "int",
                "description": "Long moving average period (must be > short_period).",
                "required": True,
                "min": 3,
                "max": 500,
                "default": 50,
                "example": 50,
            },
            "timeframe": {
                "type": "str",
                "description": "Bar timeframe used for signals.",
                "required": True,
                "allowed": TIMEFRAMES,
                "default": "1D",
                "example": "15Min",
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Optional stop-loss percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 0.50,
                "default": 0.0,
                "example": 0.03,
            },
            "take_profit_pct": {
                "type": "float",
                "description": "Optional take-profit percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
                "example": 0.06,
            },
        },
    },

    "rsi_reversion": {
        "id": "rsi_reversion",
        "name": "RSI Mean Reversion",
        "description": (
            "Mean-reversion strategy using RSI extremes. "
            "Buy when RSI <= oversold; exit when RSI >= overbought or risk rules trigger."
        ),
        "symbols_role": "symbols",
        "min_symbols": 1,
        "max_symbols": None,
        "params": {
            "rsi_period": {
                "type": "int",
                "description": "RSI lookback period.",
                "required": True,
                "min": 2,
                "max": 100,
                "default": 14,
                "example": 14,
            },
            "oversold": {
                "type": "int",
                "description": "RSI threshold to enter long positions.",
                "required": True,
                "min": 1,
                "max": 50,
                "default": 30,
                "example": 30,
            },
            "overbought": {
                "type": "int",
                "description": "RSI threshold to exit / take profit.",
                "required": True,
                "min": 50,
                "max": 99,
                "default": 70,
                "example": 70,
            },
            "timeframe": {
                "type": "str",
                "description": "Bar timeframe used for RSI calculation.",
                "required": True,
                "allowed": TIMEFRAMES,
                "default": "1D",
                "example": "1H",
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Optional stop-loss percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 0.50,
                "default": 0.0,
                "example": 0.03,
            },
            "take_profit_pct": {
                "type": "float",
                "description": "Optional take-profit percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
                "example": 0.06,
            },
        },
    },

    "macd_trend": {
        "id": "macd_trend",
        "name": "MACD Trend / Reversal",
        "description": (
            "Momentum strategy using MACD and signal line crossovers. "
            "Enter on MACD cross above signal; exit on cross below (plus optional risk rules)."
        ),
        "symbols_role": "symbols",
        "min_symbols": 1,
        "max_symbols": None,
        "params": {
            "fast_period": {
                "type": "int",
                "description": "Fast EMA period for MACD.",
                "required": True,
                "min": 2,
                "max": 100,
                "default": 12,
                "example": 12,
            },
            "slow_period": {
                "type": "int",
                "description": "Slow EMA period for MACD (must be > fast_period).",
                "required": True,
                "min": 3,
                "max": 200,
                "default": 26,
                "example": 26,
            },
            "signal_period": {
                "type": "int",
                "description": "Signal EMA period.",
                "required": True,
                "min": 2,
                "max": 100,
                "default": 9,
                "example": 9,
            },
            "timeframe": {
                "type": "str",
                "description": "Bar timeframe used for MACD calculation.",
                "required": True,
                "allowed": TIMEFRAMES,
                "default": "1D",
                "example": "30Min",
            },
            "stop_loss_pct": {
                "type": "float",
                "description": "Optional stop-loss percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 0.50,
                "default": 0.0,
                "example": 0.03,
            },
            "take_profit_pct": {
                "type": "float",
                "description": "Optional take-profit percentage (0 disables).",
                "required": False,
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
                "example": 0.06,
            },
        },
    },
}


def list_strategy_ids() -> List[str]:
    return sorted(STRATEGY_SPECS.keys())


def get_spec(strategy_id: str) -> StrategySpec:
    if strategy_id not in STRATEGY_SPECS:
        raise KeyError(f"Unknown strategy_id: {strategy_id}")
    return STRATEGY_SPECS[strategy_id]
