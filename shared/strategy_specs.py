"""shared/strategy_specs.py

FILE OVERVIEW:
This file defines the **Metadata** for all strategies.
It tells the UI:
1. What strategies are available.
2. What parameters (inputs) they require.
3. Default values and validation rules for those parameters.

It is a "Single Source of Truth". The UI reads this file to build the input forms dynamically.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


# -----------------------------
# Common options
# -----------------------------

TIMEFRAMES: List[str] = [
    "1Min",
    "5Min",
    "15Min",
    "30Min",
    "1H",
    "1D",
    "2D",
    "3D",
    "4D",
    "5D",
    "1W",
    "1M",
]


# -----------------------------
# TypedDicts (Used for type hinting)
# -----------------------------

ParamType = Literal["int", "float", "str", "bool"]


class ParamSpec(TypedDict, total=False):
    """Defines a single input field (e.g. 'Stop Loss')."""
    type: ParamType      # Data type: int, float, str, bool
    description: str     # Help text for the user
    required: bool

    default: object      # Default value
    example: object

    min: float           # Minimum allowed value
    max: float           # Maximum allowed value

    allowed: List[object] # Dropdown options (if applicable)


class StrategySpec(TypedDict):
    """Defines the full configuration for a Strategy."""
    id: str
    name: str            # Display name in UI
    description: str

    # "universe" = strategy picks from a list. "symbols" = strategy trades the list directly.
    symbols_role: Literal["universe", "symbols"]

    # Symbol count limits
    min_symbols: int
    max_symbols: Optional[int]

    params: Dict[str, ParamSpec] # Dictionary of all parameters


# -----------------------------
# Strategy Specs
# -----------------------------

STRATEGY_SPECS: Dict[str, StrategySpec] = {
    # 1. Mean Reversion
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
                "description": "Decision timeframe for calculations (daily is typical).",
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

    # 2. Moving Average
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

    # 3. RSI
    "rsi_reversion": {
        "id": "rsi_reversion",
        "name": "RSI Mean Reversion",
        "description": (
            "Mean-reversion strategy using RSI extremes. "
            "Buy when RSI <= oversold; sell/exit when RSI >= overbought or risk rules trigger."
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

    # 4. MACD
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
                "description": "Signal line EMA period.",
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
    """Returns a list of all strategy IDs (keys)."""
    return sorted(STRATEGY_SPECS.keys())


def get_spec(strategy_id: str) -> StrategySpec:
    """Returns the configuration specification for a single strategy."""
    if strategy_id not in STRATEGY_SPECS:
        raise KeyError(f"Unknown strategy_id: {strategy_id}")
    return STRATEGY_SPECS[strategy_id]