"""
shared/models.py

FILE OVERVIEW:
This file defines the Data Structures (Models) used throughout the system.
It defines what a "User", "Bot", or "Trade" looks like.
Using these classes ensures the Frontend and Backend speak the same language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime


# ==================================================
# Enums (Fixed Options)
# ==================================================

class BotStatus(Enum):
    """Possible states for a bot."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class OrderSide(Enum):
    """Buy or Sell."""
    BUY = "BUY"
    SELL = "SELL"


# ==================================================
# User
# ==================================================

@dataclass
class User:
    """
    Represents a platform user.
    Stores login info and API keys.
    """
    user_id: str
    username: str
    password_hash: str

    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_base_url: Optional[str] = None

    discord_webhook: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)


# ==================================================
# Strategy Configuration
# ==================================================

ParamValue = Union[float, int, str, bool]

@dataclass
class StrategyConfig:
    """
    Specific settings for a strategy logic.
    Example: { strategy_id: "rsi", params: { "rsi_period": 14 } }
    """
    strategy_id: str
    params: Dict[str, ParamValue] = field(default_factory=dict)


# ==================================================
# Bot Configuration
# ==================================================

@dataclass
class BotConfig:
    """
    Configuration for a Bot Instance.
    This includes its name, which strategy it uses, which symbols it trades,
    and how much money is assigned to it.
    """
    bot_id: str
    user_id: str

    # Optional friendly name (UI / list view)
    name: str = ""

    strategy: StrategyConfig = field(default_factory=lambda: StrategyConfig(strategy_id="", params={}))

    symbols: List[str] = field(default_factory=list)
    capital: float = 0.0

    created_at: datetime = field(default_factory=datetime.utcnow)


# ==================================================
# Trade Record
# ==================================================

@dataclass
class Trade:
    """
    Represents a single executed trade.
    Used for History and Backtesting results.
    """
    trade_id: str
    bot_id: str
    user_id: str

    symbol: str
    side: OrderSide

    quantity: int
    price: float

    executed_at: datetime = field(default_factory=datetime.utcnow)