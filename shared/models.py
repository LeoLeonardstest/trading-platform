"""
shared/models.py

Core domain models for the trading platform.
These models define the LANGUAGE of the system.

Rules:
- No business logic
- No database code
- No API calls
- Plain data structures only

Everything else (DB, backend, bots, UI) depends on these models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
from datetime import datetime


# ==================================================
# Enums
# ==================================================

class BotStatus(Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SLEEPING = "SLEEPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


# ==================================================
# User
# ==================================================

@dataclass
class User:
    """
    Represents a platform user.
    Owns bots and provides broker credentials.
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
    Strategy-specific configuration.

    - strategy_id identifies which predefined strategy is used
    - params contains ONLY parameters relevant to that strategy
    """
    strategy_id: str
    params: Dict[str, ParamValue] = field(default_factory=dict)


# ==================================================
# Bot Configuration (fixed at creation)
# ==================================================

@dataclass
class BotConfig:
    """
    Immutable configuration of a bot.
    Does NOT change while the bot is running.
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
    Represents a single executed trade (MVP: order submission record).
    Used for history and performance tracking.
    """
    trade_id: str
    bot_id: str
    user_id: str

    symbol: str
    side: OrderSide

    quantity: int
    price: float

    executed_at: datetime = field(default_factory=datetime.utcnow)
