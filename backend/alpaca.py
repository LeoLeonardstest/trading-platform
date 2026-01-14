"""backend/alpaca.py

FILE OVERVIEW:
This module initializes the connection to the Alpaca Broker API.
It allows us to fetch data, check account balance, and place orders.
"""

from __future__ import annotations

from typing import Any, Optional

import os

try:
    import alpaca_trade_api as tradeapi
except Exception:  # pragma: no cover
    tradeapi = None


DEFAULT_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def _require_alpaca() -> None:
    """Ensures the library is installed."""
    if tradeapi is None:
        raise RuntimeError(
            "alpaca-trade-api is not installed. Add it to requirements.txt and install:\n"
            "  pip install alpaca-trade-api"
        )


def make_alpaca_client(
    key_id: str,
    secret_key: str,
    base_url: Optional[str] = None,
) -> Any:
    """
    Creates and returns an authenticated Alpaca REST client.
    Used by the BotRunner to execute trades.
    """
    _require_alpaca()
    return tradeapi.REST(
        key_id,
        secret_key,
        base_url=base_url or DEFAULT_BASE_URL,
        api_version="v2",
    )