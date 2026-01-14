"""backend/history.py

FILE OVERVIEW:
A helper module to record trade submissions to the database.
It abstracts the database insertion logic from the trading logic.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from backend import db


def record_order_submission(
    user_id: str,
    bot_id: str,
    symbol: str,
    side: str,
    quantity: int,
    price: Optional[float],
    meta: Optional[dict] = None,
) -> None:
    """
    Records that an order was SENT to Alpaca.
    Note: Ideally we would wait for a "Fill" event, but for this MVP 
    we record it as soon as the bot decides to buy/sell.
    """
    db.insert_trade(
        trade_id=str(uuid4()),
        bot_id=bot_id,
        user_id=user_id,
        symbol=symbol,
        side=side,
        quantity=int(quantity),
        price=float(price) if price is not None else None,
        executed_at=datetime.utcnow().isoformat(),
        meta_json=json.dumps(meta or {}),
    )