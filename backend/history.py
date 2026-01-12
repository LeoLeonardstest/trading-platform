# backend/history.py
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
    MVP approach:
    - record immediately when order is submitted (not guaranteed fill price)
    Later upgrade:
    - poll Alpaca orders/activities and record actual fills.
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
