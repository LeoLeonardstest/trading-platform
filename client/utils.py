"""client/utils.py

Small UI helper functions.

These helpers keep client/app.py clean.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def parse_symbols(text: str) -> List[str]:
    """Convert "AAPL, MSFT, TSLA" -> ["AAPL","MSFT","TSLA"]"""
    items = [s.strip().upper() for s in text.split(",")]
    return [s for s in items if s]


def parse_date(text: str, fmt: str = "%Y-%m-%d") -> datetime:
    s = (text or "").strip()
    if not s:
        raise ValueError("Date is required (YYYY-MM-DD)")

    # If the user typed extra stuff, take the first YYYY-MM-DD part
    if len(s) >= 10:
        head = s[:10]
        try:
            return datetime.strptime(head, fmt)
        except ValueError:
            pass

    # Fallback: try dateutil (installed with pandas)
    try:
        from dateutil import parser as du_parser
        dt = du_parser.parse(s)
        return datetime(dt.year, dt.month, dt.day)
    except Exception:
        raise ValueError(f"Invalid date: {s!r} (expected YYYY-MM-DD)")


def parse_bool(text: str) -> bool:
    t = text.strip().lower()
    if t in ("true", "1", "yes", "y", "on"):
        return True
    if t in ("false", "0", "no", "n", "off"):
        return False
    # default: python truthiness is not safe; just raise
    raise ValueError("Expected true/false")


def collect_params(raw: Dict[str, str]) -> Dict[str, object]:
    """Convert string inputs to basic Python types when possible.

    This is a convenience; you said you don't want validation.
    Still, converting numbers/booleans helps your backtest code.
    """
    out: Dict[str, object] = {}
    for k, v in raw.items():
        s = (v or "").strip()
        if s == "":
            continue

        # bool
        if s.lower() in ("true", "false", "yes", "no", "on", "off", "1", "0"):
            try:
                out[k] = parse_bool(s)
                continue
            except Exception:
                pass

        # int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                out[k] = int(s)
                continue
        except Exception:
            pass

        # float
        try:
            if any(ch in s for ch in (".", "e", "E")):
                out[k] = float(s)
                continue
        except Exception:
            pass

        # default: string
        out[k] = s

    return out
