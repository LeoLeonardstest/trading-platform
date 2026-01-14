"""client/utils.py

FILE OVERVIEW:
This file contains small helper functions for the UI.
It cleans up user input (converting strings to proper data types).
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List


def parse_symbols(text: str) -> List[str]:
    """Convert a string like "AAPL, MSFT, TSLA" -> ["AAPL","MSFT","TSLA"]"""
    items = [s.strip().upper() for s in text.split(",")]
    return [s for s in items if s]


def parse_date(text: str, fmt: str = "%Y-%m-%d") -> datetime:
    """
    Parses a date string.
    If the user includes extra time information, it truncates to the date.
    """
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
    """Converts strings like 'true', 'yes', '1' into Python True/False."""
    t = text.strip().lower()
    if t in ("true", "1", "yes", "y", "on"):
        return True
    if t in ("false", "0", "no", "n", "off"):
        return False
    # default: python truthiness is not safe; just raise
    raise ValueError("Expected true/false")


def collect_params(raw: Dict[str, str]) -> Dict[str, object]:
    """
    Convert string inputs (from UI text boxes) to basic Python types (int, float, bool).
    This allows the strategy code to use the numbers directly for math.
    """
    out: Dict[str, object] = {}
    for k, v in raw.items():
        s = (v or "").strip()
        if s == "":
            continue

        # Try bool
        if s.lower() in ("true", "false", "yes", "no", "on", "off", "1", "0"):
            try:
                out[k] = parse_bool(s)
                continue
            except Exception:
                pass

        # Try int
        try:
            if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                out[k] = int(s)
                continue
        except Exception:
            pass

        # Try float
        try:
            if any(ch in s for ch in (".", "e", "E")):
                out[k] = float(s)
                continue
        except Exception:
            pass

        # Default: string
        out[k] = s

    return out