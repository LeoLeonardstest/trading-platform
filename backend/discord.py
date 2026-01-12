# backend/discord.py
from __future__ import annotations

from typing import Optional
import requests


def send_discord(webhook_url: Optional[str], message: str) -> None:
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception:
        # best-effort only
        return
