"""backend/discord.py

FILE OVERVIEW:
A helper to send messages to a Discord Webhook.
The bots use this to say "I bought AAPL!" or "I crashed!".
"""

from __future__ import annotations

from typing import Optional
import requests


def send_discord(webhook_url: Optional[str], message: str) -> None:
    """
    Sends a message to the provided Discord Webhook URL.
    Fails silently (best-effort) so it doesn't crash the bot if Discord is down.
    """
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception:
        # best-effort only
        return