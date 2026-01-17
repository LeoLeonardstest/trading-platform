"""backend/bot_runner.py

FILE OVERVIEW:
This file is the "Engine Room". It manages the lifecycle of trading bots.
Since we want to run multiple bots at once, each bot gets its own Thread.

Key Components:
1. `BotRunner` class: Starts, Stops, and Monitors bot threads.
2. `_run_bot_loop`: The infinite loop that every bot runs. It wakes up, checks the market, runs logic, and sleeps.
3. `AlpacaProxy`: A wrapper around the Alpaca API to intercept orders and log them to our database.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend import db
from backend.alpaca import make_alpaca_client
from backend.discord import send_discord
from backend.history import record_order_submission
from backend.strategies import build_live_strategy, make_tick_context, DailyMeanReversionTopLosers
from shared.models import BotConfig, StrategyConfig


def _utcnow() -> datetime:
    """Helper for current UTC time."""
    return datetime.now(timezone.utc)


def _bar_seconds(timeframe: str) -> int:
    """
    Decides how long to sleep based on the strategy timeframe.
    If running a 1-Minute strategy, sleep 60 seconds.
    If running a Daily strategy, sleep 24 hours (roughly).
    """
    tf = (timeframe or "").strip()
    tf_lower = tf.lower()

    if tf_lower in ("1min", "1m"):
        return 60
    if tf_lower in ("5min", "5m"):
        return 5 * 60
    if tf_lower in ("15min", "15m"):
        return 15 * 60
    if tf_lower in ("30min", "30m"):
        return 30 * 60
    if tf_lower in ("1hour", "60m", "1h"):
        return 60 * 60
    if tf_lower in ("1day", "1d", "day"):
        return 24 * 60 * 60

    return 5 * 60 # Default 5 min


class AlpacaProxy:
    """
    A "Middleman" for the Alpaca API.
    When a strategy calls `alpaca.submit_order()`, it goes through THIS class first.
    This allows us to:
    1. Log the order to our `trades` database table.
    2. Send a Discord notification.
    3. Then, pass the order to the real Alpaca API.
    """
    def __init__(self, alpaca: Any, *, user_id: str, bot_id: str, webhook: Optional[str]):
        self._alpaca = alpaca
        self._user_id = user_id
        self._bot_id = bot_id
        self._webhook = webhook

    def __getattr__(self, item: str) -> Any:
        # Pass any other method calls directly to the real Alpaca client
        return getattr(self._alpaca, item)

    def submit_order(self, *args, **kwargs):
        # Extract arguments to log them
        symbol = kwargs.get("symbol") if "symbol" in kwargs else (args[0] if len(args) > 0 else None)
        qty = kwargs.get("qty") if "qty" in kwargs else (args[1] if len(args) > 1 else None)
        side = kwargs.get("side") if "side" in kwargs else (args[2] if len(args) > 2 else None)

        # 1. Log to Database
        try:
            if symbol and qty and side:
                record_order_submission(
                    user_id=self._user_id,
                    bot_id=self._bot_id,
                    symbol=str(symbol),
                    side=str(side).upper(),
                    quantity=int(qty),
                    price=None, # Market orders have no fixed price at submission
                    meta={"source": "submit_order", "kwargs": {k: str(v) for k, v in kwargs.items()}},
                )
        except Exception:
            pass # Don't crash if logging fails

        # 2. Send Discord Alert
        try:
            if self._webhook and symbol and qty and side:
                send_discord(self._webhook, f"Bot {self._bot_id}: {str(side).upper()} {qty} {symbol} (market)")
        except Exception:
            pass

        # 3. Execute Real Order
        return self._alpaca.submit_order(*args, **kwargs)


class BotThread:
    """Holds the Thread object and a Stop Event signal for a specific bot."""
    def __init__(self, bot_id: str, stop_event: threading.Event, thread: threading.Thread):
        self.bot_id = bot_id
        self.stop_event = stop_event
        self.thread = thread


class BotRunner:
    """
    Manager Class.
    Methods:
    - start_bot(config): Spawns a new thread.
    - stop_bot(id): Signals a thread to stop.
    - is_running(id): Checks status.
    """

    def __init__(self):
        self._bots: Dict[str, BotThread] = {}
        self._lock = threading.Lock() # Prevents race conditions

    def is_running(self, bot_id: str) -> bool:
        """Checks if a bot thread exists and is alive."""
        with self._lock:
            bt = self._bots.get(bot_id)
            return bool(bt and bt.thread.is_alive())

    def start_bot(self, bot_row: Dict[str, Any]) -> None:
        """Starts a new bot in a separate thread."""
        bot_id = bot_row["bot_id"]
        if self.is_running(bot_id):
            return

        stop_event = threading.Event()
        # The target function is `_run_bot_loop`
        t = threading.Thread(target=self._run_bot_loop, args=(bot_row, stop_event), daemon=True)
        with self._lock:
            self._bots[bot_id] = BotThread(bot_id, stop_event, t)

        db.update_bot_status(bot_id, "starting")
        t.start()

    def stop_bot(self, bot_id: str) -> None:
        """Gracefully stops a bot."""
        with self._lock:
            bt = self._bots.get(bot_id)

        if not bt:
            db.update_bot_status(bot_id, "stopped")
            return

        # Signal the loop to stop
        bt.stop_event.set()
        db.update_bot_status(bot_id, "stopping")

        # Wait for thread to finish current task
        bt.thread.join(timeout=15)

        with self._lock:
            self._bots.pop(bot_id, None)

        db.update_bot_status(bot_id, "stopped")

    def _run_bot_loop(self, bot_row: Dict[str, Any], stop_event: threading.Event) -> None:
        """
        THE MAIN INFINITE LOOP.
        This runs inside the thread for as long as the bot is active.
        """
        bot_id = bot_row["bot_id"]
        user_id = bot_row["user_id"]

        # 1. Fetch Credentials
        settings = db.get_user_settings(user_id)
        alpaca_key = settings.get("alpaca_key_id")
        alpaca_secret = settings.get("alpaca_secret_key")
        alpaca_base = settings.get("alpaca_base_url")
        webhook = settings.get("discord_webhook")

        if not alpaca_key or not alpaca_secret:
            db.update_bot_status(bot_id, "error_missing_alpaca_keys")
            send_discord(webhook, f"Bot {bot_id} error: missing Alpaca keys.")
            return

        # 2. Setup Client
        alpaca_raw = make_alpaca_client(alpaca_key, alpaca_secret, alpaca_base)
        alpaca = AlpacaProxy(alpaca_raw, user_id=user_id, bot_id=bot_id, webhook=webhook)

        # 3. Configure Strategy
        strategy_id = bot_row["strategy_id"]
        symbols = json.loads(bot_row["symbols_json"])
        params = json.loads(bot_row["params_json"])

        bot_cfg = BotConfig(
            bot_id=bot_id,
            user_id=user_id,
            name=str(bot_row.get("name") or ""),
            capital=float(bot_row["capital"]),
            symbols=list(symbols),
            strategy=StrategyConfig(strategy_id=strategy_id, params=dict(params)),
        )

        strategy = build_live_strategy(bot_cfg, alpaca)

        send_discord(webhook, f"Bot started: {bot_cfg.name or bot_cfg.bot_id} strategy={strategy_id}")
        db.update_bot_status(bot_id, "running")

        # 4. Loop until Stop Signal
        while not stop_event.is_set():
            try:
                # Get current market status (Open/Closed)
                ctx = make_tick_context(alpaca)

                # Check Market Open
                if not ctx.is_market_open:
                    db.update_bot_status(bot_id, "sleeping")
                    if ctx.next_open_utc and ctx.now_utc:
                        sleep_s = max(30.0, (ctx.next_open_utc - ctx.now_utc).total_seconds())
                        # cap sleep so stop requests are responsive
                        sleep_s = min(sleep_s, 15 * 60.0)
                    else:
                        sleep_s = 15 * 60.0
                    stop_event.wait(timeout=sleep_s)
                    continue

                db.update_bot_status(bot_id, "running")

                # Special Check: Close daily positions 10 mins before market close
                if isinstance(strategy, DailyMeanReversionTopLosers) and ctx.is_market_open and ctx.next_close_utc and ctx.now_utc:
                    mins_to_close = (ctx.next_close_utc - ctx.now_utc).total_seconds() / 60.0
                    if 0 <= mins_to_close <= 10:
                        strategy.close_all_at_day_end()

                # --- EXECUTE STRATEGY ---
                strategy.tick(ctx)

                # Sleep
                tf = str(bot_cfg.strategy.params.get("timeframe", "5Min"))
                sleep_s = float(_bar_seconds(tf))
                # Cap sleep at 15 mins so the bot status updates occasionally
                sleep_s = min(sleep_s, 15 * 60.0)

                stop_event.wait(timeout=sleep_s)

            except Exception as e:
                db.update_bot_status(bot_id, "error")
                send_discord(webhook, f"Bot {bot_id} error: {e}")
                stop_event.wait(timeout=10.0) # Short wait on error before retrying

        send_discord(webhook, f"Bot stopped: {bot_cfg.name or bot_cfg.bot_id}")