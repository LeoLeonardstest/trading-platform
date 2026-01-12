# backend/db.py
from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.getenv("DB_PATH", "backend.sqlite3")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    cur = conn.cursor()

    # Users + auth token (simple token auth for MVP)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    # Per-user settings (alpaca keys + discord webhook)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            alpaca_key_id TEXT,
            alpaca_secret_key TEXT,
            alpaca_base_url TEXT,
            discord_webhook TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    # Bot configs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bots (
            bot_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            symbols_json TEXT NOT NULL,
            params_json TEXT NOT NULL,
            capital REAL NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    # Trades / order submissions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            bot_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            executed_at TEXT NOT NULL,
            meta_json TEXT,
            FOREIGN KEY(bot_id) REFERENCES bots(bot_id),
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
        """
    )

    conn.commit()
    conn.close()


# -------------------------
# Users / Auth
# -------------------------

def create_user(user_id: str, username: str, password_hash: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO users (user_id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (user_id, username, password_hash, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_token(token: str, user_id: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT OR REPLACE INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_id_by_token(token: str) -> Optional[str]:
    conn = _connect()
    row = conn.execute("SELECT user_id FROM tokens WHERE token = ?", (token,)).fetchone()
    conn.close()
    return str(row["user_id"]) if row else None


# -------------------------
# Settings
# -------------------------

def upsert_user_settings(
    user_id: str,
    alpaca_key_id: Optional[str],
    alpaca_secret_key: Optional[str],
    alpaca_base_url: Optional[str],
    discord_webhook: Optional[str],
) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO user_settings (user_id, alpaca_key_id, alpaca_secret_key, alpaca_base_url, discord_webhook, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          alpaca_key_id=excluded.alpaca_key_id,
          alpaca_secret_key=excluded.alpaca_secret_key,
          alpaca_base_url=excluded.alpaca_base_url,
          discord_webhook=excluded.discord_webhook,
          updated_at=excluded.updated_at
        """,
        (
            user_id,
            alpaca_key_id,
            alpaca_secret_key,
            alpaca_base_url,
            discord_webhook,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def get_user_settings(user_id: str) -> Dict[str, Any]:
    conn = _connect()
    row = conn.execute("SELECT * FROM user_settings WHERE user_id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {
        "user_id": user_id,
        "alpaca_key_id": None,
        "alpaca_secret_key": None,
        "alpaca_base_url": None,
        "discord_webhook": None,
        "updated_at": datetime.utcnow().isoformat(),
    }


# -------------------------
# Bots
# -------------------------

def create_bot(
    bot_id: str,
    user_id: str,
    name: str,
    strategy_id: str,
    symbols_json: str,
    params_json: str,
    capital: float,
    status: str,
) -> None:
    conn = _connect()
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO bots (bot_id, user_id, name, strategy_id, symbols_json, params_json, capital, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (bot_id, user_id, name, strategy_id, symbols_json, params_json, capital, status, now, now),
    )
    conn.commit()
    conn.close()


def update_bot_status(bot_id: str, status: str) -> None:
    conn = _connect()
    conn.execute(
        "UPDATE bots SET status=?, updated_at=? WHERE bot_id=?",
        (status, datetime.utcnow().isoformat(), bot_id),
    )
    conn.commit()
    conn.close()


def list_bots_for_user(user_id: str) -> List[Dict[str, Any]]:
    conn = _connect()
    rows = conn.execute("SELECT * FROM bots WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_bot(bot_id: str) -> Optional[Dict[str, Any]]:
    conn = _connect()
    row = conn.execute("SELECT * FROM bots WHERE bot_id=?", (bot_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# -------------------------
# Trades / History
# -------------------------

def insert_trade(
    trade_id: str,
    bot_id: str,
    user_id: str,
    symbol: str,
    side: str,
    quantity: int,
    price: Optional[float],
    executed_at: str,
    meta_json: Optional[str] = None,
) -> None:
    conn = _connect()
    conn.execute(
        """
        INSERT INTO trades (trade_id, bot_id, user_id, symbol, side, quantity, price, executed_at, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trade_id, bot_id, user_id, symbol, side, quantity, price, executed_at, meta_json),
    )
    conn.commit()
    conn.close()


def list_trades_for_user(user_id: str, bot_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _connect()
    if bot_id:
        rows = conn.execute(
            "SELECT * FROM trades WHERE user_id=? AND bot_id=? ORDER BY executed_at DESC",
            (user_id, bot_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM trades WHERE user_id=? ORDER BY executed_at DESC",
            (user_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
