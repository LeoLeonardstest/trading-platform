# backend/main.py
from __future__ import annotations

import json
import secrets
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend import db
from backend.bot_runner import BotRunner
from backend.alpaca import make_alpaca_client

# init DB + runner
db.init_db()
runner = BotRunner()

app = FastAPI(title="Trading Platform Backend", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later (domain/desktop app only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Auth helpers
# -------------------------

def _hash_password(password: str, salt: str) -> str:
    # MVP hash. Upgrade later to bcrypt/argon2.
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def _make_password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    return f"{salt}${_hash_password(password, salt)}"

def _verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, h = password_hash.split("$", 1)
    except Exception:
        return False
    return _hash_password(password, salt) == h


def get_current_user_id(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    user_id = db.get_user_id_by_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user_id


# -------------------------
# Request/Response models
# -------------------------

class RegisterReq(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=200)

class LoginReq(BaseModel):
    username: str
    password: str

class AuthResp(BaseModel):
    token: str
    user_id: str
    username: str

class SettingsReq(BaseModel):
    alpaca_key_id: Optional[str] = None
    alpaca_secret_key: Optional[str] = None
    alpaca_base_url: Optional[str] = None
    discord_webhook: Optional[str] = None

class BotStartReq(BaseModel):
    name: str = Field(default="My Bot")
    strategy_id: str
    symbols: list[str]
    params: dict = Field(default_factory=dict)
    capital: float = Field(gt=0)

class StopBotReq(BaseModel):
    bot_id: str

class BotResp(BaseModel):
    bot_id: str
    name: str
    strategy_id: str
    symbols: list[str]
    params: dict
    capital: float
    status: str
    created_at: str
    updated_at: str


# -------------------------
# Auth endpoints
# -------------------------

@app.post("/auth/register", response_model=AuthResp)
def register(req: RegisterReq):
    existing = db.get_user_by_username(req.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user_id = str(uuid4())
    pw_hash = _make_password_hash(req.password)
    db.create_user(user_id, req.username, pw_hash)

    token = secrets.token_urlsafe(32)
    db.upsert_token(token, user_id)

    return AuthResp(token=token, user_id=user_id, username=req.username)


@app.post("/auth/login", response_model=AuthResp)
def login(req: LoginReq):
    user = db.get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    db.upsert_token(token, user["user_id"])

    return AuthResp(token=token, user_id=user["user_id"], username=user["username"])


# -------------------------
# Settings endpoints
# -------------------------

@app.get("/settings")
def get_settings(user_id: str = Depends(get_current_user_id)):
    return db.get_user_settings(user_id)


@app.post("/settings")
def set_settings(req: SettingsReq, user_id: str = Depends(get_current_user_id)):
    db.upsert_user_settings(
        user_id=user_id,
        alpaca_key_id=req.alpaca_key_id,
        alpaca_secret_key=req.alpaca_secret_key,
        alpaca_base_url=req.alpaca_base_url,
        discord_webhook=req.discord_webhook,
    )
    return {"ok": True}

# Backwards-compatible alias for the desktop client
@app.post("/settings/update")
def set_settings_alias(req: SettingsReq, user_id: str = Depends(get_current_user_id)):
    return set_settings(req, user_id)


# -------------------------
# Bots endpoints
# -------------------------

@app.get("/bots", response_model=list[BotResp])
def list_bots(user_id: str = Depends(get_current_user_id)):
    rows = db.list_bots_for_user(user_id)
    out = []
    for r in rows:
        out.append(
            BotResp(
                bot_id=r["bot_id"],
                name=r["name"],
                strategy_id=r["strategy_id"],
                symbols=json.loads(r["symbols_json"]),
                params=json.loads(r["params_json"]),
                capital=float(r["capital"]),
                status=r["status"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
        )
    return out


@app.post("/bots/start")
def start_bot(req: BotStartReq, user_id: str = Depends(get_current_user_id)):
    bot_id = str(uuid4())
    db.create_bot(
        bot_id=bot_id,
        user_id=user_id,
        name=req.name,
        strategy_id=req.strategy_id,
        symbols_json=json.dumps(req.symbols),
        params_json=json.dumps(req.params),
        capital=float(req.capital),
        status="starting",
    )

    bot_row = db.get_bot(bot_id)
    if not bot_row:
        raise HTTPException(status_code=500, detail="Bot create failed")

    runner.start_bot(bot_row)
    return {"ok": True, "bot_id": bot_id}


@app.post("/bots/stop")
def stop_bot(req: StopBotReq, user_id: str = Depends(get_current_user_id)):
    bot = db.get_bot(req.bot_id)
    if not bot or bot["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Bot not found")

    runner.stop_bot(req.bot_id)
    return {"ok": True}


@app.get("/bots/status")
def bot_status(bot_id: str, user_id: str = Depends(get_current_user_id)):
    bot = db.get_bot(bot_id)
    if not bot or bot["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Bot not found")

    return {
        "bot_id": bot_id,
        "status": bot["status"],
        "is_running_in_runner": runner.is_running(bot_id),
        "updated_at": bot["updated_at"],
    }


# -------------------------
# History endpoints
# -------------------------

@app.get("/history")
def history(bot_id: Optional[str] = None, user_id: str = Depends(get_current_user_id)):
    return db.list_trades_for_user(user_id, bot_id=bot_id)


# -------------------------
# Dashboard endpoint
# -------------------------

@app.get("/dashboard")
def dashboard(user_id: str = Depends(get_current_user_id)):
    """
    MVP dashboard:
    - returns Alpaca account fields (paper) if credentials exist
    - returns bot list summary
    """
    settings = db.get_user_settings(user_id)
    alpaca_key = settings.get("alpaca_key_id")
    alpaca_secret = settings.get("alpaca_secret_key")
    alpaca_base = settings.get("alpaca_base_url")

    account = None
    if alpaca_key and alpaca_secret:
        try:
            alpaca = make_alpaca_client(alpaca_key, alpaca_secret, alpaca_base)
            acct = alpaca.get_account()
            account = {
                "cash": float(getattr(acct, "cash", 0.0)),
                "buying_power": float(getattr(acct, "buying_power", 0.0)),
                "portfolio_value": float(getattr(acct, "portfolio_value", 0.0)),
                "currency": str(getattr(acct, "currency", "USD")),
            }
        except Exception as e:
            account = {"error": str(e)}

    bots = db.list_bots_for_user(user_id)
    running = sum(1 for b in bots if (b.get("status") or "").lower() in ("running", "sleeping", "starting", "stopping"))
    return {
        "account": account,
        "bots_total": len(bots),
        "bots_running": running,
    }
