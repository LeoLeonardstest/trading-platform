"""client/api.py

Very small HTTP client for the EC2 backend API.

Goal (university-project simple):
- Provide 1 place for requests to backend :8000
- UI can call these functions without knowing requests details

Assumptions:
- Backend exposes JSON endpoints on http://<host>:8000
- Auth can be simplified (token returned from /auth/login)

You can expand/modify endpoints later to match your backend.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import requests


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None

    # -----------------------------
    # Helpers
    # -----------------------------
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    @staticmethod
    def _to_json(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return obj

    def _post(self, path: str, payload: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=self._to_json(payload), headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params or {}, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    # -----------------------------
    # Auth
    # -----------------------------
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Expected response: {"token": "...", "user_id": "..."}"""
        data = self._post("/auth/login", {"username": username, "password": password})
        if "token" in data:
            self.token = data["token"]
        return data

    def register(self, username: str, password: str) -> Dict[str, Any]:
        return self._post("/auth/register", {"username": username, "password": password})

    # -----------------------------
    # Settings
    # -----------------------------
    def update_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Payload may include alpaca keys, discord webhook, username/password changes."""
        return self._post("/settings", payload)

    def get_settings(self) -> Dict[str, Any]:
        return self._get("/settings")

    # -----------------------------
    # Bots
    # -----------------------------
    def start_bot(self, bot_config: Any) -> Dict[str, Any]:
        """bot_config should match shared BotConfig shape."""
        return self._post("/bots/start", bot_config)

    def stop_bot(self, bot_id: str) -> Dict[str, Any]:
        return self._post("/bots/stop", {"bot_id": bot_id})

    def list_bots(self) -> Dict[str, Any]:
        return self._get("/bots")

    # -----------------------------
    # History / Status
    # -----------------------------
    def get_history(self) -> Dict[str, Any]:
        return self._get("/history")

    def get_dashboard(self) -> Dict[str, Any]:
        """Expected: alpaca balance, running bots count, total pnl, etc."""
        return self._get("/dashboard")
