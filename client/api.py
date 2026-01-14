"""client/api.py

FILE OVERVIEW:
This file is the **Wrapper** for the backend API.
Instead of the UI making messy HTTP requests directly (like `requests.post("http://..."...)`),
the UI calls neat functions like `api.start_bot(...)`.

This file handles:
1. Connecting to the Server URL.
2. Managing the Authentication Token.
3. Converting Python objects to JSON and vice-versa.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import requests


class ApiClient:
    """Handles communication with the backend server."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None # Stores the JWT token after login

    # -----------------------------
    # Helpers
    # -----------------------------
    def _headers(self) -> Dict[str, str]:
        """Automatically adds the Authentication Token to the headers."""
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    @staticmethod
    def _to_json(obj: Any) -> Any:
        """Helper to convert dataclasses to dictionaries for JSON serialization."""
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        return obj

    def _post(self, path: str, payload: Any) -> Dict[str, Any]:
        """Generic POST request handler."""
        url = f"{self.base_url}{path}"
        r = requests.post(url, json=self._to_json(payload), headers=self._headers(), timeout=30)
        r.raise_for_status() # Raise error if status is not 2xx
        return r.json()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generic GET request handler."""
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params or {}, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    # -----------------------------
    # Auth
    # -----------------------------
    def login(self, username: str, password: str) -> dict:
        """Logs the user in and saves the returned token."""
        r = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data.get("token")
        return data

    def register(self, username: str, password: str) -> dict:
        """Creates a new user account."""
        r = requests.post(
            f"{self.base_url}/auth/register",
            json={"username": username, "password": password},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        self.token = data.get("token")
        return data
    
    # -----------------------------
    # Settings
    # -----------------------------
    def update_settings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Updates API keys and webhook."""
        return self._post("/settings", payload)

    def get_settings(self) -> Dict[str, Any]:
        """Retrieves current API keys."""
        return self._get("/settings")

    # -----------------------------
    # Bots
    # -----------------------------
    def start_bot(self, bot_config: Any) -> Dict[str, Any]:
        """Starts a new bot instance."""
        return self._post("/bots/start", bot_config)

    def stop_bot(self, bot_id: str) -> Dict[str, Any]:
        """Stops a running bot."""
        return self._post("/bots/stop", {"bot_id": bot_id})

    def list_bots(self) -> Dict[str, Any]:
        """Lists all bots."""
        return self._get("/bots")

    def delete_bot(self, bot_id: str) -> Dict[str, Any]:
        """Deletes a bot permanently."""
        return self._post("/bots/delete", {"bot_id": bot_id})
    
    def restart_bot(self, bot_id: str) -> Dict[str, Any]:
        """Restarts a bot."""
        return self._post("/bots/restart", {"bot_id": bot_id})
    
    # -----------------------------
    # History / Status
    # -----------------------------
    def get_history(self) -> Dict[str, Any]:
        """Gets trade history."""
        return self._get("/history")

    def get_dashboard(self) -> Dict[str, Any]:
        """Gets dashboard summary data."""
        return self._get("/dashboard")