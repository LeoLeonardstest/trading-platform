# Trading Platform (Python-only)

Client–server trading platform:

- **Desktop Client (Windows, Tkinter)**:
  - Login/Register
  - Local backtests (Yahoo Finance)
  - Start/stop live bots on the server
  - View server trade history
  - Save Alpaca keys + Discord webhook to server

- **Backend (AWS EC2, FastAPI)**:
  - Auth + user settings
  - Bot runner (threads; one bot per thread)
  - Alpaca paper trading execution (market orders)
  - Trade history logging (MVP: order submissions)

## Project structure

```
trading-platform/
  backend/
  client/
  shared/
  requirements.txt
```

## Run backend (locally or on EC2)

From repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Environment variable (optional):

- `DB_PATH` (default: `backend.sqlite3`)

## Run desktop client

From repository root (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

python -m client.app
```

In the login screen, set the backend base URL (default uses your EC2 IP):
`http://35.159.124.180:8000`

## Notes / MVP limitations

- Trade history is recorded at **order submission time** (market order price is unknown at submission).
- Bots:
  - Sleep outside US market hours (based on Alpaca clock)
  - Tick frequency is tied to strategy timeframe (capped to stay responsive)
- No HTTPS included. If you want “anyone anywhere”, add TLS (e.g., reverse proxy or AWS ALB).
