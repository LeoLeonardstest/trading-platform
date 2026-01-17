# Trading Platform 

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

