from __future__ import annotations
"""client/app.py

Finished desktop UI for the Trading Platform (University Project)
Python-only, Tkinter-based.

Scope: UI + wiring points (no real trading logic).

Key goal:
- The UI already has ALL inputs/outputs and clear "hooks" so other AI/code
  can implement the backend/backtests and simply plug results into the UI.

Backtest contract (as provided by you):
  run_backtest_from_yahoo(bot: BotConfig, start: datetime, end: datetime) -> BacktestResult
and BacktestResult contains:
  equity_curve (pd.Series), KPIs, trades (List[Trade]).

Notes:
- This file does NOT import your backtest engine or backend API on purpose.
  It exposes two hooks:
    - on_run_backtest(): call your local backtest and then call populate_backtest_result(result)
    - on_start_bot()/on_stop_bot(): call backend API endpoints

Dependencies:
- tkinter (built-in)
- matplotlib (optional, for equity curve chart)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from datetime import datetime, timezone
from client.api import ApiClient

# Optional: render equity curve using matplotlib
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False


APP_TITLE = "Trading Platform"
WINDOW_SIZE = "1280x780"
DATE_FMT = "%Y-%m-%d"  # UI date input format

class TradingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Trading Platform")
        self.geometry("1280x780")

        self.api = None
        self.user = None
        self.user_id = None

        self.container = tk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.show_login()

    def clear(self):
        for w in self.container.winfo_children():
            w.destroy()

    def show_login(self):
        self.clear()
        LoginPage(self.container, self).pack(fill=tk.BOTH, expand=True)

    def show_main(self):
        self.clear()
        MainLayout(self.container, self).pack(fill=tk.BOTH, expand=True)


class LoginPage(tk.Frame):
    def __init__(self, parent, app: TradingApp):
        super().__init__(parent)
        self.app = app

        tk.Label(self, text="Login", font=("Arial", 22)).pack(pady=20)

        tk.Label(self, text="Username").pack()
        self.username = tk.Entry(self, width=30)
        self.username.insert(0, "testuser1")
        self.username.pack(pady=5)

        tk.Label(self, text="Password").pack()
        self.password = tk.Entry(self, width=30, show="*")
        self.password.insert(0, "testpass123")
        self.password.pack(pady=5)

        btn_row = tk.Frame(self)
        btn_row.pack(pady=15)

        tk.Button(btn_row, text="Login", width=16, command=self.on_login).pack(side="left", padx=6)
        tk.Button(btn_row, text="Register", width=16, command=self.on_register).pack(side="left", padx=6)

        tk.Label(
            self,
            text="Uses EC2 backend for authentication and settings.",
            fg="#666",
        ).pack(pady=10)

    def on_login(self):
        try:
            base_url = "http://35.159.124.180:8000"
            self.app.api = ApiClient(base_url)

            username = self.username.get().strip()
            password = self.password.get().strip()

            data = self.app.api.login(username, password)

            self.app.user = data.get("username", username)
            self.app.user_id = data.get("user_id")

            self.app.show_main()
        except Exception as e:
            messagebox.showerror("Login Error", str(e))

    def on_register(self):
        try:
            base_url = "http://35.159.124.180:8000"
            self.app.api = ApiClient(base_url)

            username = self.username.get().strip()
            password = self.password.get().strip()

            data = self.app.api.register(username, password)

            # ApiClient.register already stores token internally; this is optional
            self.app.api.token = data.get("token")

            self.app.user = data.get("username", username)
            self.app.user_id = data.get("user_id")

            self.app.show_main()
        except Exception as e:
            if hasattr(e, "response") and e.response is not None:
                messagebox.showerror("Register Error", f"{e}\n\n{e.response.text}")
            else:
                messagebox.showerror("Register Error", str(e))

class MainLayout(tk.Frame):
    def __init__(self, parent, app: TradingApp):
        super().__init__(parent)
        self.app = app
        self._build_header()
        self._build_body()

    def _build_header(self):
        header = tk.Frame(self, bg="#2c3e50", height=50)
        header.pack(fill=tk.X)
        tk.Label(
            header, text=APP_TITLE, fg="white", bg="#2c3e50", font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT, padx=15)
        tk.Button(header, text="Logout", command=self.logout).pack(side=tk.RIGHT, padx=10)
        tk.Label(header, text=f"User: {self.app.user}", fg="white", bg="#2c3e50").pack(side=tk.RIGHT, padx=10)

    def logout(self):
        self.app.user = None
        self.app.api.token = None
        self.app.show_login()

    def _build_body(self):
        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        # Sidebar Navigation
        nav = tk.Frame(body, width=220, bg="#ecf0f1")
        nav.pack(side=tk.LEFT, fill=tk.Y)
        
        btn_config = {"width": 24, "pady": 6}
        tk.Button(nav, text="Home", command=self.show_home, **btn_config).pack()
        tk.Button(nav, text="Test (Backtest)", command=self.show_test, **btn_config).pack()
        tk.Button(nav, text="Live Bots", command=self.show_bots, **btn_config).pack()
        tk.Button(nav, text="History", command=self.show_history, **btn_config).pack()
        tk.Button(nav, text="Settings", command=self.show_settings, **btn_config).pack()

        # Content Area
        self.content = tk.Frame(body)
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.show_home()

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # -----------------------------
    # 1. REAL DASHBOARD
    # -----------------------------
    def show_home(self):
        self.clear_content()
        tk.Label(self.content, text="Dashboard", font=("Arial", 20)).pack(pady=14)

        # 1. Fetch Real Data
        try:
            dash_data = self.app.api.get_dashboard()  #
            # Dashboard returns: { "account": {...}, "bots_total": int, "bots_running": int }
            
            # Helper to safely get nested account data
            acct = dash_data.get("account") or {}
            if "error" in acct:
                cash = "Error"
                equity = acct["error"]
                bp = "Error"
            else:
                cash_val = acct.get("cash", 0.0)
                eq_val = acct.get("portfolio_value", 0.0)
                bp_val = acct.get("buying_power", 0.0)
                
                cash = f"${float(cash_val):,.2f}"
                equity = f"${float(eq_val):,.2f}"
                bp = f"${float(bp_val):,.2f}"

            running_count = str(dash_data.get("bots_running", 0))
            total_bots = str(dash_data.get("bots_total", 0))

        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to fetch dashboard: {e}")
            return

        # 2. Render Cards
        cards_frame = tk.Frame(self.content)
        cards_frame.pack(pady=10)

        metrics = [
            ("Cash", cash),
            ("Portfolio Value", equity),
            ("Buying Power", bp),
            ("Active / Total Bots", f"{running_count} / {total_bots}"),
        ]

        for title, value in metrics:
            card = tk.LabelFrame(cards_frame, text=title, width=240, height=90)
            card.pack(side=tk.LEFT, padx=10)
            card.pack_propagate(False)
            tk.Label(card, text=value, font=("Arial", 16, "bold"), fg="#27ae60").pack(expand=True)

        # 3. Real Running Bots List
        tk.Label(self.content, text="Active Bot Instances", font=("Arial", 12, "bold")).pack(pady=(20, 5))
        
        # We need a Treeview to show the bots cleanly
        columns = ("name", "strategy", "status", "pnl")
        tree = ttk.Treeview(self.content, columns=columns, show="headings", height=8)
        tree.heading("name", text="Bot Name")
        tree.heading("strategy", text="Strategy")
        tree.heading("status", text="Status")
        tree.heading("pnl", text="Capital") # Using capital as proxy for now, PnL requires calculation
        
        tree.column("name", width=150)
        tree.column("strategy", width=150)
        tree.column("status", width=100)
        tree.column("pnl", width=100)
        tree.pack(fill=tk.X, padx=20)

        # Fetch full bot list to populate this table
        try:
            all_bots = self.app.api.list_bots() #
            # Filter for running/starting/stopping
            active_statuses = ["running", "starting", "stopping", "sleeping"]
            
            for b in all_bots:
                status = str(b.get("status", "")).lower()
                if status in active_statuses:
                    tree.insert("", "end", values=(
                        b.get("name", "Unnamed"),
                        b.get("strategy_id", ""),
                        b.get("status", "").upper(),
                        f"${b.get('capital', 0):,.2f}"
                    ))
        except Exception:
            pass # Fail silently for the list if dashboard loaded
    # -----------------------------
    # Test (Backtesting)
    # -----------------------------
    def show_test(self):
        self.clear_content()
        tk.Label(self.content, text="Backtesting (Local)", font=("Arial", 20)).pack(pady=12)

        tk.Label(
            self.content,
            text=(
                "UI builds a BotConfig and calls: run_backtest_from_yahoo(bot, start, end).\n"
                "Outputs are displayed below (equity curve chart, KPIs, trade list)."
            ),
            fg="#555",
            justify="left",
        ).pack(pady=(0, 10))

        input_panel = tk.LabelFrame(self.content, text="Backtest Inputs")
        input_panel.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(input_panel, text="Strategy").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.bt_strategy = ttk.Combobox(
            input_panel,
            width=38,
            values=[
                "mean_reversion_losers",
                "moving_average",
                "rsi_reversion",
                "macd_trend",
            ],
        )
        self.bt_strategy.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.bt_strategy.set("moving_average")

        self.bt_strategy_desc = tk.Label(input_panel, text="", fg="#555", wraplength=760)
        self.bt_strategy_desc.grid(row=0, column=2, sticky="w", padx=6)

        tk.Label(input_panel, text="Symbols (comma-separated)").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        self.bt_symbols = tk.Entry(input_panel, width=42)
        self.bt_symbols.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        self.bt_symbols.insert(0, "AAPL, MSFT, TSLA")
        tk.Label(input_panel, text="Example: AAPL, MSFT, TSLA", fg="#666").grid(
            row=1, column=2, sticky="w", padx=6
        )

        tk.Label(input_panel, text="Initial Capital ($)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.bt_capital = tk.Entry(input_panel, width=20)
        self.bt_capital.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self.bt_capital.insert(0, "10000")
        tk.Label(input_panel, text="Example: 10000", fg="#666").grid(row=2, column=2, sticky="w", padx=6)

        tk.Label(input_panel, text="Start (YYYY-MM-DD)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.bt_start = tk.Entry(input_panel, width=20)
        self.bt_start.grid(row=3, column=1, sticky="w", padx=6, pady=4)
        self.bt_start.insert(0, "2022-01-01")

        tk.Label(input_panel, text="End (YYYY-MM-DD)").grid(row=3, column=2, sticky="w", padx=6, pady=4)
        self.bt_end = tk.Entry(input_panel, width=20)
        self.bt_end.grid(row=3, column=3, sticky="w", padx=6, pady=4)
        self.bt_end.insert(0, "2023-01-01")

        self.bt_params_frame = tk.LabelFrame(self.content, text="Strategy Parameters")
        self.bt_params_frame.pack(fill=tk.X, padx=12, pady=8)

        btn_row = tk.Frame(self.content)
        btn_row.pack(fill=tk.X, padx=12)
        tk.Button(btn_row, text="Run Backtest", width=18, command=self.on_run_backtest).pack(side=tk.LEFT)
        tk.Label(
            btn_row,
            text="Hook: call run_backtest_from_yahoo(...) then populate_backtest_result(result)",
            fg="#666",
        ).pack(side=tk.LEFT, padx=10)

        results = tk.Frame(self.content)
        results.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        chart_box = tk.LabelFrame(results, text="Equity Curve")
        chart_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.figure = None
        self.ax = None
        self.canvas = None

        if MATPLOTLIB_OK:
            self.figure = Figure(figsize=(5.6, 3.8), dpi=100)
            self.ax = self.figure.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.figure, master=chart_box)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(
                chart_box,
                text="Chart placeholder (install matplotlib to render real equity curve)",
                fg="#666",
            ).pack(expand=True)

        right = tk.Frame(results)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        kpis = tk.LabelFrame(right, text="KPIs")
        kpis.pack(fill=tk.X, pady=(0, 8))

        self.kpi_final_equity = tk.StringVar(value="Final equity: —")
        self.kpi_return = tk.StringVar(value="Total return: —")
        self.kpi_dd = tk.StringVar(value="Max drawdown: —")
        self.kpi_trades = tk.StringVar(value="Total trades: —")

        tk.Label(kpis, textvariable=self.kpi_final_equity).pack(anchor="w", padx=8, pady=2)
        tk.Label(kpis, textvariable=self.kpi_return).pack(anchor="w", padx=8, pady=2)
        tk.Label(kpis, textvariable=self.kpi_dd).pack(anchor="w", padx=8, pady=2)
        tk.Label(kpis, textvariable=self.kpi_trades).pack(anchor="w", padx=8, pady=2)

        trades_box = tk.LabelFrame(right, text="Trades")
        trades_box.pack(fill=tk.BOTH, expand=True)

        columns = ("executed_at", "symbol", "side", "qty", "price")
        self.trades_table = ttk.Treeview(trades_box, columns=columns, show="headings", height=14)

        self.trades_table.heading("executed_at", text="Executed At")
        self.trades_table.heading("symbol", text="Symbol")
        self.trades_table.heading("side", text="Side")
        self.trades_table.heading("qty", text="Qty")
        self.trades_table.heading("price", text="Price")

        self.trades_table.column("executed_at", width=150)
        self.trades_table.column("symbol", width=80)
        self.trades_table.column("side", width=70)
        self.trades_table.column("qty", width=60)
        self.trades_table.column("price", width=90)

        scrollbar = ttk.Scrollbar(trades_box, orient="vertical", command=self.trades_table.yview)
        self.trades_table.configure(yscrollcommand=scrollbar.set)
        self.trades_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.bt_strategy.bind("<<ComboboxSelected>>", lambda e: self._render_backtest_params())
        self._render_backtest_params()

    def _render_backtest_params(self):
        for w in self.bt_params_frame.winfo_children():
            w.destroy()

        sid = self.bt_strategy.get().strip()
        self.bt_param_entries = {}

        desc_map = {
            "mean_reversion_losers": "Daily: buy worst N losers at open and sell at close (optional stop-loss).",
            "moving_average": "Trend: buy/sell on SMA/EMA crossover (optional stop-loss/take-profit).",
            "rsi_reversion": "Mean reversion: buy when RSI <= oversold, sell when RSI >= overbought.",
            "macd_trend": "Momentum: buy/sell on MACD vs signal crossover (optional risk rules).",
        }
        self.bt_strategy_desc.config(text=desc_map.get(sid, ""))

        if sid == "mean_reversion_losers":
            specs = [
                ("losers_to_buy", "Losers to Buy", "5"),
                ("enable_stop_loss", "Enable Stop Loss (true/false)", "false"),
                ("stop_loss_pct", "Stop Loss % (e.g. 0.03)", "0.03"),
                ("timeframe", "Timeframe", "1D"),
            ]
        elif sid == "moving_average":
            specs = [
                ("ma_type", "MA Type (SMA/EMA)", "SMA"),
                ("short_period", "Short Period", "20"),
                ("long_period", "Long Period", "50"),
                ("stop_loss_pct", "Stop Loss % (0 disables)", "0.0"),
                ("take_profit_pct", "Take Profit % (0 disables)", "0.0"),
                ("timeframe", "Timeframe", "1D"),
            ]
        elif sid == "rsi_reversion":
            specs = [
                ("rsi_period", "RSI Period", "14"),
                ("oversold", "Oversold", "30"),
                ("overbought", "Overbought", "70"),
                ("stop_loss_pct", "Stop Loss % (0 disables)", "0.0"),
                ("take_profit_pct", "Take Profit % (0 disables)", "0.0"),
                ("timeframe", "Timeframe", "1D"),
            ]
        else:
            specs = [
                ("fast_period", "Fast Period", "12"),
                ("slow_period", "Slow Period", "26"),
                ("signal_period", "Signal Period", "9"),
                ("stop_loss_pct", "Stop Loss % (0 disables)", "0.0"),
                ("take_profit_pct", "Take Profit % (0 disables)", "0.0"),
                ("timeframe", "Timeframe", "1D"),
            ]

        for r, (key, label, example) in enumerate(specs):
            tk.Label(self.bt_params_frame, text=label).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            ent = tk.Entry(self.bt_params_frame, width=26)
            ent.grid(row=r, column=1, sticky="w", padx=8, pady=4)
            ent.insert(0, example)
            tk.Label(self.bt_params_frame, text=f"Example: {example}", fg="#666").grid(
                row=r, column=2, sticky="w", padx=8
            )
            self.bt_param_entries[key] = ent

    def on_run_backtest(self):
        try:
            from shared.models import BotConfig, StrategyConfig
            from client.utils import parse_symbols, parse_date, collect_params
            from client.backtest import run_backtest_from_yahoo

            sid = self.bt_strategy.get().strip()
            symbols = parse_symbols(self.bt_symbols.get())
            cap_text = self.bt_capital.get().strip()
            try:
                capital = float(cap_text)
            except ValueError:
                raise ValueError(f"Initial Capital must be a number, got: {cap_text!r}")
            start = parse_date(self.bt_start.get().strip(), "%Y-%m-%d")
            end = parse_date(self.bt_end.get().strip(), "%Y-%m-%d")

            raw_params = {k: ent.get() for k, ent in self.bt_param_entries.items()}
            params = collect_params(raw_params)

            bot = BotConfig(
                bot_id="local_backtest",
                user_id=str(getattr(self.app, "user", "local_user")),
                symbols=symbols,
                capital=capital,
                strategy=StrategyConfig(strategy_id=sid, params=params),
            )

            result = run_backtest_from_yahoo(bot, start=start, end=end)
            self.populate_backtest_result(result)

        except Exception as e:
            messagebox.showerror("Backtest Error", str(e))

    def populate_backtest_result(self, result):
        # KPIs
        self.kpi_final_equity.set(f"Final equity: {getattr(result, 'final_equity', '—')}")
        self.kpi_return.set(f"Total return: {getattr(result, 'total_return_pct', '—')}%")
        self.kpi_dd.set(f"Max drawdown: {getattr(result, 'max_drawdown_pct', '—')}%")
        self.kpi_trades.set(f"Total trades: {getattr(result, 'total_trades', '—')}")

        # Trades table
        for row in self.trades_table.get_children():
            self.trades_table.delete(row)

        trades = getattr(result, "trades", []) or []
        for t in trades:
            executed_at = getattr(t, "executed_at", None)
            executed_at_str = executed_at.strftime("%Y-%m-%d %H:%M") if isinstance(executed_at, datetime) else str(executed_at or "")
            side = getattr(t, "side", "")
            side_str = getattr(side, "value", str(side))

            self.trades_table.insert(
                "",
                "end",
                values=(
                    executed_at_str,
                    getattr(t, "symbol", ""),
                    side_str,
                    getattr(t, "quantity", ""),
                    getattr(t, "price", ""),
                ),
            )

        # Equity curve chart
        if MATPLOTLIB_OK and self.ax and self.canvas:
            self.ax.clear()
            eq = getattr(result, "equity_curve", None)
            try:
                xs = list(eq.index)
                ys = list(eq.values)
                self.ax.plot(xs, ys)
                self.ax.set_title("Equity Curve")
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Equity")
                self.figure.autofmt_xdate()
                self.canvas.draw()
            except Exception:
                self.ax.text(0.5, 0.5, "Equity curve format not supported", ha="center")
                self.canvas.draw()

    # -----------------------------
    # 3. REAL BOTS PAGE (Management)
    # -----------------------------
    def show_bots(self):
        self.clear_content()
        tk.Label(self.content, text="Live Bots Management", font=("Arial", 20)).pack(pady=12)

        # -- Top: Create Bot Form --
        create_frame = tk.LabelFrame(self.content, text="Launch New Bot")
        create_frame.pack(fill=tk.X, padx=12, pady=5)

        # Row 1: Main Config
        r1 = tk.Frame(create_frame)
        r1.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(r1, text="Strategy:").pack(side=tk.LEFT)
        self.bot_strategy = ttk.Combobox(r1, values=["mean_reversion_losers", "moving_average", "rsi_reversion", "macd_trend"], width=25)
        self.bot_strategy.pack(side=tk.LEFT, padx=5)
        self.bot_strategy.set("moving_average")

        tk.Label(r1, text="Symbols:").pack(side=tk.LEFT, padx=(10, 0))
        self.bot_symbols = tk.Entry(r1, width=20)
        self.bot_symbols.insert(0, "AAPL,MSFT")
        self.bot_symbols.pack(side=tk.LEFT, padx=5)

        tk.Label(r1, text="Capital ($):").pack(side=tk.LEFT, padx=(10, 0))
        self.bot_capital = tk.Entry(r1, width=10)
        self.bot_capital.insert(0, "2000")
        self.bot_capital.pack(side=tk.LEFT, padx=5)

        tk.Button(r1, text="START BOT", bg="#2ecc71", fg="white", font=("Arial", 10, "bold"), command=self.on_start_bot).pack(side=tk.LEFT, padx=20)

        # Row 2: Dynamic Variables (The "Wariables" section)
        self.bot_params_frame = tk.LabelFrame(create_frame, text="Strategy Variables (Params)")
        self.bot_params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.bot_param_entries = {}
        
        # Bind change event
        self.bot_strategy.bind("<<ComboboxSelected>>", lambda e: self._render_bot_params())
        self._render_bot_params() # Initial render

        # -- Bottom: Real Bot List --
        list_frame = tk.LabelFrame(self.content, text="Your Bots")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Toolbar with Normal Buttons
        toolbar = tk.Frame(list_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 1. Refresh
        tk.Button(toolbar, text="Refresh List", command=self.refresh_bot_list).pack(side=tk.LEFT)
        
        # 2. Restart (Green)
        tk.Button(toolbar, text="Restart", bg="#2ecc71", fg="white", width=10, command=self.on_restart_bot_btn).pack(side=tk.LEFT, padx=10)

        # 3. Stop (Orange)
        tk.Button(toolbar, text="Stop", bg="#f39c12", fg="white", width=10, command=self.on_stop_only_btn).pack(side=tk.LEFT, padx=0)
        
        # 4. Delete (Red)
        tk.Button(toolbar, text="Delete", bg="#c0392b", fg="white", width=10, command=self.on_delete_bot_btn).pack(side=tk.RIGHT, padx=10)

        # Table
        cols = ("bot_id", "name", "strategy", "symbols", "capital", "status")
        self.bots_tree = ttk.Treeview(list_frame, columns=cols, show="headings")
        self.bots_tree.heading("bot_id", text="ID")
        self.bots_tree.heading("name", text="Name")
        self.bots_tree.heading("strategy", text="Strategy")
        self.bots_tree.heading("symbols", text="Symbols")
        self.bots_tree.heading("capital", text="Capital")
        self.bots_tree.heading("status", text="Status")
        
        self.bots_tree.column("bot_id", width=60)
        self.bots_tree.column("name", width=120)
        self.bots_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.refresh_bot_list()

    def _render_bot_params(self):
        # Clear existing
        for w in self.bot_params_frame.winfo_children():
            w.destroy()
        
        self.bot_param_entries = {}
        sid = self.bot_strategy.get().strip()

        # Define fields based on strategy
        specs = []
        if sid == "mean_reversion_losers":
            specs = [
                ("losers_to_buy", "Losers Count", "5"),
                ("timeframe", "Timeframe", "1D"),
                ("stop_loss_pct", "Stop Loss %", "0.03")
            ]
        elif sid == "moving_average":
            specs = [
                ("ma_type", "Type (SMA/EMA)", "SMA"),
                ("short_period", "Short Period", "20"),
                ("long_period", "Long Period", "50"),
                ("timeframe", "Timeframe", "5Min")
            ]
        elif sid == "rsi_reversion":
            specs = [
                ("rsi_period", "RSI Period", "14"),
                ("oversold", "Oversold", "30"),
                ("overbought", "Overbought", "70"),
                ("timeframe", "Timeframe", "15Min")
            ]
        elif sid == "macd_trend":
            specs = [
                ("fast_period", "Fast", "12"),
                ("slow_period", "Slow", "26"),
                ("signal_period", "Signal", "9"),
                ("timeframe", "Timeframe", "30Min")
            ]
        
        # Draw fields
        for i, (key, label, default) in enumerate(specs):
            f = tk.Frame(self.bot_params_frame)
            f.pack(side=tk.LEFT, padx=10, pady=5)
            tk.Label(f, text=label).pack(anchor="w")
            ent = tk.Entry(f, width=10)
            ent.insert(0, default)
            ent.pack()
            self.bot_param_entries[key] = ent

    # --- ACTION HANDLERS ---

    def on_start_bot(self):
        try:
            sid = self.bot_strategy.get()
            syms = [s.strip().upper() for s in self.bot_symbols.get().split(",") if s.strip()]
            cap = float(self.bot_capital.get())
            
            # Collect dynamic params
            params = {k: v.get() for k, v in self.bot_param_entries.items()}
            
            payload = {
                "name": f"{sid}-{len(syms)}s",
                "strategy_id": sid,
                "symbols": syms,
                "params": params,
                "capital": cap
            }
            self.app.api.start_bot(payload)
            messagebox.showinfo("Success", "Bot started successfully!")
            self.refresh_bot_list()
        except Exception as e:
            messagebox.showerror("Start Error", str(e))

    def on_restart_bot_btn(self):
        sel = self.bots_tree.selection()
        if not sel:
            messagebox.showwarning("Select Bot", "Please select a bot to restart.")
            return
        bot_id = sel[0]
        try:
            self.app.api.restart_bot(bot_id)
            messagebox.showinfo("Success", "Bot restarted.")
            self.refresh_bot_list()
        except Exception as e:
            messagebox.showerror("Restart Error", str(e))

    def on_stop_only_btn(self):
        sel = self.bots_tree.selection()
        if not sel:
            messagebox.showwarning("Select Bot", "Please select a bot to stop.")
            return
        bot_id = sel[0]
        try:
            self.app.api.stop_bot(bot_id)
            messagebox.showinfo("Success", "Bot stopped (Paused).")
            self.refresh_bot_list()
        except Exception as e:
            messagebox.showerror("Stop Error", str(e))

    def on_delete_bot_btn(self):
        sel = self.bots_tree.selection()
        if not sel:
            messagebox.showwarning("Select Bot", "Please select a bot to delete.")
            return
        
        bot_id = sel[0]
        confirm = messagebox.askyesno("Confirm Delete", "Are you sure? This will STOP the bot and DELETE it from the database permanently.")
        
        if confirm:
            try:
                self.app.api.delete_bot(bot_id)
                messagebox.showinfo("Success", "Bot deleted.")
                self.refresh_bot_list()
            except Exception as e:
                messagebox.showerror("Delete Error", str(e))

    def refresh_bot_list(self):
        for row in self.bots_tree.get_children():
            self.bots_tree.delete(row)
        try:
            bots = self.app.api.list_bots()
            for b in bots:
                sym_str = ",".join(b.get("symbols", []))
                self.bots_tree.insert("", "end", values=(
                    b.get("bot_id")[:8],
                    b.get("name"),
                    b.get("strategy_id"),
                    sym_str,
                    f"${b.get('capital')}",
                    b.get("status").upper()
                ), iid=b.get("bot_id"))
        except:
            pass

    # -----------------------------
    # 3. REAL HISTORY PAGE
    # -----------------------------
    def show_history(self):
        self.clear_content()
        tk.Label(self.content, text="Trade History", font=("Arial", 20)).pack(pady=12)

        # Table
        cols = ("time", "symbol", "side", "qty", "price", "bot")
        tree = ttk.Treeview(self.content, columns=cols, show="headings", height=20)
        tree.heading("time", text="Time (UTC)")
        tree.heading("symbol", text="Symbol")
        tree.heading("side", text="Side")
        tree.heading("qty", text="Qty")
        tree.heading("price", text="Price")
        tree.heading("bot", text="Bot ID")
        
        tree.column("time", width=160)
        tree.column("symbol", width=80)
        tree.column("side", width=60)
        tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Refresh Button
        tk.Button(self.content, text="Refresh Trades", command=lambda: self._load_history(tree)).pack(pady=5)
        
        self._load_history(tree)

    def _load_history(self, tree):
        for row in tree.get_children():
            tree.delete(row)
        try:
            trades = self.app.api.get_history() #
            for t in trades:
                tree.insert("", "end", values=(
                    t.get("executed_at", "").replace("T", " ")[:19],
                    t.get("symbol"),
                    t.get("side"),
                    t.get("quantity"),
                    f"${t.get('price')}",
                    t.get("bot_id")[:8]
                ))
        except Exception as e:
            messagebox.showerror("Error", f"Could not load history: {e}")

    # -----------------------------
    # 4. SETTINGS PAGE (Auto-Fill)
    # -----------------------------
    def show_settings(self):
        self.clear_content()
        tk.Label(self.content, text="Settings", font=("Arial", 20)).pack(pady=12)

        form = tk.LabelFrame(self.content, text="Account Configuration")
        form.pack(fill=tk.X, padx=12, pady=10)

        # -- Restored Username/Password Fields --
        tk.Label(form, text="Change Username").grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.set_username = tk.Entry(form, width=30)
        self.set_username.insert(0, self.app.user or "")
        self.set_username.grid(row=0, column=1, sticky="w")

        tk.Label(form, text="New Password").grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.set_pw1 = tk.Entry(form, width=30, show="*")
        self.set_pw1.grid(row=1, column=1, sticky="w")

        tk.Label(form, text="Confirm Password").grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.set_pw2 = tk.Entry(form, width=30, show="*")
        self.set_pw2.grid(row=2, column=1, sticky="w")

        tk.Frame(form, height=2, bg="#ddd").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        # -- API Keys --
        # Pre-fill logic
        curr = {}
        try: curr = self.app.api.get_settings()
        except: pass

        tk.Label(form, text="Alpaca API Key").grid(row=4, column=0, sticky="e", padx=10, pady=5)
        self.alpaca_key = tk.Entry(form, width=40)
        self.alpaca_key.grid(row=4, column=1, sticky="w")
        if curr.get("alpaca_key_id"): self.alpaca_key.insert(0, curr["alpaca_key_id"])

        tk.Label(form, text="Alpaca Secret").grid(row=5, column=0, sticky="e", padx=10, pady=5)
        self.alpaca_secret = tk.Entry(form, width=40, show="*")
        self.alpaca_secret.grid(row=5, column=1, sticky="w")
        if curr.get("alpaca_secret_key"): self.alpaca_secret.insert(0, curr["alpaca_secret_key"])

        tk.Label(form, text="Discord Webhook").grid(row=6, column=0, sticky="e", padx=10, pady=5)
        self.discord_webhook = tk.Entry(form, width=50)
        self.discord_webhook.grid(row=6, column=1, sticky="w")
        if curr.get("discord_webhook"): self.discord_webhook.insert(0, curr["discord_webhook"])

        tk.Button(form, text="Save Settings", bg="#3498db", fg="white", command=self.on_save_settings).grid(row=7, column=1, sticky="w", pady=15)

    def on_save_settings(self):
        try:
            # Note: Backend might not support username/pass update yet, 
            # but we send the API key payload as requested.
            payload = {
                "alpaca_key_id": self.alpaca_key.get().strip(),
                "alpaca_secret_key": self.alpaca_secret.get().strip(),
                "discord_webhook": self.discord_webhook.get().strip(),
            }
            self.app.api.update_settings(payload)
            
            # Simple check for password match (Client side validation)
            p1 = self.set_pw1.get()
            p2 = self.set_pw2.get()
            if p1 and p1 != p2:
                messagebox.showerror("Error", "Passwords do not match!")
                return
            
            messagebox.showinfo("Saved", "Settings updated (Keys saved).")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    TradingApp().mainloop()
