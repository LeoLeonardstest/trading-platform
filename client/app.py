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
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.resizable(False, False)

        self.user = None  # logged-in username
        self.user_id = None
        self.api = None  # ApiClient

        self.container = tk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.show_login()

    def clear(self):
        for widget in self.container.winfo_children():
            widget.destroy()

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

        tk.Label(self, text="Login", font=("Arial", 22)).pack(pady=40)

        tk.Label(self, text="Username").pack()
        self.username = tk.Entry(self, width=30)
        self.username.insert(0, "leo")
        self.username.pack(pady=5)

        tk.Label(self, text="Password").pack()
        self.password = tk.Entry(self, width=30, show="*")
        self.password.insert(0, "password")
        self.password.pack(pady=5)

        tk.Button(self, text="Login", width=25, command=self.login).pack(pady=20)

        tk.Label(
            self,
            text="Authentication is a placeholder. Backend integration goes in client/api.py",
            fg="#666",
        ).pack(pady=10)

    def login(self):
        if not self.username.get() or not self.password.get():
            messagebox.showerror("Error", "Username and password required")
            return
        self.app.user = self.username.get().strip()
        self.app.show_main()



def on_login(self):
    try:
        base_url = self.base_url.get().strip()
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
        base_url = self.base_url.get().strip()
        self.app.api = ApiClient(base_url)
        username = self.username.get().strip()
        password = self.password.get().strip()
        data = self.app.api.register(username, password)
        self.app.api.token = data.get("token")
        self.app.user = data.get("username", username)
        self.app.user_id = data.get("user_id")
        self.app.show_main()
    except Exception as e:
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
            header,
            text=APP_TITLE,
            fg="white",
            bg="#2c3e50",
            font=("Arial", 14, "bold"),
        ).pack(side=tk.LEFT, padx=15)

        tk.Button(header, text="Logout", command=self.logout).pack(side=tk.RIGHT, padx=10)
        tk.Label(header, text=f"User: {self.app.user}", fg="white", bg="#2c3e50").pack(
            side=tk.RIGHT, padx=10
        )

    def logout(self):
        self.app.user = None
        self.app.show_login()

    def _build_body(self):
        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)

        nav = tk.Frame(body, width=220, bg="#ecf0f1")
        nav.pack(side=tk.LEFT, fill=tk.Y)

        for text, cmd in [
            ("Home", self.show_home),
            ("Test", self.show_test),
            ("Bots", self.show_bots),
            ("History", self.show_history),
            ("Settings", self.show_settings),
        ]:
            tk.Button(nav, text=text, width=24, command=cmd).pack(pady=6)

        self.content = tk.Frame(body)
        self.content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.show_home()

    def clear_content(self):
        for w in self.content.winfo_children():
            w.destroy()

    # -----------------------------
    # Home (Dashboard)
    # -----------------------------
    def show_home(self):
        self.clear_content()
        tk.Label(self.content, text="Dashboard", font=("Arial", 20)).pack(pady=14)

        cards = tk.Frame(self.content)
        cards.pack(pady=10)

        for title, value in [
            ("Alpaca Account Balance", "$12,345"),
            ("Running Bots", "3"),
            ("Total Profit", "+$1,240"),
            ("Today PnL", "+$120"),
        ]:
            card = tk.LabelFrame(cards, text=title, width=260, height=90)
            card.pack(side=tk.LEFT, padx=10)
            card.pack_propagate(False)
            tk.Label(card, text=value, font=("Arial", 16, "bold")).pack(expand=True)

        tk.Label(self.content, text="Running Bots (placeholder)", font=("Arial", 12, "bold")).pack(
            pady=(18, 6)
        )
        bots_box = tk.Text(self.content, height=14, width=125)
        bots_box.insert(
            "1.0",
            "BotID | Strategy | Status | PnL\n"
            "------------------------------------------------------------\n"
            "b1    | moving_average | RUNNING | +$45\n"
            "b2    | rsi_reversion   | STOPPED | +$10\n",
        )
        bots_box.config(state=tk.DISABLED)
        bots_box.pack(padx=12)

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
            capital = float(self.bt_capital.get().strip())
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
    # Bots (Live)
    # -----------------------------
    def show_bots(self):
        self.clear_content()
        tk.Label(self.content, text="Live Bots (Server)", font=("Arial", 20)).pack(pady=12)

        create = tk.LabelFrame(self.content, text="Create / Start Bot")
        create.pack(fill=tk.X, padx=12, pady=8)

        tk.Label(create, text="Strategy ID").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.bot_strategy = ttk.Combobox(
            create,
            width=38,
            values=[
                "mean_reversion_losers",
                "moving_average",
                "rsi_reversion",
                "macd_trend",
            ],
        )
        self.bot_strategy.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        self.bot_strategy.set("moving_average")

        self.bot_strategy_desc = tk.Label(create, text="", fg="#555", wraplength=760)
        self.bot_strategy_desc.grid(row=0, column=2, sticky="w", padx=6)

        tk.Label(create, text="Symbols (comma-separated)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.bot_symbols = tk.Entry(create, width=42)
        self.bot_symbols.grid(row=1, column=1, sticky="w", padx=6, pady=4)
        self.bot_symbols.insert(0, "AAPL, MSFT, TSLA")

        tk.Label(create, text="Capital Allocation ($)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.bot_capital = tk.Entry(create, width=20)
        self.bot_capital.grid(row=2, column=1, sticky="w", padx=6, pady=4)
        self.bot_capital.insert(0, "5000")

        self.bot_params_frame = tk.LabelFrame(self.content, text="Strategy Parameters (same keys as Backtest)")
        self.bot_params_frame.pack(fill=tk.X, padx=12, pady=8)

        btns = tk.Frame(self.content)
        btns.pack(fill=tk.X, padx=12)
        tk.Button(btns, text="Start Bot", width=14, command=self.on_start_bot).pack(side=tk.LEFT)
        tk.Button(btns, text="Stop Bot", width=14, command=self.on_stop_bot).pack(side=tk.LEFT, padx=8)
        tk.Label(btns, text="Hooks: POST /bots/start and POST /bots/stop", fg="#666").pack(side=tk.LEFT, padx=10)

        tk.Label(self.content, text="Running Bots (placeholder)", font=("Arial", 12, "bold")).pack(
            pady=(12, 6)
        )
        bots_box = tk.Text(self.content, height=18, width=125)
        bots_box.insert(
            "1.0",
            "BotID | Strategy | Status | PnL | Started\n"
            "------------------------------------------------------------\n"
            "b1    | moving_average | RUNNING | +$45 | 2026-01-12 09:30\n",
        )
        bots_box.config(state=tk.DISABLED)
        bots_box.pack(padx=12)

        self.bot_strategy.bind("<<ComboboxSelected>>", lambda e: self._render_bot_params())
        self._render_bot_params()

    def _render_bot_params(self):
        for w in self.bot_params_frame.winfo_children():
            w.destroy()

        sid = self.bot_strategy.get().strip()
        self.bot_param_entries = {}

        desc_map = {
            "mean_reversion_losers": "Daily: buy worst N losers at open and sell at close (optional stop-loss).",
            "moving_average": "Trend: buy/sell on SMA/EMA crossover (optional stop-loss/take-profit).",
            "rsi_reversion": "Mean reversion: buy when RSI <= oversold, sell when RSI >= overbought.",
            "macd_trend": "Momentum: buy/sell on MACD vs signal crossover (optional risk rules).",
        }
        self.bot_strategy_desc.config(text=desc_map.get(sid, ""))

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
            tk.Label(self.bot_params_frame, text=label).grid(row=r, column=0, sticky="w", padx=8, pady=4)
            ent = tk.Entry(self.bot_params_frame, width=26)
            ent.grid(row=r, column=1, sticky="w", padx=8, pady=4)
            ent.insert(0, example)
            tk.Label(self.bot_params_frame, text=f"Example: {example}", fg="#666").grid(
                row=r, column=2, sticky="w", padx=8
            )
            self.bot_param_entries[key] = ent

    def on_start_bot(self):
        try:
            from client.utils import parse_symbols, collect_params
            if not self.app.api:
                raise ValueError("Not connected. Please login again.")
            sid = self.bot_strategy.get().strip()
            symbols = parse_symbols(self.bot_symbols.get())
            capital = float(self.bot_capital.get().strip())
            raw_params = {k: ent.get() for k, ent in self.bot_param_entries.items()}
            params = collect_params(raw_params)
            name = f"{sid}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            payload = {"name": name, "strategy_id": sid, "symbols": symbols, "params": params, "capital": capital}
            resp = self.app.api.start_bot(payload)
            messagebox.showinfo("Bot Started", f"Started bot_id: {resp.get('bot_id')}")
        except Exception as e:
            messagebox.showerror("Start Bot Error", str(e))

    def on_stop_bot(self):
        try:
            if not self.app.api:
                raise ValueError("Not connected. Please login again.")
            bot_id = self.stop_bot_id.get().strip() if hasattr(self, "stop_bot_id") else ""
            if not bot_id:
                raise ValueError("Enter Bot ID to stop.")
            self.app.api.stop_bot(bot_id)
            messagebox.showinfo("Bot Stopped", f"Stopped bot_id: {bot_id}")
        except Exception as e:
            messagebox.showerror("Stop Bot Error", str(e))

    # -----------------------------
    # History
    # -----------------------------
    def show_history(self):
        self.clear_content()
        tk.Label(self.content, text="Trade History (Server)", font=("Arial", 20)).pack(pady=12)

        tk.Label(self.content, text="Placeholder: backend returns trades grouped by bot.", fg="#555").pack(
            pady=(0, 10)
        )

        hist = tk.Text(self.content, height=30, width=125)
        try:
            if not self.app.api:
                raise ValueError("Not connected. Please login again.")
            data = self.app.api.get_history()
            hist.insert("1.0", "Timestamp | BotID | Symbol | Side | Qty | Price
")
            hist.insert("end", "-" * 80 + "
")
            for t in data:
                hist.insert("end", f"{t.get('executed_at','')} | {t.get('bot_id','')} | {t.get('symbol','')} | {t.get('side','')} | {t.get('quantity','')} | {t.get('price','')}
")
        except Exception as e:
            hist.insert("1.0", f"Error loading history: {e}
")
        hist.config(state=tk.DISABLED)
        hist.pack(padx=12)

    # -----------------------------
    # Settings
    # -----------------------------
    def show_settings(self):
        self.clear_content()
        tk.Label(self.content, text="Settings", font=("Arial", 20)).pack(pady=12)

        form = tk.LabelFrame(self.content, text="Account & Integrations")
        form.pack(fill=tk.X, padx=12, pady=10)

        tk.Label(form, text="Change Username").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        self.set_username = tk.Entry(form, width=40)
        self.set_username.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        self.set_username.insert(0, self.app.user or "")
        tk.Label(form, text="Example: leo", fg="#666").grid(row=0, column=2, sticky="w")

        tk.Label(form, text="New Password").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.set_pw1 = tk.Entry(form, width=40, show="*")
        self.set_pw1.grid(row=1, column=1, sticky="w", padx=8, pady=4)
        tk.Label(form, text="Example: strongpassword", fg="#666").grid(row=1, column=2, sticky="w")

        tk.Label(form, text="Confirm Password").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.set_pw2 = tk.Entry(form, width=40, show="*")
        self.set_pw2.grid(row=2, column=1, sticky="w", padx=8, pady=4)

        tk.Label(form, text="Alpaca API Key").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.alpaca_key = tk.Entry(form, width=40, show="*")
        self.alpaca_key.grid(row=3, column=1, sticky="w", padx=8, pady=4)
        tk.Label(form, text="Example: AK...", fg="#666").grid(row=3, column=2, sticky="w")

        tk.Label(form, text="Alpaca API Secret").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        self.alpaca_secret = tk.Entry(form, width=40, show="*")
        self.alpaca_secret.grid(row=4, column=1, sticky="w", padx=8, pady=4)
        tk.Label(form, text="Example: SK...", fg="#666").grid(row=4, column=2, sticky="w")

        tk.Label(form, text="Discord Webhook URL").grid(row=5, column=0, sticky="w", padx=8, pady=4)
        self.discord_webhook = tk.Entry(form, width=60)
        self.discord_webhook.grid(row=5, column=1, sticky="w", padx=8, pady=4)
        tk.Label(form, text="Example: https://discord.com/api/webhooks/...", fg="#666").grid(
            row=5, column=2, sticky="w"
        )

        btns = tk.Frame(self.content)
        btns.pack(fill=tk.X, padx=12)
        tk.Button(btns, text="Save Settings", width=16, command=self.on_save_settings).pack(side=tk.LEFT)
        tk.Label(btns, text="Hook: POST /settings/update", fg="#666").pack(side=tk.LEFT, padx=10)

    def on_save_settings(self):
        try:
            if not self.app.api:
                raise ValueError("Not connected. Please login again.")
            payload = {
                "alpaca_key_id": (self.alpaca_key.get().strip() or None),
                "alpaca_secret_key": (self.alpaca_secret.get().strip() or None),
                "alpaca_base_url": None,
                "discord_webhook": (self.discord_webhook.get().strip() or None),
            }
            self.app.api.update_settings(payload)
            messagebox.showinfo("Saved", "Settings saved to server.")
        except Exception as e:
            messagebox.showerror("Settings Error", str(e))


if __name__ == "__main__":
    TradingApp().mainloop()
