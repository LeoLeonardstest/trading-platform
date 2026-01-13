"""client/yahoo_data.py

Yahoo Finance OHLCV loader for backtests.

Design goals:
- Download using yfinance
- Optional local CSV cache to speed up repeated backtests
- Return a pandas DataFrame with datetime index and OHLCV columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None


@dataclass
class YahooLoadOptions:
    cache_dir: Path = Path("./.cache/yahoo")
    use_cache: bool = True


def _cache_path(symbol: str, interval: str, start: str, end: str, cache_dir: Path) -> Path:
    safe = symbol.replace("/", "_")
    return cache_dir / f"{safe}__{interval}__{start}__{end}.csv"

def _download_with_retries(symbol: str, start: str, end: str, interval: str, tries: int = 3) -> pd.DataFrame:
    last_err = None
    for i in range(tries):
        try:
            df = yf.download(
                tickers=symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            if df is not None and not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(1.5 * (i + 1))
    if last_err:
        raise last_err
    return pd.DataFrame()

def _to_yf_interval(timeframe: str) -> str:
    """
    Map shared/live-style timeframes to yfinance interval strings.
    yfinance expects: 1m,5m,15m,30m,1h,1d
    """
    tf = (timeframe or "1D").strip()
    tf_lower = tf.lower()

    # Support canonical shared values like "1H" and "1D"
    tf_upper = tf.strip().upper()
    if tf_upper == "1H":
        return "1h"
    if tf_upper == "1D":
        return "1d"

    mapping = {
        "1min": "1m",
        "1m": "1m",
        "5min": "5m",
        "5m": "5m",
        "15min": "15m",
        "15m": "15m",
        "30min": "30m",
        "30m": "30m",
        "1h": "1h",
        "1hour": "1h",
        "60m": "1h",
        "1d": "1d",
        "1day": "1d",
        "day": "1d",
    }
    return mapping.get(tf_lower, "1d")


def _read_cached_csv(path: Path) -> pd.DataFrame:
    """
    Robust cache reader:
    - Supports old caches with 'Date' column
    - Supports caches with unnamed first column
    - Supports current caches with 'Datetime' column
    """
    # Read header first without parsing to detect columns
    preview = pd.read_csv(path, nrows=1)
    cols = list(preview.columns)

    # Preferred: Datetime column
    if "Datetime" in cols:
        df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
        df.index.name = "Datetime"
        return df

    # Common older format: Date column
    if "Date" in cols:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.index.name = "Datetime"
        return df

    # Fallback: assume first column is datetime index
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "Datetime"
    return df


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1D",
    options: Optional[YahooLoadOptions] = None,
) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is required. pip install yfinance")

    options = options or YahooLoadOptions()
    options.cache_dir.mkdir(parents=True, exist_ok=True)

    interval = _to_yf_interval(timeframe)

    p = _cache_path(symbol, interval, start, end, options.cache_dir)
    if options.use_cache and p.exists():
        df = _read_cached_csv(p)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    # 1. LEJUPIELĀDE
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    print(f"DEBUG: Saņemtas {len(df)} rindas")
    print(f"DEBUG: Kolonnu nosaukumi: {df.columns.tolist()}")

    # 2. PĀRBAUDE: Vai vispār ir dati (Yahoo 60 dienu limits)
    if df is None or df.empty:
        raise ValueError(f"KĻŪDA: Yahoo Finance neatgrieza datus priekš {symbol}. "
                         f"Pārbaudi, vai 5m datus neprasi senākus par 60 dienām!")

    # 3. FIX: Jaunais yfinance MultiIndex (saplacinām kolonnas)
    # Ja kolonnas ir ('MSFT', 'Open'), pārvēršam tās par 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # 2. Drošības pēc noņemam tukšumus no nosaukumiem
    df.columns = [str(c).strip() for c in df.columns]
    # 4. Standartizējam indeksu
    df.index.name = "Datetime"

    # 5. Filtrējam tikai vajadzīgās kolonnas
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # 6. Pēdējā pārbaude pirms atgriešanas
    if df.empty or len(df) < 2:
        raise ValueError("Datu rāmis pēc apstrādes ir tukšs vai par īsu backtestam.")

    if options.use_cache:
        df.to_csv(p, index_label="Datetime")

    return df
