"""client/yahoo_data.py

FILE OVERVIEW:
This module is responsible for fetching historical stock data (OHLCV).
It uses the `yfinance` library to download data from Yahoo Finance.

It includes a caching system:
- When you download data for "AAPL", it saves it as a CSV file in a `.cache` folder.
- Next time you run a test, it reads the CSV instead of downloading again.
- This speeds up testing and prevents being blocked by Yahoo.
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
    """Configuration for data loading."""
    cache_dir: Path = Path("./.cache/yahoo") # Directory to save CSVs
    use_cache: bool = True                   # Whether to use the cache or force download


def _cache_path(symbol: str, interval: str, start: str, end: str, cache_dir: Path) -> Path:
    """Generates a unique filename for the cache based on symbol and dates."""
    safe = symbol.replace("/", "_")
    return cache_dir / f"{safe}__{interval}__{start}__{end}.csv"

def _download_with_retries(symbol: str, start: str, end: str, interval: str, tries: int = 3) -> pd.DataFrame:
    """Attempts to download data, retrying if it fails (e.g. network error)."""
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
        time.sleep(1.5 * (i + 1)) # Wait longer after each failure
    if last_err:
        raise last_err
    return pd.DataFrame()

def _to_yf_interval(timeframe: str) -> str:
    """
    Maps our internal timeframe format (e.g. "1Day") to Yahoo's format (e.g. "1d").
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
    Reads a CSV file from the cache. 
    It handles different date column names to ensure compatibility with older cache files.
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
    """
    Main function to get data.
    1. Check cache.
    2. If missing, download from Yahoo.
    3. Clean and normalize data.
    4. Save to cache.
    """
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

    # 1. DOWNLOAD (LEJUPIELĀDE)
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

    # 2. CHECK: Is data empty? (PĀRBAUDE)
    if df is None or df.empty:
        raise ValueError(f"KĻŪDA: Yahoo Finance neatgrieza datus priekš {symbol}. "
                         f"Pārbaudi, vai 5m datus neprasi senākus par 60 dienām!")

    # 3. FIX: Handle MultiIndex columns from new yfinance versions
    # If columns look like ('MSFT', 'Open'), flatten them to just 'Open'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Clean column names (remove whitespace)
    df.columns = [str(c).strip() for c in df.columns]
    
    # 4. Standardize Index
    df.index.name = "Datetime"

    # 5. Filter only required columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # 6. Final validity check
    if df.empty or len(df) < 2:
        raise ValueError("Datu rāmis pēc apstrādes ir tukšs vai par īsu backtestam.")

    if options.use_cache:
        df.to_csv(p, index_label="Datetime")

    return df