"""Cross-asset data loader using yfinance with shared cache (Phase 12B).

Fetches daily data for DXY, S&P 500, VIX, Gold, and ETH-USD.
All API calls go through ``data.cache``. On failure, falls back to
stale cache. If no cache exists, returns an empty DataFrame with the
correct schema.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd

from config import CACHE_TTL_CROSSASSET
from data.cache import cache_get, cache_get_stale, cache_put

logger = logging.getLogger(__name__)

NAMESPACE = "crossasset"

# Tickers and their cache keys.
CROSS_ASSET_TICKERS = {
    "DX-Y.NYB": "dxy",
    "^GSPC": "sp500",
    "^VIX": "vix",
    "GC=F": "gold",
    "ETH-USD": "eth",
}

# Expected columns in the output DataFrame (daily, UTC-indexed).
CROSS_ASSET_COLUMNS = ["dxy", "sp500", "vix", "gold", "eth"]


def _fetch_ticker(ticker: str, start: str = "2014-01-01") -> pd.DataFrame:
    """Fetch daily close data from yfinance for a single ticker."""
    import yfinance as yf

    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pd.DataFrame()

    # yfinance may return MultiIndex columns for single ticker
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data[["Close"]].copy()
    close.index = pd.to_datetime(close.index, utc=True)
    close.index.name = "date"
    return close


def _load_single(ticker: str, cache_key: str) -> pd.Series:
    """Load a single ticker with cache-first strategy."""
    # Try cache first
    cached = cache_get(NAMESPACE, cache_key, ttl_seconds=CACHE_TTL_CROSSASSET)
    if cached is not None:
        logger.debug("Cache hit for %s", cache_key)
        df = pd.read_csv(io.BytesIO(cached), index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df["Close"]

    # Cache miss — fetch from API
    try:
        df = _fetch_ticker(ticker)
        if df.empty:
            raise ValueError(f"Empty data returned for {ticker}")
        csv_bytes = df.to_csv().encode("utf-8")
        cache_put(NAMESPACE, cache_key, csv_bytes)
        logger.info("Fetched and cached %s (%d rows)", cache_key, len(df))
        return df["Close"]
    except Exception as e:
        logger.warning("API fetch failed for %s: %s — trying stale cache", ticker, e)
        stale = cache_get_stale(NAMESPACE, cache_key)
        if stale is not None:
            logger.info("Using stale cache for %s", cache_key)
            df = pd.read_csv(io.BytesIO(stale), index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df["Close"]
        logger.warning("No cache available for %s — returning empty series", cache_key)
        return pd.Series(dtype=float, name="Close")


def load_cross_asset_data() -> pd.DataFrame:
    """Load all cross-asset daily close prices.

    Returns a DataFrame with daily UTC index and columns:
    ``dxy``, ``sp500``, ``vix``, ``gold``, ``eth``.

    Traditional market data is forward-filled through weekends/holidays.
    """
    frames = {}
    for ticker, cache_key in CROSS_ASSET_TICKERS.items():
        series = _load_single(ticker, cache_key)
        if not series.empty:
            frames[cache_key] = series

    if not frames:
        return pd.DataFrame(columns=CROSS_ASSET_COLUMNS)

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined.index.name = "date"

    # Forward-fill weekends/holidays (Friday close through Sat/Sun)
    combined = combined.asfreq("D").ffill()

    # Ensure all expected columns exist
    for col in CROSS_ASSET_COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    return combined[CROSS_ASSET_COLUMNS]
