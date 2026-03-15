"""Microstructure data loader — Binance funding rate history (Phase 12E).

Fetches historical funding rates from the Binance Futures API with shared
cache. Funding rates are published every 8 hours. On failure, falls back
to stale cache. If no cache exists, returns an empty DataFrame.

Scope: funding rate only. Open interest is excluded per the roadmap due
to limited free historical depth (~30 days).
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import requests

from config import CACHE_TTL_MICROSTRUCTURE
from data.cache import cache_get, cache_get_stale, cache_put

logger = logging.getLogger(__name__)

NAMESPACE = "microstructure"
CACHE_KEY = "binance_funding_rate"

_API_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_SYMBOL = "BTCUSDT"
_MAX_LIMIT = 1000  # Binance max per request


def _fetch_funding_rates(start_ms: int | None = None) -> pd.DataFrame:
    """Fetch funding rate history from Binance Futures API.

    Paginates through the API to get full history from 2019 onwards.
    """
    all_rows = []
    params = {"symbol": _SYMBOL, "limit": _MAX_LIMIT}
    if start_ms is not None:
        params["startTime"] = start_ms

    while True:
        resp = requests.get(_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        # Paginate forward from the last timestamp
        last_ts = data[-1]["fundingTime"]
        params["startTime"] = last_ts + 1
        if len(data) < _MAX_LIMIT:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df.set_index("date")[["funding_rate"]]
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def load_funding_rate_data() -> pd.DataFrame:
    """Load Binance BTCUSDT funding rate history.

    Returns a DataFrame with UTC DatetimeIndex and column ``funding_rate``.
    Funding rates are published every 8 hours.
    """
    cached = cache_get(NAMESPACE, CACHE_KEY, ttl_seconds=CACHE_TTL_MICROSTRUCTURE)
    if cached is not None:
        logger.debug("Cache hit for funding rate")
        df = pd.read_csv(io.BytesIO(cached), index_col=0)
        df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
        return df

    try:
        df = _fetch_funding_rates()
        if df.empty:
            raise ValueError("Empty funding rate data returned")
        csv_bytes = df.to_csv().encode("utf-8")
        cache_put(NAMESPACE, CACHE_KEY, csv_bytes)
        logger.info("Fetched and cached funding rates (%d rows)", len(df))
        return df
    except Exception as e:
        logger.warning("API fetch failed for funding rates: %s — trying stale cache", e)
        stale = cache_get_stale(NAMESPACE, CACHE_KEY)
        if stale is not None:
            logger.info("Using stale cache for funding rates")
            df = pd.read_csv(io.BytesIO(stale), index_col=0)
            df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
            return df
        logger.warning("No cache available for funding rates — returning empty DataFrame")
        return pd.DataFrame(columns=["funding_rate"])
