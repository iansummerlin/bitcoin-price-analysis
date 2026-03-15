"""On-chain data loader using blockchain.com API with shared cache (Phase 12C).

Fetches daily on-chain metrics: hash rate, difficulty, transaction count,
estimated transaction volume USD. All API calls go through ``data.cache``.
On failure, falls back to stale cache. If no cache exists, returns an
empty DataFrame with the correct schema.

Rate limit: blockchain.com allows ~1 request per 10 seconds. We fetch
each metric separately with a small delay between calls.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import CACHE_TTL_ONCHAIN
from data.cache import cache_get, cache_get_stale, cache_put

logger = logging.getLogger(__name__)

NAMESPACE = "onchain"

# Blockchain.com chart API metrics and their cache keys.
ONCHAIN_METRICS = {
    "hash-rate": "hashrate",
    "difficulty": "difficulty",
    "n-transactions": "tx_count",
    "estimated-transaction-volume-usd": "tx_volume_usd",
}

ONCHAIN_COLUMNS = ["hashrate", "difficulty", "tx_count", "tx_volume_usd"]

_API_BASE = "https://api.blockchain.info/charts"
_REQUEST_DELAY = 11  # seconds between API calls to respect rate limit


def _fetch_metric(metric: str, start: str = "2014-01-01") -> pd.DataFrame:
    """Fetch a single daily metric from blockchain.com charts API."""
    url = f"{_API_BASE}/{metric}"
    params = {
        "timespan": "all",
        "format": "json",
        "cors": "true",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    values = data.get("values", [])
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.set_index("date")[["y"]].rename(columns={"y": "value"})
    df = df[~df.index.duplicated(keep="last")]
    return df


def _load_single(metric: str, cache_key: str) -> pd.Series:
    """Load a single on-chain metric with cache-first strategy."""
    cached = cache_get(NAMESPACE, cache_key, ttl_seconds=CACHE_TTL_ONCHAIN)
    if cached is not None:
        logger.debug("Cache hit for %s", cache_key)
        df = pd.read_csv(io.BytesIO(cached), index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df["value"]

    try:
        df = _fetch_metric(metric)
        if df.empty:
            raise ValueError(f"Empty data returned for {metric}")
        csv_bytes = df.to_csv().encode("utf-8")
        cache_put(NAMESPACE, cache_key, csv_bytes)
        logger.info("Fetched and cached %s (%d rows)", cache_key, len(df))
        return df["value"]
    except Exception as e:
        logger.warning("API fetch failed for %s: %s — trying stale cache", metric, e)
        stale = cache_get_stale(NAMESPACE, cache_key)
        if stale is not None:
            logger.info("Using stale cache for %s", cache_key)
            df = pd.read_csv(io.BytesIO(stale), index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            return df["value"]
        logger.warning("No cache available for %s — returning empty series", cache_key)
        return pd.Series(dtype=float, name="value")


def load_onchain_data() -> pd.DataFrame:
    """Load all on-chain daily metrics.

    Returns a DataFrame with daily UTC index and columns:
    ``hashrate``, ``difficulty``, ``tx_count``, ``tx_volume_usd``.
    """
    frames = {}
    metrics_list = list(ONCHAIN_METRICS.items())
    for i, (metric, cache_key) in enumerate(metrics_list):
        series = _load_single(metric, cache_key)
        if not series.empty:
            frames[cache_key] = series
        # Respect rate limit between API calls (skip delay for cache hits and last item)
        if i < len(metrics_list) - 1:
            # Only delay if we actually hit the API (cache miss)
            cached = cache_get(NAMESPACE, cache_key, ttl_seconds=CACHE_TTL_ONCHAIN)
            if cached is None:
                time.sleep(_REQUEST_DELAY)

    if not frames:
        return pd.DataFrame(columns=ONCHAIN_COLUMNS)

    combined = pd.DataFrame(frames)
    combined.index = pd.to_datetime(combined.index, utc=True)
    combined.index.name = "date"

    # Forward-fill gaps
    combined = combined.asfreq("D").ffill()

    for col in ONCHAIN_COLUMNS:
        if col not in combined.columns:
            combined[col] = float("nan")

    return combined[ONCHAIN_COLUMNS]
