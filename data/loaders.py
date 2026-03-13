"""Canonical loaders for historical price and sentiment data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import BTC_PRICEDATA_PATH, BTC_SENTIMENTDATA_PATH
from data.validation import assert_valid_ohlcv_frame


def _normalize_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    if "unix" in normalized.columns:
        normalized = normalized.drop(columns=["unix"], errors="ignore")
    if "symbol" in normalized.columns:
        normalized = normalized.drop(columns=["symbol"], errors="ignore")

    if "Date" in normalized.columns:
        normalized = normalized.rename(columns={"Date": "date"})
    if "Volume BTC" not in normalized.columns and "Volume BTC" not in normalized.columns:
        pass

    normalized["date"] = pd.to_datetime(normalized["date"], utc=True)
    normalized = normalized.set_index("date").sort_index()

    numeric_columns = ["open", "high", "low", "close", "Volume BTC", "Volume USD"]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.dropna(subset=numeric_columns)
    return normalized


def load_price_history(path: str | Path = BTC_PRICEDATA_PATH) -> pd.DataFrame:
    """Load canonical historical price data from the local artifact."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Price data not found at {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    skiprows = 1 if first_line.startswith("http") and "date" not in first_line.lower() else 0
    raw = pd.read_csv(file_path, skiprows=skiprows)
    normalized = _normalize_price_columns(raw)
    normalized = normalized[(normalized["open"] > 0) & (normalized["high"] > 0) & (normalized["low"] > 0) & (normalized["close"] > 0)]
    assert_valid_ohlcv_frame(normalized)
    return normalized


def load_sentiment_history(path: str | Path = BTC_SENTIMENTDATA_PATH) -> pd.DataFrame:
    """Load canonical historical Fear and Greed data."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame(columns=["fng_value"])

    sentiment = pd.read_csv(file_path, index_col=0)
    sentiment.index = pd.to_datetime(sentiment.index, utc=True)
    sentiment = sentiment.sort_index()
    if "fng_value" in sentiment.columns:
        sentiment["fng_value"] = pd.to_numeric(sentiment["fng_value"], errors="coerce")
    sentiment = sentiment[["fng_value"]].dropna().rename_axis("date")
    return sentiment
