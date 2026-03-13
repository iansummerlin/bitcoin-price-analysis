"""Backfill-first live analytics preparation for optional signal monitoring."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import websocket

from config import (
    BTC_SENTIMENTDATA_LIVE_URL,
    BTCUSD_STREAM_URL,
    TRADING_DATA_PATH,
    EXOG_COLUMNS,
)
from data.pipeline import build_dataset
from features.pipeline import apply_feature_pipeline

logger = logging.getLogger(__name__)


def validate_row(row: dict) -> bool:
    """Validate a candle/feature row for impossible values."""
    required = ["close", "high", "low"]
    try:
        for field in required:
            if float(row.get(field, 0)) <= 0:
                return False
        if float(row["high"]) < float(row["low"]):
            return False
        rsi = row.get("rsi")
        if rsi is not None and not pd.isna(rsi):
            rsi_value = float(rsi)
            if rsi_value < 0 or rsi_value > 100:
                return False
    except (TypeError, ValueError, KeyError):
        return False
    return True


def get_daily_sentiment_data() -> tuple[float | None, datetime | None, datetime | None]:
    """Fetch the latest Fear and Greed reading."""
    try:
        response = requests.get(BTC_SENTIMENTDATA_LIVE_URL, timeout=10)
        response.raise_for_status()
        payload = response.json()["data"][0]
        value = float(payload["value"])
        current_time = datetime.now(timezone.utc)
        update_at = current_time
        return value, current_time, update_at
    except Exception:
        return None, None, None


def handle_is_enough_data(df: pd.DataFrame, minimum_rows: int = 120) -> bool:
    """Check whether enough history exists to compute the longest lookback."""
    return len(df) >= minimum_rows


def prepare_exog_data(df: pd.DataFrame, stream_data: dict, fng_value: float | None) -> pd.DataFrame:
    """Append a streamed candle and recompute the latest canonical feature state."""
    event_time = stream_data.get("k", {}).get("t", stream_data.get("E"))
    candle = stream_data.get("k", stream_data)
    row = pd.DataFrame(
        {
            "open": [float(candle["o"])],
            "high": [float(candle["h"])],
            "low": [float(candle["l"])],
            "close": [float(candle["c"])],
            "Volume BTC": [float(candle["v"])],
            "Volume USD": [float(candle["q"])],
            "fng_value": [float(fng_value) if fng_value is not None else 50.0],
        },
        index=[pd.to_datetime(int(event_time), unit="ms", utc=True)],
    )
    row.index.name = "date"
    combined = pd.concat([df, row])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if handle_is_enough_data(combined):
        combined = apply_feature_pipeline(combined, dropna=False)

    return combined


def bootstrap_live_dataset(output_path: str | Path = TRADING_DATA_PATH) -> pd.DataFrame:
    """Build a historical backfill so live analytics can start immediately."""
    dataset, _ = build_dataset()
    dataset.to_csv(output_path)
    return dataset


def on_message_with_model(filename: str | Path, df: pd.DataFrame):
    """Create the websocket callback that appends closed hourly candles."""
    last_timestamp = df.index.max() if not df.empty else None
    last_fng_value = None

    def on_message(ws, message):
        nonlocal df, last_timestamp, last_fng_value
        payload = json.loads(message)
        candle = payload.get("k", {})
        if not candle or not candle.get("x"):
            return
        current_timestamp = pd.to_datetime(int(candle["t"]), unit="ms", utc=True)
        if last_timestamp is not None and current_timestamp <= last_timestamp:
            return

        if last_fng_value is None:
            last_fng_value, _, _ = get_daily_sentiment_data()
        df = prepare_exog_data(df, payload, last_fng_value)
        if validate_row(df.iloc[-1].to_dict()):
            df.to_csv(filename)
            last_timestamp = current_timestamp
            logger.info("appended live candle at %s", current_timestamp.isoformat())

    return on_message


def on_error_with_timestamp(filename: str | Path):
    """Create an error logger callback."""
    path = Path(f"{filename}.txt")

    def on_error(ws, error):
        timestamp = datetime.now(timezone.utc).isoformat()
        path.write_text(f"{timestamp}: {error}\n", encoding="utf-8")
        logger.error("websocket error: %s", error)

    return on_error


def on_close(ws, close_status_code, close_msg):
    logger.warning("websocket closed: %s %s", close_status_code, close_msg)


def on_open(ws):
    logger.info("websocket opened")


def on_ping(ws, message):
    ws.send(message, websocket.ABNF.OPCODE_PONG)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    dataset = bootstrap_live_dataset()
    logger.info("bootstrapped %s rows with %s feature columns", len(dataset), len(EXOG_COLUMNS))
    ws = websocket.WebSocketApp(
        BTCUSD_STREAM_URL,
        on_message=on_message_with_model(TRADING_DATA_PATH, dataset),
        on_error=on_error_with_timestamp("BTCUSD_trading_errors"),
        on_close=on_close,
        on_open=on_open,
        on_ping=on_ping,
    )
    ws.run_forever()


if __name__ == "__main__":
    main()
