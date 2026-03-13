"""Dataset validation utilities."""

from __future__ import annotations

import pandas as pd

from config import OHLCV_COLUMNS


def validate_ohlcv_frame(df: pd.DataFrame) -> list[str]:
    """Return validation errors for a canonical OHLCV frame."""
    errors: list[str] = []

    missing_columns = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append(f"missing columns: {missing_columns}")
        return errors

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("index must be a DatetimeIndex")
        return errors

    if not df.index.is_monotonic_increasing:
        errors.append("timestamps are not monotonic increasing")

    if df.index.has_duplicates:
        errors.append("duplicate timestamps detected")

    if len(df.index) >= 3:
        timestamp_deltas = df.index.to_series().diff().dropna()
        if not timestamp_deltas.empty:
            most_common_delta = timestamp_deltas.mode().iloc[0]
            if most_common_delta != pd.Timedelta(hours=1):
                errors.append(f"unexpected dominant timestep: {most_common_delta}")
            missing_steps = int((timestamp_deltas / pd.Timedelta(hours=1)).sub(1).clip(lower=0).sum())
            if missing_steps > 0:
                errors.append(f"missing timestamps detected: {missing_steps}")

    if (df["high"] < df["low"]).any():
        errors.append("high below low detected")

    if (df["high"] < df["open"]).any() or (df["high"] < df["close"]).any():
        errors.append("high below open/close detected")

    if (df["low"] > df["open"]).any() or (df["low"] > df["close"]).any():
        errors.append("low above open/close detected")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("non-positive OHLC values detected")
    if (df[["Volume BTC", "Volume USD"]] < 0).any().any():
        errors.append("negative volume values detected")

    null_density = df[OHLCV_COLUMNS].isna().mean().max()
    if null_density > 0.0:
        errors.append(f"unexpected null density in OHLCV columns: {null_density:.3f}")

    return errors


def assert_valid_ohlcv_frame(df: pd.DataFrame) -> None:
    """Raise ValueError if the frame fails canonical validation."""
    errors = validate_ohlcv_frame(df)
    blocking_errors = [error for error in errors if not error.startswith("missing timestamps detected")]
    if blocking_errors:
        raise ValueError("; ".join(blocking_errors))
