"""Global liquidity artifact loader for downstream integration.

Reads the JSON artifact produced by global-liquidity-analysis and extracts
the time-series data into a monthly DataFrame. Consumers merge this into
the hourly dataset by date, forward-filling monthly values across hours.

The artifact path defaults to the sibling repo's output. Override with
the LIQUIDITY_ARTIFACT_PATH environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "global-liquidity-analysis"
    / "artifacts"
    / "liquidity_regime.json"
)

ARTIFACT_PATH = Path(os.environ.get("LIQUIDITY_ARTIFACT_PATH", str(_DEFAULT_ARTIFACT_PATH)))

LIQUIDITY_FEATURE_COLUMNS = [
    "liquidity_global_usd_t",
    "liquidity_m2_roc_3m",
    "liquidity_regime_expanding",
    "liquidity_regime_neutral",
    "liquidity_regime_contracting",
]


def load_liquidity_artifact(path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    """Load the liquidity regime artifact and return (time_series_df, metadata).

    Returns
    -------
    df : pd.DataFrame
        Monthly DatetimeIndex (UTC) with columns matching LIQUIDITY_FEATURE_COLUMNS.
        Empty DataFrame with correct columns if artifact is missing or invalid.
    metadata : dict
        Top-level artifact fields (is_stale, sources_missing, etc.).
        Empty dict on load failure.
    """
    fpath = Path(path) if path else ARTIFACT_PATH

    if not fpath.exists():
        logger.warning("Liquidity artifact not found at %s", fpath)
        return _empty_frame(), {}

    try:
        with open(fpath) as f:
            artifact = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read liquidity artifact: %s", e)
        return _empty_frame(), {}

    if artifact.get("schema_version") != "1.0.0":
        logger.warning("Unsupported liquidity artifact schema: %s", artifact.get("schema_version"))
        return _empty_frame(), {}

    is_stale = artifact.get("is_stale", True)
    if is_stale:
        logger.warning("Liquidity artifact is stale (data_lag_days=%s)", artifact.get("data_lag_days"))

    ts = artifact.get("time_series", [])
    if not ts:
        logger.warning("Liquidity artifact has no time series data")
        return _empty_frame(), artifact

    df = pd.DataFrame(ts)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()

    # Build feature columns
    result = pd.DataFrame(index=df.index)
    result["liquidity_global_usd_t"] = pd.to_numeric(df["global_liquidity_usd_t"], errors="coerce")
    result["liquidity_m2_roc_3m"] = pd.to_numeric(df["m2_roc_3m"], errors="coerce")

    # One-hot encode regime
    regime = df["regime"]
    result["liquidity_regime_expanding"] = (regime == "EXPANDING").astype(float)
    result["liquidity_regime_neutral"] = (regime == "NEUTRAL").astype(float)
    result["liquidity_regime_contracting"] = (regime == "CONTRACTING").astype(float)

    return result, artifact


def merge_liquidity_features(hourly_df: pd.DataFrame, liquidity_df: pd.DataFrame) -> pd.DataFrame:
    """Merge monthly liquidity features into an hourly BTC DataFrame.

    Monthly values are forward-filled across hourly rows (each month's
    observation applies to all hours in that month and forward until the
    next observation).
    """
    result = hourly_df.copy()

    if liquidity_df.empty:
        for col in LIQUIDITY_FEATURE_COLUMNS:
            result[col] = float("nan")
        return result

    # Use merge_asof to align monthly data to hourly timestamps.
    # Each hourly row gets the most recent monthly observation <= its timestamp.
    liq = liquidity_df.copy().sort_index()
    result = result.sort_index()

    merged = pd.merge_asof(
        result,
        liq,
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Ensure all columns exist
    for col in LIQUIDITY_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")

    return merged


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LIQUIDITY_FEATURE_COLUMNS)
