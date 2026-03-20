"""Dataset assembly, target generation, and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import logging

import pandas as pd

from config import (
    BTC_PRICEDATA_PATH,
    BTC_SENTIMENTDATA_PATH,
    CROSSASSET_COLUMNS,
    DATA_SCHEMA_VERSION,
    DATA_TIMEFRAME,
    EXOG_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    LIQUIDITY_COLUMNS,
    MICROSTRUCTURE_COLUMNS,
    MODELING_MARKET_SOURCE,
    ONCHAIN_COLUMNS,
    TIMEZONE,
)
from data.loaders import load_price_history, load_sentiment_history
from evaluation.targets import add_targets
from features.pipeline import apply_feature_pipeline, active_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    generated_at: str
    timeframe: str
    timezone: str
    market_source: str
    data_schema_version: str
    feature_schema_version: str
    row_count: int
    start: str
    end: str
    feature_columns: list[str]
    price_path: str
    sentiment_path: str


def build_dataset(
    price_path: str | Path = BTC_PRICEDATA_PATH,
    sentiment_path: str | Path = BTC_SENTIMENTDATA_PATH,
    include_crossasset: bool = True,
    include_onchain: bool = True,
    include_microstructure: bool = True,
    include_liquidity: bool = True,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Build the canonical historical research dataset.

    Parameters
    ----------
    include_crossasset : bool
        If True, load cross-asset data and compute features. Set to False
        to build the dataset with only price/sentiment features (useful
        for ablation and fallback).
    include_onchain : bool
        If True, load on-chain data and compute features.
    include_microstructure : bool
        If True, load funding rate data and compute features.
    include_liquidity : bool
        If True, load global liquidity artifact and merge features.
    """
    price = load_price_history(price_path)
    sentiment = load_sentiment_history(sentiment_path)

    dataset = price.join(sentiment, how="left")
    dataset["fng_value"] = dataset["fng_value"].ffill().fillna(50.0)
    dataset = apply_feature_pipeline(dataset, dropna=False)

    # Cross-asset features (Phase 12B)
    if include_crossasset:
        try:
            from data.crossasset import load_cross_asset_data
            from features.crossasset import compute_cross_asset_features

            cross_data = load_cross_asset_data()
            if not cross_data.empty:
                dataset = compute_cross_asset_features(dataset, cross_data)
                logger.info("Cross-asset features added (%d columns)", len(CROSSASSET_COLUMNS))
            else:
                logger.warning("Cross-asset data empty — features will be NaN")
                for col in CROSSASSET_COLUMNS:
                    dataset[col] = float("nan")
        except Exception as e:
            logger.warning("Failed to load cross-asset data: %s — continuing without", e)
            for col in CROSSASSET_COLUMNS:
                dataset[col] = float("nan")
    else:
        # Fill with 0 (neutral) when cross-asset data is excluded —
        # preserves column schema so models don't error on missing features.
        for col in CROSSASSET_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = 0.0

    # On-chain features (Phase 12C)
    if include_onchain:
        try:
            from data.onchain import load_onchain_data
            from features.onchain import compute_onchain_features

            onchain_data = load_onchain_data()
            if not onchain_data.empty:
                dataset = compute_onchain_features(dataset, onchain_data)
                logger.info("On-chain features added (%d columns)", len(ONCHAIN_COLUMNS))
            else:
                logger.warning("On-chain data empty — features will be zero")
                for col in ONCHAIN_COLUMNS:
                    dataset[col] = 0.0
        except Exception as e:
            logger.warning("Failed to load on-chain data: %s — continuing without", e)
            for col in ONCHAIN_COLUMNS:
                dataset[col] = 0.0
    else:
        for col in ONCHAIN_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = 0.0

    # Microstructure features (Phase 12E)
    if include_microstructure:
        try:
            from data.microstructure import load_funding_rate_data
            from features.microstructure import compute_microstructure_features

            funding_data = load_funding_rate_data()
            if not funding_data.empty:
                dataset = compute_microstructure_features(dataset, funding_data)
                logger.info("Microstructure features added (%d columns)", len(MICROSTRUCTURE_COLUMNS))
            else:
                logger.warning("Funding rate data empty — features will be zero")
                for col in MICROSTRUCTURE_COLUMNS:
                    dataset[col] = 0.0
        except Exception as e:
            logger.warning("Failed to load microstructure data: %s — continuing without", e)
            for col in MICROSTRUCTURE_COLUMNS:
                dataset[col] = 0.0
    else:
        for col in MICROSTRUCTURE_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = 0.0

    # Liquidity features (global-liquidity-analysis integration)
    if include_liquidity:
        try:
            from data.liquidity import load_liquidity_artifact, merge_liquidity_features

            liq_df, liq_meta = load_liquidity_artifact()
            if not liq_df.empty:
                dataset = merge_liquidity_features(dataset, liq_df)
                logger.info("Liquidity features added (%d columns)", len(LIQUIDITY_COLUMNS))
                if liq_meta.get("is_stale"):
                    logger.warning("Liquidity artifact is stale — features may be outdated")
            else:
                logger.warning("Liquidity data empty — features will be NaN")
                for col in LIQUIDITY_COLUMNS:
                    dataset[col] = float("nan")
        except Exception as e:
            logger.warning("Failed to load liquidity data: %s — continuing without", e)
            for col in LIQUIDITY_COLUMNS:
                dataset[col] = float("nan")
    else:
        for col in LIQUIDITY_COLUMNS:
            if col not in dataset.columns:
                dataset[col] = 0.0

    dataset = add_targets(dataset)

    # Drop rows with NaN in any active feature column
    all_feature_cols = active_feature_columns()
    drop_cols = [c for c in all_feature_cols if c in dataset.columns]
    dataset = dataset.dropna(subset=drop_cols).sort_index()

    feat_cols = [c for c in all_feature_cols if c in dataset.columns]

    metadata = DatasetMetadata(
        generated_at=datetime.now(timezone.utc).isoformat(),
        timeframe=DATA_TIMEFRAME,
        timezone=TIMEZONE,
        market_source=MODELING_MARKET_SOURCE,
        data_schema_version=DATA_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        row_count=len(dataset),
        start=dataset.index.min().isoformat(),
        end=dataset.index.max().isoformat(),
        feature_columns=feat_cols,
        price_path=str(Path(price_path).resolve()),
        sentiment_path=str(Path(sentiment_path).resolve()),
    )
    return dataset, metadata


def write_dataset_metadata(metadata: DatasetMetadata, output_path: str | Path) -> None:
    """Persist dataset metadata as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
