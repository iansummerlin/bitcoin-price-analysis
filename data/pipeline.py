"""Dataset assembly, target generation, and metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from config import (
    BTC_PRICEDATA_PATH,
    BTC_SENTIMENTDATA_PATH,
    DATA_SCHEMA_VERSION,
    DATA_TIMEFRAME,
    FEATURE_SCHEMA_VERSION,
    MODELING_MARKET_SOURCE,
    TIMEZONE,
)
from data.loaders import load_price_history, load_sentiment_history
from evaluation.targets import add_targets
from features.pipeline import apply_feature_pipeline, feature_columns


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
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Build the canonical historical research dataset."""
    price = load_price_history(price_path)
    sentiment = load_sentiment_history(sentiment_path)

    dataset = price.join(sentiment, how="left")
    dataset["fng_value"] = dataset["fng_value"].ffill().fillna(50.0)
    dataset = apply_feature_pipeline(dataset, dropna=False)
    dataset = add_targets(dataset)
    dataset = dataset.dropna().sort_index()

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
        feature_columns=feature_columns(),
        price_path=str(Path(price_path).resolve()),
        sentiment_path=str(Path(sentiment_path).resolve()),
    )
    return dataset, metadata


def write_dataset_metadata(metadata: DatasetMetadata, output_path: str | Path) -> None:
    """Persist dataset metadata as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

