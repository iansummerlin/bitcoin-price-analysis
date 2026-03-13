"""Versioned downstream signal export contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from config import FEATURE_SCHEMA_VERSION, MODEL_STALE_AFTER_DAYS, SIGNAL_SCHEMA_VERSION


@dataclass
class SignalArtifact:
    timestamp: str
    instrument: str
    model_version: str
    feature_schema_version: str
    signal_schema_version: str
    prediction: float
    probability: float | None
    actionable: bool
    generated_at: str


def export_latest_signal(
    predictions: pd.DataFrame,
    output_path: str | Path,
    instrument: str,
    model_version: str,
) -> SignalArtifact:
    """Export the latest model prediction in a self-describing JSON contract."""
    if predictions.empty:
        raise ValueError("No predictions available for export")

    latest = predictions.sort_index().iloc[-1]
    artifact = SignalArtifact(
        timestamp=str(predictions.sort_index().index[-1].isoformat()),
        instrument=instrument,
        model_version=model_version,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        signal_schema_version=SIGNAL_SCHEMA_VERSION,
        prediction=float(latest["prediction"]),
        probability=float(latest["probability"]) if "probability" in latest and pd.notna(latest["probability"]) else None,
        actionable=bool(latest.get("actionable", int(float(latest["prediction"]) > 0))),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")
    return artifact


def validate_signal_artifact(path: str | Path) -> dict:
    """Validate freshness and required fields for a signal artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required_fields = {
        "timestamp",
        "instrument",
        "model_version",
        "feature_schema_version",
        "signal_schema_version",
        "prediction",
        "generated_at",
    }
    missing = required_fields - payload.keys()
    if missing:
        raise ValueError(f"Signal artifact missing required fields: {sorted(missing)}")

    generated_at = datetime.fromisoformat(payload["generated_at"])
    age_days = (datetime.now(timezone.utc) - generated_at).days
    payload["is_stale"] = age_days > MODEL_STALE_AFTER_DAYS
    payload["age_days"] = age_days
    return payload
