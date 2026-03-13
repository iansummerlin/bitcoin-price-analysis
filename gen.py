"""Training entrypoint for the canonical signal research dataset."""

from __future__ import annotations

import json
from pathlib import Path

from config import DEFAULT_CLASSIFICATION_MODEL, DEFAULT_TARGET_COLUMN
from data.pipeline import build_dataset, write_dataset_metadata
from evaluation.walk_forward import MODEL_REGISTRY


def main() -> None:
    dataset, metadata = build_dataset()
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(metadata, artifacts_dir / "dataset_metadata.json")

    model = MODEL_REGISTRY[DEFAULT_CLASSIFICATION_MODEL](n_estimators=100, max_depth=3)
    fit_metrics = model.fit(dataset, target_column=DEFAULT_TARGET_COLUMN)
    model_path = artifacts_dir / f"{DEFAULT_CLASSIFICATION_MODEL}.joblib"
    model.save(model_path)

    summary = {
        "trained_model": DEFAULT_CLASSIFICATION_MODEL,
        "target_column": DEFAULT_TARGET_COLUMN,
        "dataset_rows": metadata.row_count,
        "model_path": str(model_path.resolve()),
        "fit_metrics": fit_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
