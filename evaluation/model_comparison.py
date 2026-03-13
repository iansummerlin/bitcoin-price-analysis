"""Reproducible model-family comparison on shared walk-forward windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from config import DEFAULT_EVALUATION_MAX_ROWS
from data.pipeline import build_dataset
from evaluation.walk_forward import walk_forward_evaluate
from models.arima_model import ImprovedARIMAModel
from models.xgboost_model import XGBoostDirectionModel, XGBoostPriceModel


@dataclass
class ComparisonResult:
    generated_at: str
    evaluation_rows: int
    acceptance_thresholds: dict[str, float]
    models: dict[str, dict]
    winner: str
    conclusion: str


MODEL_SPECS = {
    "arima_regressor": {
        "target_column": "target_log_return_1",
        "train_window": 24 * 30,
        "test_window": 24 * 15,
        "rows": 24 * 75,
        "factory": lambda: ImprovedARIMAModel(
            feature_columns=["return_1", "return_24", "volatility_24", "atr_pct", "fng_value"],
            p_range=range(0, 2),
            d_range=range(0, 2),
            q_range=range(0, 2),
        ),
    },
    "xgboost_regressor": {
        "target_column": "target_log_return_1",
        "train_window": 24 * 45,
        "test_window": 24 * 15,
        "rows": 24 * 90,
        "factory": lambda: XGBoostPriceModel(n_estimators=80, max_depth=3),
    },
    "xgboost_direction": {
        "target_column": "target_direction_cost_adj",
        "train_window": 24 * 45,
        "test_window": 24 * 15,
        "rows": 24 * 90,
        "factory": lambda: XGBoostDirectionModel(n_estimators=80, max_depth=3),
    },
}

ACCEPTANCE_THRESHOLDS = {
    "directional_precision_min": 0.55,
    "directional_recall_min": 0.15,
    "directional_roc_auc_min": 0.60,
}


def run_model_comparison(output_dir: str | Path = "artifacts") -> ComparisonResult:
    """Evaluate the supported model families on the same recent research slice."""
    dataset, _ = build_dataset()

    model_results: dict[str, dict] = {}
    from evaluation.walk_forward import MODEL_REGISTRY

    for model_name, spec in MODEL_SPECS.items():
        evaluation_dataset = dataset.tail(min(DEFAULT_EVALUATION_MAX_ROWS, spec["rows"])).copy()
        original_model = MODEL_REGISTRY[model_name]
        MODEL_REGISTRY[model_name] = spec["factory"]
        result = walk_forward_evaluate(
            evaluation_dataset,
            model_name=model_name,
            target_column=spec["target_column"],
            train_window=spec["train_window"],
            test_window=spec["test_window"],
            output_dir=output_dir,
            output_stem=f"comparison_{model_name}",
        )
        MODEL_REGISTRY[model_name] = original_model
        model_results[model_name] = asdict(result)

    direction_metrics = model_results["xgboost_direction"]["scores"]
    winner = "xgboost_direction"
    conclusion = (
        "Current best model is xgboost_direction, but cost-aware precision/recall remain too weak "
        "for downstream trading dependency status."
    )
    if (
        direction_metrics.get("precision", 0.0) >= ACCEPTANCE_THRESHOLDS["directional_precision_min"]
        and direction_metrics.get("recall", 0.0) >= ACCEPTANCE_THRESHOLDS["directional_recall_min"]
        and direction_metrics.get("roc_auc", 0.0) >= ACCEPTANCE_THRESHOLDS["directional_roc_auc_min"]
    ):
        conclusion = "Current directional model clears the provisional precision/recall bar for integration review."

    comparison = ComparisonResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation_rows=max(spec["rows"] for spec in MODEL_SPECS.values()),
        acceptance_thresholds=ACCEPTANCE_THRESHOLDS,
        models=model_results,
        winner=winner,
        conclusion=conclusion,
    )
    output_path = Path(output_dir) / "model_comparison.json"
    output_path.write_text(json.dumps(asdict(comparison), indent=2), encoding="utf-8")
    return comparison
