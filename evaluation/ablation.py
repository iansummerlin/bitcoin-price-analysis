"""Feature-family ablation reporting."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

from config import DEFAULT_EVALUATION_MAX_ROWS
from data.pipeline import build_dataset
from evaluation.walk_forward import walk_forward_evaluate
from models.xgboost_model import XGBoostDirectionModel


FEATURE_GROUPS = {
    "core_price": [
        "close_lag_1",
        "close_lag_6",
        "close_lag_24",
        "close_lag_90",
        "close_lag_120",
        "return_1",
        "return_24",
        "log_return_1",
    ],
    "trend": ["MA_7", "MA_24", "MA_72", "ma_spread_7_24", "ma_spread_24_72", "trend_regime_24", "trend_regime_72"],
    "volatility": ["volatility_24", "volatility_ewma_24", "parkinson_volatility", "atr_24", "atr_pct"],
    "volume": ["volume_btc_zscore_24", "volume_usd_zscore_24"],
    "sentiment": ["fng_value"],
    "momentum": ["rsi"],
}


@dataclass
class AblationResult:
    generated_at: str
    evaluation_rows: int
    scores: dict[str, dict]


ABLATION_ROWS = min(DEFAULT_EVALUATION_MAX_ROWS, 24 * 240)
ABLATION_TRAIN_WINDOW = 24 * 90
ABLATION_TEST_WINDOW = 24 * 30


def run_ablation_report(output_dir: str | Path = "artifacts") -> AblationResult:
    """Run directional walk-forward evaluation on cumulative feature families."""
    dataset, _ = build_dataset()
    evaluation_dataset = dataset.tail(ABLATION_ROWS).copy()

    scores: dict[str, dict] = {}
    selected_features: list[str] = []

    for family_name, features in FEATURE_GROUPS.items():
        selected_features.extend(features)
        def factory(n_estimators=100, max_depth=3, family_features=list(selected_features)):
            return XGBoostDirectionModel(
                feature_columns=family_features,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )
        # Inline copy of walk_forward with custom model instance would be overkill.
        from evaluation.walk_forward import MODEL_REGISTRY
        original = MODEL_REGISTRY["xgboost_direction"]
        MODEL_REGISTRY["xgboost_direction"] = factory
        try:
            result = walk_forward_evaluate(
                evaluation_dataset,
                model_name="xgboost_direction",
                target_column="target_direction_cost_adj",
                train_window=ABLATION_TRAIN_WINDOW,
                test_window=ABLATION_TEST_WINDOW,
                output_dir=output_dir,
                output_stem=f"ablation_{family_name}",
            )
        finally:
            MODEL_REGISTRY["xgboost_direction"] = original
        scores[family_name] = result.scores

    report = AblationResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        evaluation_rows=len(evaluation_dataset),
        scores=scores,
    )
    output_path = Path(output_dir) / "feature_ablation.json"
    output_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return report
