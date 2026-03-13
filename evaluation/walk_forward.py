"""Strict walk-forward model evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
)
from evaluation.baselines import add_baseline_predictions
from evaluation.reporting import score_predictions
from models.arima_model import ImprovedARIMAModel
from models.xgboost_model import XGBoostDirectionModel, XGBoostPriceModel


MODEL_REGISTRY = {
    "arima_regressor": ImprovedARIMAModel,
    "xgboost_regressor": XGBoostPriceModel,
    "xgboost_direction": XGBoostDirectionModel,
}


@dataclass
class WalkForwardResult:
    model_name: str
    target_column: str
    train_window: int
    test_window: int
    windows_evaluated: int
    scores: dict[str, float]
    baseline_scores: dict[str, dict[str, float]]
    prediction_path: str
    generated_at: str


def iter_walk_forward_slices(
    df: pd.DataFrame,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
    min_train_rows: int | None = DEFAULT_MIN_TRAIN_ROWS,
):
    """Yield strict train/test slices."""
    required_train_rows = min_train_rows if min_train_rows is not None else min(train_window, DEFAULT_MIN_TRAIN_ROWS)
    required_train_rows = min(required_train_rows, train_window)
    start = 0
    while start + train_window + test_window <= len(df):
        train = df.iloc[start : start + train_window]
        test = df.iloc[start + train_window : start + train_window + test_window]
        if len(train) >= required_train_rows:
            yield train, test
        start += test_window


def _build_model(model_name: str):
    model_cls = MODEL_REGISTRY[model_name]
    if not isinstance(model_cls, type):
        return model_cls()
    if model_name == "arima_regressor":
        return model_cls(p_range=range(0, 2), d_range=range(0, 2), q_range=range(0, 2))
    return model_cls(n_estimators=100, max_depth=3)


def walk_forward_evaluate(
    df: pd.DataFrame,
    model_name: str = DEFAULT_CLASSIFICATION_MODEL,
    target_column: str = DEFAULT_TARGET_COLUMN,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
    output_dir: str | Path = "artifacts",
    output_stem: str | None = None,
) -> WalkForwardResult:
    """Run strict walk-forward evaluation and persist predictions."""
    prediction_rows: list[pd.DataFrame] = []
    window_scores: list[dict[str, float]] = []
    baseline_scores_accumulator: dict[str, list[dict[str, float]]] = {}

    for window_id, (train_df, test_df) in enumerate(
        iter_walk_forward_slices(
            df,
            train_window=train_window,
            test_window=test_window,
            min_train_rows=min(train_window, DEFAULT_MIN_TRAIN_ROWS),
        ),
        start=1,
    ):
        model = _build_model(model_name)
        fit_metrics = model.fit(train_df, target_column=target_column)
        predictions = model.predict_frame(test_df)
        predictions["window_id"] = window_id
        predictions["fit_model_name"] = model_name
        predictions["fit_target_column"] = target_column
        prediction_rows.append(predictions)

        probability_column = "probability" if "probability" in predictions.columns else None
        score = score_predictions(test_df, predictions, target_column, "prediction", probability_column)
        score.update({"window_id": float(window_id)})
        score.update({f"fit_{key}": float(value) for key, value in fit_metrics.items() if isinstance(value, (int, float))})
        window_scores.append(score)

        baselines = add_baseline_predictions(test_df)
        baseline_specs = {
            "persistence_price": ("target_close_next", "baseline_persistence_price", None),
            "zero_return": ("target_log_return_1", "baseline_zero_return", None),
            "momentum_return": ("target_log_return_1", "baseline_momentum_return", None),
            "mean_reversion_return": ("target_log_return_1", "baseline_mean_reversion_return", None),
            "momentum_direction": ("target_direction_cost_adj", "baseline_momentum_direction", None),
            "mean_reversion_direction": ("target_direction_cost_adj", "baseline_mean_reversion_direction", None),
        }
        for name, (baseline_target, prediction_column, probability_column) in baseline_specs.items():
            baseline_prediction_frame = baselines[[prediction_column]].rename(columns={prediction_column: "prediction"})
            baseline_prediction_frame["probability"] = baseline_prediction_frame["prediction"]
            score_dict = score_predictions(
                baselines,
                baseline_prediction_frame,
                baseline_target,
                "prediction",
                "probability" if baseline_target.startswith("target_direction") else None,
            )
            baseline_scores_accumulator.setdefault(name, []).append(score_dict)

    if not prediction_rows:
        raise ValueError("Not enough rows for walk-forward evaluation")

    prediction_frame = pd.concat(prediction_rows).sort_index()
    stem = output_stem or model_name
    output_path = Path(output_dir) / f"{stem}_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path)

    mean_scores = pd.DataFrame(window_scores).drop(columns=["window_id"]).mean(numeric_only=True).to_dict()
    mean_baseline_scores = {
        name: pd.DataFrame(scores).mean(numeric_only=True).to_dict()
        for name, scores in baseline_scores_accumulator.items()
    }

    summary = WalkForwardResult(
        model_name=model_name,
        target_column=target_column,
        train_window=train_window,
        test_window=test_window,
        windows_evaluated=len(window_scores),
        scores={key: float(value) for key, value in mean_scores.items()},
        baseline_scores={name: {k: float(v) for k, v in metrics.items()} for name, metrics in mean_baseline_scores.items()},
        prediction_path=str(output_path.resolve()),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    return summary
