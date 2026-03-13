"""Forecast and signal scoring helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Regression-oriented forecast metrics."""
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def directional_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    probabilities: pd.Series | None = None,
) -> dict[str, float]:
    """Classification-oriented directional metrics."""
    metrics = {
        "directional_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(np.mean(y_pred)),
    }
    if probabilities is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return metrics


def score_predictions(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    probability_column: str | None = None,
) -> dict[str, float]:
    """Unified model/baseline scoring entry point."""
    aligned = actuals.join(predictions[[prediction_column] + ([probability_column] if probability_column else [])], how="inner")
    required_columns = [target_column, prediction_column]
    if probability_column:
        required_columns.append(probability_column)
    aligned = aligned.dropna(subset=required_columns)
    y_true = aligned[target_column]
    y_pred = aligned[prediction_column]

    if set(pd.Series(y_true).dropna().unique()).issubset({0, 1}) and set(pd.Series(y_pred).dropna().unique()).issubset({0, 1}):
        probabilities = aligned[probability_column] if probability_column else None
        return directional_metrics(y_true, y_pred, probabilities=probabilities)
    return forecast_metrics(y_true, y_pred)
