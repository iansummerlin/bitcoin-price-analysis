"""Phase 12A: Multi-horizon walk-forward analysis.

Evaluates 1h, 4h, and 24h prediction horizons with the existing feature set
to determine which horizon to optimize for in the rest of Phase 12.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EVALUATION_MAX_ROWS,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    HORIZON_CONFIGS,
)
from data.pipeline import build_dataset
from evaluation.targets import add_horizon_targets, horizon_target_column
from evaluation.walk_forward import walk_forward_evaluate


def run_horizon_analysis() -> dict:
    """Run walk-forward evaluation on 1h, 4h, and 24h horizons."""
    print("Building dataset...")
    dataset, metadata = build_dataset()

    horizons = [1, 4, 24]
    results = {}

    # Add all horizon targets to the dataset
    for h in horizons:
        dataset = add_horizon_targets(dataset, horizon=h)

    # Use only rows where all horizons have valid targets
    eval_dataset = dataset.dropna(
        subset=[horizon_target_column(h) for h in horizons]
    )
    eval_dataset = eval_dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    print(f"Evaluation dataset: {len(eval_dataset)} rows")
    print(f"Date range: {eval_dataset.index.min()} to {eval_dataset.index.max()}")
    print()

    for h in horizons:
        target_col = horizon_target_column(h)
        positive_rate = eval_dataset[target_col].mean()
        print(f"--- Horizon {h}h ---")
        print(f"Target column: {target_col}")
        print(f"Positive rate: {positive_rate:.4f}")

        result = walk_forward_evaluate(
            eval_dataset,
            model_name=DEFAULT_CLASSIFICATION_MODEL,
            target_column=target_col,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            output_dir="artifacts",
            output_stem=f"horizon_{h}h",
        )

        scores = result.scores
        print(f"  Precision:  {scores.get('precision', 0):.4f}")
        print(f"  Recall:     {scores.get('recall', 0):.4f}")
        print(f"  ROC-AUC:    {scores.get('roc_auc', 0):.4f}")
        print(f"  Dir. Acc.:  {scores.get('directional_accuracy', 0):.4f}")
        print(f"  F1:         {scores.get('f1', 0):.4f}")
        print(f"  Windows:    {result.windows_evaluated}")
        print()

        results[f"{h}h"] = {
            "horizon": h,
            "target_column": target_col,
            "positive_rate": float(positive_rate),
            "windows_evaluated": result.windows_evaluated,
            "scores": scores,
            "baseline_scores": result.baseline_scores,
        }

    # Probability threshold analysis on each horizon
    print("=== Probability Threshold Analysis ===")
    print()
    for h in horizons:
        target_col = horizon_target_column(h)
        pred_path = Path("artifacts") / f"horizon_{h}h_predictions.csv"
        if not pred_path.exists():
            continue

        preds = pd.read_csv(pred_path, index_col=0, parse_dates=True)
        if "probability" not in preds.columns:
            continue

        # Merge with actual targets
        actuals = eval_dataset[target_col].reindex(preds.index).dropna()
        valid = preds.loc[actuals.index]

        if len(valid) == 0:
            continue

        y_true = actuals.values
        y_prob = valid["probability"].values

        print(f"--- Horizon {h}h threshold sweep ---")
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        threshold_results = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            n_predicted = int(y_pred.sum())
            row = {
                "threshold": t,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "n_predicted": n_predicted,
                "n_correct": int(tp),
            }
            threshold_results.append(row)
            print(f"  t={t:.1f}  prec={precision:.4f}  rec={recall:.4f}  n={n_predicted}")

        results[f"{h}h"]["threshold_analysis"] = threshold_results
        print()

    # Summary
    print("=== Summary ===")
    best_horizon = None
    best_roc = -1.0
    for key, res in results.items():
        roc = res["scores"].get("roc_auc", 0)
        prec = res["scores"].get("precision", 0)
        rec = res["scores"].get("recall", 0)
        print(f"{key}: ROC-AUC={roc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}")
        if roc > best_roc:
            best_roc = roc
            best_horizon = key

    print(f"\nBest ROC-AUC: {best_horizon} ({best_roc:.4f})")

    # Save results
    output_path = Path("artifacts") / "horizon_analysis.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_horizon_analysis()
