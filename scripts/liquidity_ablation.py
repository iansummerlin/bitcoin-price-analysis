"""Liquidity feature family ablation.

Runs walk-forward evaluation with and without liquidity features to
determine whether the global liquidity data adds incremental value.
Uses the canonical 1h target and default model (same as `make backtest`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EVALUATION_MAX_ROWS,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
)
from data.pipeline import build_dataset
from evaluation.walk_forward import walk_forward_evaluate


def run_evaluation(include_liquidity: bool, label: str) -> dict:
    """Build dataset and run walk-forward evaluation."""
    print(f"\n=== {label} ===")
    print(f"Building dataset (include_liquidity={include_liquidity})...")
    dataset, metadata = build_dataset(include_liquidity=include_liquidity)

    eval_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    print(f"Evaluation rows: {len(eval_dataset)}")
    print(f"Date range: {eval_dataset.index.min()} to {eval_dataset.index.max()}")

    result = walk_forward_evaluate(
        eval_dataset,
        model_name=DEFAULT_CLASSIFICATION_MODEL,
        target_column=DEFAULT_TARGET_COLUMN,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
        output_dir="artifacts",
        output_stem=f"liquidity_ablation_{label}",
    )

    scores = result.scores
    print(f"  Precision:  {scores.get('precision', 0):.4f}")
    print(f"  Recall:     {scores.get('recall', 0):.4f}")
    print(f"  ROC-AUC:    {scores.get('roc_auc', 0):.4f}")
    print(f"  Dir. Acc.:  {scores.get('directional_accuracy', 0):.4f}")
    print(f"  F1:         {scores.get('f1', 0):.4f}")
    print(f"  Windows:    {result.windows_evaluated}")

    return {
        "label": label,
        "include_liquidity": include_liquidity,
        "scores": scores,
        "windows_evaluated": result.windows_evaluated,
        "dataset_rows": len(eval_dataset),
    }


def main():
    results = {}

    # Run with liquidity features
    results["with_liquidity"] = run_evaluation(
        include_liquidity=True, label="with_liquidity"
    )

    # Run without liquidity features (ablation)
    results["without_liquidity"] = run_evaluation(
        include_liquidity=False, label="without_liquidity"
    )

    # Compare
    w = results["with_liquidity"]["scores"]
    wo = results["without_liquidity"]["scores"]

    print("\n=== Liquidity Ablation Comparison ===")
    print(f"{'Metric':<20} {'With Liq':>12} {'Without Liq':>14} {'Delta':>10}")
    print("-" * 60)
    for key in ["precision", "recall", "roc_auc", "directional_accuracy", "f1"]:
        with_val = w.get(key, 0)
        without_val = wo.get(key, 0)
        delta = with_val - without_val
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {with_val:>12.4f} {without_val:>14.4f} {sign}{delta:>9.4f}")

    # Verdict
    prec_delta = w.get("precision", 0) - wo.get("precision", 0)
    recall_delta = w.get("recall", 0) - wo.get("recall", 0)
    auc_delta = w.get("roc_auc", 0) - wo.get("roc_auc", 0)

    print("\n=== Verdict ===")
    improved = prec_delta >= 0.01 or recall_delta >= 0.01 or auc_delta >= 0.01
    degraded = prec_delta < -0.005 or recall_delta < -0.005 or auc_delta < -0.005

    if improved and not degraded:
        verdict = "INCREMENTALLY USEFUL"
    elif degraded and not improved:
        verdict = "HARMFUL"
    elif improved and degraded:
        verdict = "TRADEOFF (mixed)"
    else:
        verdict = "NEUTRAL"

    print(f"Liquidity feature family: {verdict}")
    print(f"  Precision delta: {prec_delta:+.4f}")
    print(f"  Recall delta:    {recall_delta:+.4f}")
    print(f"  ROC-AUC delta:   {auc_delta:+.4f}")

    results["verdict"] = verdict
    results["deltas"] = {
        "precision": prec_delta,
        "recall": recall_delta,
        "roc_auc": auc_delta,
    }

    output_path = Path("artifacts") / "liquidity_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
