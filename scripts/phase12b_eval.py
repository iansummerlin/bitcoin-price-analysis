"""Phase 12B: Cross-asset evaluation and ablation.

Runs walk-forward on the 4h horizon with cross-asset features included,
then without, to isolate their contribution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_CLASSIFICATION_MODEL, DEFAULT_EVALUATION_MAX_ROWS, DEFAULT_TEST_WINDOW, DEFAULT_TRAIN_WINDOW
from data.pipeline import build_dataset
from evaluation.targets import add_horizon_targets, horizon_target_column
from evaluation.walk_forward import walk_forward_evaluate


HORIZON = 4
TARGET_COL = horizon_target_column(HORIZON)


def run_evaluation(include_crossasset: bool, label: str) -> dict:
    """Build dataset and run walk-forward evaluation."""
    print(f"\n=== {label} ===")
    print(f"Building dataset (include_crossasset={include_crossasset})...")
    dataset, metadata = build_dataset(include_crossasset=include_crossasset)

    # Add horizon targets
    dataset = add_horizon_targets(dataset, horizon=HORIZON)
    dataset = dataset.dropna(subset=[TARGET_COL])
    eval_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    print(f"Evaluation rows: {len(eval_dataset)}")
    print(f"Date range: {eval_dataset.index.min()} to {eval_dataset.index.max()}")

    result = walk_forward_evaluate(
        eval_dataset,
        model_name=DEFAULT_CLASSIFICATION_MODEL,
        target_column=TARGET_COL,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
        output_dir="artifacts",
        output_stem=f"phase12b_{label}",
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
        "include_crossasset": include_crossasset,
        "scores": scores,
        "windows_evaluated": result.windows_evaluated,
        "dataset_rows": len(eval_dataset),
    }


def main():
    # Run with cross-asset features
    with_cross = run_evaluation(include_crossasset=True, label="with_crossasset")

    # Run without cross-asset features (ablation)
    without_cross = run_evaluation(include_crossasset=False, label="without_crossasset")

    # Compare
    print("\n=== Ablation Comparison ===")
    print(f"{'Metric':<20} {'With Cross':>12} {'Without Cross':>14} {'Delta':>10}")
    print("-" * 60)
    for key in ["precision", "recall", "roc_auc", "directional_accuracy", "f1"]:
        with_val = with_cross["scores"].get(key, 0)
        without_val = without_cross["scores"].get(key, 0)
        delta = with_val - without_val
        sign = "+" if delta >= 0 else ""
        print(f"{key:<20} {with_val:>12.4f} {without_val:>14.4f} {sign}{delta:>9.4f}")

    results = {"with_crossasset": with_cross, "without_crossasset": without_cross}
    output_path = Path("artifacts") / "phase12b_ablation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
