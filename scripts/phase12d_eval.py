"""Phase 12D: LightGBM vs XGBoost comparison on expanded features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DEFAULT_EVALUATION_MAX_ROWS, DEFAULT_TEST_WINDOW, DEFAULT_TRAIN_WINDOW
from data.pipeline import build_dataset
from evaluation.targets import add_horizon_targets, horizon_target_column
from evaluation.walk_forward import walk_forward_evaluate


HORIZON = 4
TARGET_COL = horizon_target_column(HORIZON)


def main():
    print("Building dataset with full features...")
    dataset, metadata = build_dataset(include_crossasset=True, include_onchain=True)
    dataset = add_horizon_targets(dataset, horizon=HORIZON)
    dataset = dataset.dropna(subset=[TARGET_COL])
    eval_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    print(f"Evaluation rows: {len(eval_dataset)}")

    models = ["xgboost_direction", "lightgbm_direction"]
    results = {}

    for model_name in models:
        print(f"\n--- {model_name} ---")
        result = walk_forward_evaluate(
            eval_dataset,
            model_name=model_name,
            target_column=TARGET_COL,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            output_dir="artifacts",
            output_stem=f"phase12d_{model_name}",
        )
        scores = result.scores
        print(f"  Precision:  {scores.get('precision', 0):.4f}")
        print(f"  Recall:     {scores.get('recall', 0):.4f}")
        print(f"  ROC-AUC:    {scores.get('roc_auc', 0):.4f}")
        print(f"  Dir. Acc.:  {scores.get('directional_accuracy', 0):.4f}")
        print(f"  F1:         {scores.get('f1', 0):.4f}")
        results[model_name] = {"scores": scores, "windows": result.windows_evaluated}

    # Also run on base features only for comparison
    print("\n--- xgboost_direction (base features only) ---")
    base_dataset, _ = build_dataset(include_crossasset=False, include_onchain=False)
    base_dataset = add_horizon_targets(base_dataset, horizon=HORIZON)
    base_dataset = base_dataset.dropna(subset=[TARGET_COL])
    base_eval = base_dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    result = walk_forward_evaluate(
        base_eval,
        model_name="xgboost_direction",
        target_column=TARGET_COL,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
        output_dir="artifacts",
        output_stem="phase12d_xgb_base",
    )
    scores = result.scores
    print(f"  Precision:  {scores.get('precision', 0):.4f}")
    print(f"  Recall:     {scores.get('recall', 0):.4f}")
    print(f"  ROC-AUC:    {scores.get('roc_auc', 0):.4f}")
    print(f"  Dir. Acc.:  {scores.get('directional_accuracy', 0):.4f}")
    print(f"  F1:         {scores.get('f1', 0):.4f}")
    results["xgboost_base"] = {"scores": scores, "windows": result.windows_evaluated}

    print("\n=== Comparison ===")
    header = f"{'Metric':<20} {'XGB+Full':>10} {'LGB+Full':>10} {'XGB Base':>10}"
    print(header)
    print("-" * len(header))
    for key in ["precision", "recall", "roc_auc", "directional_accuracy", "f1"]:
        v1 = results["xgboost_direction"]["scores"].get(key, 0)
        v2 = results["lightgbm_direction"]["scores"].get(key, 0)
        v3 = results["xgboost_base"]["scores"].get(key, 0)
        print(f"{key:<20} {v1:>10.4f} {v2:>10.4f} {v3:>10.4f}")

    Path("artifacts/phase12d_comparison.json").write_text(json.dumps(results, indent=2))
    print("\nResults saved to artifacts/phase12d_comparison.json")


if __name__ == "__main__":
    main()
