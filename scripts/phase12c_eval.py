"""Phase 12C: On-chain data evaluation and ablation.

Runs walk-forward on the 4h horizon with on-chain features included vs excluded,
holding cross-asset features constant (enabled).
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


def run_eval(include_crossasset: bool, include_onchain: bool, label: str) -> dict:
    print(f"\n=== {label} ===")
    print(f"Building dataset (cross={include_crossasset}, onchain={include_onchain})...")
    dataset, metadata = build_dataset(include_crossasset=include_crossasset, include_onchain=include_onchain)
    dataset = add_horizon_targets(dataset, horizon=HORIZON)
    dataset = dataset.dropna(subset=[TARGET_COL])
    eval_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    print(f"Evaluation rows: {len(eval_dataset)}")

    result = walk_forward_evaluate(
        eval_dataset,
        model_name=DEFAULT_CLASSIFICATION_MODEL,
        target_column=TARGET_COL,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
        output_dir="artifacts",
        output_stem=f"phase12c_{label}",
    )

    scores = result.scores
    print(f"  Precision:  {scores.get('precision', 0):.4f}")
    print(f"  Recall:     {scores.get('recall', 0):.4f}")
    print(f"  ROC-AUC:    {scores.get('roc_auc', 0):.4f}")
    print(f"  Dir. Acc.:  {scores.get('directional_accuracy', 0):.4f}")
    print(f"  F1:         {scores.get('f1', 0):.4f}")

    return {"label": label, "scores": scores, "windows": result.windows_evaluated}


def main():
    # With everything (cross-asset + on-chain)
    full = run_eval(True, True, "full")

    # Without on-chain (cross-asset only = 12B baseline)
    no_onchain = run_eval(True, False, "no_onchain")

    # Without cross-asset (on-chain only)
    no_cross = run_eval(False, True, "no_crossasset")

    # Base (neither)
    base = run_eval(False, False, "base_only")

    print("\n=== Comparison ===")
    header = f"{'Metric':<20} {'Full':>8} {'NoCross':>8} {'NoOnch':>8} {'Base':>8}"
    print(header)
    print("-" * len(header))
    for key in ["precision", "recall", "roc_auc", "directional_accuracy", "f1"]:
        vals = [r["scores"].get(key, 0) for r in [full, no_cross, no_onchain, base]]
        print(f"{key:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    results = {r["label"]: r for r in [full, no_onchain, no_cross, base]}
    Path("artifacts/phase12c_ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nResults saved to artifacts/phase12c_ablation.json")


if __name__ == "__main__":
    main()
