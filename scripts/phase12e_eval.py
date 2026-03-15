"""Phase 12E: Microstructure (funding rate) evaluation and ablation."""

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


def run_eval(cross: bool, onchain: bool, micro: bool, label: str, model: str = "xgboost_direction") -> dict:
    print(f"\n--- {label} ({model}) ---")
    dataset, _ = build_dataset(include_crossasset=cross, include_onchain=onchain, include_microstructure=micro)
    dataset = add_horizon_targets(dataset, horizon=HORIZON)
    dataset = dataset.dropna(subset=[TARGET_COL])
    eval_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    print(f"Rows: {len(eval_dataset)}")

    result = walk_forward_evaluate(
        eval_dataset, model_name=model, target_column=TARGET_COL,
        train_window=DEFAULT_TRAIN_WINDOW, test_window=DEFAULT_TEST_WINDOW,
        output_dir="artifacts", output_stem=f"phase12e_{label}_{model}",
    )
    s = result.scores
    print(f"  Prec={s.get('precision',0):.4f} Rec={s.get('recall',0):.4f} ROC={s.get('roc_auc',0):.4f} Acc={s.get('directional_accuracy',0):.4f}")
    return {"label": label, "model": model, "scores": s, "windows": result.windows_evaluated}


def main():
    # Full features (cross + onchain + micro)
    full_xgb = run_eval(True, True, True, "full", "xgboost_direction")
    full_lgb = run_eval(True, True, True, "full", "lightgbm_direction")

    # Without microstructure
    no_micro_xgb = run_eval(True, True, False, "no_micro", "xgboost_direction")
    no_micro_lgb = run_eval(True, True, False, "no_micro", "lightgbm_direction")

    # Base only
    base_xgb = run_eval(False, False, False, "base", "xgboost_direction")
    base_lgb = run_eval(False, False, False, "base", "lightgbm_direction")

    print("\n=== Summary ===")
    all_results = [full_xgb, full_lgb, no_micro_xgb, no_micro_lgb, base_xgb, base_lgb]
    print(f"{'Label':<20} {'Model':<20} {'Prec':>6} {'Rec':>6} {'ROC':>6} {'Acc':>6}")
    print("-" * 70)
    for r in all_results:
        s = r["scores"]
        print(f"{r['label']:<20} {r['model']:<20} {s.get('precision',0):>6.4f} {s.get('recall',0):>6.4f} {s.get('roc_auc',0):>6.4f} {s.get('directional_accuracy',0):>6.4f}")

    results = {f"{r['label']}_{r['model']}": r for r in all_results}
    Path("artifacts/phase12e_ablation.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to artifacts/phase12e_ablation.json")


if __name__ == "__main__":
    main()
