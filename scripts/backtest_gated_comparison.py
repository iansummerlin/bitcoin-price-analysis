"""Compare ungated vs liquidity-gated walk-forward evaluation.

Runs the standard walk-forward evaluation (identical to `make backtest`),
then applies the directional liquidity regime gate post-prediction and
re-scores the filtered subset. Reports both side by side.

The gate is applied post-prediction only — the model trains and predicts
on the full dataset. Gating determines which predictions are evaluated.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EVALUATION_MAX_ROWS,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
)
from data.pipeline import build_dataset, write_dataset_metadata
from evaluation.liquidity_gate import (
    DIRECTIONAL_REGIMES,
    apply_directional_gate,
    assign_regimes,
)
from evaluation.reporting import score_predictions
from evaluation.walk_forward import (
    _build_model,
    iter_walk_forward_slices,
    walk_forward_evaluate,
)

METRICS_KEYS = ["precision", "recall", "roc_auc", "directional_accuracy", "f1", "positive_rate"]


def main():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset (with liquidity features for gating regime data)
    print("Building dataset...")
    dataset, metadata = build_dataset(include_liquidity=True)
    write_dataset_metadata(metadata, artifacts_dir / "dataset_metadata.json")
    eval_ds = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    print(f"Evaluation rows: {len(eval_ds)}")
    print(f"Date range: {eval_ds.index.min()} to {eval_ds.index.max()}")

    # --- Standard walk-forward (ungated) ---
    print("\n=== Running standard walk-forward evaluation ===")
    result = walk_forward_evaluate(
        eval_ds,
        model_name=DEFAULT_CLASSIFICATION_MODEL,
        target_column=DEFAULT_TARGET_COLUMN,
        output_dir=artifacts_dir,
    )
    ungated_scores = result.scores
    print(f"Windows: {result.windows_evaluated}")

    # --- Gated evaluation: re-run walk-forward manually to collect per-row data ---
    print("\n=== Applying directional liquidity gate ===")

    all_actuals = []
    all_predictions = []
    gated_window_scores = []
    ungated_window_scores = []

    for window_id, (train_df, test_df) in enumerate(
        iter_walk_forward_slices(
            eval_ds,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            min_train_rows=min(DEFAULT_TRAIN_WINDOW, DEFAULT_MIN_TRAIN_ROWS),
        ),
        start=1,
    ):
        model = _build_model(DEFAULT_CLASSIFICATION_MODEL)
        model.fit(train_df, target_column=DEFAULT_TARGET_COLUMN)
        predictions = model.predict_frame(test_df)

        all_actuals.append(test_df)
        all_predictions.append(predictions)

        # Per-window ungated score
        prob_col = "probability" if "probability" in predictions.columns else None
        ungated_score = score_predictions(
            test_df, predictions, DEFAULT_TARGET_COLUMN, "prediction", prob_col
        )
        ungated_window_scores.append(ungated_score)

        # Per-window gated score
        gated_a, gated_p, coverage = apply_directional_gate(test_df, predictions)
        if len(gated_a) > 0:
            gated_prob_col = "probability" if "probability" in gated_p.columns else None
            gated_score = score_predictions(
                gated_a, gated_p, DEFAULT_TARGET_COLUMN, "prediction", gated_prob_col
            )
            gated_score["coverage"] = coverage
            gated_score["gated_rows"] = len(gated_a)
        else:
            gated_score = {k: 0.0 for k in METRICS_KEYS}
            gated_score["coverage"] = 0.0
            gated_score["gated_rows"] = 0
        gated_window_scores.append(gated_score)

        regime_dist = assign_regimes(test_df).value_counts().to_dict()
        print(f"  Window {window_id}: coverage={coverage:.1%}, regimes={regime_dist}")

    # Aggregate scores (mean across windows)
    ungated_mean = pd.DataFrame(ungated_window_scores).mean(numeric_only=True).to_dict()
    gated_mean = pd.DataFrame(gated_window_scores).mean(numeric_only=True).to_dict()

    # Also compute pooled scores across all rows
    pooled_actuals = pd.concat(all_actuals)
    pooled_predictions = pd.concat(all_predictions)
    pooled_prob_col = "probability" if "probability" in pooled_predictions.columns else None

    ungated_pooled = score_predictions(
        pooled_actuals, pooled_predictions, DEFAULT_TARGET_COLUMN, "prediction", pooled_prob_col
    )

    gated_a_pooled, gated_p_pooled, pooled_coverage = apply_directional_gate(
        pooled_actuals, pooled_predictions
    )
    gated_pooled = score_predictions(
        gated_a_pooled, gated_p_pooled, DEFAULT_TARGET_COLUMN, "prediction",
        "probability" if "probability" in gated_p_pooled.columns else None,
    )
    gated_pooled["coverage"] = pooled_coverage
    gated_pooled["signal_count"] = int(gated_p_pooled["prediction"].sum())
    ungated_pooled["signal_count"] = int(pooled_predictions["prediction"].sum())

    # --- Report ---
    print("\n" + "=" * 70)
    print("COMPARISON: Ungated vs Directional Liquidity Gate")
    print("=" * 70)

    print("\n--- Window-Mean Scores ---")
    col_w = 16
    header = f"{'Metric':<22} {'Ungated':>{col_w}} {'Gated':>{col_w}} {'Delta':>{col_w}}"
    print(header)
    print("-" * len(header))
    for key in METRICS_KEYS:
        u = ungated_mean.get(key, 0)
        g = gated_mean.get(key, 0)
        d = g - u
        print(f"{key:<22} {u:>{col_w}.4f} {g:>{col_w}.4f} {d:>+{col_w}.4f}")

    print(f"\n--- Pooled Scores (all rows across all windows) ---")
    print(header)
    print("-" * len(header))
    for key in METRICS_KEYS:
        u = ungated_pooled.get(key, 0)
        g = gated_pooled.get(key, 0)
        d = g - u
        print(f"{key:<22} {u:>{col_w}.4f} {g:>{col_w}.4f} {d:>+{col_w}.4f}")
    print(f"{'coverage':<22} {'100.0%':>{col_w}} {pooled_coverage:>{col_w}.1%}")
    print(f"{'signal_count':<22} {ungated_pooled['signal_count']:>{col_w}} {gated_pooled['signal_count']:>{col_w}}")

    # Verdict
    prec_delta = gated_pooled.get("precision", 0) - ungated_pooled.get("precision", 0)
    recall_delta = gated_pooled.get("recall", 0) - ungated_pooled.get("recall", 0)
    f1_delta = gated_pooled.get("f1", 0) - ungated_pooled.get("f1", 0)

    print(f"\n--- Verdict ---")
    print(f"Precision delta (pooled): {prec_delta:+.4f}")
    print(f"Recall delta (pooled):    {recall_delta:+.4f}")
    print(f"F1 delta (pooled):        {f1_delta:+.4f}")
    print(f"Coverage:                 {pooled_coverage:.1%}")

    if prec_delta > 0 and f1_delta > 0:
        print("Gate improves precision and F1. Keep as optional research filter.")
    elif prec_delta > 0:
        print("Gate improves precision with tradeoffs. Keep for further research.")
    else:
        print("Gate does not improve precision. Not recommended.")

    # Save results
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": DEFAULT_CLASSIFICATION_MODEL,
        "target_column": DEFAULT_TARGET_COLUMN,
        "evaluation_rows": len(eval_ds),
        "windows_evaluated": result.windows_evaluated,
        "gate_type": "directional",
        "allowed_regimes": sorted(DIRECTIONAL_REGIMES),
        "ungated": {
            "window_mean": {k: float(v) for k, v in ungated_mean.items()},
            "pooled": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                       for k, v in ungated_pooled.items()},
        },
        "gated": {
            "window_mean": {k: float(v) for k, v in gated_mean.items()},
            "pooled": {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                       for k, v in gated_pooled.items()},
            "coverage": float(pooled_coverage),
        },
        "deltas_pooled": {
            "precision": float(prec_delta),
            "recall": float(recall_delta),
            "f1": float(f1_delta),
        },
    }

    output_path = artifacts_dir / "backtest_gated_comparison.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
