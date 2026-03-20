"""Liquidity regime gating experiment.

Tests whether suppressing model signals during weak macro regimes
improves precision and overall usefulness. The gate is applied
post-prediction — the model trains and predicts normally, then
predictions outside allowed regimes are suppressed.

Gating variants:
  ungated:              all predictions scored (baseline)
  directional:          allow EXPANDING + CONTRACTING, suppress NEUTRAL
  expanding_only:       allow EXPANDING only
  contracting_only:     allow CONTRACTING only

For each variant, we report:
  - metrics on the gated-in subset only (honest: "when you trade, how good?")
  - coverage: fraction of evaluation rows where the gate is active
  - signal_count: how many positive predictions survive the gate
"""

from __future__ import annotations

import json
import sys
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
from data.pipeline import build_dataset
from evaluation.reporting import score_predictions
from evaluation.walk_forward import _build_model, iter_walk_forward_slices

METRICS_KEYS = ["precision", "recall", "roc_auc", "directional_accuracy", "f1", "positive_rate"]

# Gate definitions: {name: set of allowed regimes}
GATE_CONFIGS = {
    "ungated": {"EXPANDING", "NEUTRAL", "CONTRACTING", "UNKNOWN"},
    "directional": {"EXPANDING", "CONTRACTING"},
    "expanding_only": {"EXPANDING"},
    "contracting_only": {"CONTRACTING"},
}


def row_regime(row: pd.Series) -> str:
    """Return the liquidity regime for a single row."""
    if row.get("liquidity_regime_expanding", 0) == 1.0:
        return "EXPANDING"
    if row.get("liquidity_regime_contracting", 0) == 1.0:
        return "CONTRACTING"
    if row.get("liquidity_regime_neutral", 0) == 1.0:
        return "NEUTRAL"
    return "UNKNOWN"


def assign_regimes(df: pd.DataFrame) -> pd.Series:
    """Vectorized regime assignment for all rows."""
    regime = pd.Series("UNKNOWN", index=df.index)
    if "liquidity_regime_expanding" in df.columns:
        regime = regime.where(df["liquidity_regime_expanding"] != 1.0, "EXPANDING")
    if "liquidity_regime_contracting" in df.columns:
        regime = regime.where(df["liquidity_regime_contracting"] != 1.0, "CONTRACTING")
    if "liquidity_regime_neutral" in df.columns:
        regime = regime.where(df["liquidity_regime_neutral"] != 1.0, "NEUTRAL")
    return regime


def score_gated(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    regimes: pd.Series,
    allowed_regimes: set[str],
    target_column: str,
) -> dict:
    """Score predictions on the subset where the regime gate is active."""
    mask = regimes.isin(allowed_regimes)
    gated_actuals = actuals.loc[mask]
    gated_preds = predictions.loc[mask]

    total_rows = len(actuals)
    gated_rows = len(gated_actuals)
    coverage = gated_rows / total_rows if total_rows > 0 else 0.0

    if gated_rows == 0:
        return {
            "coverage": 0.0,
            "gated_rows": 0,
            "total_rows": total_rows,
            "signal_count": 0,
            **{k: 0.0 for k in METRICS_KEYS},
        }

    prob_col = "probability" if "probability" in gated_preds.columns else None
    scores = score_predictions(
        gated_actuals, gated_preds, target_column, "prediction", prob_col
    )

    signal_count = int(gated_preds["prediction"].sum())

    return {
        "coverage": coverage,
        "gated_rows": gated_rows,
        "total_rows": total_rows,
        "signal_count": signal_count,
        **scores,
    }


def main():
    print("Building dataset with liquidity features...")
    dataset, meta = build_dataset(include_liquidity=True)
    eval_ds = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    print(f"Evaluation rows: {len(eval_ds)}")
    print(f"Date range: {eval_ds.index.min()} to {eval_ds.index.max()}")

    # Collect all per-row predictions and actuals across walk-forward windows
    all_actuals = []
    all_predictions = []
    all_regimes = []

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

        regimes = assign_regimes(test_df)

        all_actuals.append(test_df)
        all_predictions.append(predictions)
        all_regimes.append(regimes)

        print(f"  Window {window_id}: {len(test_df)} rows, "
              f"regimes: {regimes.value_counts().to_dict()}")

    actuals = pd.concat(all_actuals)
    predictions = pd.concat(all_predictions)
    regimes = pd.concat(all_regimes)

    print(f"\nTotal evaluation rows across windows: {len(actuals)}")
    print(f"Regime distribution: {regimes.value_counts().to_dict()}")

    # Score each gating variant
    results = {}
    for gate_name, allowed in GATE_CONFIGS.items():
        scores = score_gated(actuals, predictions, regimes, allowed, DEFAULT_TARGET_COLUMN)
        results[gate_name] = {"label": gate_name, "allowed_regimes": sorted(allowed), **scores}

    # Print comparison table
    print("\n" + "=" * 90)
    print("REGIME GATING COMPARISON")
    print("=" * 90)

    col_w = 18
    gate_names = list(GATE_CONFIGS.keys())
    header = f"{'Metric':<22}" + "".join(f"{g:>{col_w}}" for g in gate_names)
    print(header)
    print("-" * len(header))

    for key in ["coverage", "signal_count"] + METRICS_KEYS:
        line = f"{key:<22}"
        for g in gate_names:
            val = results[g].get(key, 0)
            if key == "signal_count":
                line += f"{int(val):>{col_w}}"
            elif key == "coverage":
                line += f"{val:>{col_w}.1%}"
            else:
                line += f"{val:>{col_w}.4f}"
        print(line)

    # Delta table vs ungated
    print("\n--- Deltas vs Ungated ---")
    ungated = results["ungated"]
    header2 = f"{'Metric':<22}" + "".join(f"{g:>{col_w}}" for g in gate_names[1:])
    print(header2)
    print("-" * len(header2))
    for key in METRICS_KEYS:
        if key == "positive_rate":
            continue
        line = f"{key:<22}"
        for g in gate_names[1:]:
            delta = results[g].get(key, 0) - ungated.get(key, 0)
            line += f"{delta:>+{col_w}.4f}"
        print(line)

    # Precision per signal analysis
    print("\n--- Precision-Coverage Tradeoff ---")
    for g in gate_names:
        r = results[g]
        prec = r.get("precision", 0)
        cov = r.get("coverage", 0)
        signals = r.get("signal_count", 0)
        print(f"  {g:<20} precision={prec:.4f}  coverage={cov:.1%}  signals={signals}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    best_gate = None
    best_precision = ungated.get("precision", 0)
    for g in gate_names[1:]:
        gp = results[g].get("precision", 0)
        gc = results[g].get("coverage", 0)
        gs = results[g].get("signal_count", 0)
        if gp > best_precision and gc >= 0.20 and gs >= 10:
            best_gate = g
            best_precision = gp

    if best_gate:
        delta_p = results[best_gate]["precision"] - ungated["precision"]
        print(f"Best gate: {best_gate}")
        print(f"  Precision improvement: {delta_p:+.4f}")
        print(f"  Coverage: {results[best_gate]['coverage']:.1%}")
        print(f"  Signals: {results[best_gate]['signal_count']}")
        print(f"  Recommendation: Gating IMPROVES model usefulness")
    else:
        print("No gating variant improved precision with acceptable coverage.")
        print("Recommendation: Gating does NOT improve model usefulness")

    # Save results
    output_path = Path("artifacts") / "liquidity_gating.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.write_text(json.dumps(results, indent=2, default=convert), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
