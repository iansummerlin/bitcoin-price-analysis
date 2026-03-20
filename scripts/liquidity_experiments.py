"""Liquidity feature experiments: horizon, representation, and regime context.

Experiment 1 — Horizon test:
  Compare liquidity-enabled vs disabled on 4h target (previous ablation was 1h).

Experiment 2 — Representation test:
  Compare four liquidity representations on 1h target, all else fixed:
    full:       all 5 liquidity columns
    regime_only: 3 regime one-hot columns only (zero out level + m2_roc)
    m2_roc_only: m2_roc_3m only (zero out everything else)
    none:       all liquidity columns zeroed

Experiment 3 — Regime context test:
  Run walk-forward with full liquidity, then break down per-window scores
  by the dominant liquidity regime in each test window. Determines whether
  the model performs differently across macro regimes.
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
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
    LIQUIDITY_COLUMNS,
)
from data.pipeline import build_dataset
from evaluation.reporting import score_predictions
from evaluation.targets import add_horizon_targets, horizon_target_column
from evaluation.walk_forward import (
    WalkForwardResult,
    _build_model,
    iter_walk_forward_slices,
    walk_forward_evaluate,
)

METRICS_KEYS = ["precision", "recall", "roc_auc", "directional_accuracy", "f1"]

# Liquidity column subsets
REGIME_COLS = [
    "liquidity_regime_expanding",
    "liquidity_regime_neutral",
    "liquidity_regime_contracting",
]
M2_ROC_COL = "liquidity_m2_roc_3m"
LEVEL_COL = "liquidity_global_usd_t"


def print_scores(scores: dict, indent: int = 2) -> None:
    pad = " " * indent
    for key in METRICS_KEYS:
        print(f"{pad}{key:<22} {scores.get(key, 0):.4f}")


def print_comparison(rows: list[dict], label_key: str = "label") -> None:
    """Print a comparison table from a list of {label, scores} dicts."""
    labels = [r[label_key] for r in rows]
    col_w = max(14, max(len(l) for l in labels) + 2)

    header = f"{'Metric':<22}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * len(header))
    for key in METRICS_KEYS:
        line = f"{key:<22}"
        for r in rows:
            line += f"{r['scores'].get(key, 0):>{col_w}.4f}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Experiment 1: Horizon test (4h)
# ---------------------------------------------------------------------------

def run_horizon_experiment() -> dict:
    """Compare liquidity on/off at 4h horizon."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Liquidity on 4h horizon")
    print("=" * 70)

    horizon = 4
    target_col = horizon_target_column(horizon)
    results = {}

    for include_liq, label in [(True, "4h_with_liq"), (False, "4h_without_liq")]:
        print(f"\n--- {label} ---")
        dataset, meta = build_dataset(include_liquidity=include_liq)
        dataset = add_horizon_targets(dataset, horizon=horizon)
        dataset = dataset.dropna(subset=[target_col])
        eval_ds = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
        print(f"  Rows: {len(eval_ds)}, Range: {eval_ds.index.min()} to {eval_ds.index.max()}")

        result = walk_forward_evaluate(
            eval_ds,
            model_name=DEFAULT_CLASSIFICATION_MODEL,
            target_column=target_col,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            output_dir="artifacts",
            output_stem=f"liq_exp1_{label}",
        )
        print_scores(result.scores)
        results[label] = {
            "label": label,
            "scores": result.scores,
            "windows": result.windows_evaluated,
            "rows": len(eval_ds),
        }

    print("\n--- 4h Horizon Comparison ---")
    print_comparison([results["4h_with_liq"], results["4h_without_liq"]])

    w = results["4h_with_liq"]["scores"]
    wo = results["4h_without_liq"]["scores"]
    results["deltas_4h"] = {k: w.get(k, 0) - wo.get(k, 0) for k in METRICS_KEYS}
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Representation test (1h)
# ---------------------------------------------------------------------------

def run_representation_experiment() -> dict:
    """Compare four liquidity representations on 1h target."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Liquidity representation comparison (1h)")
    print("=" * 70)

    # Build dataset once with full liquidity
    dataset_full, meta = build_dataset(include_liquidity=True)
    eval_full = dataset_full.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    # Build dataset with no liquidity (for baseline)
    dataset_none, _ = build_dataset(include_liquidity=False)
    eval_none = dataset_none.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    representations = {
        "full": eval_full,
        "regime_only": _zero_cols(eval_full, [LEVEL_COL, M2_ROC_COL]),
        "m2_roc_only": _zero_cols(eval_full, [LEVEL_COL] + REGIME_COLS),
        "none": eval_none,
    }

    results = {}
    for name, ds in representations.items():
        print(f"\n--- {name} ---")
        result = walk_forward_evaluate(
            ds,
            model_name=DEFAULT_CLASSIFICATION_MODEL,
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            output_dir="artifacts",
            output_stem=f"liq_exp2_{name}",
        )
        print_scores(result.scores)
        results[name] = {
            "label": name,
            "scores": result.scores,
            "windows": result.windows_evaluated,
        }

    print("\n--- Representation Comparison ---")
    print_comparison([results[k] for k in ["full", "regime_only", "m2_roc_only", "none"]])
    return results


def _zero_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy with specified columns zeroed out."""
    result = df.copy()
    for col in cols:
        if col in result.columns:
            result[col] = 0.0
    return result


# ---------------------------------------------------------------------------
# Experiment 3: Regime context test
# ---------------------------------------------------------------------------

def run_regime_context_experiment() -> dict:
    """Break down per-window model performance by dominant liquidity regime."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Per-window performance by liquidity regime")
    print("=" * 70)

    dataset, meta = build_dataset(include_liquidity=True)
    eval_ds = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()

    window_results = []

    for window_id, (train_df, test_df) in enumerate(
        iter_walk_forward_slices(
            eval_ds,
            train_window=DEFAULT_TRAIN_WINDOW,
            test_window=DEFAULT_TEST_WINDOW,
            min_train_rows=min(DEFAULT_TRAIN_WINDOW, 24 * 60),
        ),
        start=1,
    ):
        model = _build_model(DEFAULT_CLASSIFICATION_MODEL)
        model.fit(train_df, target_column=DEFAULT_TARGET_COLUMN)
        predictions = model.predict_frame(test_df)

        prob_col = "probability" if "probability" in predictions.columns else None
        scores = score_predictions(
            test_df, predictions, DEFAULT_TARGET_COLUMN, "prediction", prob_col
        )

        # Determine dominant regime in this test window
        regime = _dominant_regime(test_df)
        window_results.append({
            "window_id": window_id,
            "regime": regime,
            "scores": scores,
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
        })

        print(f"  Window {window_id}: regime={regime:<13} prec={scores.get('precision', 0):.4f}  "
              f"recall={scores.get('recall', 0):.4f}  roc_auc={scores.get('roc_auc', 0):.4f}  "
              f"f1={scores.get('f1', 0):.4f}")

    # Aggregate by regime
    regime_groups: dict[str, list[dict]] = {}
    for wr in window_results:
        regime_groups.setdefault(wr["regime"], []).append(wr["scores"])

    regime_summary = {}
    print("\n--- Regime-Level Averages ---")
    print(f"{'Regime':<15} {'Windows':>8} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10} {'F1':>10}")
    print("-" * 65)
    for regime, scores_list in sorted(regime_groups.items()):
        avg = pd.DataFrame(scores_list).mean(numeric_only=True).to_dict()
        n = len(scores_list)
        print(f"{regime:<15} {n:>8} {avg.get('precision', 0):>10.4f} {avg.get('recall', 0):>10.4f} "
              f"{avg.get('roc_auc', 0):>10.4f} {avg.get('f1', 0):>10.4f}")
        regime_summary[regime] = {"n_windows": n, "avg_scores": avg}

    return {"per_window": window_results, "by_regime": regime_summary}


def _dominant_regime(df: pd.DataFrame) -> str:
    """Return the regime label that covers the most rows in df."""
    if "liquidity_regime_expanding" not in df.columns:
        return "UNKNOWN"

    expanding = df["liquidity_regime_expanding"].sum()
    neutral = df["liquidity_regime_neutral"].sum()
    contracting = df["liquidity_regime_contracting"].sum()

    counts = {"EXPANDING": expanding, "NEUTRAL": neutral, "CONTRACTING": contracting}
    dominant = max(counts, key=counts.get)

    # If all zeros (no liquidity data overlaps), mark as UNKNOWN
    if counts[dominant] == 0:
        return "UNKNOWN"
    return dominant


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_results = {}

    all_results["experiment_1_horizon"] = run_horizon_experiment()
    all_results["experiment_2_representation"] = run_representation_experiment()
    all_results["experiment_3_regime_context"] = run_regime_context_experiment()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Exp 1: 4h vs 1h comparison
    # (Load 1h results from the previous ablation artifact if available)
    exp1 = all_results["experiment_1_horizon"]
    d4 = exp1["deltas_4h"]
    print("\nExp 1 — 4h horizon deltas (with_liq - without_liq):")
    for k in METRICS_KEYS:
        print(f"  {k:<22} {d4[k]:+.4f}")

    # Exp 2
    exp2 = all_results["experiment_2_representation"]
    print("\nExp 2 — Best representation (by precision):")
    best_repr = max(exp2.items(), key=lambda x: x[1]["scores"].get("precision", 0))
    print(f"  {best_repr[0]}: precision={best_repr[1]['scores'].get('precision', 0):.4f}")
    print("  Full comparison above.")

    # Exp 3
    exp3 = all_results["experiment_3_regime_context"]
    print("\nExp 3 — Regime breakdown:")
    for regime, data in exp3["by_regime"].items():
        avg = data["avg_scores"]
        print(f"  {regime}: n={data['n_windows']}, prec={avg.get('precision', 0):.4f}, "
              f"recall={avg.get('recall', 0):.4f}, auc={avg.get('roc_auc', 0):.4f}")

    output_path = Path("artifacts") / "liquidity_experiments.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path.write_text(json.dumps(all_results, indent=2, default=convert), encoding="utf-8")
    print(f"\nAll results saved to {output_path}")


if __name__ == "__main__":
    main()
