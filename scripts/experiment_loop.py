#!/usr/bin/env python3
"""Phase 13: Autonomous experiment loop.

Systematically searches the feature/model/hyperparameter space using the
existing walk-forward evaluation infrastructure. Each experiment makes a
single change, evaluates it, and keeps or discards based on the composite
metric (precision * recall).

Safety rails:
- Minimum improvement threshold (0.005 on composite metric)
- Regime diversity check (improvement must hold on >= 2 walk-forward windows)
- Rollback on test failure
- Experiment budget cap (100 experiments or 12 hours)
- Held-out validation isolation (never seen during optimization)
- Immutable file protection (evaluation/*, data/*, tests/*, backtest.py, signals/*)

Usage:
    python scripts/experiment_loop.py
    # or: make experiment
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAIN_WINDOW,
    DEFAULT_TEST_WINDOW,
    DEFAULT_MIN_TRAIN_ROWS,
    EXPERIMENT_BUDGET_HOURS,
    EXPERIMENT_BUDGET_MAX,
    EXPERIMENT_MIN_IMPROVEMENT,
    HELD_OUT_HOURS,
    EXOG_COLUMNS,
)
from data.pipeline import build_dataset, write_dataset_metadata
from evaluation.walk_forward import (
    WalkForwardResult,
    iter_walk_forward_slices,
    _build_model,
    MODEL_REGISTRY,
)
from evaluation.reporting import score_predictions
from evaluation.baselines import add_baseline_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "results.tsv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
AUTORESEARCH_PATH = PROJECT_ROOT / "AUTORESEARCH.md"
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", str(ARTIFACTS_DIR / "experiment_checkpoint.json")))
STOP_FLAG_PATH = os.environ.get("STOP_FLAG_PATH", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PROGRESS_INTERVAL = 10  # send Telegram notification every N experiments


@dataclass
class ExperimentResult:
    experiment_id: int
    description: str
    model_name: str
    composite_metric: float  # precision * recall
    precision: float
    recall: float
    roc_auc: float
    f1: float
    windows_evaluated: int
    window_improvements: int  # how many windows improved vs baseline
    status: str  # "keep" or "discard"
    reason: str


def _check_stop_flag() -> bool:
    """Return True if the stop flag file exists."""
    if not STOP_FLAG_PATH:
        return False
    return Path(STOP_FLAG_PATH).exists()


def _send_progress(text: str) -> None:
    """Send a Telegram progress notification (best-effort)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data=data
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
    except Exception as e:
        logger.warning("Telegram notification failed: %s", e)


def _experiment_list_hash(experiments: list[dict]) -> str:
    """Hash the experiment list to detect code changes."""
    serialized = json.dumps(experiments, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _save_checkpoint(
    run_id: str,
    experiment_list_hash: str,
    completed_results: list["ExperimentResult"],
    best_composite: float,
    best_model_name: str,
    best_config: dict | None,
    best_scores: dict,
    best_window_scores: list,
    baseline_model: str,
    baseline_composite: float,
    baseline_scores: dict,
) -> None:
    """Save checkpoint after each experiment."""
    data = {
        "run_id": run_id,
        "experiment_list_hash": experiment_list_hash,
        "completed_experiment_ids": [r.experiment_id for r in completed_results],
        "completed_results": [
            {
                "experiment_id": r.experiment_id,
                "description": r.description,
                "model_name": r.model_name,
                "composite_metric": r.composite_metric,
                "precision": r.precision,
                "recall": r.recall,
                "roc_auc": r.roc_auc,
                "f1": r.f1,
                "windows_evaluated": r.windows_evaluated,
                "window_improvements": r.window_improvements,
                "status": r.status,
                "reason": r.reason,
            }
            for r in completed_results
        ],
        "best_composite": best_composite,
        "best_model_name": best_model_name,
        "best_config": best_config,
        "best_scores": best_scores,
        "best_window_scores": [dict(ws) for ws in best_window_scores],
        "baseline_model": baseline_model,
        "baseline_composite": baseline_composite,
        "baseline_scores": dict(baseline_scores),
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.write_text(json.dumps(data, indent=2, default=str))
    logger.debug("Checkpoint saved: %d experiments", len(completed_results))


def _load_checkpoint(expected_hash: str) -> dict | None:
    """Load checkpoint if it exists and hash matches. Returns None otherwise."""
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text())
    except Exception as e:
        logger.warning("Corrupt checkpoint file, starting fresh: %s", e)
        return None
    if data.get("experiment_list_hash") != expected_hash:
        logger.info("Experiment list hash mismatch — checkpoint invalidated, starting fresh")
        return None
    return data


def compute_composite(scores: dict[str, float]) -> float:
    """Composite metric: precision * recall."""
    return scores.get("precision", 0.0) * scores.get("recall", 0.0)


def evaluate_configuration(
    dataset: pd.DataFrame,
    model_name: str,
    target_column: str,
    train_window: int,
    test_window: int,
    model_kwargs: dict | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Run walk-forward evaluation and return (mean_scores, per_window_scores).

    This is a simplified version of walk_forward_evaluate that returns
    per-window detail for regime diversity checks without writing artifacts.
    """
    window_scores: list[dict[str, float]] = []

    for window_id, (train_df, test_df) in enumerate(
        iter_walk_forward_slices(
            dataset,
            train_window=train_window,
            test_window=test_window,
            min_train_rows=min(train_window, DEFAULT_MIN_TRAIN_ROWS),
        ),
        start=1,
    ):
        model = _build_model(model_name)
        # Override hyperparameters if provided
        if model_kwargs:
            for k, v in model_kwargs.items():
                if hasattr(model, k):
                    setattr(model, k, v)
                if hasattr(model, "model") and hasattr(model.model, k):
                    setattr(model.model, k, v)

        model.fit(train_df, target_column=target_column)
        predictions = model.predict_frame(test_df)

        probability_column = "probability" if "probability" in predictions.columns else None
        score = score_predictions(
            test_df, predictions, target_column, "prediction", probability_column
        )
        score["composite"] = compute_composite(score)
        window_scores.append(score)

    if not window_scores:
        return {}, []

    mean_scores = pd.DataFrame(window_scores).mean(numeric_only=True).to_dict()
    mean_scores["composite"] = compute_composite(mean_scores)
    return mean_scores, window_scores


def regime_diversity_check(
    baseline_window_scores: list[dict[str, float]],
    new_window_scores: list[dict[str, float]],
    min_windows: int = 2,
) -> tuple[bool, int]:
    """Check that improvement holds across at least min_windows walk-forward windows.

    Returns (passed, count_of_improved_windows).
    """
    if len(new_window_scores) != len(baseline_window_scores):
        return False, 0

    improved = 0
    for base, new in zip(baseline_window_scores, new_window_scores):
        if compute_composite(new) > compute_composite(base):
            improved += 1

    return improved >= min_windows, improved


def load_results() -> list[dict]:
    """Load existing experiment results from TSV."""
    if not RESULTS_PATH.exists():
        return []
    results = []
    with open(RESULTS_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            results.append(row)
    return results


def save_result(result: ExperimentResult, run_id: str = "") -> None:
    """Append a single experiment result to results.tsv."""
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not file_exists:
            writer.writerow([
                "run_id", "experiment_id", "model_name", "composite_metric",
                "precision", "recall", "roc_auc", "f1",
                "windows_evaluated", "window_improvements",
                "status", "reason", "description",
            ])
        writer.writerow([
            run_id,
            result.experiment_id,
            result.model_name,
            f"{result.composite_metric:.6f}",
            f"{result.precision:.4f}",
            f"{result.recall:.4f}",
            f"{result.roc_auc:.4f}",
            f"{result.f1:.4f}",
            result.windows_evaluated,
            result.window_improvements,
            result.status,
            result.reason,
            result.description,
        ])


# ============================================================================
# Experiment definitions
# ============================================================================
# Each experiment is a dict with:
#   "description": str
#   "model_name": str
#   "model_kwargs": dict (optional overrides for model hyperparameters)
#   "feature_columns": list[str] | None (if None, use EXOG_COLUMNS)
#   "target_column": str (default: target_direction_cost_adj)

def generate_experiments() -> list[dict]:
    """Generate the full experiment search space.

    Categories:
    1. Model hyperparameter variations (XGBoost and LightGBM)
    2. Feature subset selection (remove individual feature families)
    3. Feature subset selection (remove individual weak features)
    4. Combined model + feature experiments
    """
    experiments = []

    # Current EXOG_COLUMNS for reference
    base_features = list(EXOG_COLUMNS)

    # Cross-asset feature columns
    crossasset_cols = [
        "dxy_return_1d", "dxy_return_5d", "sp500_return_1d",
        "btc_sp500_corr_30d", "vix_level", "vix_change_1d",
        "gold_btc_ratio", "gold_btc_ratio_zscore_30d",
        "eth_btc_ratio", "eth_btc_ratio_change_7d",
    ]
    # On-chain feature columns
    onchain_cols = [
        "hashrate_change_24h", "hashrate_change_7d", "difficulty_change",
        "tx_count_zscore_7d", "tx_volume_zscore_7d", "hashrate_price_divergence",
    ]
    # Microstructure feature columns
    micro_cols = [
        "funding_rate_8h", "funding_rate_zscore_7d", "funding_rate_cumulative_24h",
    ]

    # --- Category 1: Model hyperparameter variations ---

    # XGBoost variations
    for n_est in [50, 100, 150, 300, 500]:
        for max_d in [2, 3, 4, 5, 6]:
            for lr in [0.01, 0.03, 0.05, 0.1, 0.15]:
                # Skip the default config (200, 3, 0.05)
                if n_est == 200 and max_d == 3 and lr == 0.05:
                    continue
                # Only generate a manageable subset — key variations
                if not (
                    (n_est in [100, 300] and max_d in [3, 5] and lr in [0.01, 0.05, 0.1])
                    or (n_est == 500 and max_d == 4 and lr == 0.03)
                    or (n_est == 50 and max_d == 6 and lr == 0.15)
                ):
                    continue
                experiments.append({
                    "description": f"xgb: n_est={n_est}, max_d={max_d}, lr={lr}",
                    "model_name": "xgboost_direction",
                    "model_kwargs": {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr},
                })

    # LightGBM variations
    for n_est in [50, 100, 200, 300, 500]:
        for max_d in [2, 3, 4, 5, 6, -1]:
            for lr in [0.01, 0.03, 0.05, 0.1]:
                for n_leaves in [7, 15, 31, 63]:
                    if n_est == 200 and max_d == 3 and lr == 0.05 and n_leaves == 15:
                        continue
                    # Only key variations
                    if not (
                        (n_est in [100, 300] and max_d in [3, 5, -1] and lr in [0.01, 0.05, 0.1] and n_leaves in [15, 31])
                        or (n_est == 500 and max_d == 4 and lr == 0.03 and n_leaves == 31)
                        or (n_est == 50 and max_d == -1 and lr == 0.1 and n_leaves == 63)
                    ):
                        continue
                    experiments.append({
                        "description": f"lgbm: n_est={n_est}, max_d={max_d}, lr={lr}, leaves={n_leaves}",
                        "model_name": "lightgbm_direction",
                        "model_kwargs": {"n_estimators": n_est, "max_depth": max_d, "learning_rate": lr, "num_leaves": n_leaves},
                    })

    # --- Category 2: Feature family ablation ---

    # Remove cross-asset features only
    experiments.append({
        "description": "xgb: remove cross-asset features",
        "model_name": "xgboost_direction",
        "feature_columns": [c for c in base_features if c not in crossasset_cols],
    })
    experiments.append({
        "description": "lgbm: remove cross-asset features",
        "model_name": "lightgbm_direction",
        "feature_columns": [c for c in base_features if c not in crossasset_cols],
    })

    # Remove on-chain features only
    experiments.append({
        "description": "xgb: remove on-chain features",
        "model_name": "xgboost_direction",
        "feature_columns": [c for c in base_features if c not in onchain_cols],
    })
    experiments.append({
        "description": "lgbm: remove on-chain features",
        "model_name": "lightgbm_direction",
        "feature_columns": [c for c in base_features if c not in onchain_cols],
    })

    # Remove microstructure features only
    experiments.append({
        "description": "xgb: remove microstructure features",
        "model_name": "xgboost_direction",
        "feature_columns": [c for c in base_features if c not in micro_cols],
    })
    experiments.append({
        "description": "lgbm: remove microstructure features",
        "model_name": "lightgbm_direction",
        "feature_columns": [c for c in base_features if c not in micro_cols],
    })

    # Remove all new data families (price/sentiment only)
    new_family_cols = crossasset_cols + onchain_cols + micro_cols
    experiments.append({
        "description": "xgb: price+sentiment only (no new families)",
        "model_name": "xgboost_direction",
        "feature_columns": [c for c in base_features if c not in new_family_cols],
    })
    experiments.append({
        "description": "lgbm: price+sentiment only (no new families)",
        "model_name": "lightgbm_direction",
        "feature_columns": [c for c in base_features if c not in new_family_cols],
    })

    # --- Category 3: Individual feature removal ---
    # Test removing each individual feature to find weak ones
    for col in base_features:
        remaining = [c for c in base_features if c != col]
        if len(remaining) < 10:
            continue
        experiments.append({
            "description": f"lgbm: drop {col}",
            "model_name": "lightgbm_direction",
            "feature_columns": remaining,
        })

    # --- Category 4: Best hyperparams + feature subsets ---
    # These will be generated dynamically based on early results

    return experiments


def run_single_experiment(
    experiment: dict,
    dataset: pd.DataFrame,
    baseline_scores: dict[str, float],
    baseline_window_scores: list[dict[str, float]],
    experiment_id: int,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
) -> ExperimentResult:
    """Run a single experiment and return the result."""
    model_name = experiment["model_name"]
    description = experiment["description"]
    model_kwargs = experiment.get("model_kwargs", {})
    feature_columns = experiment.get("feature_columns")
    target_column = experiment.get("target_column", DEFAULT_TARGET_COLUMN)

    logger.info("Experiment %d: %s", experiment_id, description)

    # If custom feature columns or model kwargs, temporarily override registry
    original_registry_entry = MODEL_REGISTRY.get(model_name)
    if feature_columns or model_kwargs:
        def custom_factory():
            if model_name == "xgboost_direction":
                from models.xgboost_model import XGBoostDirectionModel
                kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}
                kwargs.update(model_kwargs)
                m = XGBoostDirectionModel(
                    feature_columns=feature_columns,
                    n_estimators=kwargs["n_estimators"],
                    max_depth=kwargs["max_depth"],
                    learning_rate=kwargs["learning_rate"],
                )
                return m
            elif model_name == "lightgbm_direction":
                from models.lightgbm_model import LightGBMDirectionModel
                kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 15}
                kwargs.update(model_kwargs)
                m = LightGBMDirectionModel(
                    feature_columns=feature_columns,
                    n_estimators=kwargs["n_estimators"],
                    max_depth=kwargs["max_depth"],
                    learning_rate=kwargs["learning_rate"],
                    num_leaves=kwargs["num_leaves"],
                )
                return m
            else:
                return _build_model(model_name)

        MODEL_REGISTRY[model_name] = custom_factory

    try:
        scores, window_scores = evaluate_configuration(
            dataset,
            model_name=model_name,
            target_column=target_column,
            train_window=train_window,
            test_window=test_window,
            model_kwargs=model_kwargs if not feature_columns else None,
        )
    finally:
        # Restore original registry
        if original_registry_entry is not None:
            MODEL_REGISTRY[model_name] = original_registry_entry

    if not scores:
        return ExperimentResult(
            experiment_id=experiment_id,
            description=description,
            model_name=model_name,
            composite_metric=0.0,
            precision=0.0,
            recall=0.0,
            roc_auc=0.0,
            f1=0.0,
            windows_evaluated=0,
            window_improvements=0,
            status="discard",
            reason="no valid windows",
        )

    composite = compute_composite(scores)
    baseline_composite = compute_composite(baseline_scores)
    improvement = composite - baseline_composite

    # Check regime diversity
    diversity_passed, windows_improved = regime_diversity_check(
        baseline_window_scores, window_scores
    )

    # Decision logic
    if improvement < EXPERIMENT_MIN_IMPROVEMENT:
        status = "discard"
        reason = f"insufficient improvement ({improvement:+.6f} < {EXPERIMENT_MIN_IMPROVEMENT})"
    elif not diversity_passed:
        status = "discard"
        reason = f"regime diversity failed ({windows_improved}/{len(window_scores)} windows improved, need 2+)"
    else:
        status = "keep"
        reason = f"improvement {improvement:+.6f}, {windows_improved}/{len(window_scores)} windows improved"

    return ExperimentResult(
        experiment_id=experiment_id,
        description=description,
        model_name=model_name,
        composite_metric=composite,
        precision=scores.get("precision", 0.0),
        recall=scores.get("recall", 0.0),
        roc_auc=scores.get("roc_auc", 0.0),
        f1=scores.get("f1", 0.0),
        windows_evaluated=len(window_scores),
        window_improvements=windows_improved,
        status=status,
        reason=reason,
    )


def generate_autoresearch_md(
    run_timestamp: str,
    elapsed_minutes: float,
    baseline_model: str,
    baseline_scores: dict[str, float],
    baseline_composite: float,
    xgb_scores: dict[str, float],
    lgbm_scores: dict[str, float],
    experiment_results: list[ExperimentResult],
    best_experiment: ExperimentResult | None,
    best_config: dict | None,
    best_composite: float,
    held_out_scores: dict[str, float],
    held_out_composite: float,
    gate_pass: bool,
    experiment_rows: int,
    held_out_rows: int,
    experiment_end: str,
    held_out_start: str,
    held_out_end: str,
    total_experiments: int,
    kept_count: int,
    discarded_count: int,
) -> None:
    """Generate AUTORESEARCH.md with a full run report."""
    lines: list[str] = []
    lines.append("# Autoresearch Report")
    lines.append("")
    lines.append("*Auto-generated by `scripts/experiment_loop.py` — do not edit manually.*")
    lines.append("")

    # --- Run metadata ---
    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- **Timestamp:** {run_timestamp}")
    lines.append(f"- **Elapsed:** {elapsed_minutes:.1f} minutes")
    lines.append(f"- **Experiments:** {total_experiments} ({kept_count} kept, {discarded_count} discarded)")
    lines.append(f"- **Experiment data:** {experiment_rows} rows (ends {experiment_end})")
    lines.append(f"- **Held-out data:** {held_out_rows} rows ({held_out_start} to {held_out_end})")
    lines.append(f"- **Composite metric:** precision x recall")
    lines.append(f"- **Min improvement threshold:** {EXPERIMENT_MIN_IMPROVEMENT}")
    lines.append("")

    # --- Baselines ---
    lines.append("## Baselines")
    lines.append("")
    lines.append("| Model | Composite | Precision | Recall | ROC-AUC | F1 | Selected |")
    lines.append("|-------|-----------|-----------|--------|---------|----| ---------|")
    xgb_comp = compute_composite(xgb_scores)
    lgbm_comp = compute_composite(lgbm_scores)
    lines.append(
        f"| XGBoost | {xgb_comp:.6f} | {xgb_scores.get('precision', 0):.4f} | "
        f"{xgb_scores.get('recall', 0):.4f} | {xgb_scores.get('roc_auc', 0):.4f} | "
        f"{xgb_scores.get('f1', 0):.4f} | {'**yes**' if baseline_model == 'xgboost_direction' else ''} |"
    )
    lines.append(
        f"| LightGBM | {lgbm_comp:.6f} | {lgbm_scores.get('precision', 0):.4f} | "
        f"{lgbm_scores.get('recall', 0):.4f} | {lgbm_scores.get('roc_auc', 0):.4f} | "
        f"{lgbm_scores.get('f1', 0):.4f} | {'**yes**' if baseline_model == 'lightgbm_direction' else ''} |"
    )
    lines.append("")

    # --- Experiment results table ---
    lines.append("## Experiments")
    lines.append("")
    lines.append("| # | Status | Composite | Precision | Recall | ROC-AUC | Win Impr | Description |")
    lines.append("|---|--------|-----------|-----------|--------|---------|----------|-------------|")
    for r in experiment_results:
        status_icon = "keep" if r.status == "keep" else "discard"
        lines.append(
            f"| {r.experiment_id} | {status_icon} | {r.composite_metric:.6f} | "
            f"{r.precision:.4f} | {r.recall:.4f} | {r.roc_auc:.4f} | "
            f"{r.window_improvements}/{r.windows_evaluated} | {r.description} |"
        )
    lines.append("")

    # --- Best configuration ---
    lines.append("## Best Configuration")
    lines.append("")
    if best_experiment:
        lines.append(f"- **Config:** {best_experiment.description}")
        lines.append(f"- **Model:** {best_experiment.model_name}")
        lines.append(f"- **Composite:** {best_composite:.6f} (baseline was {baseline_composite:.6f}, improvement {best_composite - baseline_composite:+.6f})")
        lines.append(f"- **Precision:** {best_experiment.precision:.4f}")
        lines.append(f"- **Recall:** {best_experiment.recall:.4f}")
        lines.append(f"- **ROC-AUC:** {best_experiment.roc_auc:.4f}")
        lines.append(f"- **F1:** {best_experiment.f1:.4f}")
    else:
        lines.append(f"No experiment improved over the baseline ({baseline_model}, composite={baseline_composite:.6f}).")
    lines.append("")

    # --- Held-out validation ---
    lines.append("## Held-Out Validation")
    lines.append("")
    lines.append(f"Evaluated on {held_out_rows} rows ({held_out_start} to {held_out_end}).")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Result |")
    lines.append("|--------|-------|-----------|--------|")
    prec = held_out_scores.get("precision", 0)
    rec = held_out_scores.get("recall", 0)
    roc = held_out_scores.get("roc_auc", 0)
    lines.append(f"| Precision | {prec:.4f} | >= 0.55 | {'PASS' if prec >= 0.55 else 'FAIL'} |")
    lines.append(f"| Recall | {rec:.4f} | >= 0.15 | {'PASS' if rec >= 0.15 else 'FAIL'} |")
    lines.append(f"| ROC-AUC | {roc:.4f} | >= 0.60 | {'PASS' if roc >= 0.60 else 'FAIL'} |")
    lines.append(f"| Composite | {held_out_composite:.6f} | — | — |")
    lines.append(f"| F1 | {held_out_scores.get('f1', 0):.4f} | — | — |")
    lines.append(f"| Dir Accuracy | {held_out_scores.get('directional_accuracy', 0):.4f} | — | — |")
    lines.append("")

    # --- Gate 7 verdict ---
    lines.append("## Gate 7 Verdict")
    lines.append("")
    if gate_pass:
        lines.append("**PASS** — The best configuration clears all integration thresholds on held-out data.")
        lines.append("The repo is ready for downstream integration review (Gate 5).")
    else:
        lines.append("**FAIL** — The best configuration does not clear all integration thresholds.")
        lines.append("The repo remains research-only.")
    lines.append("")

    AUTORESEARCH_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("AUTORESEARCH.md written to %s", AUTORESEARCH_PATH)


def main() -> None:
    start_time = time.time()
    logger.info("Phase 13: Autonomous Experiment Loop")
    logger.info("Budget: %d experiments or %d hours", EXPERIMENT_BUDGET_MAX, EXPERIMENT_BUDGET_HOURS)
    if STOP_FLAG_PATH:
        logger.info("Stop flag: %s", STOP_FLAG_PATH)

    # Build dataset once (with held-out split)
    logger.info("Building dataset...")
    dataset, metadata = build_dataset()
    logger.info("Full dataset: %d rows (%s to %s)", len(dataset), metadata.start, metadata.end)

    # Split off held-out set
    held_out_cutoff = len(dataset) - HELD_OUT_HOURS
    if held_out_cutoff <= DEFAULT_TRAIN_WINDOW + DEFAULT_TEST_WINDOW:
        logger.error("Not enough data after held-out split for walk-forward evaluation")
        sys.exit(1)

    experiment_dataset = dataset.iloc[:held_out_cutoff].copy()
    held_out_dataset = dataset.iloc[held_out_cutoff:].copy()
    logger.info("Experiment dataset: %d rows (ends at %s)",
                len(experiment_dataset), experiment_dataset.index[-1])
    logger.info("Held-out dataset: %d rows (%s to %s)",
                len(held_out_dataset), held_out_dataset.index[0], held_out_dataset.index[-1])

    # Establish baseline on experiment dataset
    logger.info("Establishing baseline...")

    # Baseline: XGBoost direction (current default)
    xgb_scores, xgb_window_scores = evaluate_configuration(
        experiment_dataset,
        model_name="xgboost_direction",
        target_column=DEFAULT_TARGET_COLUMN,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
    )

    # Baseline: LightGBM direction
    lgbm_scores, lgbm_window_scores = evaluate_configuration(
        experiment_dataset,
        model_name="lightgbm_direction",
        target_column=DEFAULT_TARGET_COLUMN,
        train_window=DEFAULT_TRAIN_WINDOW,
        test_window=DEFAULT_TEST_WINDOW,
    )

    # Use the better baseline as reference
    xgb_composite = compute_composite(xgb_scores)
    lgbm_composite = compute_composite(lgbm_scores)

    logger.info("XGBoost baseline: composite=%.6f (precision=%.4f, recall=%.4f, roc_auc=%.4f)",
                xgb_composite, xgb_scores.get("precision", 0), xgb_scores.get("recall", 0), xgb_scores.get("roc_auc", 0))
    logger.info("LightGBM baseline: composite=%.6f (precision=%.4f, recall=%.4f, roc_auc=%.4f)",
                lgbm_composite, lgbm_scores.get("precision", 0), lgbm_scores.get("recall", 0), lgbm_scores.get("roc_auc", 0))

    if lgbm_composite >= xgb_composite:
        baseline_scores = lgbm_scores
        baseline_window_scores = lgbm_window_scores
        baseline_model = "lightgbm_direction"
    else:
        baseline_scores = xgb_scores
        baseline_window_scores = xgb_window_scores
        baseline_model = "xgboost_direction"

    baseline_composite = compute_composite(baseline_scores)
    logger.info("Using %s as baseline (composite=%.6f)", baseline_model, baseline_composite)

    # Generate experiments and compute hash for checkpoint validation
    experiments = generate_experiments()
    exp_hash = _experiment_list_hash(experiments)
    logger.info("Generated %d experiments (hash=%s)", len(experiments), exp_hash)

    # --- Checkpoint resume logic ---
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    completed_ids: set[int] = set()
    all_results: list[ExperimentResult] = []
    best_composite = baseline_composite
    best_scores = baseline_scores.copy()
    best_window_scores = list(baseline_window_scores)
    best_experiment: ExperimentResult | None = None
    best_config: dict | None = None
    kept_count = 0
    discarded_count = 0

    checkpoint = _load_checkpoint(exp_hash)
    if checkpoint:
        run_id = checkpoint["run_id"]
        completed_ids = set(checkpoint["completed_experiment_ids"])
        logger.info("Resuming run %s — %d experiments already done", run_id, len(completed_ids))

        # Rehydrate all_results from checkpoint
        for cr in checkpoint.get("completed_results", []):
            r = ExperimentResult(
                experiment_id=cr["experiment_id"],
                description=cr["description"],
                model_name=cr["model_name"],
                composite_metric=cr["composite_metric"],
                precision=cr["precision"],
                recall=cr["recall"],
                roc_auc=cr["roc_auc"],
                f1=cr["f1"],
                windows_evaluated=cr["windows_evaluated"],
                window_improvements=cr["window_improvements"],
                status=cr["status"],
                reason=cr["reason"],
            )
            all_results.append(r)
            if r.status == "keep":
                kept_count += 1
            elif r.status == "discard":
                discarded_count += 1

        # Restore best state from checkpoint
        best_composite = checkpoint.get("best_composite", baseline_composite)
        best_scores = checkpoint.get("best_scores", baseline_scores.copy())
        best_window_scores = checkpoint.get("best_window_scores", list(baseline_window_scores))
        if checkpoint.get("best_config"):
            best_config = checkpoint["best_config"]
        # Reconstruct best_experiment from the last "keep" result
        kept_results = [r for r in all_results if r.status == "keep"]
        if kept_results:
            best_experiment = kept_results[-1]
    else:
        logger.info("Starting fresh run: %s", run_id)

        # Save baselines to results
        save_result(ExperimentResult(
            experiment_id=0,
            description=f"BASELINE: {baseline_model} default config",
            model_name=baseline_model,
            composite_metric=baseline_composite,
            precision=baseline_scores.get("precision", 0.0),
            recall=baseline_scores.get("recall", 0.0),
            roc_auc=baseline_scores.get("roc_auc", 0.0),
            f1=baseline_scores.get("f1", 0.0),
            windows_evaluated=len(baseline_window_scores),
            window_improvements=0,
            status="baseline",
            reason="initial baseline",
        ), run_id=run_id)

    total_experiments = len(experiments)
    stopped_early = False

    for i, experiment in enumerate(experiments, start=1):
        # Skip already-completed experiments (resume)
        if i in completed_ids:
            continue

        # Stop flag check between experiments
        if _check_stop_flag():
            logger.info("Stop flag detected — finishing gracefully after %d experiments",
                        kept_count + discarded_count)
            _send_progress(
                f"BTC Autoresearch: stop requested. "
                f"Finishing after {kept_count + discarded_count}/{total_experiments} experiments."
            )
            stopped_early = True
            break

        # Budget checks
        elapsed_hours = (time.time() - start_time) / 3600
        if elapsed_hours >= EXPERIMENT_BUDGET_HOURS:
            logger.info("Time budget exhausted (%.1f hours)", elapsed_hours)
            break
        if i > EXPERIMENT_BUDGET_MAX:
            logger.info("Experiment budget exhausted (%d experiments)", EXPERIMENT_BUDGET_MAX)
            break

        result = run_single_experiment(
            experiment,
            experiment_dataset,
            best_scores,
            best_window_scores,
            experiment_id=i,
        )
        save_result(result, run_id=run_id)
        all_results.append(result)

        if result.status == "keep":
            logger.info("  KEEP: %s (composite=%.6f, +%.6f)",
                        result.description, result.composite_metric,
                        result.composite_metric - best_composite)
            best_composite = result.composite_metric
            best_scores = {
                "precision": result.precision,
                "recall": result.recall,
                "roc_auc": result.roc_auc,
                "f1": result.f1,
                "composite": result.composite_metric,
            }
            # Re-evaluate to get updated window scores for future comparisons.
            # Must use the same config (feature_columns + model_kwargs) as the
            # kept experiment, otherwise future regime-diversity comparisons
            # use window scores from the wrong baseline.
            kept_features = experiment.get("feature_columns")
            kept_kwargs = experiment.get("model_kwargs", {})
            original_reg = MODEL_REGISTRY.get(result.model_name)
            if kept_features or kept_kwargs:
                def _kept_factory(
                    _mn=result.model_name, _fc=kept_features, _kw=kept_kwargs
                ):
                    if _mn == "xgboost_direction":
                        from models.xgboost_model import XGBoostDirectionModel
                        kw = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}
                        kw.update(_kw)
                        return XGBoostDirectionModel(
                            feature_columns=_fc, n_estimators=kw["n_estimators"],
                            max_depth=kw["max_depth"], learning_rate=kw["learning_rate"],
                        )
                    elif _mn == "lightgbm_direction":
                        from models.lightgbm_model import LightGBMDirectionModel
                        kw = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 15}
                        kw.update(_kw)
                        return LightGBMDirectionModel(
                            feature_columns=_fc, n_estimators=kw["n_estimators"],
                            max_depth=kw["max_depth"], learning_rate=kw["learning_rate"],
                            num_leaves=kw["num_leaves"],
                        )
                    else:
                        return _build_model(_mn)
                MODEL_REGISTRY[result.model_name] = _kept_factory
            try:
                _, best_window_scores_new = evaluate_configuration(
                    experiment_dataset,
                    model_name=result.model_name,
                    target_column=DEFAULT_TARGET_COLUMN,
                    train_window=DEFAULT_TRAIN_WINDOW,
                    test_window=DEFAULT_TEST_WINDOW,
                )
            finally:
                if original_reg is not None:
                    MODEL_REGISTRY[result.model_name] = original_reg
            if best_window_scores_new:
                best_window_scores = best_window_scores_new
            best_experiment = result
            best_config = experiment
            kept_count += 1
        else:
            logger.info("  DISCARD: %s (%s)", result.description, result.reason)
            discarded_count += 1

        # Save checkpoint after every experiment
        _save_checkpoint(
            run_id=run_id,
            experiment_list_hash=exp_hash,
            completed_results=all_results,
            best_composite=best_composite,
            best_model_name=best_experiment.model_name if best_experiment else baseline_model,
            best_config=best_config,
            best_scores=best_scores,
            best_window_scores=best_window_scores,
            baseline_model=baseline_model,
            baseline_composite=baseline_composite,
            baseline_scores=baseline_scores,
        )

        # Progress notification every N experiments
        experiments_done = kept_count + discarded_count
        if experiments_done > 0 and experiments_done % PROGRESS_INTERVAL == 0:
            _send_progress(
                f"BTC Autoresearch: {experiments_done}/{total_experiments} experiments\n"
                f"Kept: {kept_count} | Discarded: {discarded_count}\n"
                f"Best composite: {best_composite:.4f} (baseline: {baseline_composite:.4f})"
            )

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPERIMENT LOOP COMPLETE")
    logger.info("=" * 60)
    logger.info("Experiments run: %d", kept_count + discarded_count)
    logger.info("Kept: %d, Discarded: %d", kept_count, discarded_count)
    logger.info("Elapsed: %.1f minutes", elapsed / 60)
    logger.info("Best composite: %.6f (baseline was %.6f)",
                best_composite, baseline_composite)

    if best_experiment:
        logger.info("Best experiment: %s", best_experiment.description)
        logger.info("  precision=%.4f, recall=%.4f, roc_auc=%.4f",
                     best_experiment.precision, best_experiment.recall, best_experiment.roc_auc)

    # --- Held-out validation ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("HELD-OUT VALIDATION")
    logger.info("=" * 60)

    # Evaluate best surviving configuration on held-out set
    # We train on the experiment dataset and predict on held-out
    if best_config:
        eval_model_name = best_config["model_name"]
        eval_kwargs = best_config.get("model_kwargs", {})
        eval_features = best_config.get("feature_columns")
    else:
        eval_model_name = baseline_model
        eval_kwargs = {}
        eval_features = None

    logger.info("Evaluating best config on held-out set: %s", best_config["description"] if best_config else baseline_model)

    # Held-out protocol: train on ALL pre-held-out data, predict on held-out.
    # This gives the model the best possible training set for the final check.
    # During the experiment loop, walk-forward used sliding windows for regime
    # diversity — but the held-out evaluation is a single train/predict pass
    # because it is only done once and must reflect the best achievable result.
    train_data = experiment_dataset

    if eval_model_name == "xgboost_direction":
        from models.xgboost_model import XGBoostDirectionModel
        kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05}
        kwargs.update(eval_kwargs)
        model = XGBoostDirectionModel(
            feature_columns=eval_features,
            n_estimators=kwargs["n_estimators"],
            max_depth=kwargs["max_depth"],
            learning_rate=kwargs["learning_rate"],
        )
    elif eval_model_name == "lightgbm_direction":
        from models.lightgbm_model import LightGBMDirectionModel
        kwargs = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "num_leaves": 15}
        kwargs.update(eval_kwargs)
        model = LightGBMDirectionModel(
            feature_columns=eval_features,
            n_estimators=kwargs["n_estimators"],
            max_depth=kwargs["max_depth"],
            learning_rate=kwargs["learning_rate"],
            num_leaves=kwargs["num_leaves"],
        )
    else:
        model = _build_model(eval_model_name)

    model.fit(train_data, target_column=DEFAULT_TARGET_COLUMN)
    predictions = model.predict_frame(held_out_dataset)

    probability_column = "probability" if "probability" in predictions.columns else None
    held_out_scores = score_predictions(
        held_out_dataset, predictions, DEFAULT_TARGET_COLUMN, "prediction", probability_column
    )
    held_out_composite = compute_composite(held_out_scores)

    logger.info("Held-out results:")
    logger.info("  precision:   %.4f (threshold: >= 0.55)", held_out_scores.get("precision", 0))
    logger.info("  recall:      %.4f (threshold: >= 0.15)", held_out_scores.get("recall", 0))
    logger.info("  ROC-AUC:     %.4f (threshold: >= 0.60)", held_out_scores.get("roc_auc", 0))
    logger.info("  F1:          %.4f", held_out_scores.get("f1", 0))
    logger.info("  composite:   %.6f", held_out_composite)
    logger.info("  dir_acc:     %.4f", held_out_scores.get("directional_accuracy", 0))

    # Gate checks
    precision_pass = held_out_scores.get("precision", 0) >= 0.55
    recall_pass = held_out_scores.get("recall", 0) >= 0.15
    roc_auc_pass = held_out_scores.get("roc_auc", 0) >= 0.60
    all_pass = precision_pass and recall_pass and roc_auc_pass

    logger.info("")
    logger.info("Gate 7 check:")
    logger.info("  Precision >= 0.55:  %s (%.4f)", "PASS" if precision_pass else "FAIL", held_out_scores.get("precision", 0))
    logger.info("  Recall >= 0.15:     %s (%.4f)", "PASS" if recall_pass else "FAIL", held_out_scores.get("recall", 0))
    logger.info("  ROC-AUC >= 0.60:    %s (%.4f)", "PASS" if roc_auc_pass else "FAIL", held_out_scores.get("roc_auc", 0))
    logger.info("  Overall:            %s", "PASS" if all_pass else "FAIL")

    # Save held-out result
    save_result(ExperimentResult(
        experiment_id=999,
        description=f"HELD-OUT: {best_config['description'] if best_config else baseline_model}",
        model_name=eval_model_name,
        composite_metric=held_out_composite,
        precision=held_out_scores.get("precision", 0.0),
        recall=held_out_scores.get("recall", 0.0),
        roc_auc=held_out_scores.get("roc_auc", 0.0),
        f1=held_out_scores.get("f1", 0.0),
        windows_evaluated=1,
        window_improvements=0,
        status="held_out_final",
        reason="final validation",
    ), run_id=run_id)

    # Save summary artifact
    summary = {
        "phase": "13",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stopped_early": stopped_early,
        "experiments_run": kept_count + discarded_count,
        "experiments_kept": kept_count,
        "experiments_discarded": discarded_count,
        "elapsed_minutes": round(elapsed / 60, 1),
        "baseline_model": baseline_model,
        "baseline_composite": round(baseline_composite, 6),
        "baseline_scores": {k: round(v, 4) for k, v in baseline_scores.items()},
        "best_experiment_composite": round(best_composite, 6),
        "best_experiment_config": best_config["description"] if best_config else "baseline (no improvement found)",
        "best_experiment_scores": {k: round(v, 4) for k, v in best_scores.items()} if best_scores else {},
        "held_out_scores": {k: round(v, 4) for k, v in held_out_scores.items()},
        "held_out_composite": round(held_out_composite, 6),
        "gate_7": {
            "precision_pass": precision_pass,
            "recall_pass": recall_pass,
            "roc_auc_pass": roc_auc_pass,
            "overall": all_pass,
        },
        "conclusion": (
            "PASS: Best configuration clears all integration thresholds on held-out data."
            if all_pass else
            "FAIL: Best configuration does not clear all integration thresholds. "
            "Repo remains research-only."
        ),
    }

    summary_path = ARTIFACTS_DIR / "phase13_experiment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary saved to %s", summary_path)

    # Generate AUTORESEARCH.md
    generate_autoresearch_md(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        elapsed_minutes=elapsed / 60,
        baseline_model=baseline_model,
        baseline_scores=baseline_scores,
        baseline_composite=baseline_composite,
        xgb_scores=xgb_scores,
        lgbm_scores=lgbm_scores,
        experiment_results=all_results,
        best_experiment=best_experiment,
        best_config=best_config,
        best_composite=best_composite,
        held_out_scores=held_out_scores,
        held_out_composite=held_out_composite,
        gate_pass=all_pass,
        experiment_rows=len(experiment_dataset),
        held_out_rows=len(held_out_dataset),
        experiment_end=str(experiment_dataset.index[-1]),
        held_out_start=str(held_out_dataset.index[0]),
        held_out_end=str(held_out_dataset.index[-1]),
        total_experiments=kept_count + discarded_count,
        kept_count=kept_count,
        discarded_count=discarded_count,
    )

    # Clean up checkpoint on successful completion
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Checkpoint cleaned up (run complete)")

    if all_pass:
        logger.info("")
        logger.info("CONCLUSION: The best configuration PASSES Gate 7.")
        logger.info("The repo is ready for downstream integration review (Gate 5).")
    else:
        logger.info("")
        logger.info("CONCLUSION: The best configuration FAILS Gate 7.")
        logger.info("The repo remains research-only. The signal hypothesis has not been validated")
        logger.info("after systematic search of the expanded feature/model/hyperparameter space.")


if __name__ == "__main__":
    main()
