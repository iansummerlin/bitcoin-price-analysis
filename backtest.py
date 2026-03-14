"""Signal evaluation entrypoint and compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EVALUATION_MAX_ROWS,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_TARGET_COLUMN,
)
from data.pipeline import build_dataset, write_dataset_metadata
from evaluation.history import save_backtest_report
from evaluation.regression_gate import print_verdict, run_regression_gate
from evaluation.walk_forward import walk_forward_evaluate
from features.pipeline import apply_feature_pipeline
from evaluation.cost_model import CostSimulator
from evaluation.signal_rules import MultiFactorRule, Signal, SignalRule


def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the canonical feature pipeline."""
    return apply_feature_pipeline(df, dropna=True)


def generate_naive_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy baseline helper retained for tests and baseline comparisons."""
    baseline = df.copy()
    baseline["predicted_price"] = baseline["close"]
    baseline.loc[baseline.index[1:], "predicted_price"] = (
        baseline["close"].iloc[1:] + baseline["close"].diff().iloc[1:]
    )
    return baseline


def run_backtest(
    df: pd.DataFrame,
    strategy: SignalRule,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
) -> tuple[CostSimulator, list[float]]:
    """Evaluate how a downstream consumer would react to a prediction stream."""
    simulator = CostSimulator(
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        slippage_pct=slippage_pct,
    )
    buy_and_hold_curve = [initial_capital]
    buy_and_hold_btc = initial_capital / df["close"].iloc[0]

    for _, row in df.iloc[1:].iterrows():
        current_price = row["close"]
        signal = strategy.on_candle(row, simulator)
        if signal == Signal.BUY and not simulator.has_position():
            amount = strategy.get_position_size_usd(row, simulator)
            if amount > 0:
                simulator.simulate_entry(current_price, amount, timestamp=row.name)
        elif signal == Signal.SELL and simulator.has_position():
            simulator.simulate_exit(current_price, timestamp=row.name)
        simulator.record_equity(current_price)
        buy_and_hold_curve.append(buy_and_hold_btc * current_price)

    return simulator, buy_and_hold_curve


def main() -> None:
    """Run the canonical walk-forward evaluation flow."""
    dataset, metadata = build_dataset()
    evaluation_dataset = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_dataset_metadata(metadata, artifacts_dir / "dataset_metadata.json")

    result = walk_forward_evaluate(
        evaluation_dataset,
        model_name=DEFAULT_CLASSIFICATION_MODEL,
        target_column=DEFAULT_TARGET_COLUMN,
        output_dir=artifacts_dir,
    )

    # Append to history and regenerate BACKTEST.md.
    save_backtest_report(
        result,
        dataset_rows=metadata.row_count,
        dataset_start=metadata.start,
        dataset_end=metadata.end,
    )

    # Run regression gate against previous result.
    verdict = run_regression_gate()
    print_verdict(verdict)

    summary = {
        "dataset_rows": metadata.row_count,
        "evaluation_rows": len(evaluation_dataset),
        "dataset_start": metadata.start,
        "dataset_end": metadata.end,
        "evaluation": result.__dict__,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
