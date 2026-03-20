"""Trading-aligned comparison: ungated vs directional liquidity gate.

Bridges the walk-forward direction classifier to trading metrics.
For each prediction=1 signal, the "trade" earns the actual next-hour
return minus round-trip costs (fee + slippage). This directly measures
whether the gate improves operational outcomes.

Trading metrics computed:
  - trade count (number of signals taken)
  - win rate (fraction of trades with positive net return)
  - average return per trade (net of costs)
  - cumulative return (compounded)
  - max drawdown (from cumulative equity curve)
  - profit factor (gross wins / abs(gross losses))
  - Sharpe ratio (annualized, of per-trade returns)
  - total cost drag (fees + slippage paid)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_EVALUATION_MAX_ROWS,
    DEFAULT_FEE_PCT,
    DEFAULT_MIN_TRAIN_ROWS,
    DEFAULT_SLIPPAGE_PCT,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TEST_WINDOW,
    DEFAULT_TRAIN_WINDOW,
)
from data.pipeline import build_dataset
from evaluation.liquidity_gate import assign_regimes, DIRECTIONAL_REGIMES
from evaluation.walk_forward import _build_model, iter_walk_forward_slices

# Round-trip cost: entry fee + slippage + exit fee + slippage
ROUND_TRIP_COST = 2 * (DEFAULT_FEE_PCT + DEFAULT_SLIPPAGE_PCT)


def compute_trading_metrics(
    signals_df: pd.DataFrame,
    label: str,
) -> dict:
    """Compute trading metrics from a DataFrame of signal outcomes.

    Expects columns: prediction, actual_return_1h (the simple return of
    the next hour's close vs current close).
    """
    # Only rows where model signals "buy"
    trades = signals_df[signals_df["prediction"] == 1].copy()
    n_trades = len(trades)

    if n_trades == 0:
        return {
            "label": label,
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_return_per_trade_bps": 0.0,
            "cumulative_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "total_cost_drag_pct": 0.0,
            "total_rows": len(signals_df),
            "signal_rate": 0.0,
        }

    # Net return per trade = gross return - round-trip costs
    trades["net_return"] = trades["actual_return_1h"] - ROUND_TRIP_COST

    wins = trades[trades["net_return"] > 0]
    losses = trades[trades["net_return"] <= 0]

    win_rate = len(wins) / n_trades
    avg_return = trades["net_return"].mean()

    # Cumulative return (compounded)
    cumulative = (1 + trades["net_return"]).prod() - 1

    # Max drawdown from equity curve
    equity = (1 + trades["net_return"]).cumprod()
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0

    # Profit factor
    gross_wins = wins["net_return"].sum() if len(wins) > 0 else 0.0
    gross_losses = abs(losses["net_return"].sum()) if len(losses) > 0 else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    # Sharpe ratio (annualized assuming hourly trades)
    std = trades["net_return"].std()
    if std > 0:
        sharpe = (avg_return / std) * math.sqrt(8760)  # annualize from hourly
    else:
        sharpe = 0.0

    total_cost_drag = ROUND_TRIP_COST * n_trades

    return {
        "label": label,
        "trade_count": n_trades,
        "win_rate": win_rate,
        "avg_return_per_trade_bps": avg_return * 10000,  # basis points
        "cumulative_return_pct": cumulative * 100,
        "max_drawdown_pct": max_drawdown * 100,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "total_cost_drag_pct": total_cost_drag * 100,
        "total_rows": len(signals_df),
        "signal_rate": n_trades / len(signals_df) if len(signals_df) > 0 else 0.0,
    }


def main():
    print("Building dataset with liquidity features...")
    dataset, metadata = build_dataset(include_liquidity=True)
    eval_ds = dataset.tail(DEFAULT_EVALUATION_MAX_ROWS).copy()
    print(f"Evaluation rows: {len(eval_ds)}")
    print(f"Date range: {eval_ds.index.min()} to {eval_ds.index.max()}")
    print(f"Round-trip cost: {ROUND_TRIP_COST * 10000:.1f} bps")

    # Collect per-row predictions across walk-forward windows
    all_rows = []

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

        # Compute actual next-hour return for each row
        actual_return = test_df["close"].pct_change().shift(-1)

        combined = test_df[["close"]].copy()
        combined["prediction"] = predictions["prediction"].values
        combined["probability"] = predictions.get("probability", predictions["prediction"]).values
        combined["actual_return_1h"] = actual_return
        combined["regime"] = assign_regimes(test_df).values
        combined["window_id"] = window_id
        combined = combined.dropna(subset=["actual_return_1h"])

        all_rows.append(combined)
        print(f"  Window {window_id}: {len(combined)} rows, "
              f"signals={int(combined['prediction'].sum())}, "
              f"regime_dist={combined['regime'].value_counts().to_dict()}")

    signals_df = pd.concat(all_rows).sort_index()
    print(f"\nTotal rows: {len(signals_df)}, Total signals: {int(signals_df['prediction'].sum())}")

    # --- Ungated metrics ---
    ungated = compute_trading_metrics(signals_df, "ungated")

    # --- Gated: suppress signals during NEUTRAL ---
    gated_df = signals_df.copy()
    neutral_mask = ~gated_df["regime"].isin(DIRECTIONAL_REGIMES)
    gated_df.loc[neutral_mask, "prediction"] = 0
    gated = compute_trading_metrics(gated_df, "gated_directional")

    # --- Print comparison ---
    print("\n" + "=" * 76)
    print("TRADING-ALIGNED COMPARISON: Ungated vs Directional Liquidity Gate")
    print("=" * 76)

    metrics_order = [
        ("trade_count", "Trade count", "", "d"),
        ("signal_rate", "Signal rate", "", ".1%"),
        ("win_rate", "Win rate", "", ".1%"),
        ("avg_return_per_trade_bps", "Avg return/trade", "bps", ".2f"),
        ("cumulative_return_pct", "Cumulative return", "%", ".2f"),
        ("max_drawdown_pct", "Max drawdown", "%", ".2f"),
        ("profit_factor", "Profit factor", "", ".3f"),
        ("sharpe_ratio", "Sharpe ratio", "", ".2f"),
        ("total_cost_drag_pct", "Total cost drag", "%", ".2f"),
    ]

    col_w = 16
    print(f"\n{'Metric':<26} {'Ungated':>{col_w}} {'Gated':>{col_w}} {'Delta':>{col_w}}")
    print("-" * (26 + col_w * 3))

    for key, label, unit, fmt in metrics_order:
        u_val = ungated[key]
        g_val = gated[key]
        delta = g_val - u_val

        if fmt == "d":
            u_str = f"{int(u_val)}"
            g_str = f"{int(g_val)}"
            d_str = f"{int(delta):+d}"
        elif fmt == ".1%":
            u_str = f"{u_val:.1%}"
            g_str = f"{g_val:.1%}"
            d_str = f"{delta:+.1%}"
        elif fmt == ".2f":
            u_str = f"{u_val:.2f}{unit}"
            g_str = f"{g_val:.2f}{unit}"
            d_str = f"{delta:+.2f}{unit}"
        elif fmt == ".3f":
            u_str = f"{u_val:.3f}"
            g_str = f"{g_val:.3f}"
            d_str = f"{delta:+.3f}"
        else:
            u_str = f"{u_val}"
            g_str = f"{g_val}"
            d_str = f"{delta}"

        print(f"{label:<26} {u_str:>{col_w}} {g_str:>{col_w}} {d_str:>{col_w}}")

    # Coverage
    gated_rows = len(signals_df[signals_df["regime"].isin(DIRECTIONAL_REGIMES)])
    coverage = gated_rows / len(signals_df)
    print(f"\n{'Coverage':<26} {'100.0%':>{col_w}} {coverage:>{col_w}.1%}")
    print(f"{'Suppressed signals':<26} {'0':>{col_w}} {int(ungated['trade_count'] - gated['trade_count']):>{col_w}}")

    # --- Verdict ---
    print("\n" + "=" * 76)
    print("VERDICT")
    print("=" * 76)

    # The key question: does the gate make you richer?
    cum_delta = gated["cumulative_return_pct"] - ungated["cumulative_return_pct"]
    wr_delta = gated["win_rate"] - ungated["win_rate"]
    avg_r_delta = gated["avg_return_per_trade_bps"] - ungated["avg_return_per_trade_bps"]
    dd_delta = gated["max_drawdown_pct"] - ungated["max_drawdown_pct"]
    sharpe_delta = gated["sharpe_ratio"] - ungated["sharpe_ratio"]
    pf_delta = gated["profit_factor"] - ungated["profit_factor"]

    print(f"\nCumulative return delta:       {cum_delta:+.2f}%")
    print(f"Win rate delta:               {wr_delta:+.1%}")
    print(f"Avg return per trade delta:   {avg_r_delta:+.2f} bps")
    print(f"Max drawdown delta:           {dd_delta:+.2f}%")
    print(f"Sharpe ratio delta:           {sharpe_delta:+.2f}")
    print(f"Profit factor delta:          {pf_delta:+.3f}")

    # Classification
    practically_helpful = (
        cum_delta > 0
        and avg_r_delta > 0
        and (pf_delta > 0 or wr_delta > 0)
    )
    classification_only = (
        not practically_helpful
        and (wr_delta > 0 or avg_r_delta > 0)
    )

    if practically_helpful:
        verdict = "PRACTICALLY HELPFUL"
        detail = "Gate improves cumulative return, avg return per trade, and trade quality."
    elif classification_only:
        verdict = "CLASSIFICATION-ONLY HELPFUL"
        detail = "Gate improves some per-trade metrics but does not improve cumulative outcome."
    else:
        verdict = "NOT WORTH KEEPING"
        detail = "Gate does not improve trading outcomes."

    print(f"\nClassification: {verdict}")
    print(f"  {detail}")

    # Save results
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": DEFAULT_CLASSIFICATION_MODEL,
        "target_column": DEFAULT_TARGET_COLUMN,
        "round_trip_cost_bps": ROUND_TRIP_COST * 10000,
        "evaluation_rows": len(signals_df),
        "coverage": float(coverage),
        "ungated": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                    for k, v in ungated.items()},
        "gated": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
                  for k, v in gated.items()},
        "deltas": {
            "cumulative_return_pct": float(cum_delta),
            "win_rate": float(wr_delta),
            "avg_return_per_trade_bps": float(avg_r_delta),
            "max_drawdown_pct": float(dd_delta),
            "sharpe_ratio": float(sharpe_delta),
            "profit_factor": float(pf_delta),
        },
        "verdict": verdict,
    }

    output_path = Path("artifacts") / "backtest_trading_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
