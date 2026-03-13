# Bitcoin Signal Research

This repository is a Bitcoin signal-generation, model-research, evaluation, and signal-export repo for a separate trading strategy/execution system.

It does not place trades. It exists to answer one question honestly: is there a Bitcoin signal here that is strong enough to be worth consuming downstream after realistic costs?

The canonical execution plan and progress tracker live in [ROADMAP.md](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/ROADMAP.md).

## Current Judgment

As of March 13, 2026, this repo is materially more credible as a research system than it was, but it is still `research-only`, not a justified trading-strategy dependency.

The latest verified walk-forward result in this repo used the local historical dataset ending on July 17, 2025 and evaluated the cost-adjusted directional classifier over the most recent 365 days in that dataset. The current `xgboost_direction` model beat naive directional baselines on headline accuracy and ROC-AUC, but its actionable precision/recall remain weak:

- directional accuracy: `0.829`
- precision on cost-adjusted up moves: `0.215`
- recall on cost-adjusted up moves: `0.030`
- ROC-AUC: `0.634`

That is not good enough to claim a robust deployable signal. The repo now fails honestly and exports artifacts cleanly, but the present evidence does not justify downstream strategy integration.

The routine family-comparison artifact in `artifacts/model_comparison.json` is stricter because it uses a faster repeated-check comparison slice and explicit acceptance thresholds. On that comparison run, `xgboost_direction` remained the best family, but still missed the integration bar:

- precision: `0.097`
- recall: `0.051`
- ROC-AUC: `0.596`

Current provisional integration thresholds:

- precision >= `0.55`
- recall >= `0.15`
- ROC-AUC >= `0.60`

## Architecture

The current architecture is:

```text
.
├── data/         # canonical loaders, validation, dataset assembly
├── evaluation/   # targets, baselines, walk-forward evaluation, ablation/comparison
├── features/     # deterministic feature engineering functions + canonical pipeline
├── models/       # model interface, ARIMA baseline, tree-based models
├── signals/      # downstream signal export and validation
├── scripts/      # thin CLI wrappers for comparison, ablation, export
├── tests/        # unit + integration coverage for the research workflow
├── backtest.py   # evaluation entrypoint
├── collect.py    # backfill-first live analytics preparation
├── gen.py        # training entrypoint
├── README.md
└── ROADMAP.md
```

## Source Policy

- Research/training market: `Gemini BTCUSD spot 1h`
- Optional live monitoring market: `Binance BTCUSDT spot 1h`
- Sentiment: `alternative.me Fear & Greed`
- Canonical timezone: `UTC`

Training and evaluation use one canonical dataset builder in [data/pipeline.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/data/pipeline.py). Live analytics bootstrap from that same historical feature state before appending new candles in [collect.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/collect.py).

## Core Workflow

Setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Train the current default model and write model metadata:

```bash
.venv/bin/python gen.py
```

Run the default walk-forward evaluation:

```bash
.venv/bin/python backtest.py
```

Run the full test suite:

```bash
.venv/bin/python -m pytest -q
```

Run feature ablation:

```bash
.venv/bin/python scripts/run_ablation.py
```

Run the routine model-family comparison:

```bash
.venv/bin/python scripts/compare_models.py
```

Export the latest prediction to the downstream contract:

```bash
.venv/bin/python scripts/export_latest_signal.py
```

Or use `make test`, `make train`, `make backtest`, `make compare`, `make ablate`, and `make export-signal`.

## Data, Feature, and Target Contracts

Canonical feature generation lives in [features/pipeline.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/features/pipeline.py). The active feature set includes:

- lagged closes across short and medium lookbacks,
- return and log-return features,
- multi-horizon moving averages and MA spreads,
- RSI,
- volatility and ATR features,
- volume z-scores,
- multi-timeframe trend regime features,
- Fear & Greed sentiment.

Trading-aligned targets live in [evaluation/targets.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/evaluation/targets.py):

- `target_close_next`
- `target_simple_return_1`
- `target_log_return_1`
- `target_direction_1`
- `target_direction_cost_adj`
- `target_actionable_move`

The default training target is the cost-adjusted directional label, not raw next-close prediction.

## Evaluation Contract

[evaluation/walk_forward.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/evaluation/walk_forward.py) is the canonical scoring harness. It provides:

- strict train-on-past / predict-on-future walk-forward windows,
- baseline comparisons on identical windows,
- timestamped prediction artifacts,
- forecast metrics for regression targets,
- directional metrics for trading-aligned classification targets.

Naive prediction logic remains only as explicit baseline scaffolding in [backtest.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/backtest.py).

## Downstream Signal Contract

The first integration mode is a versioned file artifact written by [signals/export.py](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/signals/export.py).

`artifacts/latest_signal.json` contains:

- `timestamp`
- `instrument`
- `model_version`
- `feature_schema_version`
- `signal_schema_version`
- `prediction`
- `probability`
- `actionable`
- `generated_at`

Artifact validation checks required fields and freshness before a downstream repo should accept the signal.

## Artifacts

Current generated artifacts live in `artifacts/`:

- `dataset_metadata.json`
- `xgboost_direction.joblib`
- `xgboost_direction.metadata.json`
- `xgboost_direction_predictions.csv`
- `xgboost_direction_predictions.summary.json`
- `feature_ablation.json`
- `model_comparison.json`
- `latest_signal.json`

Historical 2025 outputs such as old plots and the legacy ARIMA pickle were removed because they were stale clutter, not evidence.

## Limitations

- The local historical dataset currently ends on July 17, 2025. Results are structurally stale until fresher data is ingested and re-evaluated.
- The current best model does not yet show strong enough cost-aware precision/recall to justify integration into a trading strategy repo.
- `scripts/compare_models.py` exists for reproducible family comparison, but ARIMA evaluation is still materially slower than the tree-based path and should be treated as research tooling, not a fast daily check.

## Decision

This repo is now a useful research and signal-export foundation, but not yet a genuinely useful production signal provider for a trading strategy repo. It should remain research-only until it shows materially better cost-aware edge on fresher data.
