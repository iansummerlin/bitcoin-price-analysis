# Bitcoin Signal Research

This repository is a Bitcoin signal-generation, model-research, evaluation, and signal-export repo for a separate trading strategy/execution system.

It does not place trades. It exists to answer one question honestly: is there a Bitcoin signal here that is strong enough to be worth consuming downstream after realistic costs?

The canonical execution plan and progress tracker live in [ROADMAP.md](/home/ixn/Documents/code/crypto/bitcoin-price-analysis/ROADMAP.md).

## Current Judgment

As of March 14, 2026, this repo is `research-only`, not a justified trading-strategy dependency.

The dataset was refreshed on March 14, 2026 (price data through March 13, 2026; sentiment through March 14, 2026). The latest walk-forward evaluation on fresh data:

- directional accuracy: `0.721`
- precision on cost-adjusted up moves: `0.169`
- recall on cost-adjusted up moves: `0.261`
- ROC-AUC: `0.641`

Recall now clears the threshold (>= 0.15) and ROC-AUC is above threshold (>= 0.60), but precision remains far below (needs >= 0.55). The model catches real moves but cannot distinguish them from noise with the current feature set. This confirms the next priority is expanding the data universe (on-chain, cross-asset, microstructure) — see Phase 12 in `ROADMAP.md`.

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

- The dataset was last refreshed on March 14, 2026 (price through March 13, sentiment through March 14).
- The current best model does not yet show strong enough cost-aware precision/recall to justify integration into a trading strategy repo.
- The current feature set is derived almost entirely from price/volume data and a single sentiment index. These are lagging, widely available indicators that the market has already priced in. Reaching the integration bar almost certainly requires alternative data sources (on-chain metrics, cross-asset signals, market microstructure). See Phase 12 in `ROADMAP.md`.
- `scripts/compare_models.py` exists for reproducible family comparison, but ARIMA evaluation is still materially slower than the tree-based path and should be treated as research tooling, not a fast daily check.

## Autonomous Experiment Loop — Limitations

This repo uses (or will use) an autonomous AI experiment loop inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to systematically search the feature/model/hyperparameter space. See Phase 13 in `ROADMAP.md` for full details. The following limitations apply:

**Overfitting risk.** Financial data is far noisier than LLM training data. A metric improvement on walk-forward windows may not generalize. The loop mitigates this with: a held-out validation set never seen during optimization, regime diversity checks (improvements must hold across multiple market conditions), and minimum improvement thresholds to filter noise.

**The loop cannot fix bad inputs.** If the feature set doesn't contain genuine predictive signal, no amount of automated experimentation will find edge. The loop optimizes the search space it's given — it doesn't create new data sources. Phase 12 (expand data universe) must come first.

**Metric gaming.** An agent optimizing a single composite metric over many iterations will find configurations that score well on that metric but may not represent robust trading signals. The held-out validation set is the final check against this, but it can only be used once.

**Compute cost.** Each experiment runs a full walk-forward evaluation (~10-15 minutes). Running 100 experiments takes ~20 hours. This is a meaningful compute commitment.

**Not a replacement for domain reasoning.** The loop finds what works empirically but doesn't explain why. A feature that improves metrics could be capturing a real market dynamic or could be a coincidence in the evaluation window. Human review of the surviving experiments is still necessary before trusting the results for live trading.

## Decision

This repo is a useful research and signal-export foundation, but not yet a genuinely useful production signal provider. Fresh data (March 2026) confirmed that recall is achievable but precision is the bottleneck — the model needs richer data sources to filter noise from signal. It should remain research-only until Phase 12 (data universe expansion) and Phase 13 (experiment loop) have been completed and evaluated.
