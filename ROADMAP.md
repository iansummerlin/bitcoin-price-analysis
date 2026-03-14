# Bitcoin Price Analysis Roadmap

## Document Status

- Owner intent: turn this repository into a reliable **Bitcoin signal-generation and model-research repo** for a separate trading strategy/execution repository.
- Audience: a fresh engineering agent with zero prior context.
- Primary constraint: this repo should prove whether it can produce a usable predictive signal. It should not become an execution bot.
- Current date context: this roadmap was last updated on March 14, 2026. Data was refreshed through March 13, 2026.

## Progress Tracker

Use these checkboxes to track progress directly in this file.

### Foundation (complete)

- [x] Remove direct execution pathways from the repository.
- [x] Remove obvious standalone-trading legacy surface (`trade.py`, `trade.html`, redundant docs/specs).
- [x] Finish Phase 0 repo framing cleanup.
- [x] Finish Phase 1 deterministic data pipeline.
- [x] Finish Phase 2 baseline evaluation harness.
- [x] Finish Phase 3 real model-backed walk-forward validation.
- [x] Finish Phase 4 trading-aligned target redesign.
- [ ] Finish Phase 5 feature engineering improvements.
- [x] Finish Phase 6 model family comparison.
- [x] Finish Phase 7 downstream signal contract.
- [x] Finish Phase 8 artifact reliability and freshness.
- [x] Finish Phase 9 documentation quality hardening.
- [x] Finish Phase 10 testing and reproducibility hardening.

### Signal quality (current priority)

- [x] Finish Phase 11 data refresh.
- [ ] Finish Phase 12 expand data universe and model families.
- [ ] Finish Phase 13 autonomous experiment loop.
- [ ] Pass all decision gates and declare repo ready for downstream integration review.

### Current Status After Execution

- The repo was refactored into a research/evaluation/export shape with canonical `data/`, `evaluation/`, `features/`, `models/`, `signals/`, and `scripts/` modules.
- `gen.py`, `backtest.py`, and `collect.py` are now thin entrypoints on top of shared pipelines.
- The repository now builds a deterministic dataset, evaluates models with strict walk-forward splits, exports self-describing signal artifacts, and ships an A-tier test suite.
- The current evidence still does **not** justify downstream trading-repo integration.
- Data was refreshed on March 14, 2026. The dataset now covers **2015-10-13 to 2026-03-13** (91,302 rows after feature pipeline).
- Latest verified walk-forward result on March 14, 2026 used the refreshed dataset and found `xgboost_direction` directional accuracy `0.721`, precision `0.169`, recall `0.261`, ROC-AUC `0.641` on the cost-adjusted target over the default recent evaluation slice. Recall now clears the threshold (>= 0.15) but precision remains far below (needs >= 0.55). The model catches real moves but can’t distinguish them from noise.
- Latest routine model comparison on March 14, 2026 confirmed `xgboost_direction` remains the best model family but still fails the integration bar.

### What blocks progress now

The foundation infrastructure (Phases 0–10) is solid. The three blockers are:

1. ~~**Stale data**~~ — resolved March 14, 2026. Dataset now extends through March 13, 2026.
2. **Narrow data universe** — the model only sees price-derived technical indicators and a single sentiment index. These are lagging, widely available, and largely priced in. The market has this information already. Fresh data confirmed this: recall improved to 0.261 (above threshold) but precision dropped to 0.169 (far below 0.55). The model detects moves but can't filter noise from signal with current features. Reaching the integration bar almost certainly requires data sources the market hasn't fully absorbed: on-chain metrics, cross-asset signals, order flow / microstructure data.
3. **Unexplored search space** — even with better data, the combination of features, hyperparameters, target thresholds, and model architectures is too large for manual exploration.

Phase 11 (data refresh) is complete. Phase 12 expands what the model can see. Phase 13 systematically searches the expanded space.

---

## Current Known Facts

These facts are important for a fresh agent and should be assumed true unless the code has changed materially:

- This repo is intended to be a **signal-generation/model-research repo**, not a trading bot.
- Direct execution code and obvious standalone-trading surface have already been removed.
- `backtest.py` is now the canonical signal-evaluation entrypoint and uses naive predictions only as explicit baseline logic.
- `collect.py` is being retained only because live signal analytics may be useful.
- If `collect.py` remains, it should operate as a **backfill-first live analytics pipeline**, not a "wait for lag windows to fill naturally" process.
- Training and live/inference data alignment now goes through shared schema and feature pipeline code, though live analytics should still be treated as monitoring until fresh data is revalidated.
- Data was refreshed on March 14, 2026 (price through March 13, sentiment through March 14). Historical artifacts prior to that date are stale.
- The roadmap is the canonical planning and progress-tracking document for this repository.

---

## Start Here

If you are a fresh agent with no other context, start in this order:

1. Read this roadmap fully once.
2. Read `README.md` and confirm it matches the roadmap.
3. Check the Progress Tracker above. Phases 0–10 are complete (except one remaining Phase 5 item). The current work is Phases 11, 12, and 13.
4. Phase 11 (data refresh) is complete. Data now extends through March 13, 2026.
5. Start with **Phase 12: Expand Data Universe & Model Families**. This is the biggest remaining phase and the one most likely to determine whether the repo succeeds or fails.
6. After Phase 12 delivers new data sources and model families integrated into the pipeline, move to **Phase 13: Autonomous Experiment Loop** to systematically search the expanded space.
7. The remaining Phase 5 item (remove weak features) will be handled automatically by the experiment loop in Phase 13.

If unsure what the next practical task is, the default next task is:

- **Phase 12** — integrate on-chain data as the first new data family, following the implementation order in the Phase 12 checklist.

---

## Mission

This repository should answer, with evidence:

1. Can we generate a Bitcoin prediction signal with repeatable out-of-sample edge?
2. Is that edge large enough to survive realistic trading costs?
3. Can we package that signal cleanly for use in a separate trading strategy repo?

If the evidence says "no", the correct outcome is to document that clearly and stop treating this repo as a production dependency.

---

## What This Repo Should Become

This repo should become a **research, training, evaluation, and signal-export system**.

It should eventually:

- build a clean, reproducible dataset,
- generate features deterministically,
- train candidate models,
- evaluate them with strict no-lookahead walk-forward validation,
- compare them against simple baselines,
- estimate whether the signal is tradable after costs,
- export a versioned artifact or prediction interface for another repo.

It should **not** own:

- exchange execution,
- wallet/key management,
- DEX integration,
- order routing,
- portfolio/risk orchestration beyond what is necessary to evaluate signal usefulness.

Those concerns belong in the downstream trading repo.

---

## Current Repo Assessment

### What works (Phases 0–10 complete)

- Deterministic data pipeline with schema validation.
- Walk-forward evaluation with strict no-lookahead splits.
- 6 naive baselines for honest comparison.
- Cost-adjusted trading-aligned targets.
- Model comparison and ablation reporting.
- Versioned signal export contract.
- Comprehensive test suite.
- A-tier documentation.

### What doesn't work yet

- **Data is stale** — dataset ends July 2025 (Phase 11 fixes this).
- **Feature universe is too narrow** — only price-derived technicals and one sentiment index. The market already has this information (Phase 12 fixes this).
- **Model roster is too narrow** — only XGBoost and ARIMA. Need LightGBM, CatBoost, ensemble methods (Phase 12 fixes this).
- **Search space is unexplored** — no systematic exploration of the feature/model/hyperparameter space (Phase 13 fixes this).
- **Model metrics fail the integration bar** — precision 0.215, recall 0.030 vs thresholds of 0.55, 0.15.

### Current bottom line

This repo has **solid research infrastructure** but is trying to find signal in data that doesn't contain enough of it. The infrastructure is not the problem — the inputs are.

---

## Working Principles

### 1. Out-of-sample evidence is the only real evidence

Do not make decisions from training error, fitted-value error, or single split results.

### 2. Data consistency is mandatory

Training, validation, and inference must use the same market semantics, schema, timestamp rules, and feature definitions.

### 3. Simpler models get priority

A simple stable signal is more valuable than a complex fragile model.

### 4. Signal usefulness matters more than forecast elegance

A model that predicts returns/direction well enough to support trades is more valuable than a model with slightly lower price RMSE but no tradable edge.

### 5. The repo must fail honestly

If the model is stale, unsupported, or not useful, the repo should say so explicitly.

### 6. Remove what is unnecessary

This repository should not keep dead UI, execution code, duplicate docs, stale planning files, or generated clutter as "legacy". If something does not support the signal-generation mission, remove it.

---

## Explicit Success Criteria

This repo is only considered seriously useful if all of the following are true:

- A fresh clone can build the dataset and run the evaluation flow without manual CSV editing.
- Walk-forward validation uses only past data to predict future windows.
- Candidate models are evaluated against naive baselines over the same periods.
- The chosen signal beats baselines materially and consistently out of sample.
- Cost-aware evaluation suggests the signal could survive realistic fees/slippage.
- Model artifacts or exported signals are versioned, validated, and self-describing.
- Another repo can consume outputs without depending on internal implementation details.

If those conditions are not met, this repo remains research-only.

## Provisional Acceptance Thresholds

These are the current minimum bars for recommending downstream integration review:

- cost-adjusted directional precision >= `0.55`
- cost-adjusted directional recall >= `0.15`
- cost-adjusted directional ROC-AUC >= `0.60`
- the best model should stay above those bars across repeated recent walk-forward windows, not just one split

If the best current model fails these thresholds, the repo remains research-only even if the software stack itself is solid.

---

## Immediate Structural Decision

Treat this repository as a **signal repo**, not a trading bot repo.

That means:

- keep backtesting only as a method of evaluating signal usefulness,
- remove direct execution pathways from this repository,
- identify and delete files that only supported the old standalone-trading shape,
- optimize for model evaluation and export,
- ensure all documentation reflects that role.

---

## Canonical Deliverables

By the end of the roadmap, this repo should provide these deliverables:

### Deliverable A: Reproducible dataset build

- A documented command or script that builds the full training/evaluation dataset.
- No manual file cleanup.
- Metadata on data source, date range, frequency, and generation time.

### Deliverable B: Deterministic feature pipeline

- A canonical feature builder used by training, evaluation, and inference.
- Shared schema validation.
- Tests that guard against drift and leakage.

### Deliverable C: Model training pipeline

- One or more candidate models trained from a stable interface.
- Versioned artifacts with metadata.
- Reproducible hyperparameters and training windows.

### Deliverable D: Evaluation pipeline

- Baselines plus candidate models.
- Walk-forward validation.
- Forecast metrics plus trading-relevant metrics.
- A concise summary of whether the signal is actually useful.

### Deliverable E: Signal export contract

- A CLI, file format, or API contract that downstream systems can consume.
- Stable input/output schema.
- Artifact validation and staleness checks.

### Deliverable F: A-tier documentation

- Clear repo purpose and scope.
- Accurate setup and workflow docs.
- A current architecture overview.
- A documented data/model/evaluation contract.
- A fresh-agent handoff document that matches the actual codebase.

---

## Current Repo Inventory

This section exists so a fresh agent can map the current repository shape to the intended future architecture.

### Core files to keep and refactor

- `gen.py`
  - Current role: main historical training entrypoint for ARIMA-style modeling.
  - Problems: manual data cleanup assumptions, training-centric structure, limited evaluation discipline.
  - Intended future role: thin training entrypoint built on shared data/training modules.

- `collect.py`
  - Current role: live-ish data collection and feature generation from Binance stream data.
  - Problems: market/schema mismatch with training path, mixed responsibilities, repeated feature computation.
  - Intended future role: canonical live inference/analytics data-preparation path with historical backfill plus live continuation, or remove it entirely if this repo will not own live signal analytics.

- `backtest.py`
  - Current role: strategy/backtest runner with naive prediction generation.
  - Problems: currently misleading because it does not validate trained model outputs.
  - Intended future role: evaluation harness for walk-forward model-backed signal testing.

- `backtest_metrics.py`
  - Current role: performance metric calculations.
  - Problems: useful but currently attached to a partially misleading evaluation flow.
  - Intended future role: shared evaluation/reporting metrics module.

- `config.py`
  - Current role: shared constants.
  - Problems: currently mixes concerns and bakes in assumptions that may drift.
  - Intended future role: explicit config/schema definitions for data, features, evaluation, and artifacts.

- `features/`
  - Current role: feature engineering modules.
  - Problems: features are usable, but pipeline consistency and leakage safeguards need tightening.
  - Intended future role: canonical feature pipeline shared by training, evaluation, and inference.

- `models/base.py`
  - Current role: model abstraction and scaler utilities.
  - Problems: too thin for the evaluation contract the repo needs.
  - Intended future role: common interface for fit/predict/save/load/metadata validation.

- `models/arima_model.py`
  - Current role: improved ARIMA wrapper.
  - Problems: still not integrated into a proper model-backed evaluation flow.
  - Intended future role: baseline time-series model implementation with strict out-of-sample reporting.

- `models/xgboost_model.py`
  - Current role: boosted-tree regression/classification wrappers.
  - Problems: current metrics are largely training-side, not deployment-grade evaluation.
  - Intended future role: primary non-linear candidate model family with proper validation.

- `tests/`
  - Current role: mixed unit/integration coverage.
  - Problems: does not yet prove the key repo claims end to end.
  - Intended future role: reproducibility and regression safety net for the real workflow.

### Files likely to keep as secondary support

- `portfolio.py`
  - Current role: trade/accounting simulation.
  - Intended future role: only keep if needed for signal usefulness evaluation; do not let it drive repo scope.

- `strategies.py`
  - Current role: strategy logic for threshold/ATR/multi-factor decisions.
  - Intended future role: keep only insofar as needed to test downstream signal consumption assumptions.

- `utils.py`
  - Current role: small utilities.
  - Intended future role: retain only generic shared helpers.

### Legacy product surface already removed

The following were intentionally removed to keep the repository focused:

- `trade.py`
- `trade.html`
- `documentation.md`
- `specs/`
- duplicate legacy tests/docs that did not support the current mission

This should be treated as the default cleanup policy: after identifying non-core legacy surface, remove it rather than preserving it.

### Data and artifact files present in the repo

- `BTCUSD_1H.csv`
- `BTC_sentiment.csv`
- `BTCUSD_trading.csv`
- `arima-btc-closing-price.pkl`

These should be treated as local artifacts and reference material, not as proof of current model validity.

---

## Recommended Final Repository Shape

This is the target shape, not necessarily the immediate next commit:

```text
.
├── data/
├── evaluation/
├── features/
├── models/
├── signals/
├── scripts/
├── tests/
├── README.md
├── ROADMAP.md
└── requirements.txt
```

---

## Execution Order

Work through phases in order. Do not skip ahead to more advanced model work until earlier gates are met.

1. ~~Repo framing and workflow cleanup.~~ (done)
2. ~~Deterministic data pipeline.~~ (done)
3. ~~Baselines and honest evaluation harness.~~ (done)
4. ~~Real walk-forward model-backed backtesting/evaluation.~~ (done)
5. ~~Better targets and features.~~ (mostly done — final cleanup folded into Phase 13)
6. ~~Model comparison.~~ (done)
7. ~~Downstream signal contract.~~ (done)
8. ~~Reliability, testing, and artifact hardening.~~ (done)
9. ~~Data refresh.~~ (done)
10. **Expand data universe and model families.** ← you are here
11. **Autonomous experiment loop.**
12. Final gate re-evaluation and integration decision.

---

## Phase 0: Reframe the Repo Around Signal Generation

### Checklist

- [x] Remove direct execution from the repository.
- [x] Remove obvious standalone-trading docs/UI/specs.
- [x] Reframe `README.md` around signal generation.
- [x] Reframe `ROADMAP.md` around signal generation.
- [x] Audit remaining comments/docstrings for outdated trading-bot language.
- [x] Remove any remaining non-core generated clutter from the repository tree.
- [x] Confirm `backtest.py` is documented purely as signal evaluation.

### Objective

Make the codebase and docs match the real goal.

### Why this phase exists

Right now the repo is conceptually muddled: part model research, part paper trading, part execution placeholder. That increases the chance of bad architectural decisions.

### Required outcomes

- The README describes this repo as a signal-generation/model-evaluation repo.
- Direct execution is removed from the repository workflow.
- The main scripts and directory structure reflect research and export, not trade execution.
- No unnecessary legacy artifacts remain in the repository after the cleanup pass.

### Concrete tasks

- Audit docs:
  - `README.md`
  - any planning/spec docs
  - comments and module docstrings that imply this repo places trades
- Inventory all top-level files, scripts, generated artifacts, logs, and UI remnants.
- Classify each item as either:
  - core to the signal mission, or
  - remove.
- Remove execution-oriented files that do not belong in a signal repo.
- Remove redundant docs once their useful content has been consolidated into canonical files.
- Remove stale generated artifacts and log files that should not remain in the repository.
- Ensure `backtest.py` is framed as signal evaluation, not proof of strategy profitability.

### Suggested file targets

- `README.md`
- `backtest.py`
- `ROADMAP.md`

### Exit criteria

- A new reader can identify the repo’s purpose in under two minutes.
- No code or doc path implies this repo is responsible for executing trades.
- No knowingly retained non-core legacy artifacts remain.

---

## Phase 1: Build a Deterministic Data Pipeline

### Checklist

- [x] Remove all manual CSV-editing assumptions from the core workflow.
- [x] Create one canonical historical price loader.
- [x] Create one canonical sentiment loader.
- [x] Create one canonical merged dataset builder.
- [x] Add data validation checks for timestamps, schema, and OHLC sanity.
- [x] Decide and document the single market/source policy for modeling.
- [x] Unify training and inference schema definitions.
- [x] Redesign `collect.py` around historical backfill plus live continuation.
- [x] Ensure live analytics can start immediately after bootstrapping from historical data.

### Objective

Create one canonical, reproducible data flow for training and evaluation.

### Why this phase exists

Manual cleanup and data-source mismatch make all later results suspect.

### Required outcomes

- No manual CSV editing.
- One canonical loader for historical data.
- One canonical sentiment merge path.
- Clear timestamp and timezone handling.
- Consistent market semantics between training and inference.
- If live analytics are retained, the pipeline supports historical backfill before live updates so the model can produce immediate predictions without waiting for lag windows to fill in real time.

### Concrete tasks

- Replace manual dataset preparation with scripted cleanup/parsing.
- Build canonical loaders:
  - price data loader,
  - sentiment loader,
  - merged dataset builder.
- Decide whether this repo will own live signal analytics:
  - if yes, keep `collect.py` and redesign it around backfill plus live continuation,
  - if no, remove `collect.py`.
- Add data validation for:
  - duplicate timestamps,
  - missing timestamps,
  - non-monotonic index,
  - impossible OHLC values,
  - missing required columns,
  - unexpected null density.
- Decide on a single market source for modeling:
  - Gemini spot,
  - Binance spot/futures,
  - or a deliberate normalized abstraction.
- Ensure inference uses the same schema as training.
- If `collect.py` is retained:
  - backfill at least the maximum required lookback window before live inference begins,
  - compute feature state immediately from the backfill,
  - append live candles incrementally,
  - log predictions together with later realized outcomes for live model monitoring.

### Suggested file targets

- `gen.py`
- `collect.py`
- `config.py`
- new shared loader module if needed
- tests for data loading and validation

### Exit criteria

- A fresh environment can build the dataset from source.
- No human cleanup is required.
- Train/eval/inference schema is explicitly defined and validated.
- If live analytics are in scope, the system can start producing model outputs immediately after bootstrapping from historical data.

---

## Phase 2: Establish Honest Baselines

### Checklist

- [x] Implement persistence price baseline.
- [x] Implement zero-return baseline.
- [x] Implement momentum baseline.
- [x] Implement mean-reversion baseline.
- [x] Add buy-and-hold comparison where appropriate.
- [x] Build a unified scoring harness for baseline/model comparison.
- [x] Document reproducible baseline results.

### Objective

Prove that any future model actually adds value.

### Why this phase exists

Without baselines, model improvements are mostly narrative.

### Required baselines

- persistence price baseline: next close = current close,
- zero-return baseline,
- momentum baseline,
- mean-reversion baseline,
- buy-and-hold benchmark for downstream strategy comparison.

### Concrete tasks

- Create a unified scoring harness.
- Evaluate every baseline over identical windows.
- Report:
  - RMSE/MAE where relevant,
  - return error where relevant,
  - directional accuracy,
  - precision/recall on actionable moves,
  - coverage and turnover proxies.

### Suggested file targets

- `backtest.py`
- `backtest_metrics.py`
- `models/`
- new evaluation module if needed

### Exit criteria

- Every candidate model is compared directly with naive alternatives.
- Baseline results are reproducible and documented.

---

## Phase 3: Replace Fake Backtesting With Real Walk-Forward Validation

### Checklist

- [x] Refactor prediction generation behind a common model interface.
- [x] Use real model predictions in evaluation instead of naive placeholders.
- [x] Implement train-on-past / predict-on-future walk-forward windows.
- [x] Store predictions with timestamps and metadata.
- [x] Add no-lookahead tests for splitting, scaling, and alignment.
- [x] Remove or isolate naive prediction code as baseline-only logic.

### Objective

Evaluate actual model predictions, not synthetic placeholders.

### Why this phase exists

The current backtest uses naive momentum predictions rather than trained model output. That makes the reported strategy metrics unsuitable for deciding whether the model is useful.

### Required outcomes

- Walk-forward windows train only on past data.
- Predictions are generated by the real model interface.
- Predictions flow into downstream signal evaluation.
- No lookahead leakage in features, scaling, or target construction.

### Concrete tasks

- Refactor prediction generation behind a common model API.
- For each walk-forward split:
  - fit on training window,
  - generate predictions on next unseen window,
  - store predictions with timestamps and metadata,
  - evaluate both forecast quality and signal usefulness.
- Remove or demote naive prediction code to explicit baseline-only status.
- Add tests for:
  - no-lookahead splits,
  - scaler fit only on training data,
  - output alignment between predictions and timestamps.

### Suggested file targets

- `backtest.py`
- `models/base.py`
- `models/arima_model.py`
- `models/xgboost_model.py`
- tests for evaluation flow

### Exit criteria

- Reported backtest/evaluation metrics correspond to real model outputs.
- There is no accidental dependence on future information.

---

## Phase 4: Align the Prediction Target With Trading Use

### Checklist

- [x] Add log-return target support.
- [x] Add simple-return target support.
- [x] Add directional label support.
- [x] Add cost-adjusted directional label support.
- [x] Define actionable-move threshold from fees/slippage/buffer.
- [x] Prefer return/direction targets over raw next-price prediction by default.

### Objective

Optimize for a signal that another trading repo can actually use.

### Why this phase exists

Predicting raw next-hour price is often a poor target for tradable signal quality.

### Preferred target order

1. Direction of next-period move above a cost-aware threshold.
2. Expected next-period return.
3. Probability that next-period move clears cost buffer.
4. Raw next price only as a benchmark.

### Concrete tasks

- Add target builders for:
  - log return,
  - simple return,
  - direction label,
  - cost-adjusted direction label.
- Define "actionable move" using estimated fee + slippage + minimum safety buffer.
- Score classification models on:
  - precision,
  - recall,
  - F1,
  - ROC-AUC if useful,
  - calibration if probabilities are emitted.
- Keep price-level prediction only for comparison, not as default objective.

### Suggested file targets

- `models/`
- training/evaluation scripts
- tests for label construction

### Exit criteria

- The default target is explicitly trading-aligned.
- Thresholds reflect costs, not arbitrary constants alone.

---

## Phase 5: Rational Feature Engineering

### Checklist

- [x] Audit current features for redundancy and leakage.
- [x] Add return-based features.
- [x] Add volume-based features.
- [x] Add multi-timeframe trend/regime features.
- [x] Add ablation reporting.
- [ ] Remove weak features when evidence supports removal.

### Objective

Improve predictive power without inflating fragility.

### Why this phase exists

The current feature set is a reasonable start, but it is still limited and not clearly optimized for out-of-sample signal quality.

### Priority feature families

- return-based features,
- volume-based features,
- multi-timeframe trend features,
- volatility-regime features,
- sentiment regime features,
- interaction features only where justified.

### Concrete tasks

- Audit existing features for redundancy and leakage risk.
- Add feature families in small controlled batches.
- Add ablation reports to measure whether each feature family helps.
- Add model-side feature importance reporting where appropriate.
- Prefer robust features over clever but brittle ones.

### Suggested file targets

- `features/`
- `config.py`
- `models/`
- tests for new features

### Exit criteria

- Added features improve out-of-sample performance over multiple windows.
- Weak or redundant features are removed when evidence supports removal.

---

## Phase 6: Evaluate Model Families in the Right Order

### Checklist

- [x] Upgrade ARIMA evaluation to strict out-of-sample reporting.
- [x] Add proper out-of-sample validation for boosted-tree models.
- [x] Compare models on forecast, directional, and cost-aware signal metrics.
- [x] Produce a reproducible model comparison summary.
- [x] Conclude whether a model family is genuinely good enough.

### Objective

Use engineering time efficiently.

### Model priority order

1. baselines,
2. ARIMA/SARIMAX on returns,
3. boosted-tree regression/classification,
4. volatility-aware or regime-aware models,
5. sequence neural nets only if simpler models fail and data quality supports them.

### Concrete tasks

- Upgrade ARIMA evaluation to be fully out of sample.
- Add proper validation to boosted-tree models; current training-only metrics are not enough.
- Compare models on:
  - forecast performance,
  - directional edge,
  - signal precision after cost thresholds,
  - stability across windows/regimes.
- Record model comparison results in a reproducible summary.

### Suggested file targets

- `models/arima_model.py`
- `models/xgboost_model.py`
- model comparison reporting module or script

### Exit criteria

- One model family is clearly best by evidence, or the repo concludes no model is good enough.

---

## Phase 7: Define the Downstream Signal Contract

### Checklist

- [x] Decide first integration mode: artifact, file export, CLI, or service.
- [x] Define canonical signal output fields.
- [x] Document input and output schemas.
- [x] Add validation for stale or incompatible artifacts/signals.
- [x] Verify another repo can consume outputs without internal coupling.

### Objective

Make this repo easy for the other trading repo to consume.

### Why this phase exists

A good model is still not usable if its outputs are unstable or undocumented.

### Required signal fields

At minimum, every exported prediction/signal should include:

- timestamp,
- instrument identifier,
- model version,
- feature/schema version,
- predicted return or direction,
- confidence/probability score if available,
- optional "actionable" flag after cost thresholding.

### Concrete tasks

- Decide on first integration mode:
  - versioned artifact load,
  - scored CSV/parquet output,
  - CLI command,
  - minimal service.
- Prefer a CLI or file-based interface first for simplicity.
- Document input schema and output schema.
- Add validation so downstream systems can reject:
  - stale models,
  - mismatched features,
  - wrong symbol/source assumptions.

### Suggested file targets

- new `signals/` or `scripts/` modules
- README integration section
- tests for schema validation

### Exit criteria

- Another repo can consume outputs without knowing internal code structure.

---

## Phase 8: Artifact Reliability and Freshness

### Checklist

- [x] Standardize save/load interfaces.
- [x] Attach metadata to artifacts.
- [x] Add artifact staleness checks.
- [x] Add schema compatibility checks before prediction.
- [x] Review and improve artifact size/storage format.

### Objective

Ensure exported models/signals are safe to depend on.

### Why this phase exists

Large opaque artifacts and stale models create operational risk.

### Concrete tasks

- Standardize save/load interfaces.
- Attach metadata to artifacts:
  - training window,
  - feature set,
  - source market,
  - evaluation date,
  - evaluation summary,
  - creation date.
- Add artifact staleness checks.
- Add schema compatibility checks before prediction.
- Review artifact size and storage format; current large pickle should be treated as a problem to solve.

### Suggested file targets

- `models/base.py`
- model-specific save/load logic
- new metadata schema module if needed

### Exit criteria

- Artifacts are self-describing.
- Downstream consumers can validate them automatically.

---

## Phase 9: Documentation Quality

### Checklist

- [x] Keep `README.md` fully aligned with current code and workflow.
- [x] Keep `ROADMAP.md` fully aligned with current code and workflow.
- [x] Add or improve architecture documentation.
- [x] Document data sources, feature schema, model interfaces, and evaluation flow.
- [x] Document the downstream signal contract.
- [x] Remove redundant or stale documentation.
- [x] Confirm a fresh agent can operate the repo from docs alone.

### Objective

Make the repository understandable, trustworthy, and easy to pick up cold.

### Why this phase exists

Even a good model repo becomes hard to use if the documentation is stale, vague, or inconsistent with the code. Documentation quality needs to be treated as a first-class engineering deliverable.

### Concrete tasks

- Keep `README.md` aligned with the current architecture and workflow.
- Keep `ROADMAP.md` aligned with the current codebase and cleanup decisions.
- Add or improve documentation for:
  - repository purpose,
  - current status and limitations,
  - data sources,
  - feature schema,
  - model interfaces,
  - evaluation workflow,
  - downstream signal contract,
  - artifact metadata expectations.
- Add a concise architecture section that explains how data, features, models, evaluation, and signal export fit together.
- Remove redundant docs instead of letting multiple partially-correct documents coexist.
- Treat documentation drift as a defect to fix alongside code drift.

### Exit criteria

- A fresh agent can understand the repo structure and intended workflow from docs alone.
- There is one canonical source of truth for repo direction and one for day-to-day usage.
- There are no stale docs describing removed workflows or deleted files.

---

## Phase 10: Testing and Reproducibility

### Checklist

- [x] Ensure documented setup installs everything needed for core workflows.
- [x] Add tests for data loading and schema validation.
- [x] Add tests for feature determinism.
- [x] Add tests for no-lookahead behavior.
- [x] Add tests for model interface consistency.
- [x] Add tests for artifact/schema compatibility.
- [x] Add tests for downstream signal export.
- [x] Add one lightweight end-to-end workflow test.
- [x] Verify a fresh environment can run the documented workflow.

### Objective

Make repo claims verifiable by a fresh agent or engineer.

### Why this phase exists

A roadmap is not actionable if the documented workflow cannot be reproduced in a clean environment.

### Concrete tasks

- Ensure documented setup installs everything required to run tests and core workflows.
- Add tests for:
  - data loading,
  - feature determinism,
  - no-lookahead splitting,
  - model interface consistency,
  - artifact/schema compatibility,
  - downstream signal export.
- Add one lightweight end-to-end test covering:
  - data load,
  - feature build,
  - model fit on toy data,
  - walk-forward prediction,
  - evaluation report creation.
- Keep tests lightweight enough for frequent execution.

### Suggested file targets

- `requirements.txt`
- `Makefile`
- `tests/`

### Exit criteria

- A fresh environment can run the documented test/evaluation workflow successfully.

---

## Phase 11: Data Refresh

### Checklist

- [x] Identify and document current data sources and their coverage gaps (dataset ends July 17, 2025).
- [x] Download fresh Gemini BTCUSD spot 1h candle data through the current date.
- [x] Validate the fresh data passes all existing schema and OHLCV sanity checks.
- [x] Merge the fresh data into the canonical `BTCUSD_1H.csv` without duplicates or gaps.
- [x] Download fresh Fear & Greed sentiment data through the current date.
- [x] Merge fresh sentiment into `BTC_sentiment.csv`.
- [x] Run `build_dataset()` and confirm the full pipeline completes without errors.
- [x] Run the existing test suite and confirm all tests pass on the extended dataset.
- [x] Run `make backtest` on the fresh data and record new baseline metrics.
- [x] Run `make compare` on the fresh data and record new model comparison metrics.
- [x] Update `artifacts/dataset_metadata.json` with new date range.
- [x] Update this roadmap and `README.md` with the new data end date and fresh metric baselines.

### Objective

Get the dataset current so that all subsequent research produces trustworthy, non-stale results.

### Why this phase exists

The dataset ends July 17, 2025. Every model result since then is structurally suspect. No feature engineering, hyperparameter tuning, or experiment loop can produce reliable conclusions on 8-month-old data. This is the single highest-priority blocker.

### Concrete tasks

- Determine whether CryptoDataDownload still provides the Gemini BTCUSD 1h dataset, or whether an alternative source is needed.
- If the source has changed, update `data/loaders.py` to handle the new format while preserving schema compatibility.
- Download, validate, and merge fresh price data.
- Download and merge fresh sentiment data.
- Re-run the full evaluation pipeline on the extended dataset.
- Record the fresh baseline metrics — these become the new reference point for Phases 12 and 13.
- Do **not** retune any model parameters in this phase. The goal is to establish a clean, current baseline before experimentation begins.

### Exit criteria

- The dataset extends to within 7 days of the current date.
- All existing tests pass on the extended dataset.
- Fresh walk-forward and model comparison metrics are recorded and documented.
- No model parameters were changed — only data was updated.

---

## Phase 12: Expand Data Universe & Model Families

### Checklist

#### On-chain data integration

- [ ] Research available on-chain data sources and their API access (Glassnode, CryptoQuant, Coin Metrics, free alternatives).
- [ ] Select a primary on-chain data provider based on: cost, API reliability, historical depth, and feature coverage.
- [ ] Build a canonical on-chain data loader in `data/loaders.py` (or a new `data/onchain.py` module).
- [ ] Integrate on-chain features into the dataset pipeline with proper timestamp alignment and forward-fill policy.
- [ ] Add the following on-chain feature families to `features/`:
  - [ ] Exchange flow features: net exchange inflows/outflows, exchange reserve changes.
  - [ ] Whale/large holder activity: large transaction count, wallet concentration changes.
  - [ ] Miner metrics: miner revenue, hash rate changes, miner outflows.
  - [ ] Valuation metrics: MVRV ratio, NVT signal, realized price vs market price.
  - [ ] Network activity: active addresses, transaction count, new address momentum.
- [ ] Add schema validation for on-chain data (null handling, staleness, expected ranges).
- [ ] Add tests for on-chain feature computation and pipeline integration.

#### Cross-asset data integration

- [ ] Identify and document cross-asset data sources (Yahoo Finance, FRED, exchange APIs).
- [ ] Build a canonical cross-asset data loader.
- [ ] Add the following cross-asset feature families to `features/`:
  - [ ] USD strength: DXY index or USD basket proxy.
  - [ ] Equity correlation: S&P 500 returns, rolling BTC/SPX correlation.
  - [ ] Risk appetite: VIX level and changes, gold/BTC ratio.
  - [ ] Crypto relative strength: ETH/BTC ratio, BTC dominance.
- [ ] Handle timezone and market-hours alignment (traditional markets close on weekends; crypto doesn't).
- [ ] Add schema validation and tests for cross-asset features.

#### Market microstructure data integration

- [ ] Identify and document microstructure data sources (Binance futures API, Coinglass, exchange APIs).
- [ ] Build a canonical microstructure data loader.
- [ ] Add the following microstructure feature families to `features/`:
  - [ ] Derivatives signals: perpetual funding rate, open interest changes, long/short ratio.
  - [ ] Liquidation data: liquidation volume, cascading liquidation detection.
  - [ ] Order book features: bid-ask spread, order book depth imbalance (if available at 1h resolution).
- [ ] Add schema validation and tests for microstructure features.

#### Multi-timeframe target exploration

- [ ] Add 4-hour forward target (direction and cost-adjusted direction).
- [ ] Add daily (24h) forward target (direction and cost-adjusted direction).
- [ ] Adjust cost thresholds per timeframe (longer horizons tolerate higher costs).
- [ ] Run walk-forward evaluation on each timeframe and compare which horizon produces the strongest signal.
- [ ] Document the best-performing timeframe and update the default target if warranted.

#### Alternative model families

- [ ] Add LightGBM classifier implementing `BaseModel` interface.
- [ ] Add CatBoost classifier implementing `BaseModel` interface.
- [ ] Add at least one temporal model (LSTM or lightweight transformer) implementing `BaseModel` interface, if data volume justifies it.
- [ ] Add an ensemble method that combines predictions from multiple model families.
- [ ] Run `make compare` with the expanded model roster and record results.
- [ ] Update `requirements.txt` with new dependencies.

#### Integration and validation

- [ ] Confirm all new data sources merge cleanly into `build_dataset()` without breaking existing features.
- [ ] Confirm all existing tests still pass after the expanded feature set is integrated.
- [ ] Run a full walk-forward evaluation with the expanded feature set and record metrics.
- [ ] Compare the expanded-feature results against the Phase 11 baseline to measure the impact of new data.
- [ ] Add ablation reporting for each new data family (on-chain, cross-asset, microstructure) so the experiment loop knows which families carry signal.
- [ ] Update `config.py` with the expanded `EXOG_COLUMNS` list and any new configuration.
- [ ] Update `README.md` source policy section with new data sources.
- [ ] Update this roadmap with results.

### Objective

Give the model access to data sources that carry genuine predictive signal — information the market hasn't fully priced in — rather than trying to extract edge from price-derived indicators the entire market already sees.

### Why this phase exists

The current model uses 23 features derived almost entirely from price and volume data (lagged closes, returns, RSI, ATR, moving averages) plus a single sentiment index. These are lagging indicators that every market participant has access to. Trying to predict BTC direction from RSI and moving averages is like trying to predict tomorrow's weather from today's temperature alone — technically correlated, but the market has already priced that correlation in.

The integration bar requires precision >= 0.55 and recall >= 0.15. Getting there from 0.215 / 0.030 is not a tuning problem — it's a **data problem**. The model needs to see things the market hasn't fully digested:

- **On-chain data** provides supply/demand signals invisible in price (exchange outflows signal accumulation before price moves).
- **Cross-asset data** captures macro regime shifts that drive BTC (DXY strength, risk-on/risk-off transitions).
- **Microstructure data** captures positioning and leverage that create short-term predictable dynamics (funding rate extremes, liquidation cascades).
- **Multi-timeframe targets** may reveal that 1-hour is simply the wrong prediction horizon — daily direction after costs may be far more predictable.

Without this phase, the experiment loop (Phase 13) would be searching a narrow, low-signal space. This phase expands the space so the loop has something worth finding.

### Critical design constraints

1. **Every new data source must go through the existing validation pipeline.** No raw API responses flowing directly into features. Schema validation, null handling, and staleness checks apply to all new data equally.

2. **New features must be deterministic and reproducible.** The same input data must produce the same features every time. No API calls during feature computation — data is downloaded and cached first, then features are computed from the cache.

3. **Lookback periods must be documented and respected.** On-chain data may have different availability frequencies (daily vs hourly). The pipeline must handle this cleanly via forward-fill with explicit documentation of the fill policy.

4. **Paid API dependencies must be documented.** If a data source requires a paid API key, document the cost, the free tier limits, and what features are lost without it. The repo should degrade gracefully to free data sources.

5. **Each data family is added and validated independently before combining.** Do not add on-chain + cross-asset + microstructure all at once. Add one family, validate it works, measure its impact, then add the next. This prevents debugging nightmares and lets ablation isolate which families carry signal.

### Suggested implementation order within this phase

1. **On-chain data first** — highest expected signal-to-noise ratio, most unique information relative to price.
2. **Cross-asset data second** — captures macro context, relatively easy to source.
3. **Microstructure data third** — highest noise, most complex to integrate, but captures short-term dynamics.
4. **Multi-timeframe targets** — can be done in parallel with any data work.
5. **Alternative models last** — better to have rich features with one model than poor features with five models.

### Suggested file targets

- `data/onchain.py` (new — on-chain data loader)
- `data/crossasset.py` (new — cross-asset data loader)
- `data/microstructure.py` (new — microstructure data loader)
- `data/loaders.py` (extend with new source registration)
- `data/pipeline.py` (extend `build_dataset()` to merge new sources)
- `features/onchain.py` (new — on-chain feature computation)
- `features/crossasset.py` (new — cross-asset feature computation)
- `features/microstructure.py` (new — microstructure feature computation)
- `features/pipeline.py` (extend to include new feature families)
- `models/lightgbm_model.py` (new)
- `models/catboost_model.py` (new)
- `models/temporal_model.py` (new — LSTM/transformer, optional)
- `models/ensemble_model.py` (new)
- `config.py` (expanded feature lists, new data source configs)
- `tests/` (new tests for each data family and model)

### Exit criteria

- At least two new data families (on-chain + one other) are integrated into the pipeline and producing features.
- At least one alternative model family (LightGBM or CatBoost) is implemented and passing the model comparison harness.
- Ablation results show at least one new data family contributes measurable out-of-sample improvement over the Phase 11 baseline.
- All existing tests still pass.
- The expanded feature set and model roster are documented in README.md.
- The repo is ready for the Phase 13 experiment loop to search the expanded space.

---

## Phase 13: Autonomous Experiment Loop

### Checklist

#### Infrastructure

- [ ] Create `program.md` with research directives, scope boundaries, and safety rails.
- [ ] Create `scripts/experiment_loop.py` — the autonomous runner script.
- [ ] Define the single composite metric used to judge experiments (e.g., cost-adjusted F1, or precision × recall product).
- [ ] Create `results.tsv` format for experiment logging (commit hash, metric, status, description).
- [ ] Designate immutable files the agent must never modify (evaluation harness, data loaders, test suite).
- [ ] Designate mutable files the agent may modify (features, model hyperparameters, target thresholds, config).
- [ ] Reserve a held-out final validation set that the experiment loop never sees during optimization.
- [ ] Add a `make experiment` target to the Makefile.

#### Safety rails

- [ ] Implement minimum improvement threshold — discard changes below noise floor (e.g., < 0.005 metric improvement).
- [ ] Implement maximum complexity budget — reject changes that add more than N lines without proportional metric gain.
- [ ] Implement regime diversity check — require the improvement to hold across at least 2 distinct walk-forward windows, not just the aggregate.
- [ ] Implement rollback on test failure — if `pytest` fails after a change, revert automatically.
- [ ] Implement experiment budget cap — stop after N experiments or M hours to prevent runaway compute.

#### Experiment scope

- [ ] Feature engineering: add, remove, or modify features in `features/pipeline.py` and `config.py`.
- [ ] Feature selection: enable/disable feature families from Phase 12 (on-chain, cross-asset, microstructure).
- [ ] Hyperparameter tuning: modify model parameters across all model families (XGBoost, LightGBM, CatBoost, ensemble).
- [ ] Target threshold tuning: adjust cost buffer, actionable move threshold, prediction timeframe in `config.py`.
- [ ] Probability threshold tuning: adjust classification decision boundary.
- [ ] Model selection: switch between or combine Phase 12 model families.
- [ ] New model architectures: add new model classes in `models/` that conform to the `BaseModel` interface.

#### Validation

- [ ] Run the experiment loop end-to-end on at least one full cycle and confirm keep/discard logic works.
- [ ] Confirm `results.tsv` accurately logs all experiments with correct metrics.
- [ ] Confirm the held-out validation set was never used during the experiment loop.
- [ ] Evaluate the best surviving configuration on the held-out set and record final metrics.
- [ ] If final metrics clear the integration bar, update the progress tracker and gate results.

### Objective

Systematically explore the feature/hyperparameter/target search space using an autonomous AI experiment loop, inspired by Karpathy's autoresearch pattern, layered on top of the existing walk-forward evaluation infrastructure.

### Why this phase exists

After Phase 12 expands the data universe and model roster, the search space is large: dozens of features across multiple data families × hyperparameter grids × target thresholds × multiple model architectures. That is too large for manual exploration. An autonomous loop can run ~12 experiments/hour, covering overnight what would take weeks of manual iteration.

### How it works

The experiment loop follows the autoresearch pattern adapted for financial signal research:

```
1. Agent reads current state (code, metrics, experiment history)
2. Agent proposes a single, minimal change to a mutable file
3. Agent commits the change to a git branch
4. Agent runs `make backtest` and extracts the composite metric
5. Agent runs `pytest -q` to confirm no tests broke
6. If metric improved AND tests pass AND regime check passes:
   → keep the commit, log as "keep" in results.tsv
7. If metric did not improve OR tests failed:
   → git reset, log as "discard" in results.tsv
8. Repeat from step 1
```

### Critical design constraints

These exist because financial data is fundamentally different from LLM training data:

1. **Held-out validation set is sacred.** Split the most recent N months of data as a final validation set. The experiment loop trains and evaluates on everything before that. The held-out set is only used once, at the end, to check if the best configuration generalizes. This prevents the loop from overfitting to the walk-forward evaluation windows.

2. **Regime diversity is mandatory.** A change that improves the aggregate metric but only works in one market regime (e.g., bull run) is not a real improvement. The regime check ensures improvements hold across at least 2 distinct walk-forward windows.

3. **Complexity has a cost.** Unlike LLM training where a 0.001 improvement is always real, a 0.005 improvement in crypto signal metrics could be noise. The minimum improvement threshold and complexity budget prevent the loop from accumulating fragile micro-optimizations.

4. **The evaluation harness is never modified.** The loop can change what goes into the model (features, parameters) but never how the model is judged (walk-forward splits, cost model, baselines, metrics). This is the key invariant that makes the results trustworthy.

### Scope boundaries

**The agent MAY modify:**
- `features/pipeline.py` — add, remove, or modify feature computation
- `features/price.py`, `features/technical.py`, `features/volatility.py` — individual feature modules
- `features/onchain.py`, `features/crossasset.py`, `features/microstructure.py` — Phase 12 feature modules
- `config.py` — feature lists (`EXOG_COLUMNS`), cost thresholds, model hyperparameters, target timeframe
- `models/xgboost_model.py` — hyperparameters, model architecture
- `models/lightgbm_model.py`, `models/catboost_model.py`, `models/ensemble_model.py` — Phase 12 model families
- `models/` — new model files that implement `BaseModel`

**The agent must NEVER modify:**
- `evaluation/walk_forward.py` — the scoring harness
- `evaluation/baselines.py` — baseline implementations
- `evaluation/targets.py` — target construction
- `evaluation/reporting.py` — metric computation
- `data/loaders.py` — data loading
- `data/pipeline.py` — dataset assembly
- `data/validation.py` — data validation
- `tests/` — test suite
- `backtest.py` — evaluation entrypoint
- `signals/` — signal export contract

### `program.md` template

The `program.md` file instructs the AI agent running the experiment loop. It should contain:

```markdown
# Bitcoin Signal Research Program

## Your role
You are an autonomous research agent optimizing Bitcoin price signal models.
You propose changes, run experiments, and keep or discard based on results.

## Metric
Your optimization target is: [composite metric, e.g., cost_adj_precision * cost_adj_recall].
Extract it from the walk-forward summary JSON after each run.

## Budget
- Each experiment must complete in under 15 minutes (backtest + tests).
- Stop after 100 experiments or 12 hours, whichever comes first.

## Scope
You may modify: features/*, config.py (EXOG_COLUMNS, thresholds, hyperparams), models/*.
You must NOT modify: evaluation/*, data/*, tests/*, backtest.py, signals/*.

## Rules
1. One change per experiment. Keep changes minimal and targeted.
2. Always run pytest after backtest. If tests fail, revert immediately.
3. A change must improve the metric by at least 0.005 to be kept.
4. If a change adds more than 20 lines of code, the improvement must be proportionally larger.
5. Log every experiment to results.tsv with: commit hash, metric value, status, description.
6. Never touch the held-out validation set. It is only used at the end.
7. Prefer removing weak features over adding new ones when both options score similarly.
8. Read results.tsv before proposing a new experiment — do not repeat failed approaches.
```

### Suggested file targets

- `program.md` (new — research directives for the AI agent)
- `scripts/experiment_loop.py` (new — autonomous runner)
- `results.tsv` (new — experiment log, git-tracked)
- `Makefile` (add `experiment` target)
- `config.py` (add held-out validation split config)

### Exit criteria

- The experiment loop has run at least 50 experiments with proper logging.
- The best surviving configuration has been evaluated on the held-out validation set.
- If the held-out metrics clear the integration bar: declare the repo ready for Gate 5 re-evaluation.
- If the held-out metrics do not clear the bar after sustained experimentation: document the conclusion honestly and consider whether the signal hypothesis should be abandoned.

---

## Prioritized Task List

If a fresh agent needs a concrete execution sequence, follow this order:

### Already complete (Phases 0–10)

1. ~~Repo framing, cleanup, data pipeline, baselines, walk-forward validation, targets, model comparison, signal contract, artifact hardening, documentation, and testing.~~

### Current priority sequence

2. **Refresh the dataset** (Phase 11). Download fresh price and sentiment data through the current date. Re-run all evaluations. Record new baseline metrics. Do not change any model parameters.
3. **Expand the data universe** (Phase 12). Integrate on-chain data first, then cross-asset, then microstructure. Add each family independently, validate, measure impact. Add alternative model families (LightGBM, CatBoost). Explore multi-timeframe targets.
4. **Build the experiment loop infrastructure** (Phase 13). Create `program.md`, `scripts/experiment_loop.py`, held-out validation split, and `results.tsv` format. Add safety rails (minimum improvement threshold, regime diversity check, complexity budget, test-failure rollback).
5. **Run the experiment loop** (Phase 13). Let the autonomous agent explore the expanded feature/model/target space. Monitor `results.tsv` for progress.
6. **Evaluate on held-out data** (Phase 13). After the loop completes, evaluate the best surviving configuration on the held-out validation set exactly once.
7. **Final gate decision.** If held-out metrics clear the integration bar, declare the repo ready for downstream integration review. If not, document the conclusion honestly.

---

## Decision Gates

These gates should be checked explicitly.

### Gate 0: After cleanup

Question: Does the repository still contain files that do not support the signal-generation mission?

If yes:

- do not move on and leave them behind as "legacy",
- remove them or justify them explicitly as core.

### Gate 1: After Phase 1

Question: Is the data pipeline reproducible and consistent?

If no:

- do not continue model optimization,
- fix data quality and source consistency first.

### Gate 2: After Phase 3

Question: Do real model predictions beat baselines out of sample?

If no:

- do not proceed to downstream integration,
- either improve targets/features/models or conclude the repo is not useful enough.

### Gate 3: After Phase 7

Question: Can another repo consume the outputs safely and simply?

If no:

- do not integrate yet,
- finish the artifact and signal contract work first.

### Gate 4: After documentation hardening

Question: Can a fresh agent understand and operate the repo correctly from the documentation alone?

If no:

- do not call the repo complete,
- improve the docs until they accurately reflect the code and workflow.

### Gate 5: Final go/no-go

Question: Does the best signal survive realistic costs with stable out-of-sample performance?

If yes:

- integrate with the separate trading strategy repo.

If no:

- keep this repo as research-only, or
- stop using it as a dependency.

### Gate 6: After data universe expansion

Question: Did expanding the data universe produce at least one new feature family that measurably improves out-of-sample metrics over the Phase 11 baseline?

If yes:

- proceed to Phase 13 (experiment loop) to systematically optimize the expanded space.

If no:

- re-evaluate whether the data sources chosen are the right ones,
- consider alternative sources before running the experiment loop on a still-narrow feature set.

### Gate 7: After experiment loop

Question: Did the best surviving experiment configuration clear the integration bar on the held-out validation set?

If yes:

- proceed to Gate 5 re-evaluation with confidence.

If no after sustained experimentation (50+ experiments):

- document the conclusion honestly,
- consider whether the signal hypothesis should be abandoned entirely or whether the problem requires a fundamentally different approach (different asset, different market, different prediction task).

### Current gate results

Last updated: March 14, 2026.

- Gate 0: pass
- Gate 1: pass
- Gate 2: fail on signal quality (precision 0.169, recall 0.261, ROC-AUC 0.641 on March 14, 2026 fresh data — recall now passes but precision needs >= 0.55)
- Gate 3: pass for interface design, fail for integration readiness because Gate 2 failed
- Gate 4: pass
- Gate 5: fail
- Gate 6: not yet attempted — blocked on Phase 12 (data universe expansion)
- Gate 7: not yet attempted — blocked on Phase 13 (experiment loop)

---

## Notes for a Fresh Agent

When picking this roadmap up with zero context:

- Assume the repo is **not** production-ready.
- The foundation infrastructure (Phases 0–10) is solid and complete. Do not redo this work.
- The dataset is stale (ends July 2025). **Phase 11 (data refresh) is the first thing to do.**
- The current model only sees price-derived technicals and one sentiment index. That is not enough data to reach the integration bar. **Phase 12 (expand data universe) is the critical phase** — it determines whether this repo can succeed at all.
- Do not run the experiment loop (Phase 13) before Phase 12 is complete. Searching a narrow feature space is a waste of compute.
- Do not manually tune model parameters outside the experiment loop. That defeats the purpose of systematic exploration.
- If the experiment loop fails to find a configuration that clears the bar after 50+ experiments on an expanded feature set, that is a valid and important result — it means the signal hypothesis should be reconsidered.
- The evaluation harness, data pipeline, and test suite are immutable during experimentation. If you think they need changes, propose them as a separate phase.
- Add new data families one at a time. Validate each independently before combining. Do not integrate everything at once.

Your first job is to refresh the data (Phase 11).
Your second job is to expand what the model can see (Phase 12).
Your third job is to systematically search the expanded space (Phase 13).
Do not skip phases.
