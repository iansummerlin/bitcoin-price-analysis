# Bitcoin Price Analysis Roadmap

## Document Status

- Owner intent: turn this repository into a reliable **Bitcoin signal-generation and model-research repo** for a separate trading strategy/execution repository.
- Audience: a fresh engineering agent with zero prior context.
- Primary constraint: this repo should prove whether it can produce a usable predictive signal. It should not become an execution bot.
- Current date context: this roadmap was written on March 13, 2026. Any model artifacts or results from 2025 are stale until revalidated.

## Progress Tracker

Use these checkboxes to track progress directly in this file.

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
- [ ] Pass all decision gates and declare repo ready for downstream integration review.

### Current Status After Execution

- The repo was refactored into a research/evaluation/export shape with canonical `data/`, `evaluation/`, `features/`, `models/`, `signals/`, and `scripts/` modules.
- `gen.py`, `backtest.py`, and `collect.py` are now thin entrypoints on top of shared pipelines.
- The repository now builds a deterministic dataset, evaluates models with strict walk-forward splits, exports self-describing signal artifacts, and ships an A-tier test suite.
- The current evidence still does **not** justify downstream trading-repo integration.
- Latest verified walk-forward result on March 13, 2026 used local data through **July 17, 2025** and found `xgboost_direction` directional accuracy `0.829`, precision `0.215`, recall `0.030`, ROC-AUC `0.634` on the cost-adjusted target over the default recent evaluation slice. That is not strong enough to call the signal genuinely useful yet.
- Latest routine model comparison artifact on March 13, 2026 found `xgboost_direction` remained the best model family, but only at precision `0.097`, recall `0.051`, ROC-AUC `0.596` on the comparison slice. That fails the repo’s provisional integration bar.

---

## Current Known Facts

These facts are important for a fresh agent and should be assumed true unless the code has changed materially:

- This repo is intended to be a **signal-generation/model-research repo**, not a trading bot.
- Direct execution code and obvious standalone-trading surface have already been removed.
- `backtest.py` is now the canonical signal-evaluation entrypoint and uses naive predictions only as explicit baseline logic.
- `collect.py` is being retained only because live signal analytics may be useful.
- If `collect.py` remains, it should operate as a **backfill-first live analytics pipeline**, not a "wait for lag windows to fill naturally" process.
- Training and live/inference data alignment now goes through shared schema and feature pipeline code, though live analytics should still be treated as monitoring until fresh data is revalidated.
- Historical artifacts from 2025 are stale until revalidated.
- The roadmap is the canonical planning and progress-tracking document for this repository.

---

## Start Here

If you are a fresh agent with no other context, start in this order:

1. Read this roadmap fully once.
2. Read `README.md` and confirm it matches the roadmap.
3. Complete any remaining unchecked items in Phase 0.
4. Start Phase 1 by deciding and documenting the retained role of `collect.py`.
5. If `collect.py` stays, redesign it around historical backfill plus live continuation so the system can produce immediate live analytics without waiting for lag windows to fill.
6. Do not move to model optimization until the Phase 1 data pipeline and schema alignment work is complete.

If unsure what the next practical task is, the default next task is:

- finish Phase 0 cleanup,
- then refactor the data pipeline and `collect.py` under Phase 1.

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

### Useful pieces already present

- Historical data download and loading.
- Modular feature engineering.
- ARIMA training path.
- Alternative model scaffolding for boosted-tree regression/classification.
- Portfolio/backtest framework.
- A meaningful test suite around features, portfolio logic, strategies, and metrics.

### Critical shortcomings

- The current backtest does **not** test the trained model; it uses naive momentum predictions.
- Training and inference data sources are inconsistent.
- The repo still contains "trading bot" framing and execution placeholders that do not match the intended role.
- Current reported model metrics are not sufficient to justify downstream usage.
- Some code and docs still imply production readiness that the repo does not have.

### Current bottom line

This repo is a **promising research base** but **not yet a trustworthy model provider**.

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

1. Repo framing and workflow cleanup.
2. Deterministic data pipeline.
3. Baselines and honest evaluation harness.
4. Real walk-forward model-backed backtesting/evaluation.
5. Better targets and features.
6. Model comparison.
7. Downstream signal contract.
8. Reliability, testing, and artifact hardening.

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

## Prioritized Task List

If a fresh agent needs a concrete execution sequence, follow this order:

1. Update `README.md` so the repo is described as a signal/model repo, not a trading bot.
2. Remove any remaining files, docs, generated artifacts, logs, or UI remnants that do not support the signal/model mission.
3. Refactor data loading so `gen.py` and evaluation use one canonical data pipeline with no manual cleanup.
4. Unify training and inference schema assumptions in shared config/utilities.
5. Add honest baselines and make them first-class citizens in evaluation.
6. Refactor `backtest.py` so it evaluates real model predictions, not only naive momentum.
7. Add walk-forward evaluation that trains on past windows and predicts unseen future windows.
8. Move modeling focus toward returns/direction instead of raw next close.
9. Add cost-aware signal metrics and define an actionable signal contract.
10. Harden artifact metadata, validation, and staleness checks.
11. Bring documentation to A-tier quality and keep it aligned with the real codebase.
12. Expand tests to cover end-to-end evaluation and downstream compatibility.
13. Only after all of the above, assess whether this repo should be integrated into the trading strategy repo.

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

Current gate result on March 13, 2026:

- Gate 0: pass
- Gate 1: pass
- Gate 2: fail on signal quality
- Gate 3: pass for interface design, fail for integration readiness because Gate 2 failed
- Gate 4: pass
- Gate 5: fail

---

## Notes for a Fresh Agent

When picking this roadmap up with zero context:

- Assume the repo is **not** production-ready.
- Assume all historical performance claims must be revalidated.
- Assume 2025 artifacts are stale.
- Treat naive backtesting results as scaffolding, not proof.
- Prefer deleting confusing legacy behavior over preserving it.
- Optimize for clarity, reproducibility, and honest evaluation.

Your first job is not to improve model sophistication.
Your first job is to make the repo trustworthy enough that future model results mean something.
