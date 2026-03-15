# Bitcoin Price Analysis Roadmap

## Document Status

- Owner intent: turn this repository into a reliable **Bitcoin signal-generation and model-research repo** for a separate trading strategy/execution repository.
- Audience: a fresh engineering agent with zero prior context.
- Primary constraint: this repo should prove whether it can produce a usable predictive signal. It should not become an execution bot.
- Current date context: this roadmap was last updated on March 15, 2026. Data was refreshed through March 13, 2026.

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
- [x] Finish Phase 12 rebuild information set and prediction framing.
- [ ] Finish Phase 13 autonomous experiment loop.
- [ ] Pass all decision gates and declare repo ready for downstream integration review.

### Current Status After Execution

- The repo was refactored into a research/evaluation/export shape with canonical `data/`, `evaluation/`, `features/`, `models/`, `signals/`, and `scripts/` modules.
- `gen.py`, `backtest.py`, and `collect.py` are now thin entrypoints on top of shared pipelines.
- The repository now builds a deterministic dataset, evaluates models with strict walk-forward splits, exports self-describing signal artifacts, and ships an A-tier test suite.
- The current evidence still does **not** justify downstream trading-repo integration.
- Data was refreshed on March 14, 2026. The dataset now covers **2015-10-13 to 2026-03-13** (91,302 rows after feature pipeline).
- Phase 12 completed on March 15, 2026. Three new data families (cross-asset, on-chain, microstructure) and one alternative model (LightGBM) were integrated and tested. None of the new data families showed clear measurable improvement individually — they were integrated into the pipeline but not validated as signal sources.
- **12A horizon analysis:** 4h horizon is the most promising prediction target (precision=0.487, recall=0.218, ROC-AUC=0.563). 1h has best ROC-AUC but very low precision. 24h is near-random.
- **12B cross-asset:** Tradeoff result — improves recall (+0.138) and ROC-AUC (+0.019) but degrades precision (-0.117). Not a clear improvement by the measurable-improvement rules.
- **12C on-chain:** Negative to neutral. Precision degrades in isolation, marginal ROC-AUC improvement when combined with cross-asset.
- **12D LightGBM:** Different precision-recall tradeoff from XGBoost. LightGBM+expanded features achieves ROC-AUC=0.603 and recall=0.183. Precision=0.428 remains the blocker. Data appears to be the main bottleneck for precision, but learner choice still affects the tradeoff.
- **12E microstructure:** Neutral — funding rate features have negligible impact on either model.
- **12F dynamic features:** Skipped — no data family showed clear measurable improvement, so full contract redesign was not needed.
- **Best known configuration:** LightGBM with full expanded features on 4h horizon — ROC-AUC=0.603, recall=0.183, precision=0.428. Clears 2 of 3 integration thresholds. Precision gap (0.428 vs 0.55 needed) is the remaining bottleneck.
- **Note for Phase 13:** some sub-phase comparisons had minor baseline drift due to evolving evaluation setups (e.g. cross-asset data affected row counts slightly across runs). Phase 13 should keep comparisons tightly standardized — identical dataset, identical splits, identical evaluation slice for every experiment pair.
- **Phase 13 infrastructure** completed on March 15, 2026. Experiment loop (`scripts/experiment_loop.py`), research directives (`program.md`), held-out validation split (last 3 months, 2160 hours), composite metric (precision × recall), safety rails, and `make experiment` target are all in place.
- **Phase 13 initial full run** completed on March 15, 2026. 100 experiments were evaluated in 284.3 minutes; 3 were kept and 97 discarded. The best surviving walk-forward configuration was `xgb: n_est=300, max_d=5, lr=0.1` with composite=0.099656, precision=0.2248, recall=0.4433, ROC-AUC=0.5521.
- **Phase 13 held-out result:** FAIL. On the held-out set, the best surviving configuration achieved precision=0.4074, recall=0.0595, ROC-AUC=0.7033. This clears ROC-AUC but fails precision and recall thresholds, so the repo remains research-only.
- **Interpretation:** the loop improved the search metric but did not produce a deployable operating point. The next Phase 13 work should focus on precision-oriented threshold tuning and probability-boundary calibration, not on promoting the current best config into repo defaults.

### What blocks progress now

The foundation infrastructure (Phases 0–10) is solid. The three blockers are:

1. ~~**Stale data**~~ — resolved March 14, 2026. Dataset now extends through March 13, 2026.
2. **Narrow data universe** — the model only sees price-derived technical indicators and a single sentiment index. These are lagging, widely available, and largely priced in. The market has this information already. Fresh data confirmed this: recall improved to 0.261 (above threshold) but precision dropped to 0.169 (far below 0.55). The model detects moves but can't filter noise from signal with current features. Reaching the integration bar almost certainly requires data sources the market hasn't fully absorbed: on-chain metrics, cross-asset signals, order flow / microstructure data.
3. **Unexplored search space** — even with better data, the combination of features, hyperparameters, target thresholds, and model architectures is too large for manual exploration. The broad search has now been exercised once; the remaining high-value gap is operating-point search with an explicit precision focus.

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
5. Treat **Phase 12** as complete. The repo now has a richer information set, a 4h target decision, and a larger feature/model search space.
6. Continue with **Phase 13: Autonomous Experiment Loop**, but do so narrowly: prioritize target-threshold tuning, probability-threshold tuning, and precision-oriented operating-point selection before more generic hyperparameter search.
7. The remaining Phase 5 item (remove weak features) can still be informed by Phase 13 evidence, but Phase 13 should not auto-promote winning configs into repo defaults.

If unsure what the next practical task is, the default next task is:

- **Phase 13 threshold search** — tune the actionable threshold / cost buffer and classification decision boundary with an explicit goal of improving held-out precision without collapsing recall below the acceptance bar.

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

- ~~**Data is stale**~~ — resolved March 14, 2026. Dataset now extends through March 13, 2026.
- ~~**Feature universe is too narrow**~~ — addressed in Phase 12. Now includes cross-asset (10 features), on-chain (6 features), and microstructure (3 features) alongside price/sentiment. None individually showed clear measurable improvement, but the expanded space is available for Phase 13 to search in combination.
- ~~**Model roster is too narrow**~~ — addressed in Phase 12D. LightGBM added alongside XGBoost and ARIMA. LightGBM shows a different precision-recall tradeoff on some configurations.
- **Search space is unexplored** — no systematic exploration of the feature/model/hyperparameter space (Phase 13 fixes this).
- **Model metrics fail the integration bar** — best known: LightGBM 4h precision 0.428, recall 0.183, ROC-AUC 0.603. Clears recall and ROC-AUC thresholds but precision remains below 0.55.

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

### Files moved into evaluation/

- `evaluation/cost_model.py` (was `portfolio.py`)
  - Role: cost simulation for signal-usefulness evaluation. Tracks hypothetical cash, holdings, and equity to compute cost-adjusted metrics.

- `evaluation/signal_rules.py` (was `strategies.py`)
  - Role: signal-consumption rules that translate predictions into discrete ENTRY/EXIT/HOLD decisions for the evaluation harness.

- `portfolio.py` / `strategies.py` — thin backward-compatibility shims that re-export from the evaluation modules. Will be removed once no external callers remain.

### Files likely to keep as secondary support

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

## Phase 12: Rebuild the Information Set and Prediction Framing

### Objective

Rebuild the repo's information set and prediction framing so that Phase 13 can search a materially better problem. This phase is not only about richer features — it is also about finding the prediction horizon where those features can plausibly matter.

### Why this phase exists

The current model uses 23 features derived almost entirely from price and volume data (lagged closes, returns, RSI, ATR, moving averages) plus a single sentiment index. These are lagging indicators that every market participant has access to. The integration bar requires precision >= 0.55 and recall >= 0.15. The evidence strongly suggests that better data and/or better problem framing are more important than incremental tuning of the current feature set — though some headroom may exist in probability threshold calibration, it is unlikely to close a gap of this magnitude on its own.

The candidate data families below have plausible mechanisms for carrying predictive signal, but they are hypotheses to test, not known sources of edge. Each will be integrated, evaluated, and either kept or documented as non-contributing.

- **Cross-asset data** — macro regime context (DXY strength, risk-on/risk-off transitions) with plausible connections to BTC regime shifts.
- **On-chain data** — network activity and miner behaviour as candidate supply/demand proxies not directly visible in price.
- **Microstructure data** — derivatives positioning (funding rates) as a candidate short-term sentiment signal.
- **Multi-timeframe targets** — the current 1h target may be structurally mismatched to the cadence of the most informative new data. Finding the right horizon is central to this phase, not auxiliary.

Without this phase, the experiment loop (Phase 13) would be searching a narrow, low-signal space.

### Implementation constraints

1. **Free sources only.** Phase 12 implementations must use free data sources only. Paid providers are documented as reference options for future phases and are out of scope for implementation here.

2. **All external data goes through the shared cache.** No external loader — whether free or paid in later phases — may bypass the shared cache layer (`data/cache.py`). Every remote API call must check the cache first and write results back to it. Repeated dataset builds within TTL must not hit the remote API again.

3. **Stale cache over retries.** On API failure or rate limiting, loaders must fall back to stale cache (expired but present) rather than repeatedly retrying. A single retry with backoff is acceptable; a retry loop is not. If no cache exists at all, return an empty DataFrame with the correct column schema so the pipeline degrades gracefully.

4. **Hypotheses, not assumptions.** The candidate data families have plausible mechanisms but are not proven signal sources for this repo. Treat every integration as a hypothesis to test. If ablation shows no improvement, document the null result and move on.

---

### Phase 12A: Problem framing check

**What it proves:** Whether the current prediction task is the right task before investing in new data.

**Work:**
- [x] Add 4h and 24h forward targets with horizon-scaled cost thresholds to `evaluation/targets.py`.
- [x] Add horizon cost threshold config to `config.py`.
- [x] Run walk-forward evaluation on all three horizons (1h, 4h, 24h) with the existing feature set.
- [x] Check the precision-recall tradeoff from the current model's predicted probabilities — is there meaningful headroom by adjusting the classification threshold?
- [x] Record which horizon produces the best cost-adjusted metrics.
- [x] Add tests for multi-timeframe targets.

**Exit criterion:** A documented answer to "which horizon should we optimise for?" backed by walk-forward numbers across all three. The rest of Phase 12 uses that horizon as the default primary evaluation target — but later sub-phases may also evaluate on a secondary horizon when the data cadence justifies it (e.g., microstructure features tested on a shorter horizon than cross-asset features). Results recorded in `BACKTEST.md`.

**Why first:** If the best candidate features are daily-cadence (on-chain, cross-asset) and the target remains 1h cost-adjusted direction, there is a structural cadence mismatch. Answering the horizon question first prevents wasted effort evaluating slow-moving features against a fast target where they cannot plausibly help.

---

### Phase 12B: Cross-asset data integration

**What it proves:** Whether macro/regime context improves the model on the chosen horizon.

**Work:**
- [x] Build `data/cache.py` — generic TTL-based file cache with atomic writes.
- [x] Add cache config to `config.py` (TTLs, cache directory path).
- [x] Add `data/cache/` to `.gitignore`.
- [x] Add tests for cache (write/read/TTL expiry/invalidation).
- [x] Build `data/crossasset.py` — loader for yfinance (DXY, S&P 500, VIX, Gold, ETH).
- [x] Build `features/crossasset.py` — cross-asset feature computation (10 features).
- [x] Add cross-asset feature columns to `config.EXOG_COLUMNS`.
- [x] Handle weekend/holiday gaps with explicit forward-fill.
- [x] Wire into `data/pipeline.py` and `features/pipeline.py`.
- [x] Add tests for cross-asset loader and feature computation.
- [x] Run walk-forward on chosen horizon, compare against 12A baseline.
- [x] Run ablation to isolate cross-asset contribution.

**Why first:** Cleanest data source, most reliable API (yfinance, no key required, lighter rate limits than the on-chain APIs), best historical depth (30+ years daily). Validates the caching and pipeline integration pattern before hitting more constrained APIs.

**Exit criterion:** Ablation result showing cross-asset contribution (positive, neutral, or negative) on the chosen horizon. Documented in `BACKTEST.md` history with comparison against 12A baseline.

#### Cross-asset data — free sources

| Source | API | Rate limit | Historical depth | Metrics |
|---|---|---|---|---|
| yfinance | Python `yfinance` package | ~360 req/hr, no key | 30+ years daily | DXY, S&P 500, VIX, Gold, ETH-USD |

**Tickers:**
- `DX-Y.NYB` — US Dollar Index (USD strength)
- `^GSPC` — S&P 500 (equity risk appetite)
- `^VIX` — CBOE Volatility Index (fear gauge)
- `GC=F` — Gold futures (safe haven proxy)
- `ETH-USD` — Ether (crypto relative strength)

**Proposed features** (`features/crossasset.py`):
- `dxy_return_1d`, `dxy_return_5d` — USD strength momentum
- `sp500_return_1d` — equity market daily return
- `btc_sp500_corr_30d` — rolling 30-day BTC/S&P correlation
- `vix_level` — raw VIX (regime indicator)
- `vix_change_1d` — VIX daily change (risk sentiment shift)
- `gold_btc_ratio` — Gold/BTC price ratio
- `gold_btc_ratio_zscore_30d` — z-score of ratio over 30 days
- `eth_btc_ratio` — ETH/BTC ratio (crypto relative strength)
- `eth_btc_ratio_change_7d` — 7-day change in ETH/BTC ratio

**Fill policy:** Traditional markets are closed on weekends. Friday close forward-filled through Saturday/Sunday. All data aligned to UTC daily, then forward-filled to hourly.

#### Cross-asset data — paid alternatives (not implemented)

| Provider | Cost | What it adds | When to upgrade |
|---|---|---|---|
| **FRED API** | Free (key required) | CPI, Fed funds rate, yield curve, M2 money supply. Apply at https://fred.stlouisfed.org/docs/api/api_key.html | If macro features show signal in ablation. Not paid, but requires registration. |
| **Quandl / Nasdaq Data Link** | $50/mo+ | Higher-quality cross-asset data, commodity prices, bond yields | Only justified for production-grade reliability. |

---

### Phase 12C: On-chain data integration

**What it proves:** Whether on-chain network metrics add signal beyond cross-asset context.

**Work:**
- [x] Build `data/onchain.py` — loader for blockchain.com and mempool.space APIs with caching and rate limiting.
- [x] Build `features/onchain.py` — on-chain feature computation (6 features).
- [x] Add on-chain feature columns to `config.EXOG_COLUMNS`.
- [x] Wire into `data/pipeline.py` and `features/pipeline.py`.
- [x] Add tests for on-chain loader and feature computation.
- [x] Run walk-forward on chosen horizon, compare against 12B baseline.
- [x] Run ablation to isolate on-chain contribution independently.

**Exit criterion:** Ablation result showing on-chain contribution. Documented comparison against both 12A (no new data) and 12B (cross-asset only) in `BACKTEST.md`.

**Note on expectations:** The free on-chain metrics (hash rate, difficulty, tx count, tx volume) are coarse network-health proxies. They are not the strongest on-chain features discussed in academic literature (exchange flows, MVRV, SOPR) — those require paid providers. Free on-chain is worth integrating as hypothesis testing, but it is more likely to help as regime context than as sharp directional signal. If it shows no value, that is a useful result.

#### On-chain data — free sources

| Source | API | Rate limit | Historical depth | Metrics |
|---|---|---|---|---|
| blockchain.com | `api.blockchain.info/charts/{metric}` | 1 req / 10s, no key | Full history (2009+) | Hash rate, difficulty, transaction count, estimated tx volume USD |
| mempool.space | `mempool.space/api/v1/mining/*` | Generous, no key | ~3 years | Mining pool distribution, difficulty adjustments, fee estimates |

**Proposed features** (`features/onchain.py`):
- `hashrate_change_24h` — 24-hour pct change of hash rate
- `hashrate_change_7d` — 7-day pct change of hash rate
- `difficulty_change` — pct change since last difficulty adjustment
- `tx_count_zscore_7d` — z-score of transaction count over 7-day window
- `tx_volume_zscore_7d` — z-score of transaction volume over 7-day window
- `hashrate_price_divergence` — z-score of hashrate/price ratio (candidate miner conviction signal)

**Fill policy:** Daily data forward-filled to hourly in `build_dataset()`.

#### On-chain data — paid alternatives (not implemented)

| Provider | Cost | What it adds | When to upgrade |
|---|---|---|---|
| **Glassnode** | $29/mo (Student) | MVRV, NVT, exchange net flows, whale metrics, realized price, SOPR. Most comprehensive on-chain analytics. Free tier exists but limited to 24h-delayed daily data with restricted endpoints. | If 12C ablation shows on-chain features contribute measurable signal — the free metrics are proxies for the stronger paid features. |
| **CryptoQuant** | $29/mo (Starter) | Exchange reserves, miner flows, whale alerts, fund flow ratio. Stronger on exchange-specific flow analysis. | If exchange flow granularity is needed beyond what Glassnode provides. |
| **Coin Metrics** | $100/mo+ (Pro) | Institutional-grade network data feeds, CMBI indexes, reference rates. | Only for production systems requiring SLA guarantees. Overkill for research. |

---

### Phase 12D: One alternative model family

**What it proves:** Whether any signal found in 12B/12C survives a different inductive bias.

**Work:**
- [x] Add LightGBM classifier implementing `BaseModel` interface (`models/lightgbm_model.py`).
- [x] Register in `evaluation/walk_forward.py` MODEL_REGISTRY and `evaluation/model_comparison.py` MODEL_SPECS.
- [x] Update `requirements.txt` with `lightgbm` and `yfinance`.
- [x] Add tests for the new model.
- [x] Run model comparison with expanded feature set on chosen horizon.

**Why one, not two:** XGBoost, LightGBM, and CatBoost are all tree-based gradient boosting families. They are adjacent, not fundamentally different. If the same feature improvement helps both XGBoost and LightGBM, the result is more credible — that is the value. Adding a second alternative (CatBoost) in the same phase is model shopping with lower expected return than the data work. CatBoost can be added in Phase 13 if the experiment loop benefits from more learner diversity.

**Exit criterion:** Model comparison results with the expanded feature set. If LightGBM outperforms XGBoost, note it. If roughly equal, that strengthens confidence in the features rather than the learner. Results in `BACKTEST.md`.

---

### Phase 12E: Microstructure (conditional)

**What it proves:** Whether derivatives positioning data adds signal, particularly on shorter horizons.

**Condition:** Proceed if either (a) 12B or 12C showed measurable improvement, confirming that new data families can contribute, or (b) the 12A horizon analysis showed that short-horizon prediction (1h or 4h) remains worth pursuing — since funding rate is a faster-moving signal than cross-asset or on-chain, it may help on shorter horizons even if the slower-cadence families did not.

**Work:**
- [x] Build `data/microstructure.py` — loader for Binance funding rate history only.
- [x] Build `features/microstructure.py` — funding-rate feature computation (3 features).
- [x] Wire into pipeline.
- [x] Add tests.
- [x] Run walk-forward on chosen horizon, compare against 12C baseline.
- [x] Run ablation to isolate funding rate contribution.

**Scope limitation:** Funding rate history back to ~2019 is acceptable for walk-forward research. Open interest is explicitly excluded from this sub-phase — the free Binance endpoint only serves ~30 days of OI history, which creates non-comparable walk-forward windows (recent windows would have better OI coverage than older ones, degrading historical comparability and reproducibility). OI is deferred until either CoinGlass ($49/mo) provides full history or enough data has been accumulated through live collection to support meaningful evaluation.

**Proposed features** (funding rate only):
- `funding_rate_8h` — raw funding rate (forward-filled from 8h to hourly)
- `funding_rate_zscore_7d` — z-score of funding rate over 7-day window
- `funding_rate_cumulative_24h` — cumulative funding over 24h (annualized carry cost signal)

**Exit criterion:** Ablation result for funding rate features. If no signal, document and move on.

#### Microstructure data — paid alternatives (not implemented)

| Provider | Cost | What it adds | When to upgrade |
|---|---|---|---|
| **CoinGlass** | $49/mo | Historical OI back to 2019, liquidation data, long/short ratios, multi-exchange aggregation. Highest-value microstructure upgrade. | If funding rate features show signal in ablation — OI and liquidation data are likely where the real microstructure value is. |
| **CoinAPI** | $79/mo+ | Extended funding rate history across multiple exchanges, order book snapshots. | If multi-exchange funding divergence or order book features are needed. |
| **Tardis.dev** | $50/mo+ | Tick-level historical data, order book reconstructions. | Only if the model needs sub-hourly resolution, which is unlikely given current architecture. |

---

### Phase 12F: Contract redesign for dynamic feature availability

**What it proves:** That the pipeline can handle partial data families gracefully in both training and inference.

**Condition:** Only needed if multiple data families proved useful in 12B–12E and the repo needs to support running with subsets. If only one family matters, the simpler approach is to make it required.

**Why this is substantial:** The current codebase is not built for optional feature families. Feature columns are fixed in `config.EXOG_COLUMNS`, the pipeline drops rows on that full list, and models hard-fail on missing columns. Making features dynamic implies a contract redesign, not a simple fallback:

**Work:**
- [x] Feature-family registry in `config.py` (which families are available, which are enabled). — **Skipped:** no data family showed clear measurable improvement; the `build_dataset()` API already supports toggling each family via `include_crossasset`, `include_onchain`, `include_microstructure` parameters. Full contract redesign not needed.
- [x] Dynamic `EXOG_COLUMNS` based on available data at build time. — Skipped (see above).
- [x] Conditional dropna on present columns only in `features/pipeline.py`. — Skipped (see above).
- [x] Model feature-set metadata: save the feature list used at training time, validate at inference time. — Skipped (see above).
- [x] Tests covering representative family combinations: all enabled, each family individually disabled, and price-only (all disabled). Full combinatorial coverage is not required. — Skipped (see above).

**Exit criterion:** `make test` passes for the representative combinations above. Pipeline produces correct results whether run with full data or price-only fallback.

---

### Caching infrastructure

All new data sources are rate-limited free APIs. Every loader uses a shared caching layer (`data/cache.py`) to avoid hitting rate limits during development and repeated runs. Built in Phase 12B alongside the first data integration.

- Cache directory: `data/cache/` (gitignored)
- Each data source has its own namespace and TTL
- On cache miss or expiry: fetch from API, validate, write to cache
- On cache hit within TTL: return cached data, skip API call
- On API failure: fall back to stale cache if available, else return empty DataFrame
- Atomic writes (temp file + `os.replace`) to prevent corruption

| Source family | Cache TTL | Rationale |
|---|---|---|
| Cross-asset | 6 hours | Daily data but markets close at different times |
| On-chain | 24 hours | Daily resolution data, no benefit refreshing more often |
| Microstructure | 1 hour | Funding rates update every 8h |

**Verification check:** After implementing any data loader, confirm that calling `build_dataset()` twice within TTL produces identical results without any remote API calls on the second invocation. This should be covered by cache unit tests.

---

### Decision rules

#### Defining "measurable improvement"

A new data family counts as a measurable improvement if it meets **all** of the following on the chosen primary horizon:

1. At least one of the three gated metrics (precision, recall, ROC-AUC) improves by >= 0.01 absolute over the previous sub-phase baseline.
2. None of the three gated metrics degrades by more than 0.005 absolute.
3. The improvement holds across at least 2 of the walk-forward windows individually, not just the aggregate mean.

If a family improves one metric but degrades another beyond the tolerance, it is not a clear improvement — document the tradeoff and defer the decision to Phase 13's experiment loop, which can explore whether the family helps in combination with other changes.

If a family shows no improvement and no degradation, document it as neutral and move on. Neutral is a valid result.

#### Ablation methodology

When the roadmap says "run ablation to isolate contribution," this means:

1. Use the **same model family, same walk-forward windows, same target column, and same evaluation slice** as the comparison baseline.
2. Run two evaluations: (a) with the new feature family included, (b) with only the feature family removed (all other features from prior sub-phases remain).
3. The difference between (a) and (b) is the isolated contribution of that family.
4. Record both sets of metrics in `BACKTEST.md` so the comparison is visible in the history table.

This ensures results are comparable across sub-phases. Do not change the model, the windows, or the target between the two runs.

**Artifacts:** Each ablation comparison produces two entries in `BACKTEST.md` (via `make backtest` for the with-family run and the without-family run). The walk-forward summary JSONs in `artifacts/` (`*_predictions.summary.json`) provide the machine-readable comparison. Label backtest history entries with notes indicating the sub-phase and whether the family was included or excluded (e.g., `notes="12B: cross-asset included"` and `notes="12B: cross-asset excluded"`).

---

### Overall Phase 12 exit criteria

- The prediction horizon question has a data-backed answer (12A).
- At least one new data family shows measurable out-of-sample improvement over the Phase 11 baseline, or all candidates are documented as non-contributing.
- Each sub-phase's impact is individually documented in `BACKTEST.md` history, providing clean attribution.
- One alternative model family is implemented and compared.
- All existing tests still pass.
- The repo is ready for Phase 13 to search the expanded space — or, if no data family helped, for an honest reassessment of the signal hypothesis.

---

## Phase 13: Autonomous Experiment Loop

### Checklist

#### Infrastructure

- [x] Create `program.md` with research directives, scope boundaries, and safety rails.
- [x] Create `scripts/experiment_loop.py` — the autonomous runner script.
- [x] Define the single composite metric used to judge experiments (e.g., cost-adjusted F1, or precision × recall product). — **precision × recall** chosen.
- [x] Create `results.tsv` format for experiment logging (run_id, experiment_id, model, metrics, status, description).
- [x] Designate immutable files the agent must never modify (evaluation harness, data loaders, test suite). — enforced in `program.md`.
- [x] Designate mutable files the agent may modify (features, model hyperparameters, target thresholds, config). — enforced in `program.md`.
- [x] Reserve a held-out final validation set that the experiment loop never sees during optimization. — last 3 months (2160 hours), configured in `config.py`.
- [x] Add a `make experiment` target to the Makefile.

#### Safety rails

- [x] Implement minimum improvement threshold — discard changes below noise floor (e.g., < 0.005 metric improvement). — `EXPERIMENT_MIN_IMPROVEMENT = 0.005` in `config.py`.
- [x] Implement maximum complexity budget — reject changes that add more than N lines without proportional metric gain. — `EXPERIMENT_MAX_ADDED_LINES = 20` in `config.py`; experiment loop uses fixed search space so no code-gen needed.
- [x] Implement regime diversity check — require the improvement to hold across at least 2 distinct walk-forward windows, not just the aggregate. — `regime_diversity_check()` in experiment loop, 92 walk-forward windows.
- [x] Implement rollback on test failure — if `pytest` fails after a change, revert automatically. — experiment loop does not modify files (search-based, not code-gen), so rollback is implicit.
- [x] Implement experiment budget cap — stop after N experiments or M hours to prevent runaway compute. — `EXPERIMENT_BUDGET_MAX = 100`, `EXPERIMENT_BUDGET_HOURS = 12`.

#### Experiment scope

- [x] Feature engineering: add, remove, or modify features in `features/pipeline.py` and `config.py`. — individual feature removal experiments included.
- [x] Feature selection: enable/disable feature families from Phase 12 (on-chain, cross-asset, microstructure). — family ablation experiments included.
- [x] Hyperparameter tuning: modify model parameters across all model families (XGBoost, LightGBM, CatBoost, ensemble). — XGBoost and LightGBM hyperparameter grid included (CatBoost not added in Phase 12).
- [x] Target threshold tuning: adjust cost buffer, actionable move threshold, prediction timeframe in `config.py`. — actionable-threshold and cost-buffer search added to the experiment loop.
- [x] Probability threshold tuning: adjust classification decision boundary. — decision-threshold search added to the experiment loop.
- [x] Model selection: switch between or combine Phase 12 model families. — both XGBoost and LightGBM evaluated.
- [ ] New model architectures: add new model classes in `models/` that conform to the `BaseModel` interface. — not attempted; existing families sufficient for search.

#### Validation

- [x] Run the experiment loop end-to-end on at least one full cycle and confirm keep/discard logic works. — full run completed: 100 experiments, 3 kept, 97 discarded.
- [x] Confirm `results.tsv` accurately logs all experiments with correct metrics. — verified in the completed run.
- [x] Confirm the held-out validation set was never used during the experiment loop. — held-out split at row 70797, experiment loop only uses rows 0–70796.
- [x] Evaluate the best surviving configuration on the held-out set and record final metrics. — complete; held-out precision=0.4074, recall=0.0595, ROC-AUC=0.7033.
- [x] If final metrics clear the integration bar, update the progress tracker and gate results. — complete; Gate 7 failed, repo remains research-only.

### Objective

Systematically explore the feature/hyperparameter/target search space using an autonomous AI experiment loop, inspired by Karpathy's autoresearch pattern, layered on top of the existing walk-forward evaluation infrastructure.

### Why this phase exists

After Phase 12 expands the data universe and model roster, the search space is large: dozens of features across multiple data families × hyperparameter grids × target thresholds × multiple model architectures. That is too large for manual exploration. An autonomous loop can run ~12 experiments/hour, covering overnight what would take weeks of manual iteration.

After the first full run, the remaining value is no longer broad model shopping. The highest-value unresolved question is whether a better operating point can recover enough precision on held-out data without destroying recall.

### How it works

The experiment loop is a **predefined runtime search** over model hyperparameters and feature subsets. It does not modify source files, generate code, or use git commits/resets. All variation is expressed as runtime parameters passed to the existing walk-forward evaluation infrastructure.

```
1. Build dataset once, split off held-out (last 3 months)
2. Establish baselines (XGBoost + LightGBM, default configs)
3. For each experiment in the predefined search space:
   a. Build model with the experiment's hyperparameters and feature subset
   b. Run walk-forward evaluation (92 windows) on experiment dataset
   c. Compare composite metric (precision × recall) against current best
   d. Check regime diversity (improvement must hold on ≥ 2 windows)
   e. If improved by ≥ 0.005 AND diversity passes → keep as new best
   f. Log result to results.tsv
   g. Save checkpoint (for resume on interruption)
4. After all experiments: train best config on all pre-held-out data
5. Evaluate once on held-out set
6. Generate AUTORESEARCH.md report and Gate 7 verdict
```

The search space includes:
- XGBoost and LightGBM hyperparameter grids (~50 combinations)
- Feature family ablation (remove cross-asset, on-chain, microstructure, or all)
- Individual feature removal (drop each of 42 features one at a time)
- Total: ~103 experiments, ~5 minutes each, ~8-9 hours end to end

The next search revision should prioritize:
- probability-threshold tuning for classification outputs,
- actionable-threshold / cost-buffer tuning,
- precision-constrained selection rules (for example: maximize precision subject to recall >= 0.15),
- only then additional hyperparameter expansion.

### Critical design constraints

These exist because financial data is fundamentally different from LLM training data:

1. **Held-out validation set is sacred.** Split the most recent N months of data as a final validation set. The experiment loop trains and evaluates on everything before that. The held-out set is only used once, at the end, to check if the best configuration generalizes. This prevents the loop from overfitting to the walk-forward evaluation windows.

2. **Regime diversity is mandatory.** A change that improves the aggregate metric but only works in one market regime (e.g., bull run) is not a real improvement. The regime check ensures improvements hold across at least 2 distinct walk-forward windows.

3. **Complexity has a cost.** Unlike LLM training where a 0.001 improvement is always real, a 0.005 improvement in crypto signal metrics could be noise. The minimum improvement threshold and complexity budget prevent the loop from accumulating fragile micro-optimizations.

4. **The evaluation harness is never modified.** The loop can change what goes into the model (features, parameters) but never how the model is judged (walk-forward splits, cost model, baselines, metrics). This is the key invariant that makes the results trustworthy.

5. **Winning configs are not auto-promoted.** The loop produces evidence, not codebase mutations. Any promotion into `config.py`, training defaults, or export paths must be a manual review step performed after stronger-model or human validation.

### Scope boundaries

The experiment loop does not modify files. The search dimensions are runtime parameters:

**Mutable dimensions (passed at runtime):**
- Model choice: XGBoost or LightGBM direction classifier
- Hyperparameters: n_estimators, max_depth, learning_rate, num_leaves
- Feature subsets: any subset of the 42 EXOG_COLUMNS

**Immutable (never touched by the loop):**
- `evaluation/*` — scoring harness, baselines, targets, metrics, cost model
- `data/*` — loaders, pipeline, validation, cache
- `tests/*` — test suite
- `backtest.py` — evaluation entrypoint
- `signals/` — signal export contract
- `features/*` — feature computation (features are included/excluded by column selection, not by modifying the pipeline code)

### `program.md`

The `program.md` file documents the experiment loop's operating model, metric,
budget, scope, safety rails, results format, and held-out protocol. It is the
canonical reference for understanding what `make experiment` does. See the actual
file for the current contents.

### Suggested file targets

- `program.md` — research directives and loop documentation
- `scripts/experiment_loop.py` — autonomous runner
- `results.tsv` — experiment log (append-only across runs)
- `AUTORESEARCH.md` — auto-generated human-readable report (regenerated each run)
- `Makefile` — `experiment` target
- `config.py` — held-out validation split and budget config

### Exit criteria

- The experiment loop has run at least 50 experiments with proper logging.
- The best surviving configuration has been evaluated on the held-out validation set.
- If the held-out metrics clear the integration bar: declare the repo ready for Gate 5 re-evaluation.
- If the held-out metrics do not clear the bar after sustained experimentation: document the conclusion honestly and consider whether the signal hypothesis should be abandoned.

Current status: the first full run satisfied the first two criteria and failed the third. The next iteration should be treated as a focused precision-recovery attempt, not as evidence that broad search is still untouched.

---

## Prioritized Task List

If a fresh agent needs a concrete execution sequence, follow this order:

### Already complete (Phases 0–10)

1. ~~Repo framing, cleanup, data pipeline, baselines, walk-forward validation, targets, model comparison, signal contract, artifact hardening, documentation, and testing.~~

### Current priority sequence

2. ~~**Refresh the dataset** (Phase 11).~~ Complete. Dataset extends through March 13, 2026.
3. **Rebuild information set and prediction framing** (Phase 12). Sub-phases in order:
   - 12A: Problem framing check — add multi-timeframe targets, determine best horizon.
   - 12B: Cross-asset data — caching infra + yfinance integration, ablation.
   - 12C: On-chain data — blockchain.com/mempool.space integration, ablation.
   - 12D: One alternative model family (LightGBM) for robustness checking.
   - 12E: Microstructure (conditional) — funding rate only, if 12B/12C showed signal or 12A showed short-horizon prediction is worth pursuing.
   - 12F: Dynamic feature availability (conditional) — contract redesign if multiple families proved useful.
4. **Build the experiment loop infrastructure** (Phase 13). Create `program.md`, `scripts/experiment_loop.py`, held-out validation split, and `results.tsv` format. Add safety rails.
5. **Run the experiment loop** (Phase 13). Systematically search the expanded feature/model/target space.
6. **Evaluate on held-out data** (Phase 13). Evaluate the best surviving configuration on the held-out set exactly once.
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
- Gate 6: fail — no individual data family showed measurable improvement by the strict rules. Proceeding to Phase 13 is justified because Phase 12 created a richer search space (42 features, 2 learners, a better horizon) and identified the 4h target — not because any new data family was validated as a signal source on its own.
- Gate 7: not yet attempted — blocked on Phase 13 (experiment loop)

---

## Notes for a Fresh Agent

When picking this roadmap up with zero context:

- Assume the repo is **not** production-ready.
- The foundation infrastructure (Phases 0–10) is solid and complete. Do not redo this work.
- Phase 11 (data refresh) is complete. Dataset extends through March 13, 2026.
- The current model only sees price-derived technicals and one sentiment index. That is not enough data to reach the integration bar. **Phase 12 (rebuild information set and prediction framing) is the critical phase** — it determines whether this repo can succeed at all.
- Phase 12 is internally staged (12A–12F) with explicit exit criteria per sub-phase. Do not skip sub-phases or combine them. Each produces its own evidence in `BACKTEST.md`.
- **Start with 12A (problem framing).** The horizon question must be answered before adding new data families, because daily-cadence features evaluated against a 1h target may appear useless when they would help on a 4h or 24h target.
- Add new data families one at a time. Validate each independently before combining. Do not integrate everything at once.
- Do not run the experiment loop (Phase 13) before Phase 12 is complete. Searching a narrow feature space is a waste of compute.
- Do not manually tune model parameters outside the experiment loop. That defeats the purpose of systematic exploration.
- If the experiment loop fails to find a configuration that clears the bar after 50+ experiments on an expanded feature set, that is a valid and important result — it means the signal hypothesis should be reconsidered.
- The evaluation harness, data pipeline, and test suite are immutable during experimentation. If you think they need changes, propose them as a separate phase.

Phase 11 (data refresh) is complete.
Your first job is to expand what the model can see (Phase 12).
Your second job is to systematically search the expanded space (Phase 13).
Do not skip phases.
