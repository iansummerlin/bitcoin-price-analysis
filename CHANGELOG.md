# Changelog

This file tracks high-signal repository changes that affect model behavior,
evaluation pipeline, feature engineering, data sources, or research workflow.

It is intentionally lightweight.

Add entries for:
- model or feature changes that affect prediction quality
- evaluation pipeline or gating changes
- new data sources or feature sets
- config changes (thresholds, windows, cost assumptions)
- signal export contract changes
- backtest infrastructure changes

Do not use this as a full engineering diary. Prefer short entries linked to the
relevant docs and commits.

## 2026-03-16

### Added
- Per-run autoresearch archiving (`artifacts/autoresearch_runs/`) and append-only history index

### Changed
- `generate_autoresearch_md()` returns string instead of writing file directly

## 2026-03-15 (Phase 13 experiment loop refinements)

### Changed
- Experiment loop: renamed baseline score params to `reviewed_baseline_scores`/`candidate_baseline_scores` for clarity
- Added `baseline_description` param to autoresearch report generation
- Model thresholds configurable in `config.py`

### Fixed
- Test suite: aligned `test_experiment_loop.py` with updated `generate_autoresearch_md()` signature

## 2026-03-15 (Phase 13 infrastructure)

### Added
- Multi-horizon targets (1h/4h/24h) with horizon-scaled cost thresholds (Phase 12A)
- Shared TTL-based file cache (`data/cache.py`) for all external data sources
- Cross-asset data loader and 10 features: DXY, S&P 500, VIX, Gold, ETH (Phase 12B)
- On-chain data loader and 6 features: hash rate, difficulty, tx count/volume (Phase 12C)
- LightGBM direction classifier implementing BaseModel interface (Phase 12D)
- Microstructure data loader and 3 funding rate features (Phase 12E)
- Evaluation scripts for each sub-phase with ablation comparisons

### Changed
- `EXOG_COLUMNS` expanded from 23 to 42 features
- `build_dataset()` accepts `include_crossasset`, `include_onchain`, `include_microstructure` flags
- Model comparison now includes LightGBM alongside XGBoost and ARIMA
- Best known config updated: LightGBM, 4h horizon, ROC-AUC=0.603, recall=0.183, precision=0.428

### Added (Phase 13 infrastructure)
- Experiment loop (`scripts/experiment_loop.py`): predefined search over hyperparameters and feature subsets
- `program.md`: experiment loop documentation, metric, budget, held-out protocol
- `AUTORESEARCH.md` report generation after each run
- Held-out validation split config: last 3 months reserved, trained on all pre-held-out data
- Checkpointing and stop-flag support
- `make experiment` target
- 34 tests for experiment loop (composite metric, regime diversity, TSV round-trip, AUTORESEARCH.md, held-out isolation, end-to-end mini loop)

### Fixed
- Feature-subset baseline bug: kept experiments with custom feature columns now correctly refresh per-window scores for future regime-diversity comparisons
