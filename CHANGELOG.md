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

## 2026-03-15

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
