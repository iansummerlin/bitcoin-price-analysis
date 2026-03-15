# Backtest Results

*Auto-generated from `artifacts/backtest_history.json` — do not edit manually.*


## Latest Results

**Run:** 2026-03-15T13:38:39.514833+00:00  
**Model:** xgboost_direction  
**Target:** target_direction_cost_adj  
**Windows:** 6  

### Metrics

| Metric | Value |
|--------|-------|
| Precision | 0.2161 |
| Recall | 0.3804 |
| ROC-AUC | 0.6285 |
| Dir Acc | 0.6937 |
| F1 | 0.2483 |

### Acceptance Gate

**Overall: FAIL**

| Check | Value | Target | Result |
|-------|-------|--------|--------|
| Precision | 0.2161 | >= 0.5500 | FAIL |
| Recall | 0.3804 | >= 0.1500 | PASS |
| ROC-AUC | 0.6285 | >= 0.6000 | PASS |

