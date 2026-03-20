# Bitcoin Price Analysis Roadmap

## Document Status

- **Purpose:** Turn this repository into a reliable Bitcoin signal-generation and model-research repo for a separate trading/execution repository.
- **Audience:** A fresh engineering agent with zero prior context.
- **Primary constraint:** Prove whether the repo can produce a usable predictive signal. It must not become an execution bot.
- **Last updated:** March 20, 2026 (Phase 13 concluded; post-Phase-13 liquidity integration and gating research incorporated). Data refreshed through March 13, 2026.

---

## Start Here

If you are a fresh agent with no other context:

1. Read this roadmap fully once.
2. Read `CLAUDE.md` for architecture, commands, and non-negotiable constraints.
3. Read `README.md` to confirm it matches the roadmap.
4. **Phases 0–13 are complete.** The infrastructure is solid — do not redo this work.
5. **Phase 13 (autonomous experiment loop) concluded after 3 runs and 172 experiments.** All runs failed Gate 7. The search space is exhausted — hyperparameter/threshold tuning cannot close the precision gap with the current feature set. The bottleneck is data inputs.
6. **Post-Phase-13 liquidity work is complete and documented.** The repo now consumes the sibling `global-liquidity-analysis` artifact, but the result was context-only: additive liquidity features were mixed, and an optional directional liquidity gate is modestly helpful in pooled classification metrics but operationally weak in trading terms.
7. Read `program.md` for experiment loop operating model and `AUTORESEARCH.md` for the latest run results.

**Default next task:** Expand the data universe with leading indicators beyond the now-tested liquidity artifact before running the experiment loop again. See "If Phase 12 is reopened" section and Phase 13 conclusion for context.

---

## Mission

This repository should answer, with evidence:

1. Can we generate a Bitcoin prediction signal with repeatable out-of-sample edge?
2. Is that edge large enough to survive realistic trading costs?
3. Can we package that signal cleanly for use in a separate trading strategy repo?

If the evidence says "no", the correct outcome is to document that clearly and stop treating this repo as a production dependency.

---

## Integration Thresholds

The model must clear all three on walk-forward evaluation before downstream integration:

| Metric | Threshold | Best held-out (run 1) | Best held-out (run 2) | Best held-out (run 3) |
|--------|-----------|----------------------|----------------------|----------------------|
| Precision | >= 0.55 | 0.4074 (FAIL) | 0.3333 (FAIL) | 0.4074 (FAIL) |
| Recall | >= 0.15 | 0.0595 (FAIL) | 0.0108 (FAIL) | 0.0595 (FAIL) |
| ROC-AUC | >= 0.60 | 0.7033 (PASS) | 0.7140 (PASS) | 0.7033 (PASS) |

These must hold across multiple walk-forward windows, not just one split.

---

## Working Principles

1. **Out-of-sample evidence is the only real evidence.** Do not make decisions from training error, fitted-value error, or single split results.
2. **Data consistency is mandatory.** Training, validation, and inference must use the same market semantics, schema, timestamp rules, and feature definitions.
3. **Simpler models get priority.** A simple stable signal is more valuable than a complex fragile model.
4. **Signal usefulness matters more than forecast elegance.** A model that predicts direction well enough to support trades beats one with slightly lower RMSE but no tradable edge.
5. **The repo must fail honestly.** If the model is stale, unsupported, or not useful, say so explicitly.
6. **Remove what is unnecessary.** Do not keep dead UI, execution code, duplicate docs, stale planning files, or generated clutter.

---

## Completed Work Summary (Phases 0–12)

All infrastructure phases are done. This section exists for context, not as a task list.

### Foundation (Phases 0–10)

- Repo reframed as signal-generation/research (not a trading bot).
- Deterministic data pipeline with schema validation.
- Walk-forward evaluation with strict no-lookahead splits.
- 6 naive baselines for honest comparison.
- Cost-adjusted trading-aligned targets.
- Model comparison and ablation reporting.
- Versioned signal export contract (`signals/`).
- Comprehensive test suite. A-tier documentation.
- All entrypoints: `gen.py` (train), `backtest.py` (evaluate), `collect.py` (live analytics prep).

### Data refresh (Phase 11)

Data refreshed March 14, 2026. Dataset covers 2015-10-13 to 2026-03-13 (91,302 rows after feature pipeline).

### Information set expansion (Phase 12)

Phase 12's goal was to rebuild the repo's information set and prediction framing so that Phase 13 could search a materially better problem. It was not only about richer features — it also determined the prediction horizon where those features can plausibly matter.

**Why Phase 12 exists:** The pre-Phase-12 model used 23 features derived almost entirely from price and volume data (lagged closes, returns, RSI, ATR, moving averages) plus a single sentiment index. These are lagging indicators that every market participant has access to. The hypothesis was that better data and/or better problem framing are more important than incremental tuning. Without Phase 12, the experiment loop would be searching a narrow, low-signal space.

**Implementation constraints that still apply if Phase 12 is reopened:**
1. **Free sources only** for initial implementation. Paid providers are documented below as future options.
2. **All external data goes through the shared cache** (`data/cache.py`). No loader may bypass it or retry-loop on failure.
3. **Stale cache over retries.** On API failure, fall back to stale cache. If no cache exists, return empty DataFrame with correct schema.
4. **Each new data family is a hypothesis to test**, not a known signal source. Integrate, ablate, document the result.

**Measurable improvement rules:** A new data family counts as an improvement if (1) at least one gated metric improves by >= 0.01 absolute, (2) no metric degrades by > 0.005, and (3) improvement holds across >= 2 walk-forward windows. If a family shows a tradeoff (improves one, degrades another), defer to Phase 13's experiment loop.

#### Sub-phase results

- **12A Horizon analysis:** 4h horizon is the most promising target (precision=0.487, recall=0.218, ROC-AUC=0.563). 1h has best ROC-AUC but very low precision. 24h is near-random. Multi-timeframe targets added to `evaluation/targets.py` with horizon-scaled cost thresholds in `config.py`.

- **12B Cross-asset:** 10 features via yfinance (DXY, S&P 500, VIX, Gold, ETH). Loader: `data/crossasset.py`. Features: `features/crossasset.py`. Cache TTL: 6 hours. Fill policy: Friday close forward-filled through weekends, daily aligned to UTC then forward-filled to hourly. **Result:** Tradeoff — improves recall (+0.138) and ROC-AUC (+0.019) but degrades precision (-0.117). Not a clear net improvement by the measurable-improvement rules.

- **12C On-chain:** 6 features via blockchain.com (hash rate, difficulty, tx count, tx volume, hashrate-price divergence). Loader: `data/onchain.py`. Features: `features/onchain.py`. Cache TTL: 24 hours. **Result:** Negative to neutral. Precision degrades in isolation, marginal ROC-AUC improvement when combined with cross-asset. Note: the free on-chain metrics are coarse proxies — the stronger features (exchange flows, MVRV, SOPR) require paid providers.

- **12D LightGBM:** Added as alternative model family (`models/lightgbm_model.py`), registered in walk-forward and model comparison. **Result:** Different precision-recall tradeoff from XGBoost. LightGBM+expanded features achieves ROC-AUC=0.603, recall=0.183, precision=0.428.

- **12E Microstructure:** 3 funding rate features via Binance (`data/microstructure.py`, `features/microstructure.py`). Cache TTL: 1 hour. Open interest excluded (free API only serves ~30 days of history). **Result:** Negligible impact on either model.

- **12F Dynamic features:** Skipped — no data family showed clear individual improvement, so full contract redesign was not needed. The `build_dataset()` API already supports toggling families via `include_crossasset`, `include_onchain`, `include_microstructure` flags.

#### If Phase 12 is reopened

The most likely reason to reopen Phase 12 is that Phase 13 fails to close the precision gap, suggesting the feature set itself is insufficient. The highest-value next data sources to investigate:

| Provider | Cost | What it adds | When to consider |
|----------|------|-------------|-----------------|
| **Glassnode** | $29/mo | MVRV, NVT, exchange net flows, whale metrics, realized price, SOPR — strongest on-chain analytics | If Phase 13 fails and on-chain proxy features (12C) showed any directional signal |
| **CoinGlass** | $49/mo | Historical open interest back to 2019, liquidation data, long/short ratios, multi-exchange aggregation | If funding rate features (12E) showed any signal — OI/liquidation data is likely where the real microstructure value is |
| **FRED API** | Free (key required) | CPI, Fed funds rate, yield curve, M2 money supply | If cross-asset features (12B) showed macro regime signal |
| **CryptoQuant** | $29/mo | Exchange reserves, miner flows, whale alerts, fund flow ratio | If exchange flow granularity beyond Glassnode is needed |

Any new data family added in a reopened Phase 12 must follow the same pattern: integrate, ablate in isolation, document the result in `BACKTEST.md`, then hand to Phase 13 for combinatorial search.

**Best known configuration entering Phase 13:** LightGBM with full expanded features on 4h horizon — ROC-AUC=0.603, recall=0.183, precision=0.428. Clears 2 of 3 thresholds. Precision gap (0.428 vs 0.55) is the remaining bottleneck.

**Key insight:** The infrastructure is not the problem — the inputs are. The model detects moves but can't reliably filter noise from signal with current features. All 42 features are lagging indicators that are widely available and largely priced in.

---

## Post-Phase-13 Liquidity Research (complete)

After Phase 13 concluded, the repo integrated the sibling [`global-liquidity-analysis`](/home/ixn/Documents/code/crypto/global-liquidity-analysis) artifact as an optional feature family.

What was added:

- `data/liquidity.py` artifact loader
- optional `include_liquidity` dataset flag
- `LIQUIDITY_COLUMNS` in `config.py`
- research scripts for ablation, representation tests, gating tests, and trading-aligned comparison
- optional `make backtest-gated` path

What the research found:

- additive liquidity features were a mixed tradeoff at 1h
- additive liquidity was effectively neutral at 4h
- simpler liquidity representations were cleaner than the full additive family
- liquidity regime was more useful as context than as direct additive input
- a directional liquidity gate (`EXPANDING` + `CONTRACTING`, suppress `NEUTRAL`) modestly improved pooled classification metrics
- the same gate remained operationally weak in trading-aligned terms because the base signal is still deeply negative after costs

Conclusion:

- liquidity is worth keeping as optional research infrastructure
- liquidity is not the fix for the repo’s core economics
- the binding problem remains absolute signal quality after costs

---

## Phase 13: Autonomous Experiment Loop (complete)

### Objective

Systematically explore the feature/hyperparameter/target search space using an autonomous experiment loop, layered on top of the existing walk-forward evaluation infrastructure.

### Final status

Infrastructure is built and working well. Three runs completed, 172 total experiments, search space exhausted:

- **Run 1:** 100 experiments in 284 minutes. 3 kept, 97 discarded. Best walk-forward: `xgb: n_est=300, max_d=5, lr=0.1` (composite=0.0997). Held-out: precision=0.4074, recall=0.0595, ROC-AUC=0.7033. **Gate 7: FAIL.**
- **Run 2:** 72 experiments in 728 minutes. 2 kept, 70 discarded. Best walk-forward: `xgb: n_est=100, max_d=5, lr=0.05` (composite=0.0638). Held-out: precision=0.3333, recall=0.0108, ROC-AUC=0.7140. **Gate 7: FAIL.**
- **Run 3:** 0 experiments (stopped early) in 52 minutes. No experiment beat the baseline — the search space has converged. Held-out: precision=0.4074, recall=0.0595, ROC-AUC=0.7033. **Gate 7: FAIL.**

All runs failed the held-out precision and recall gates. ROC-AUC consistently clears the bar. The search space is exhausted — no hyperparameter, threshold, or feature subset combination in the current space can close the precision gap.

### How it works

The experiment loop (`scripts/experiment_loop.py`) is a predefined runtime search over model hyperparameters and feature subsets. It does not modify source files. All variation is expressed as runtime parameters passed to the walk-forward evaluation.

```
1. Build dataset once, split off held-out (last 3 months, 2160 hours)
2. Establish baselines (XGBoost + LightGBM, default configs)
3. For each experiment in the search space:
   a. Run walk-forward evaluation (92 windows)
   b. Compare composite metric (precision x recall) against current best
   c. Check regime diversity (must hold on >= 2 windows)
   d. Keep if improved by >= 0.005 AND diversity passes
   e. Log to results.tsv
4. Train best config on all pre-held-out data
5. Evaluate once on held-out set
6. Generate AUTORESEARCH.md report and Gate 7 verdict
```

### Conclusion

The experiment loop answered its question honestly: **hyperparameter/threshold tuning cannot close the precision gap with the current feature set.** Run 2 tested probability thresholds (0.50–0.75), actionable thresholds (0.0035–0.0075), and cost buffers — none helped. Run 1 ablated all 42 features individually — no single feature removal improved the composite metric. The feature set is uniformly weak: all 42 features are lagging indicators that are widely available and largely priced in.

The experiment loop infrastructure is solid and should be reused once the data universe is expanded with leading indicators. The next high-value work is new data families (see "If Phase 12 is reopened" above), not more tuning.

### Critical design constraints

1. **Held-out validation set is sacred.** The most recent 3 months are reserved for final validation. The loop trains/evaluates on everything before that. The held-out set is used once per run.
2. **Regime diversity is mandatory.** Improvements must hold across at least 2 walk-forward windows.
3. **Complexity has a cost.** Minimum improvement threshold of 0.005 prevents accumulating noise.
4. **The evaluation harness is never modified.** The loop can change inputs (features, parameters) but never how models are judged (walk-forward splits, cost model, baselines, metrics).
5. **Winning configs are not auto-promoted.** The loop produces evidence, not codebase mutations.

### Scope boundaries

**Mutable dimensions (passed at runtime):**
- Model choice: XGBoost or LightGBM direction classifier
- Hyperparameters: n_estimators, max_depth, learning_rate, num_leaves
- Feature subsets: any subset of the 42 EXOG_COLUMNS
- Decision threshold, actionable threshold, cost buffer

**Immutable (never touched by the loop):**
- `evaluation/*`, `data/*`, `tests/*`, `backtest.py`, `signals/*`, `features/*`

### Key files

- `program.md` — experiment loop operating model and directives
- `scripts/experiment_loop.py` — the autonomous runner
- `results.tsv` — experiment log (append-only)
- `AUTORESEARCH.md` — latest run report (auto-generated)
- `artifacts/autoresearch_history.json` — run-level history
- `artifacts/autoresearch_runs/` — archived per-run artifacts
- `Makefile` — `make experiment` target

### Phase 13 checklist

- [x] Infrastructure: experiment loop, program.md, held-out split, safety rails, results format.
- [x] Run 1: 100 experiments, broad hyperparameter + feature search. Gate 7 FAIL.
- [x] Run 2: 72 experiments, threshold tuning + refined search. Gate 7 FAIL.
- [x] Run 3: 0 experiments (search space converged, stopped early). Gate 7 FAIL.
- [x] Document conclusion: search space exhausted, precision gap cannot be closed with current features. Signal hypothesis not abandoned — model has discriminative ability (ROC-AUC 0.70+) but needs better inputs to convert that into precision.

### Exit criteria

- [x] The experiment loop has run at least 50 experiments with proper logging. (172 experiments across 3 runs)
- [x] The best surviving configuration has been evaluated on the held-out validation set. (3 times)
- Held-out metrics did not clear the integration bar after sustained experimentation (172 experiments).
- Conclusion: the signal hypothesis is not abandoned (ROC-AUC 0.70+ shows discriminative ability), but the current feature set cannot produce actionable precision. New data families are needed before further experiment loop runs.

---

## Decision Gates

| Gate | Question | Status |
|------|----------|--------|
| 0 | Repo contains only signal-mission files? | PASS |
| 1 | Data pipeline reproducible and consistent? | PASS |
| 2 | Real model predictions beat baselines OOS? | FAIL (precision 0.169, recall 0.261) |
| 3 | Another repo can consume outputs safely? | PASS (interface), FAIL (Gate 2 blocks) |
| 4 | Fresh agent can operate repo from docs? | PASS |
| 5 | Best signal survives costs with stable OOS performance? | FAIL |
| 6 | Did expanding data universe improve metrics? | FAIL (no individual family helped; search space expanded for Phase 13) |
| 7 | Best experiment config clears held-out bar? | FAIL (3 runs, 172 experiments, best precision=0.41 vs 0.55 needed) |

**Gate 5 (final go/no-go)** is the ultimate decision. Gate 7 has failed after sustained effort (172 experiments across 3 runs including precision-focused search). The signal hypothesis is not abandoned — the model has discriminative ability — but the current feature set cannot close the precision gap. New data families (leading indicators) are needed before Gate 7 can be re-attempted.

---

## What Blocks Progress Now

**Precision is the bottleneck, and it's a data problem.** The model achieves decent ROC-AUC (0.70+) on held-out data, meaning it has discriminative ability. But it cannot convert that into precise actionable signals above the 0.55 bar. Phase 13 proved this empirically: 172 experiments across 3 runs — covering hyperparameter grids, decision thresholds, actionable thresholds, cost buffers, feature ablation, and both model families — failed to close the gap. The core problem is confirmed: the 42-feature set consists of lagging, widely-available indicators that are largely priced in.

What has already been tested since Phase 13:

- global liquidity as an additive feature family
- global liquidity as a simpler regime-only representation
- global liquidity as an optional directional regime gate

These did not change the final economic judgment. The gate made the model slightly less bad, not good.

One path forward:

1. **Better data** — the remaining option is data sources with genuine leading signal that the market hasn't fully absorbed beyond the now-tested liquidity artifact. Top candidates: exchange flows/MVRV/SOPR (Glassnode $29/mo), historical open interest/liquidations (CoinGlass $49/mo), or other non-price leading indicators. See "If Phase 12 is reopened" section for the full list. Once new data families are integrated and ablated, the Phase 13 experiment loop should be re-run on the expanded feature set.
