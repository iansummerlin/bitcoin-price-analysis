# Project Guidelines

## What This Repo Is

A Bitcoin signal-generation and model-research repo. It does **not** place trades. It exists to answer whether a predictive signal survives realistic costs. The canonical plan and progress live in `ROADMAP.md`.

**Current status:** research-only. Phase 12 complete (data universe expanded, no new family individually validated as a signal source). Phase 13 complete (autonomous experiment loop — 3 runs, 172 experiments, search space exhausted). All runs failed Gate 7. Best held-out: precision=0.407 (need 0.55), recall=0.059 (need 0.15), ROC-AUC=0.703 (pass). Conclusion: hyperparameter/threshold tuning cannot close the precision gap with the current 42-feature set. The bottleneck is data inputs, not model configuration. Data last refreshed March 14, 2026.

---

## Commands

```bash
make test              # run full test suite (pytest)
make train             # train default model (gen.py)
make backtest          # walk-forward evaluation (backtest.py) — auto-saves history + BACKTEST.md
make regression-gate   # compare latest backtest against previous (exit 1 on regression)
make compare           # model family comparison
make ablate            # feature ablation
make export-signal     # export latest signal artifact
make setup             # create venv + install deps
```

All commands use `.venv/bin/python`. Activate with `source .venv/bin/activate` or use `make` targets.

---

## Non-Negotiable Constraints

1. **Tests must pass before any change is considered complete.** Run `make test` after every code change. No exceptions.
2. **No lookahead leakage.** Training, scaling, and feature computation must never use future data. Walk-forward splits are train-on-past, predict-on-future only.
3. **Out-of-sample evidence is the only real evidence.** Do not make decisions from training error or single-split results.
4. **The repo must fail honestly.** If the model is stale, weak, or not useful, say so explicitly. Never overstate signal quality.
5. **Do not manually tune model parameters outside a structured experiment.** Ad-hoc tweaks without before/after comparison are not allowed.

---

## Immutable Files During Experimentation

These files define how models are judged. Modifying them during feature/hyperparameter exploration invalidates all results:

- `evaluation/walk_forward.py` — scoring harness
- `evaluation/baselines.py` — baseline implementations
- `evaluation/targets.py` — target construction
- `evaluation/reporting.py` — metric computation
- `evaluation/cost_model.py` — cost simulation
- `evaluation/signal_rules.py` — signal decision rules
- `data/loaders.py` — data loading
- `data/pipeline.py` — dataset assembly
- `data/validation.py` — data validation
- `backtest.py` — evaluation entrypoint
- `signals/` — signal export contract
- `tests/` — test suite

If these need changes, propose them as a separate task with explicit justification. Never change the ruler while measuring.

---

## Model Change Protocol

Any change to features, hyperparameters, targets, or model architecture:

1. **Baseline first** — run `make backtest` and record current metrics before touching code. The result is saved to `artifacts/backtest_history.json` and `BACKTEST.md` automatically.
2. **State hypothesis** — what you're changing, why, and what metric improvement you expect.
3. **Implement** — make the minimal change.
4. **Re-evaluate** — run `make backtest` again (auto-appends to history). Run `make test` to confirm no regressions.
5. **Check regression gate** — the backtest run automatically compares against the previous entry. If `make regression-gate` exits 1, revert unless explicitly accepted. Check `BACKTEST.md` for the side-by-side history table.
6. **Never claim improvement without data** — the history table in `BACKTEST.md` shows before/after numbers.

---

## Architecture

```
data/         # loaders, validation, dataset assembly
evaluation/   # targets, baselines, walk-forward, ablation, comparison,
              #   cost simulation (cost_model.py), signal rules (signal_rules.py)
features/     # deterministic feature engineering pipeline
models/       # model interface (base.py), XGBoost, LightGBM
signals/      # downstream signal export and validation
scripts/      # CLI wrappers (compare, ablate, export)
tests/        # unit + integration tests
config.py     # all shared constants, feature lists, thresholds
```

**Entrypoints:** `gen.py` (train), `backtest.py` (evaluate), `collect.py` (live analytics prep).

---

## Key Conventions

- **Single canonical feature pipeline** — `features/pipeline.py` is used by training, evaluation, and inference. Never duplicate feature logic.
- **Config is the source of truth** — feature lists (`EXOG_COLUMNS`), cost parameters, walk-forward windows, and schema versions live in `config.py`.
- **Artifacts go in `artifacts/`** — model files, predictions, metadata, signal exports. Never commit stale artifacts without updating `dataset_metadata.json`.
- **Backtest history is append-only** — `artifacts/backtest_history.json` is the source of truth. `BACKTEST.md` is auto-generated from it — never edit manually. Every `make backtest` run appends to history and regenerates the markdown.
- **Targets are trading-aligned** — the default target is cost-adjusted direction, not raw price. See `evaluation/targets.py`.
- **All external data sources must use the shared cache** — any loader that fetches from a remote API must go through `data/cache.py` with a defined TTL. No loader may bypass the cache or retry repeatedly on failure. On API failure, prefer stale cache over retries. Repeated `build_dataset()` calls within TTL must not hit the remote API.
- **External data families are toggleable** — `build_dataset()` accepts `include_crossasset`, `include_onchain`, `include_microstructure` flags. When disabled, feature columns are filled with 0 (neutral) to maintain schema compatibility.

---

## Integration Thresholds

The model must clear these on walk-forward evaluation before downstream integration is considered:

- cost-adjusted directional precision >= `0.55`
- cost-adjusted directional recall >= `0.15`
- cost-adjusted directional ROC-AUC >= `0.60`

These must hold across multiple walk-forward windows, not just one split.

---

## What Not to Do

- Do not add execution, order routing, or wallet management to this repo.
- Do not keep stale artifacts, dead code, or unused files — remove them.
- Do not weaken or remove tests to make the suite pass — fix the code.
- Do not retrain on fresh data without re-running the full evaluation pipeline.
- Do not commit generated artifacts (model files, predictions) without updated metadata.
