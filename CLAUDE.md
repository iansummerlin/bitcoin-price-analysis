# Project Guidelines

## What This Repo Is

A Bitcoin signal-generation and model-research repo. It does **not** place trades. It exists to answer whether a predictive signal survives realistic costs. The canonical plan and progress live in `ROADMAP.md`.

**Current status:** research-only. The model does not clear the integration bar. The dataset is stale (ends July 2025).

---

## Commands

```bash
make test              # run full test suite (pytest)
make train             # train default model (gen.py)
make backtest          # walk-forward evaluation (backtest.py)
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

1. **Baseline first** — run `make backtest` and record current metrics before touching code.
2. **State hypothesis** — what you're changing, why, and what metric improvement you expect.
3. **Implement** — make the minimal change.
4. **Re-evaluate** — run `make backtest` again. Run `make test` to confirm no regressions.
5. **Compare** — if any key metric degraded (precision, recall, ROC-AUC), revert unless explicitly accepted.
6. **Never claim improvement without data** — show before/after numbers side-by-side.

---

## Architecture

```
data/         # loaders, validation, dataset assembly
evaluation/   # targets, baselines, walk-forward, ablation, comparison
features/     # deterministic feature engineering pipeline
models/       # model interface (base.py), ARIMA, XGBoost
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
- **Targets are trading-aligned** — the default target is cost-adjusted direction, not raw price. See `evaluation/targets.py`.

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
