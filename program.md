# Bitcoin Signal Research Program

## What this loop does
The experiment loop (`scripts/experiment_loop.py`) is a **predefined search** over
model hyperparameters and feature subsets. It does not modify code or generate new
features — it evaluates a fixed list of configurations against the walk-forward
harness and keeps or discards each based on the composite metric.

The search space includes:
- XGBoost and LightGBM hyperparameter grids (n_estimators, max_depth, learning_rate, num_leaves)
- Feature family ablation (remove cross-asset, on-chain, microstructure, or all new families)
- Individual feature removal (drop each of 42 features one at a time)
- Decision-threshold tuning for probabilistic classifiers
- Actionable-threshold and cost-buffer tuning for target construction

This loop is evidence-generating only. It does not promote winning configurations
into `config.py`, training defaults, or export paths.

## Metric
Optimization target: **composite score** = `precision * recall` on the
cost-adjusted directional target (`target_direction_cost_adj`).

This product captures both precision and recall in a single number. A configuration
that achieves precision=0.55 and recall=0.15 would score 0.0825.

After the first full run, this metric should be treated as a coarse search metric,
not the final decision objective. Follow-up runs should explicitly focus on the
acceptance gate, especially held-out precision.

## Budget
- Stop after 100 experiments or 12 hours, whichever comes first.
- Each experiment runs a full walk-forward evaluation (~5 minutes with 92 windows).

## Scope
The loop evaluates configurations using `evaluation/walk_forward.py`. It does not
modify any source files. The mutable dimensions are model choice, hyperparameters,
and feature column subsets — all passed as runtime parameters.

Immutable during the search: `evaluation/*`, `data/*`, `tests/*`, `backtest.py`, `signals/*`.

## Safety rails
1. A change must improve the composite metric by at least 0.005 to be kept.
2. Regime diversity: improvement must hold across at least 2 walk-forward windows.
3. Budget cap: 100 experiments or 12 hours.
4. Checkpoint/resume: progress saved after every experiment.
5. Stop flag: touch a file to gracefully stop between experiments.
6. No automatic code mutation: findings must be reviewed and promoted manually.

## Results logging
Every experiment is logged to `results.tsv` with columns:
`run_id`, `experiment_id`, `model_name`, `composite_metric`, `precision`, `recall`,
`roc_auc`, `f1`, `windows_evaluated`, `window_improvements`, `status`, `reason`,
`description`.

## Integration thresholds
- cost-adjusted directional precision >= 0.55
- cost-adjusted directional recall >= 0.15
- cost-adjusted directional ROC-AUC >= 0.60

## Held-out validation
The most recent 3 months of data (config.HELD_OUT_HOURS) are reserved as the
final validation set. The experiment loop operates on everything before that cutoff.
The held-out set is evaluated exactly once at the end of the experiment loop.

### Held-out training protocol
At the end of the experiment loop, the best surviving configuration is trained on
**all pre-held-out data** (the full experiment dataset) and then evaluated on the
held-out set. This is intentionally different from the walk-forward protocol used
during the search phase:

- **During search:** sliding walk-forward windows (180-day train, 30-day test) for
  regime diversity checking across 92 windows.
- **Final held-out:** single train/predict pass using all available pre-held-out
  data, because (a) the held-out set is only evaluated once, (b) we want the
  model to have the best possible training set for the final verdict, and
  (c) walk-forward already validated the configuration's robustness.

## Current interpretation
The first full run improved the walk-forward composite score but failed the held-out
precision/recall gate. That means the next autoresearch pass should focus on:

- probability-threshold tuning,
- actionable threshold / cost-buffer tuning,
- precision-first operating-point search,
- only then broader hyperparameter expansion.
