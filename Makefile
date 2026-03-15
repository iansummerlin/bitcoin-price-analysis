.PHONY: test train collect backtest regression-gate compare ablate export-signal setup experiment

# Run all tests
test:
	.venv/bin/python -m pytest tests/ -v

# Run tests with coverage
test-coverage:
	.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | head -100

# Train the model
train:
	.venv/bin/python gen.py

# Start data collection
collect:
	.venv/bin/python collect.py

# Run backtests
backtest:
	.venv/bin/python backtest.py

regression-gate:
	.venv/bin/python -m evaluation.regression_gate

compare:
	.venv/bin/python scripts/compare_models.py

ablate:
	.venv/bin/python scripts/run_ablation.py

export-signal:
	.venv/bin/python scripts/export_latest_signal.py

# Run experiment loop (Phase 13)
experiment:
	.venv/bin/python scripts/experiment_loop.py

# Setup environment
setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

# Install dev dependencies (for new models)
setup-dev:
	.venv/bin/pip install xgboost pytest
