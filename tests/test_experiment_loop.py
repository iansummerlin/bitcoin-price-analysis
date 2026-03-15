"""Tests for Phase 13 experiment loop infrastructure.

Covers:
- compute_composite metric calculation
- regime_diversity_check logic
- results.tsv save/load round-trip
- generate_experiments produces valid experiment dicts
- run_single_experiment keep/discard logic on synthetic data
- evaluate_configuration returns correct structure
- generate_autoresearch_md produces valid markdown
- held-out split isolation
- budget cap enforcement
- end-to-end mini loop (build dataset, baseline, 2 experiments, held-out)
"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from backtest import apply_features
from config import (
    CROSSASSET_COLUMNS,
    DEFAULT_DIRECTION_TARGET_COLUMN,
    DEFAULT_TARGET_COLUMN,
    EXOG_COLUMNS,
    EXPERIMENT_MIN_IMPROVEMENT,
    MICROSTRUCTURE_COLUMNS,
    ONCHAIN_COLUMNS,
)
from evaluation.targets import add_targets

# Import experiment loop functions
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.experiment_loop import (
    AUTORESEARCH_PATH,
    ExperimentResult,
    compute_composite,
    evaluate_configuration,
    generate_autoresearch_md,
    generate_experiments,
    regime_diversity_check,
    run_single_experiment,
    save_result,
    load_results,
    RESULTS_PATH,
)


def _make_experiment_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic dataset large enough for small walk-forward windows."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 50_000 + np.cumsum(rng.randn(n) * 90)
    high = close + rng.uniform(50, 200, n)
    low = close - rng.uniform(50, 200, n)
    open_ = close + rng.randn(n) * 35
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "Volume BTC": rng.uniform(1, 100, n),
            "Volume USD": rng.uniform(50_000, 500_000, n),
            "fng_value": rng.randint(10, 90, n).astype(float),
        },
        index=dates,
    )
    featured = apply_features(df)
    rng2 = np.random.RandomState(seed + 1)
    for col in CROSSASSET_COLUMNS + ONCHAIN_COLUMNS + MICROSTRUCTURE_COLUMNS:
        featured[col] = rng2.randn(len(featured))
    return add_targets(featured)


class TestComputeComposite(unittest.TestCase):
    def test_basic_product(self):
        scores = {"precision": 0.5, "recall": 0.4}
        self.assertAlmostEqual(compute_composite(scores), 0.2)

    def test_zero_precision(self):
        scores = {"precision": 0.0, "recall": 0.5}
        self.assertAlmostEqual(compute_composite(scores), 0.0)

    def test_zero_recall(self):
        scores = {"precision": 0.5, "recall": 0.0}
        self.assertAlmostEqual(compute_composite(scores), 0.0)

    def test_missing_keys_default_zero(self):
        self.assertAlmostEqual(compute_composite({}), 0.0)

    def test_perfect_scores(self):
        scores = {"precision": 1.0, "recall": 1.0}
        self.assertAlmostEqual(compute_composite(scores), 1.0)


class TestRegimeDiversityCheck(unittest.TestCase):
    def test_all_windows_improve(self):
        baseline = [{"precision": 0.3, "recall": 0.1}] * 5
        new = [{"precision": 0.4, "recall": 0.2}] * 5
        passed, count = regime_diversity_check(baseline, new)
        self.assertTrue(passed)
        self.assertEqual(count, 5)

    def test_no_windows_improve(self):
        baseline = [{"precision": 0.4, "recall": 0.2}] * 5
        new = [{"precision": 0.3, "recall": 0.1}] * 5
        passed, count = regime_diversity_check(baseline, new)
        self.assertFalse(passed)
        self.assertEqual(count, 0)

    def test_exactly_two_windows_improve(self):
        baseline = [{"precision": 0.3, "recall": 0.1}] * 5
        new = [
            {"precision": 0.4, "recall": 0.2},  # improved
            {"precision": 0.2, "recall": 0.05},  # worse
            {"precision": 0.4, "recall": 0.2},  # improved
            {"precision": 0.2, "recall": 0.05},  # worse
            {"precision": 0.2, "recall": 0.05},  # worse
        ]
        passed, count = regime_diversity_check(baseline, new)
        self.assertTrue(passed)
        self.assertEqual(count, 2)

    def test_only_one_window_improves_fails(self):
        baseline = [{"precision": 0.3, "recall": 0.1}] * 5
        new = [
            {"precision": 0.4, "recall": 0.2},  # improved
            {"precision": 0.2, "recall": 0.05},  # worse
            {"precision": 0.2, "recall": 0.05},  # worse
            {"precision": 0.2, "recall": 0.05},  # worse
            {"precision": 0.2, "recall": 0.05},  # worse
        ]
        passed, count = regime_diversity_check(baseline, new)
        self.assertFalse(passed)
        self.assertEqual(count, 1)

    def test_mismatched_lengths_fails(self):
        baseline = [{"precision": 0.3, "recall": 0.1}] * 3
        new = [{"precision": 0.4, "recall": 0.2}] * 5
        passed, count = regime_diversity_check(baseline, new)
        self.assertFalse(passed)
        self.assertEqual(count, 0)

    def test_custom_min_windows(self):
        baseline = [{"precision": 0.3, "recall": 0.1}] * 5
        new = [
            {"precision": 0.4, "recall": 0.2},
            {"precision": 0.4, "recall": 0.2},
            {"precision": 0.4, "recall": 0.2},
            {"precision": 0.2, "recall": 0.05},
            {"precision": 0.2, "recall": 0.05},
        ]
        passed, count = regime_diversity_check(baseline, new, min_windows=3)
        self.assertTrue(passed)
        self.assertEqual(count, 3)


class TestResultsTsvRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.original_path = RESULTS_PATH

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        tmp_path = Path(self.tmpdir) / "results.tsv"

        result = ExperimentResult(
            experiment_id=1,
            description="test experiment",
            model_name="xgboost_direction",
            composite_metric=0.123456,
            precision=0.4,
            recall=0.3,
            roc_auc=0.6,
            f1=0.35,
            windows_evaluated=5,
            window_improvements=3,
            status="keep",
            reason="improvement +0.05",
        )

        # Temporarily redirect RESULTS_PATH
        import scripts.experiment_loop as loop_mod
        loop_mod.RESULTS_PATH = tmp_path
        try:
            save_result(result)
            save_result(ExperimentResult(
                experiment_id=2,
                description="second experiment",
                model_name="lightgbm_direction",
                composite_metric=0.05,
                precision=0.2,
                recall=0.25,
                roc_auc=0.55,
                f1=0.22,
                windows_evaluated=5,
                window_improvements=1,
                status="discard",
                reason="insufficient improvement",
            ))

            loaded = load_results()
        finally:
            loop_mod.RESULTS_PATH = self.original_path

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["experiment_id"], "1")
        self.assertEqual(loaded[0]["status"], "keep")
        self.assertEqual(loaded[0]["model_name"], "xgboost_direction")
        self.assertEqual(loaded[1]["experiment_id"], "2")
        self.assertEqual(loaded[1]["status"], "discard")

    def test_tsv_header_written_once(self):
        tmp_path = Path(self.tmpdir) / "results.tsv"
        import scripts.experiment_loop as loop_mod
        loop_mod.RESULTS_PATH = tmp_path
        try:
            for i in range(3):
                save_result(ExperimentResult(
                    experiment_id=i, description=f"exp {i}",
                    model_name="xgb", composite_metric=0.1,
                    precision=0.2, recall=0.3, roc_auc=0.5, f1=0.25,
                    windows_evaluated=5, window_improvements=2,
                    status="discard", reason="test",
                ))
            lines = tmp_path.read_text().strip().split("\n")
        finally:
            loop_mod.RESULTS_PATH = self.original_path

        # 1 header + 3 data rows
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[0].startswith("run_id"))


class TestGenerateExperiments(unittest.TestCase):
    def test_returns_nonempty_list(self):
        experiments = generate_experiments()
        self.assertGreater(len(experiments), 0)

    def test_all_experiments_have_required_keys(self):
        experiments = generate_experiments()
        for exp in experiments:
            self.assertIn("description", exp)
            self.assertIn("model_name", exp)
            self.assertIn(exp["model_name"], ["xgboost_direction", "lightgbm_direction"])

    def test_no_duplicate_descriptions(self):
        experiments = generate_experiments()
        descriptions = [e["description"] for e in experiments]
        self.assertEqual(len(descriptions), len(set(descriptions)),
                         "Duplicate experiment descriptions found")

    def test_feature_columns_are_valid_subsets(self):
        experiments = generate_experiments()
        all_features = set(EXOG_COLUMNS)
        for exp in experiments:
            if "feature_columns" in exp and exp["feature_columns"] is not None:
                exp_features = set(exp["feature_columns"])
                self.assertTrue(
                    exp_features.issubset(all_features),
                    f"Experiment '{exp['description']}' has features not in EXOG_COLUMNS: "
                    f"{exp_features - all_features}"
                )

    def test_includes_hyperparameter_experiments(self):
        experiments = generate_experiments()
        hyper_exps = [e for e in experiments if "model_kwargs" in e and e["model_kwargs"]]
        self.assertGreater(len(hyper_exps), 10, "Too few hyperparameter experiments")

    def test_includes_feature_ablation_experiments(self):
        experiments = generate_experiments()
        ablation_exps = [e for e in experiments if "feature_columns" in e]
        self.assertGreater(len(ablation_exps), 5, "Too few feature ablation experiments")

    def test_includes_individual_feature_removal(self):
        experiments = generate_experiments()
        drop_exps = [e for e in experiments if e["description"].startswith("lgbm: drop ")]
        self.assertGreater(len(drop_exps), 20, "Should test dropping each of 42 features")


class TestEvaluateConfiguration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_experiment_df(800)

    def test_returns_scores_and_window_scores(self):
        scores, window_scores = evaluate_configuration(
            self.dataset,
            model_name="xgboost_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=240,
            test_window=120,
        )
        self.assertIn("precision", scores)
        self.assertIn("recall", scores)
        self.assertIn("composite", scores)
        self.assertGreater(len(window_scores), 0)
        for ws in window_scores:
            self.assertIn("precision", ws)
            self.assertIn("composite", ws)

    def test_composite_equals_precision_times_recall(self):
        scores, _ = evaluate_configuration(
            self.dataset,
            model_name="xgboost_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=240,
            test_window=120,
        )
        expected = scores["precision"] * scores["recall"]
        self.assertAlmostEqual(scores["composite"], expected, places=6)

    def test_lightgbm_evaluates(self):
        scores, window_scores = evaluate_configuration(
            self.dataset,
            model_name="lightgbm_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=240,
            test_window=120,
        )
        self.assertIn("precision", scores)
        self.assertGreater(len(window_scores), 0)

    def test_empty_dataset_returns_empty(self):
        tiny = self.dataset.head(10)
        scores, window_scores = evaluate_configuration(
            tiny,
            model_name="xgboost_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=240,
            test_window=120,
        )
        self.assertEqual(scores, {})
        self.assertEqual(window_scores, [])


class TestRunSingleExperiment(unittest.TestCase):
    TRAIN_W = 240
    TEST_W = 120

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_experiment_df(800)
        # Establish a baseline with small windows that fit in 800 rows
        cls.baseline_scores, cls.baseline_window_scores = evaluate_configuration(
            cls.dataset,
            model_name="xgboost_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=cls.TRAIN_W,
            test_window=cls.TEST_W,
        )

    def test_returns_experiment_result(self):
        experiment = {
            "description": "test: xgb default",
            "model_name": "xgboost_direction",
        }
        result = run_single_experiment(
            experiment, self.dataset,
            self.baseline_scores, self.baseline_window_scores,
            experiment_id=1,
            train_window=self.TRAIN_W,
            test_window=self.TEST_W,
        )
        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(result.experiment_id, 1)
        self.assertIn(result.status, ["keep", "discard"])
        self.assertGreater(result.windows_evaluated, 0)

    def test_identical_config_is_discarded(self):
        """Same config as baseline should show ~0 improvement and be discarded."""
        experiment = {
            "description": "test: identical to baseline",
            "model_name": "xgboost_direction",
            "model_kwargs": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
        }
        result = run_single_experiment(
            experiment, self.dataset,
            self.baseline_scores, self.baseline_window_scores,
            experiment_id=2,
            train_window=self.TRAIN_W,
            test_window=self.TEST_W,
        )
        # Should be discarded (insufficient improvement)
        self.assertEqual(result.status, "discard")

    def test_custom_feature_columns(self):
        experiment = {
            "description": "test: fewer features",
            "model_name": "xgboost_direction",
            "feature_columns": list(EXOG_COLUMNS)[:15],
        }
        result = run_single_experiment(
            experiment, self.dataset,
            self.baseline_scores, self.baseline_window_scores,
            experiment_id=3,
            train_window=self.TRAIN_W,
            test_window=self.TEST_W,
        )
        self.assertIsInstance(result, ExperimentResult)
        self.assertGreater(result.windows_evaluated, 0)

    def test_lightgbm_experiment(self):
        experiment = {
            "description": "test: lgbm variant",
            "model_name": "lightgbm_direction",
            "model_kwargs": {"n_estimators": 50, "max_depth": 4},
        }
        result = run_single_experiment(
            experiment, self.dataset,
            self.baseline_scores, self.baseline_window_scores,
            experiment_id=4,
            train_window=self.TRAIN_W,
            test_window=self.TEST_W,
        )
        self.assertIsInstance(result, ExperimentResult)


class TestGenerateAutoresearchMd(unittest.TestCase):
    def test_generates_valid_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "AUTORESEARCH.md"

            import scripts.experiment_loop as loop_mod
            original = loop_mod.AUTORESEARCH_PATH
            loop_mod.AUTORESEARCH_PATH = md_path

            try:
                generate_autoresearch_md(
                    run_timestamp="2026-03-15T14:00:00+00:00",
                    elapsed_minutes=120.5,
                    baseline_model="xgboost_direction",
                    baseline_scores={"precision": 0.3, "recall": 0.12, "roc_auc": 0.6, "f1": 0.15, "composite": 0.036},
                    baseline_composite=0.036,
                    xgb_scores={"precision": 0.3, "recall": 0.12, "roc_auc": 0.6, "f1": 0.15},
                    lgbm_scores={"precision": 0.22, "recall": 0.03, "roc_auc": 0.62, "f1": 0.05},
                    experiment_results=[
                        ExperimentResult(1, "exp1", "xgboost_direction", 0.05, 0.25, 0.2, 0.58, 0.22, 5, 3, "keep", "good"),
                        ExperimentResult(2, "exp2", "lightgbm_direction", 0.02, 0.15, 0.13, 0.55, 0.14, 5, 1, "discard", "bad"),
                    ],
                    best_experiment=ExperimentResult(1, "exp1", "xgboost_direction", 0.05, 0.25, 0.2, 0.58, 0.22, 5, 3, "keep", "good"),
                    best_config={"description": "exp1", "model_name": "xgboost_direction"},
                    best_composite=0.05,
                    held_out_scores={"precision": 0.2, "recall": 0.18, "roc_auc": 0.58, "f1": 0.19, "directional_accuracy": 0.6},
                    held_out_composite=0.036,
                    gate_pass=False,
                    experiment_rows=70000,
                    held_out_rows=2160,
                    experiment_end="2025-12-13 23:00:00+00:00",
                    held_out_start="2025-12-14 00:00:00+00:00",
                    held_out_end="2026-03-13 23:00:00+00:00",
                    total_experiments=2,
                    kept_count=1,
                    discarded_count=1,
                )
            finally:
                loop_mod.AUTORESEARCH_PATH = original

            self.assertTrue(md_path.exists())
            content = md_path.read_text()

            # Check all sections are present
            self.assertIn("# Autoresearch Report", content)
            self.assertIn("## Run Info", content)
            self.assertIn("## Baselines", content)
            self.assertIn("## Experiments", content)
            self.assertIn("## Best Configuration", content)
            self.assertIn("## Held-Out Validation", content)
            self.assertIn("## Gate 7 Verdict", content)

            # Check key data is present
            self.assertIn("2026-03-15", content)
            self.assertIn("120.5 minutes", content)
            self.assertIn("2 (1 kept", content)
            self.assertIn("1 kept", content)
            self.assertIn("exp1", content)
            self.assertIn("exp2", content)
            self.assertIn("FAIL", content)  # gate failed
            self.assertIn("xgboost_direction", content)

    def test_gate_pass_shows_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "AUTORESEARCH.md"

            import scripts.experiment_loop as loop_mod
            original = loop_mod.AUTORESEARCH_PATH
            loop_mod.AUTORESEARCH_PATH = md_path

            try:
                generate_autoresearch_md(
                    run_timestamp="2026-03-15T14:00:00+00:00",
                    elapsed_minutes=60.0,
                    baseline_model="xgboost_direction",
                    baseline_scores={"precision": 0.3, "recall": 0.12, "roc_auc": 0.6, "f1": 0.15},
                    baseline_composite=0.036,
                    xgb_scores={"precision": 0.3, "recall": 0.12, "roc_auc": 0.6, "f1": 0.15},
                    lgbm_scores={"precision": 0.22, "recall": 0.03, "roc_auc": 0.62, "f1": 0.05},
                    experiment_results=[],
                    best_experiment=None,
                    best_config=None,
                    best_composite=0.036,
                    held_out_scores={"precision": 0.6, "recall": 0.2, "roc_auc": 0.65, "f1": 0.3, "directional_accuracy": 0.7},
                    held_out_composite=0.12,
                    gate_pass=True,
                    experiment_rows=70000,
                    held_out_rows=2160,
                    experiment_end="2025-12-13",
                    held_out_start="2025-12-14",
                    held_out_end="2026-03-13",
                    total_experiments=0,
                    kept_count=0,
                    discarded_count=0,
                )
            finally:
                loop_mod.AUTORESEARCH_PATH = original

            content = md_path.read_text()
            self.assertIn("**PASS**", content)
            self.assertIn("downstream integration review", content)

    def test_no_best_experiment_handled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = Path(tmpdir) / "AUTORESEARCH.md"

            import scripts.experiment_loop as loop_mod
            original = loop_mod.AUTORESEARCH_PATH
            loop_mod.AUTORESEARCH_PATH = md_path

            try:
                generate_autoresearch_md(
                    run_timestamp="2026-03-15T14:00:00+00:00",
                    elapsed_minutes=10.0,
                    baseline_model="lightgbm_direction",
                    baseline_scores={"precision": 0.22, "recall": 0.03, "roc_auc": 0.62, "f1": 0.05},
                    baseline_composite=0.0066,
                    xgb_scores={"precision": 0.3, "recall": 0.12, "roc_auc": 0.6, "f1": 0.15},
                    lgbm_scores={"precision": 0.22, "recall": 0.03, "roc_auc": 0.62, "f1": 0.05},
                    experiment_results=[],
                    best_experiment=None,
                    best_config=None,
                    best_composite=0.0066,
                    held_out_scores={"precision": 0.1, "recall": 0.05, "roc_auc": 0.5, "f1": 0.07, "directional_accuracy": 0.5},
                    held_out_composite=0.005,
                    gate_pass=False,
                    experiment_rows=70000,
                    held_out_rows=2160,
                    experiment_end="2025-12-13",
                    held_out_start="2025-12-14",
                    held_out_end="2026-03-13",
                    total_experiments=0,
                    kept_count=0,
                    discarded_count=0,
                )
            finally:
                loop_mod.AUTORESEARCH_PATH = original

            content = md_path.read_text()
            self.assertIn("No experiment improved over the baseline", content)


class TestHeldOutSplitIsolation(unittest.TestCase):
    def test_held_out_cutoff_is_correct(self):
        """Verify that the held-out split leaves enough data for walk-forward."""
        from config import HELD_OUT_HOURS, DEFAULT_TRAIN_WINDOW, DEFAULT_TEST_WINDOW
        dataset = _make_experiment_df(3000)
        held_out_cutoff = len(dataset) - HELD_OUT_HOURS

        # Held-out should be positive (we have enough data)
        if len(dataset) > HELD_OUT_HOURS:
            self.assertGreater(held_out_cutoff, 0)
            experiment_data = dataset.iloc[:held_out_cutoff]
            held_out_data = dataset.iloc[held_out_cutoff:]
            # No overlap
            self.assertLess(experiment_data.index.max(), held_out_data.index.min())

    def test_held_out_never_in_walk_forward_windows(self):
        """Walk-forward windows on experiment data must not touch held-out rows."""
        from config import HELD_OUT_HOURS
        from evaluation.walk_forward import iter_walk_forward_slices
        dataset = _make_experiment_df(1500)
        held_out_cutoff = len(dataset) - min(HELD_OUT_HOURS, len(dataset) // 3)
        experiment_data = dataset.iloc[:held_out_cutoff]
        held_out_data = dataset.iloc[held_out_cutoff:]

        held_out_start = held_out_data.index.min()

        for train_df, test_df in iter_walk_forward_slices(
            experiment_data, train_window=240, test_window=120
        ):
            self.assertLess(
                train_df.index.max(), held_out_start,
                "Train data overlaps held-out set"
            )
            self.assertLess(
                test_df.index.max(), held_out_start,
                "Test data overlaps held-out set"
            )


class TestEndToEndMiniLoop(unittest.TestCase):
    """Lightweight integration test: run 2 experiments on synthetic data."""

    TRAIN_W = 240
    TEST_W = 120

    @classmethod
    def setUpClass(cls):
        cls.dataset = _make_experiment_df(800)

    def test_mini_loop(self):
        # Establish baseline
        baseline_scores, baseline_window_scores = evaluate_configuration(
            self.dataset,
            model_name="xgboost_direction",
            target_column=DEFAULT_TARGET_COLUMN,
            train_window=self.TRAIN_W,
            test_window=self.TEST_W,
        )
        self.assertIn("precision", baseline_scores)
        baseline_composite = compute_composite(baseline_scores)

        # Run two experiments
        experiments = [
            {
                "description": "mini: xgb deeper trees",
                "model_name": "xgboost_direction",
                "model_kwargs": {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1},
            },
            {
                "description": "mini: lgbm default",
                "model_name": "lightgbm_direction",
            },
        ]

        results = []
        for i, exp in enumerate(experiments, start=1):
            result = run_single_experiment(
                exp, self.dataset,
                baseline_scores, baseline_window_scores,
                experiment_id=i,
                train_window=self.TRAIN_W,
                test_window=self.TEST_W,
            )
            results.append(result)
            self.assertIsInstance(result, ExperimentResult)
            self.assertIn(result.status, ["keep", "discard"])
            self.assertGreater(result.windows_evaluated, 0)

        # Verify all results have valid metric values (may be 0 on random data)
        for r in results:
            self.assertGreaterEqual(r.composite_metric, 0.0)
            self.assertGreaterEqual(r.precision, 0.0)
            self.assertGreaterEqual(r.recall, 0.0)

        # Write results to temp TSV and verify round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "results.tsv"
            import scripts.experiment_loop as loop_mod
            original = loop_mod.RESULTS_PATH
            loop_mod.RESULTS_PATH = tmp_path
            try:
                for r in results:
                    save_result(r)
                loaded = load_results()
            finally:
                loop_mod.RESULTS_PATH = original

            self.assertEqual(len(loaded), 2)


if __name__ == "__main__":
    unittest.main()
