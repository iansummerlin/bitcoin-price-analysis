"""Tests for backtest history, markdown generation, and regression gate."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.history import (
    MAX_HISTORY_ENTRIES,
    _generate_markdown,
    _load_history,
    _save_history,
    save_backtest_report,
)
from evaluation.regression_gate import GateVerdict, run_regression_gate
from evaluation.walk_forward import WalkForwardResult


def _make_result(**overrides) -> WalkForwardResult:
    defaults = dict(
        model_name="xgboost_direction",
        target_column="target_direction_cost_adj",
        train_window=4320,
        test_window=720,
        windows_evaluated=4,
        scores={
            "directional_accuracy": 0.72,
            "precision": 0.17,
            "recall": 0.26,
            "f1": 0.18,
            "roc_auc": 0.64,
        },
        baseline_scores={
            "momentum_direction": {"directional_accuracy": 0.50, "precision": 0.10},
        },
        prediction_path="/tmp/predictions.csv",
        generated_at="2026-03-14T12:00:00+00:00",
    )
    defaults.update(overrides)
    return WalkForwardResult(**defaults)


class TestHistoryAppend(unittest.TestCase):
    def test_save_creates_history_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            md_path = Path(tmp) / "BACKTEST.md"

            result = _make_result()
            save_backtest_report(
                result,
                history_path=hist_path,
                markdown_path=md_path,
            )

            self.assertTrue(hist_path.exists())
            self.assertTrue(md_path.exists())

            history = json.loads(hist_path.read_text())
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["model_name"], "xgboost_direction")

    def test_history_prepends_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            md_path = Path(tmp) / "BACKTEST.md"

            r1 = _make_result(generated_at="2026-03-14T10:00:00+00:00")
            r2 = _make_result(generated_at="2026-03-14T12:00:00+00:00")

            save_backtest_report(r1, history_path=hist_path, markdown_path=md_path)
            save_backtest_report(r2, history_path=hist_path, markdown_path=md_path)

            history = json.loads(hist_path.read_text())
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]["timestamp"], "2026-03-14T12:00:00+00:00")
            self.assertEqual(history[1]["timestamp"], "2026-03-14T10:00:00+00:00")

    def test_history_caps_at_max_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            md_path = Path(tmp) / "BACKTEST.md"

            for i in range(MAX_HISTORY_ENTRIES + 3):
                r = _make_result(generated_at=f"2026-03-{i+1:02d}T00:00:00+00:00")
                save_backtest_report(r, history_path=hist_path, markdown_path=md_path)

            history = json.loads(hist_path.read_text())
            self.assertEqual(len(history), MAX_HISTORY_ENTRIES)

    def test_gate_results_included_in_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            md_path = Path(tmp) / "BACKTEST.md"

            result = _make_result()
            save_backtest_report(result, history_path=hist_path, markdown_path=md_path)

            history = json.loads(hist_path.read_text())
            gate = history[0]["gate"]
            self.assertIsInstance(gate, list)
            self.assertGreater(len(gate), 0)
            self.assertIn("name", gate[0])
            self.assertIn("passed", gate[0])

    def test_notes_and_dataset_info_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            md_path = Path(tmp) / "BACKTEST.md"

            result = _make_result()
            save_backtest_report(
                result,
                notes="test run",
                dataset_rows=91000,
                dataset_start="2015-10-13",
                dataset_end="2026-03-13",
                history_path=hist_path,
                markdown_path=md_path,
            )

            history = json.loads(hist_path.read_text())
            self.assertEqual(history[0]["notes"], "test run")
            self.assertEqual(history[0]["dataset_rows"], 91000)


class TestMarkdownGeneration(unittest.TestCase):
    def test_empty_history_produces_placeholder(self):
        md = _generate_markdown([])
        self.assertIn("No backtest results", md)

    def test_single_entry_shows_latest_and_no_history_table(self):
        entry = {
            "timestamp": "2026-03-14T12:00:00+00:00",
            "model_name": "xgboost_direction",
            "target_column": "target_direction_cost_adj",
            "windows_evaluated": 4,
            "scores": {"precision": 0.17, "recall": 0.26, "roc_auc": 0.64},
            "gate": [
                {"name": "Precision", "value": 0.17, "op": ">=", "target": 0.55, "passed": False},
                {"name": "Recall", "value": 0.26, "op": ">=", "target": 0.15, "passed": True},
            ],
            "notes": "",
        }
        md = _generate_markdown([entry])
        self.assertIn("Latest Results", md)
        self.assertIn("FAIL", md)
        self.assertNotIn("## History", md)

    def test_two_entries_shows_history_table(self):
        entry = {
            "timestamp": "2026-03-14T12:00:00+00:00",
            "model_name": "xgboost_direction",
            "target_column": "target_direction_cost_adj",
            "windows_evaluated": 4,
            "scores": {"precision": 0.17, "recall": 0.26, "roc_auc": 0.64},
            "gate": [
                {"name": "Precision", "value": 0.17, "op": ">=", "target": 0.55, "passed": False},
            ],
            "notes": "",
        }
        md = _generate_markdown([entry, entry])
        self.assertIn("## History", md)

    def test_markdown_not_editable_warning(self):
        md = _generate_markdown([])
        self.assertIn("do not edit manually", md)


class TestRegressionGate(unittest.TestCase):
    def test_no_history_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            verdict = run_regression_gate(history_path=Path(tmp) / "nope.json")
            self.assertTrue(verdict.passed)

    def test_single_entry_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            hist_path.write_text(json.dumps([{
                "scores": {"precision": 0.17, "recall": 0.26, "roc_auc": 0.64},
            }]))
            verdict = run_regression_gate(history_path=hist_path)
            self.assertTrue(verdict.passed)

    def test_no_regression_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            hist_path.write_text(json.dumps([
                {"scores": {"precision": 0.20, "recall": 0.30, "roc_auc": 0.65}},
                {"scores": {"precision": 0.17, "recall": 0.26, "roc_auc": 0.64}},
            ]))
            verdict = run_regression_gate(history_path=hist_path)
            self.assertTrue(verdict.passed)
            self.assertIn("No regressions", verdict.summary)

    def test_regression_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            # Current (entry 0) is much worse than baseline (entry 1)
            hist_path.write_text(json.dumps([
                {"scores": {"precision": 0.05, "recall": 0.10, "roc_auc": 0.50}},
                {"scores": {"precision": 0.20, "recall": 0.30, "roc_auc": 0.65}},
            ]))
            verdict = run_regression_gate(history_path=hist_path)
            self.assertFalse(verdict.passed)
            self.assertIn("REGRESSION DETECTED", verdict.summary)

    def test_within_tolerance_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            # 3% drop on precision — within 5% relative tolerance
            hist_path.write_text(json.dumps([
                {"scores": {"precision": 0.194, "recall": 0.26, "roc_auc": 0.64}},
                {"scores": {"precision": 0.200, "recall": 0.26, "roc_auc": 0.64}},
            ]))
            verdict = run_regression_gate(history_path=hist_path)
            self.assertTrue(verdict.passed)

    def test_custom_tolerances(self):
        with tempfile.TemporaryDirectory() as tmp:
            hist_path = Path(tmp) / "history.json"
            hist_path.write_text(json.dumps([
                {"scores": {"precision": 0.18}},
                {"scores": {"precision": 0.20}},
            ]))
            # Zero tolerance — any drop fails
            strict = [{"metric": "precision", "direction": "higher", "tolerance": 0.0, "mode": "relative"}]
            verdict = run_regression_gate(history_path=hist_path, tolerances=strict)
            self.assertFalse(verdict.passed)


if __name__ == "__main__":
    unittest.main()
