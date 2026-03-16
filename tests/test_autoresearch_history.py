import json
import tempfile
import unittest
from pathlib import Path

from scripts.experiment_loop import (
    generate_autoresearch_history_md,
    load_autoresearch_history,
    save_autoresearch_history,
)


class TestAutoresearchHistory(unittest.TestCase):
    def test_load_history_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            self.assertEqual(load_autoresearch_history(path), [])

    def test_save_and_load_history_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            entries = [{"run_id": "run-1", "held_out_scores": {"precision": 0.4}}]

            save_autoresearch_history(entries, path)
            loaded = load_autoresearch_history(path)

            self.assertEqual(loaded, entries)

    def test_generate_history_markdown_empty(self):
        md = generate_autoresearch_history_md([])
        self.assertIn("No autoresearch runs recorded yet.", md)

    def test_generate_history_markdown_with_entries(self):
        history = [
            {
                "run_id": "2026-03-16T01:00:00Z",
                "generated_at": "2026-03-16T06:00:00+00:00",
                "best_experiment_config": "candidate: decision_threshold=0.70",
                "held_out_scores": {
                    "precision": 0.41,
                    "recall": 0.16,
                    "roc_auc": 0.70,
                },
                "gate_7": {"overall": False},
                "stopped_early": False,
            }
        ]

        md = generate_autoresearch_history_md(history)

        self.assertIn("Autoresearch History", md)
        self.assertIn("candidate: decision_threshold=0.70", md)
        self.assertIn("0.4100", md)
        self.assertIn("FAIL", md)


if __name__ == "__main__":
    unittest.main()
