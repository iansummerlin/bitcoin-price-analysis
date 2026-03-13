"""CLI wrapper for the feature-family ablation report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.ablation import run_ablation_report


if __name__ == "__main__":
    result = run_ablation_report("artifacts")
    print(result.scores)
