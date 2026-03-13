"""CLI wrapper for the model comparison report."""

from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.model_comparison import run_model_comparison


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = run_model_comparison("artifacts")
    print(f"winner={result.winner}")
    print(result.conclusion)
