"""Append-only backtest history and auto-generated markdown reports.

Every call to ``save_backtest_report`` appends the result to
``artifacts/backtest_history.json`` (max entries capped) and regenerates
``BACKTEST.md`` from the full history.  The JSON file is the source of
truth — the markdown is derived and should never be edited by hand.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from config import ARTIFACTS_DIR, ROOT_DIR
from evaluation.model_comparison import ACCEPTANCE_THRESHOLDS
from evaluation.walk_forward import WalkForwardResult

MAX_HISTORY_ENTRIES = 10
HISTORY_PATH = ARTIFACTS_DIR / "backtest_history.json"
BACKTEST_MD_PATH = ROOT_DIR / "BACKTEST.md"

# Metrics shown in the history table (order matters).
_TABLE_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("roc_auc", "ROC-AUC"),
    ("directional_accuracy", "Dir Acc"),
    ("f1", "F1"),
]

# Acceptance gate checks: (metric_key, operator, threshold, label).
_GATE_CHECKS: list[tuple[str, str, float, str]] = [
    ("precision", ">=", ACCEPTANCE_THRESHOLDS["directional_precision_min"], "Precision"),
    ("recall", ">=", ACCEPTANCE_THRESHOLDS["directional_recall_min"], "Recall"),
    ("roc_auc", ">=", ACCEPTANCE_THRESHOLDS["directional_roc_auc_min"], "ROC-AUC"),
]


def _atomic_write(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        os.close(fd) if not os.get_inheritable(fd) else None  # pragma: no cover
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_history(entries: list[dict], path: Path = HISTORY_PATH) -> None:
    _atomic_write(path, json.dumps(entries, indent=2))


def _gate_results(scores: dict[str, float]) -> list[dict]:
    """Evaluate acceptance gates and return a list of check dicts."""
    results = []
    for key, op, threshold, label in _GATE_CHECKS:
        value = scores.get(key, 0.0)
        if op == ">=":
            passed = value >= threshold
        else:
            passed = value <= threshold
        results.append({
            "name": label,
            "value": round(value, 4),
            "op": op,
            "target": threshold,
            "passed": passed,
        })
    return results


def save_backtest_report(
    result: WalkForwardResult,
    *,
    notes: str = "",
    dataset_rows: int | None = None,
    dataset_start: str | None = None,
    dataset_end: str | None = None,
    history_path: Path = HISTORY_PATH,
    markdown_path: Path = BACKTEST_MD_PATH,
) -> Path:
    """Append *result* to the history file and regenerate the markdown report.

    Returns the path to the updated markdown file.
    """
    entry = {
        "timestamp": result.generated_at,
        "model_name": result.model_name,
        "target_column": result.target_column,
        "train_window": result.train_window,
        "test_window": result.test_window,
        "windows_evaluated": result.windows_evaluated,
        "scores": {k: round(v, 4) for k, v in result.scores.items()},
        "baseline_scores": {
            name: {k: round(v, 4) for k, v in metrics.items()}
            for name, metrics in result.baseline_scores.items()
        },
        "gate": _gate_results(result.scores),
        "notes": notes,
    }
    if dataset_rows is not None:
        entry["dataset_rows"] = dataset_rows
    if dataset_start is not None:
        entry["dataset_start"] = dataset_start
    if dataset_end is not None:
        entry["dataset_end"] = dataset_end

    history = _load_history(history_path)
    history.insert(0, entry)
    history = history[:MAX_HISTORY_ENTRIES]
    _save_history(history, history_path)

    md = _generate_markdown(history)
    _atomic_write(markdown_path, md)
    return markdown_path


def _generate_markdown(history: list[dict]) -> str:
    """Build the full BACKTEST.md content from the history list."""
    lines: list[str] = []
    lines.append("# Backtest Results\n")
    lines.append("*Auto-generated from `artifacts/backtest_history.json` — do not edit manually.*\n")

    if not history:
        lines.append("\nNo backtest results recorded yet.\n")
        return "\n".join(lines)

    latest = history[0]
    scores = latest["scores"]
    gate = latest["gate"]

    # --- Latest results ---
    lines.append("\n## Latest Results\n")
    lines.append(f"**Run:** {latest['timestamp']}  ")
    lines.append(f"**Model:** {latest['model_name']}  ")
    lines.append(f"**Target:** {latest['target_column']}  ")
    lines.append(f"**Windows:** {latest['windows_evaluated']}  ")
    if latest.get("notes"):
        lines.append(f"**Notes:** {latest['notes']}  ")
    lines.append("")

    # Metrics table
    lines.append("### Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, label in _TABLE_METRICS:
        val = scores.get(key)
        if val is not None:
            lines.append(f"| {label} | {val:.4f} |")
    lines.append("")

    # Acceptance gate
    all_passed = all(g["passed"] for g in gate)
    status = "PASS" if all_passed else "FAIL"
    lines.append("### Acceptance Gate\n")
    lines.append(f"**Overall: {status}**\n")
    lines.append("| Check | Value | Target | Result |")
    lines.append("|-------|-------|--------|--------|")
    for g in gate:
        result_str = "PASS" if g["passed"] else "FAIL"
        lines.append(f"| {g['name']} | {g['value']:.4f} | {g['op']} {g['target']:.4f} | {result_str} |")
    lines.append("")

    # --- History table ---
    if len(history) > 1:
        lines.append("## History\n")
        header_cols = ["#", "Date", "Model"]
        header_cols += [label for _, label in _TABLE_METRICS]
        header_cols.append("Gate")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

        for idx, entry in enumerate(history, start=1):
            s = entry["scores"]
            g = entry.get("gate", [])
            gate_pass = all(c["passed"] for c in g) if g else "?"
            gate_str = "PASS" if gate_pass is True else ("FAIL" if gate_pass is False else "?")
            ts = entry["timestamp"][:19]
            cols = [str(idx), ts, entry["model_name"]]
            for key, _ in _TABLE_METRICS:
                v = s.get(key)
                cols.append(f"{v:.4f}" if v is not None else "-")
            cols.append(gate_str)
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")

    return "\n".join(lines) + "\n"
