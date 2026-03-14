"""Regression gate: compare a new backtest result against the previous run.

Checks whether key metrics have degraded beyond configurable tolerances.
Returns a verdict with per-metric details and an overall PASS/FAIL.

Exit code convention (when used as a script):
  0 = PASS (no regressions)
  1 = FAIL (at least one regression detected)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import ARTIFACTS_DIR

HISTORY_PATH = ARTIFACTS_DIR / "backtest_history.json"

# Regression tolerance per metric.
# "relative" means allow this fraction of degradation (e.g., 0.05 = 5%).
# "absolute" means allow this fixed drop.
# "direction" is "higher" (higher is better) or "lower" (lower is better).
REGRESSION_TOLERANCES: list[dict] = [
    {"metric": "precision", "direction": "higher", "tolerance": 0.05, "mode": "relative"},
    {"metric": "recall", "direction": "higher", "tolerance": 0.05, "mode": "relative"},
    {"metric": "roc_auc", "direction": "higher", "tolerance": 0.05, "mode": "relative"},
    {"metric": "directional_accuracy", "direction": "higher", "tolerance": 0.05, "mode": "relative"},
    {"metric": "f1", "direction": "higher", "tolerance": 0.10, "mode": "relative"},
]


@dataclass
class GateCheck:
    metric: str
    baseline: float
    current: float
    direction: str
    threshold: float
    passed: bool
    detail: str


@dataclass
class GateVerdict:
    passed: bool
    timestamp: str
    checks: list[dict]
    summary: str


def _load_history(path: Path = HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def run_regression_gate(
    history_path: Path = HISTORY_PATH,
    tolerances: list[dict] | None = None,
) -> GateVerdict:
    """Compare the most recent history entry against the previous one.

    Returns a ``GateVerdict`` with per-metric checks and an overall pass/fail.
    If there is only one entry (no previous baseline), the gate passes
    automatically — there is nothing to regress against.
    """
    tolerances = tolerances or REGRESSION_TOLERANCES
    history = _load_history(history_path)

    if len(history) < 2:
        return GateVerdict(
            passed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks=[],
            summary="No previous baseline to compare against — gate passes by default.",
        )

    current_scores = history[0]["scores"]
    baseline_scores = history[1]["scores"]

    checks: list[dict] = []
    failures: list[str] = []

    for tol in tolerances:
        metric = tol["metric"]
        direction = tol["direction"]
        tolerance_value = tol["tolerance"]
        mode = tol["mode"]

        current = current_scores.get(metric)
        baseline = baseline_scores.get(metric)

        if current is None or baseline is None:
            checks.append(asdict(GateCheck(
                metric=metric,
                baseline=baseline or 0.0,
                current=current or 0.0,
                direction=direction,
                threshold=0.0,
                passed=True,
                detail=f"Skipped — metric not present in both runs.",
            )))
            continue

        # Compute the minimum acceptable value.
        if direction == "higher":
            if mode == "relative":
                threshold = baseline * (1 - tolerance_value)
            else:
                threshold = baseline - tolerance_value
            passed = current >= threshold
        else:  # lower is better
            if mode == "relative":
                threshold = baseline * (1 + tolerance_value)
            else:
                threshold = baseline + tolerance_value
            passed = current <= threshold

        detail_parts = [
            f"{current:.4f} vs threshold {threshold:.4f}",
            f"(baseline {baseline:.4f}",
        ]
        if mode == "relative":
            pct = int(tolerance_value * 100)
            sign = "-" if direction == "higher" else "+"
            detail_parts.append(f"{sign} {pct}%)")
        else:
            sign = "-" if direction == "higher" else "+"
            detail_parts.append(f"{sign} {tolerance_value})")

        detail = " ".join(detail_parts)

        if not passed:
            failures.append(metric)

        checks.append(asdict(GateCheck(
            metric=metric,
            baseline=round(baseline, 4),
            current=round(current, 4),
            direction=direction,
            threshold=round(threshold, 4),
            passed=passed,
            detail=detail,
        )))

    if failures:
        summary = f"REGRESSION DETECTED: {', '.join(failures)}"
    else:
        summary = "No regressions detected."

    return GateVerdict(
        passed=len(failures) == 0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        checks=checks,
        summary=summary,
    )


def print_verdict(verdict: GateVerdict) -> None:
    """Print a human-readable verdict to stdout."""
    status = "PASS" if verdict.passed else "FAIL"
    print(f"\n[REGRESSION GATE] {status}: {verdict.summary}\n")
    for check in verdict.checks:
        icon = "+" if check["passed"] else "x"
        print(f"  [{icon}] {check['metric']}: {check['detail']}")
    print()


def main() -> None:
    """CLI entrypoint for the regression gate."""
    verdict = run_regression_gate()
    print_verdict(verdict)

    # Save verdict to artifacts
    verdict_path = ARTIFACTS_DIR / "regression_gate_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    verdict_path.write_text(
        json.dumps(asdict(verdict), indent=2), encoding="utf-8"
    )

    sys.exit(0 if verdict.passed else 1)


if __name__ == "__main__":
    main()
