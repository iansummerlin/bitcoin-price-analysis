"""Optional liquidity regime gate for post-prediction signal filtering.

Suppresses model predictions during NEUTRAL liquidity regimes, allowing
signals only when the macro liquidity cycle is directional (EXPANDING or
CONTRACTING). Applied post-prediction — no model retraining needed.

This is a research-time filter, not a default. It is never applied
automatically; callers must explicitly opt in.
"""

from __future__ import annotations

import pandas as pd

# Regimes allowed through the directional gate.
DIRECTIONAL_REGIMES = frozenset({"EXPANDING", "CONTRACTING"})


def row_regime(row: pd.Series) -> str:
    """Return the liquidity regime label for a single row."""
    if row.get("liquidity_regime_expanding", 0) == 1.0:
        return "EXPANDING"
    if row.get("liquidity_regime_contracting", 0) == 1.0:
        return "CONTRACTING"
    if row.get("liquidity_regime_neutral", 0) == 1.0:
        return "NEUTRAL"
    return "UNKNOWN"


def assign_regimes(df: pd.DataFrame) -> pd.Series:
    """Vectorized regime assignment for all rows in a DataFrame."""
    regime = pd.Series("UNKNOWN", index=df.index, dtype="object")
    if "liquidity_regime_expanding" in df.columns:
        regime = regime.where(df["liquidity_regime_expanding"] != 1.0, "EXPANDING")
    if "liquidity_regime_contracting" in df.columns:
        regime = regime.where(df["liquidity_regime_contracting"] != 1.0, "CONTRACTING")
    if "liquidity_regime_neutral" in df.columns:
        regime = regime.where(df["liquidity_regime_neutral"] != 1.0, "NEUTRAL")
    return regime


def apply_directional_gate(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    allowed_regimes: frozenset[str] = DIRECTIONAL_REGIMES,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Filter actuals and predictions to rows where regime is allowed.

    Returns
    -------
    gated_actuals : pd.DataFrame
        Subset of actuals in allowed regimes.
    gated_predictions : pd.DataFrame
        Subset of predictions in allowed regimes.
    coverage : float
        Fraction of rows that pass the gate (0.0–1.0).
    """
    regimes = assign_regimes(actuals)
    mask = regimes.isin(allowed_regimes)
    coverage = mask.sum() / len(mask) if len(mask) > 0 else 0.0
    return actuals.loc[mask], predictions.loc[mask], coverage
