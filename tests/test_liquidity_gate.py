"""Tests for the liquidity regime gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.liquidity_gate import (
    DIRECTIONAL_REGIMES,
    apply_directional_gate,
    assign_regimes,
    row_regime,
)


def _make_df(n: int = 100) -> pd.DataFrame:
    """Create a minimal DataFrame with regime columns and a close price."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.RandomState(42)

    # Assign regimes in blocks: 40 EXPANDING, 30 NEUTRAL, 30 CONTRACTING
    expanding = np.zeros(n)
    neutral = np.zeros(n)
    contracting = np.zeros(n)
    expanding[:40] = 1.0
    neutral[40:70] = 1.0
    contracting[70:] = 1.0

    return pd.DataFrame(
        {
            "close": 50000 + rng.randn(n) * 100,
            "liquidity_regime_expanding": expanding,
            "liquidity_regime_neutral": neutral,
            "liquidity_regime_contracting": contracting,
        },
        index=idx,
    )


def _make_predictions(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(99)
    return pd.DataFrame(
        {
            "prediction": rng.randint(0, 2, len(df)),
            "probability": rng.uniform(0.3, 0.7, len(df)),
        },
        index=df.index,
    )


class TestRowRegime:
    def test_expanding(self):
        row = pd.Series({"liquidity_regime_expanding": 1.0, "liquidity_regime_neutral": 0.0, "liquidity_regime_contracting": 0.0})
        assert row_regime(row) == "EXPANDING"

    def test_neutral(self):
        row = pd.Series({"liquidity_regime_expanding": 0.0, "liquidity_regime_neutral": 1.0, "liquidity_regime_contracting": 0.0})
        assert row_regime(row) == "NEUTRAL"

    def test_contracting(self):
        row = pd.Series({"liquidity_regime_expanding": 0.0, "liquidity_regime_neutral": 0.0, "liquidity_regime_contracting": 1.0})
        assert row_regime(row) == "CONTRACTING"

    def test_unknown_when_all_zero(self):
        row = pd.Series({"liquidity_regime_expanding": 0.0, "liquidity_regime_neutral": 0.0, "liquidity_regime_contracting": 0.0})
        assert row_regime(row) == "UNKNOWN"

    def test_unknown_when_missing(self):
        row = pd.Series({"close": 50000})
        assert row_regime(row) == "UNKNOWN"


class TestAssignRegimes:
    def test_vectorized_assignment(self):
        df = _make_df()
        regimes = assign_regimes(df)
        assert len(regimes) == len(df)
        assert (regimes[:40] == "EXPANDING").all()
        assert (regimes[40:70] == "NEUTRAL").all()
        assert (regimes[70:] == "CONTRACTING").all()

    def test_missing_columns_returns_unknown(self):
        df = pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
        regimes = assign_regimes(df)
        assert (regimes == "UNKNOWN").all()


class TestDirectionalGate:
    def test_filters_neutral(self):
        df = _make_df()
        preds = _make_predictions(df)
        gated_a, gated_p, coverage = apply_directional_gate(df, preds)
        # Should exclude 30 NEUTRAL rows
        assert len(gated_a) == 70
        assert len(gated_p) == 70
        assert coverage == pytest.approx(0.70)

    def test_indices_match(self):
        df = _make_df()
        preds = _make_predictions(df)
        gated_a, gated_p, _ = apply_directional_gate(df, preds)
        pd.testing.assert_index_equal(gated_a.index, gated_p.index)

    def test_custom_regimes(self):
        df = _make_df()
        preds = _make_predictions(df)
        gated_a, _, coverage = apply_directional_gate(df, preds, allowed_regimes=frozenset({"EXPANDING"}))
        assert len(gated_a) == 40
        assert coverage == pytest.approx(0.40)

    def test_empty_when_no_allowed(self):
        df = _make_df()
        preds = _make_predictions(df)
        gated_a, gated_p, coverage = apply_directional_gate(df, preds, allowed_regimes=frozenset())
        assert len(gated_a) == 0
        assert coverage == 0.0

    def test_all_through_when_all_allowed(self):
        df = _make_df()
        preds = _make_predictions(df)
        all_regimes = frozenset({"EXPANDING", "NEUTRAL", "CONTRACTING", "UNKNOWN"})
        gated_a, _, coverage = apply_directional_gate(df, preds, allowed_regimes=all_regimes)
        assert len(gated_a) == len(df)
        assert coverage == 1.0
