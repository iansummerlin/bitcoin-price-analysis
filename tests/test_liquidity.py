"""Tests for liquidity artifact loader and dataset integration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.liquidity import (
    LIQUIDITY_FEATURE_COLUMNS,
    load_liquidity_artifact,
    merge_liquidity_features,
    _empty_frame,
)


def _make_valid_artifact() -> dict:
    """Minimal valid liquidity artifact for testing."""
    return {
        "schema_version": "1.0.0",
        "generated_at": "2026-03-20T12:00:00Z",
        "last_data_date": "2026-01-31",
        "data_lag_days": 48,
        "regime": "EXPANDING",
        "m2_momentum_3m": 0.023,
        "m2_momentum_1m": 0.008,
        "m2_acceleration": 0.002,
        "global_liquidity_latest_usd_trillions": 108.5,
        "components": {},
        "sources_included": ["us_m2", "fed_bs"],
        "sources_missing": ["boj_bs"],
        "time_series": [
            {"date": "2024-01-31", "global_liquidity_usd_t": 85.2, "m2_roc_3m": 0.012, "regime": "EXPANDING"},
            {"date": "2024-02-29", "global_liquidity_usd_t": 86.0, "m2_roc_3m": 0.015, "regime": "EXPANDING"},
            {"date": "2024-03-31", "global_liquidity_usd_t": 84.5, "m2_roc_3m": -0.008, "regime": "CONTRACTING"},
            {"date": "2024-04-30", "global_liquidity_usd_t": 85.0, "m2_roc_3m": 0.003, "regime": "NEUTRAL"},
        ],
        "is_stale": False,
        "stale_after_days": 14,
    }


def _make_stale_artifact() -> dict:
    artifact = _make_valid_artifact()
    artifact["is_stale"] = True
    artifact["data_lag_days"] = 30
    return artifact


def _make_hourly_btc(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2024-02-01", periods=n, freq="h", tz="UTC")
    np.random.seed(42)
    close = 60000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({"close": close}, index=idx)


# --- load_liquidity_artifact ---

class TestLoadArtifact:
    def test_valid_artifact(self, tmp_path):
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(_make_valid_artifact()))
        df, meta = load_liquidity_artifact(path)
        assert not df.empty
        assert len(df) == 4
        for col in LIQUIDITY_FEATURE_COLUMNS:
            assert col in df.columns
        assert meta["schema_version"] == "1.0.0"

    def test_missing_artifact(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        df, meta = load_liquidity_artifact(path)
        assert df.empty
        assert meta == {}
        assert list(df.columns) == LIQUIDITY_FEATURE_COLUMNS

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{broken")
        df, meta = load_liquidity_artifact(path)
        assert df.empty
        assert meta == {}

    def test_wrong_schema_version(self, tmp_path):
        artifact = _make_valid_artifact()
        artifact["schema_version"] = "99.0.0"
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(artifact))
        df, meta = load_liquidity_artifact(path)
        assert df.empty

    def test_stale_artifact_still_loads(self, tmp_path):
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(_make_stale_artifact()))
        df, meta = load_liquidity_artifact(path)
        assert not df.empty
        assert meta["is_stale"] is True

    def test_empty_time_series(self, tmp_path):
        artifact = _make_valid_artifact()
        artifact["time_series"] = []
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(artifact))
        df, meta = load_liquidity_artifact(path)
        assert df.empty
        assert meta["schema_version"] == "1.0.0"

    def test_regime_one_hot_encoding(self, tmp_path):
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(_make_valid_artifact()))
        df, _ = load_liquidity_artifact(path)
        # First two entries are EXPANDING
        assert df["liquidity_regime_expanding"].iloc[0] == 1.0
        assert df["liquidity_regime_neutral"].iloc[0] == 0.0
        assert df["liquidity_regime_contracting"].iloc[0] == 0.0
        # Third entry is CONTRACTING
        assert df["liquidity_regime_contracting"].iloc[2] == 1.0
        # Fourth is NEUTRAL
        assert df["liquidity_regime_neutral"].iloc[3] == 1.0


# --- merge_liquidity_features ---

class TestMergeLiquidity:
    def test_merge_with_data(self, tmp_path):
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(_make_valid_artifact()))
        liq_df, _ = load_liquidity_artifact(path)
        hourly = _make_hourly_btc()
        result = merge_liquidity_features(hourly, liq_df)
        for col in LIQUIDITY_FEATURE_COLUMNS:
            assert col in result.columns
        # Should have same index as input
        pd.testing.assert_index_equal(result.index, hourly.index)
        # Should have original columns preserved
        assert "close" in result.columns

    def test_merge_empty_liquidity(self):
        hourly = _make_hourly_btc()
        result = merge_liquidity_features(hourly, _empty_frame())
        for col in LIQUIDITY_FEATURE_COLUMNS:
            assert col in result.columns
            assert result[col].isna().all()

    def test_forward_fill_across_hours(self, tmp_path):
        path = tmp_path / "artifact.json"
        path.write_text(json.dumps(_make_valid_artifact()))
        liq_df, _ = load_liquidity_artifact(path)
        hourly = _make_hourly_btc()
        result = merge_liquidity_features(hourly, liq_df)
        # Values should be forward-filled, so not all NaN
        non_nan = result["liquidity_global_usd_t"].dropna()
        assert len(non_nan) > 0


# --- empty_frame ---

class TestEmptyFrame:
    def test_has_correct_columns(self):
        df = _empty_frame()
        assert list(df.columns) == LIQUIDITY_FEATURE_COLUMNS
        assert df.empty
